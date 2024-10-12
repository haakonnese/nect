import numpy as np
import tigre
import torch
from leaptorch import Projector


class TigreGeometry:
    """
    mode: ["cone", "parallel"]
    nVoxel: [Z, Y, X], (int, int, int)
    DSD: Distance Source Detector (mm) (float)
    DSO: Distance Source Origin (mm) (float)
    nDetector: [Z, Y/X], number of pixels (px) (int, int)
    dDetector: [Z, Y/X], size of each pixel (mm) (float, float)
    sDetector: [Z, Y/X], total size of detector (mm) (float, float)
    dVoxel: [Z, Y, X], Effective voxel pitch (mm) (float, float, float)
    sVoxel: [Z, Y, X], total size of image (mm) (float, float, float)
    """

    def __init__(self, default=True, *cfg, **kwargs):
        self.valid_keys = [
            "mode",
            "nVoxel",
            "DSD",
            "DSO",
            "nDetector",
            "dDetector",
            "sDetector",
            "dVoxel",
            "sVoxel",
            "offOrigin",
            "offDetector",
            "accuracy",
            "COR",
            "rotDetector",
            "default",
        ]
        self.geo = tigre.geometry()
        if default:
            self.set_default_geometry()
        for dictionary in cfg:
            for key in dictionary:
                if self.is_valid_key(key):
                    setattr(self.geo, key, dictionary[key])
        for key in kwargs:
            if self.is_valid_key(key):
                setattr(self.geo, key, kwargs[key])

    def is_valid_key(self, key):
        if key in self.valid_keys:
            return True
        else:
            print(f"[WARNING] Key {key} is not valid. Valid keys are: {self.valid_keys}")
            return False

    def set_default_geometry(self):
        self.geo.mode = "cone"
        self.geo.nVoxel = np.array([64, 128, 128])
        self.geo.DSD = 1350
        self.geo.DSO = 930
        self.geo.nDetector = np.array([64, 128])
        self.geo.dDetector = np.array([self.geo.DSD / self.geo.DSO, self.geo.DSD / self.geo.DSO])
        self.geo.sDetector = self.geo.nDetector * self.geo.dDetector
        self.geo.dVoxel = np.array(
            [
                self.geo.dDetector[0] * self.geo.DSO / self.geo.DSD,
                self.geo.dDetector[1] * self.geo.DSO / self.geo.DSD,
                self.geo.dDetector[1] * self.geo.DSO / self.geo.DSD,
            ]
        )
        self.geo.sVoxel = self.geo.dVoxel * self.geo.nVoxel
        self.geo.offOrigin = np.array([0, 0, 0])
        self.geo.offDetector = np.array([0, 0])
        self.geo.accuracy = 0.5
        self.geo.COR = 0
        self.geo.rotDetector = np.array([0, 0, 0])

    def print_params(self):
        print(self.geo)

    def from_yaml(self, yaml_dict):
        self.geo.mode = yaml_dict["mode"]
        self.geo.nVoxel = np.array(yaml_dict["nVoxel"])
        self.geo.DSD = yaml_dict["DSD"]
        self.geo.DSO = yaml_dict["DSO"]
        self.geo.nDetector = np.array(yaml_dict["nDetector"])
        self.geo.dDetector = np.array(yaml_dict["dDetector"])
        self.geo.sDetector = np.array(yaml_dict["sDetector"])
        self.geo.dVoxel = np.array(yaml_dict["dVoxel"])
        self.geo.sVoxel = np.array(yaml_dict["sVoxel"])
        self.geo.offOrigin = np.array(yaml_dict["offOrigin"])
        self.geo.offDetector = np.array(yaml_dict["offDetector"])
        self.geo.accuracy = yaml_dict["accuracy"]
        self.geo.COR = yaml_dict["COR"]
        self.geo.rotDetector = np.array(yaml_dict["rotDetector"]) if "rotDetector" in yaml_dict else np.array([0, 0, 0])


class LeapGeometry:
    def __init__(
        self,
        default: bool = True,
        device: torch.device = torch.device("cuda:0")
        if torch.cuda.is_available()
        else torch.device("cpu"),  # TODO: Update to "cuda" when supported by LEAP
        forward_project: bool = True,
        use_static: bool = False,
        *cfg,
        **kwargs,
    ):
        self.valid_keys = [
            "mode",
            "numX",
            "numY",
            "numZ",
            "width",
            "height",
            "offsetX",
            "offsetY",
            "offsetZ",
            "numAngles",
            "numRows",
            "numCols",
            "pixelWidth",
            "pixelHeight",
            "centerRow",
            "centerCol",
            "arange",
            "phis",
            "sod",
            "sdd",
            "default",
            "device",
        ]
        self.device = device
        self.use_gpu = False if self.device.type == "cpu" else True
        self.forward_project = forward_project
        self.use_static = use_static
        if default:
            self.default_projector()
        else:
            self.proj = Projector(
                use_gpu=self.use_gpu,
                gpu_device=self.device,
                forward_project=self.forward_project,
                use_static=self.use_static,
            )

        # Check if input names are valid
        for dictionary in cfg:
            for key in dictionary:
                if self.is_valid_key(key):
                    setattr(self, key, dictionary[key])
        for key in kwargs:
            if self.is_valid_key(key):
                setattr(self, key, kwargs[key])

        self.update_projector()

    def __call__(self, volume):
        return self.proj(volume)

    def print_params(self):
        self.proj.print_param()

    def is_valid_key(self, key):
        if key in self.valid_keys:
            return True
        else:
            print(f"[WARNING] Key {key} is not valid. Valid keys are: {self.valid_keys}")
            return False

    def update_projector(self):
        self.proj.set_volume(
            numX=self.numX,
            numY=self.numY,
            numZ=self.numZ,
            voxelWidth=self.voxelWidth,
            voxelHeight=self.voxelHeight,
            offsetX=self.offsetX,
            offsetY=self.offsetY,
            offsetZ=self.offsetZ,
        )
        self.proj.set_conebeam(
            numAngles=self.numAngles,
            numRows=self.numRows,
            numCols=self.numCols,
            pixelHeight=self.pixelHeight,
            pixelWidth=self.pixelWidth,
            centerRow=self.centerRow,
            centerCol=self.centerCol,
            phis=self.phis,
            sod=self.sod,
            sdd=self.sdd,
        )

    def update_phi(self, phi: torch.Tensor) -> None:
        """Phi should be a tensor and in degrees"""
        if len(phi.shape) == 0:
            numAngles = 1
        else:
            numAngles = int(phi.shape[0])
        self.proj.set_conebeam(
            numAngles=numAngles,
            numRows=self.numRows,
            numCols=self.numCols,
            pixelHeight=self.pixelHeight,
            pixelWidth=self.pixelWidth,
            centerRow=self.centerRow,
            centerCol=self.centerCol,
            phis=phi,
            sod=self.sod,
            sdd=self.sdd,
        )

    def default_projector(self):
        self.numX = 511
        self.numY = 511
        self.numZ = 255
        self.voxelWidth = 1
        self.voxelHeight = 1
        self.offsetX = 0
        self.offsetY = 0
        self.offsetZ = 0
        self.numAngles = 1
        self.sdd = 1350
        self.sod = 930
        self.numRows = int(self.numZ * 2 * self.sdd / self.sod)
        self.numCols = int(max(self.numX, self.numY) * 2 * self.sdd / self.sod)
        self.pixelHeight = 1 / 2 + 0.025
        self.pixelWidth = 1 / 2 + 0.025
        self.arange = 360
        self.centerRow = 0.5 * float(self.numRows - 1)
        self.centerCol = 0.5 * float(self.numCols - 1)
        self.proj = Projector(
            use_gpu=self.use_gpu,
            gpu_device=self.device,
            forward_project=self.forward_project,
            use_static=self.use_static,
        )
        self.proj.set_volume(
            numX=self.numX,
            numY=self.numY,
            numZ=self.numZ,
            voxelWidth=self.voxelWidth,
            voxelHeight=self.voxelHeight,
            offsetX=self.offsetX,
            offsetY=self.offsetY,
            offsetZ=self.offsetZ,
        )
        self.phis = torch.tensor([0]).float()
        self.proj.set_conebeam(
            numAngles=self.numAngles,
            numRows=self.numRows,
            numCols=self.numCols,
            pixelHeight=self.pixelHeight,
            pixelWidth=self.pixelWidth,
            centerRow=self.centerRow,
            centerCol=self.centerCol,
            phis=self.phis,
            sod=self.sod,
            sdd=self.sdd,
        )

    def to_dict(self):
        geo = {
            "DSD": self.sdd,
            "DSO": self.sod,
            "nDetector": [self.numRows, self.numCols],
            "dDetector": [self.pixelHeight, self.pixelWidth],
            "sDetector": [
                self.numRows * self.pixelHeight,
                self.numCols * self.pixelWidth,
            ],
            "nVoxel": [self.numZ, self.numY, self.numX],
            "dVoxel": [self.voxelHeight, self.voxelWidth, self.voxelWidth],
            "sVoxel": [
                self.numZ * self.voxelHeight,
                self.numY * self.voxelWidth,
                self.numX * self.voxelWidth,
            ],
            "offOrigin": [self.offsetZ, self.offsetY, self.offsetX],
            "offDetector": [0, 0],
            "accuracy": 0.5,
            "COR": 0,
            "mode": "cone",
            "filter": None,
        }
        return geo

    def from_tigre_geometry(self, geo):
        self.numX = geo.nVoxel[1]
        self.numY = geo.nVoxel[2]
        self.numZ = geo.nVoxel[0]
        self.voxelWidth = geo.dVoxel[1]
        assert geo.dVoxel[1] == geo.dVoxel[2], "Voxel width must be a single number"
        self.voxelHeight = geo.dVoxel[0]
        self.offsetX = geo.offOrigin[2]
        self.offsetY = geo.offOrigin[1]
        self.offsetZ = geo.offOrigin[0]

        self.numRows = geo.nDetector[0]
        self.numCols = geo.nDetector[1]
        self.pixelHeight = geo.dDetector[0]
        self.pixelWidth = geo.dDetector[1]
        self.centerRow = 0.5 * float(self.numRows - 1)
        self.centerCol = 0.5 * float(self.numCols - 1)
        self.sod = geo.DSO
        self.sdd = geo.DSD

        self.update_projector()
