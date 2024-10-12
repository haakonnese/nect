import napari
import numpy as np
from skimage import data
from qtpy.QtWidgets import QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFileDialog, QLineEdit, QSlider, QComboBox, QDoubleSpinBox
from qtpy.QtGui import QIntValidator, QDoubleValidator
import tifffile as tif
from tqdm import tqdm
from pathlib import Path
import torch

class  GUI():
    def __init__(self, dummy = False):
        self.viewer = napari.Viewer()
        self.dummy = dummy
        if not dummy:
            from nect.config import get_cfg
            base_path = Path(QFileDialog.getExistingDirectory(caption="Select Directory of model"))
            self.config = get_cfg(base_path / "config.yaml")
            assert self.config.mode == "dynamic", "Only dynamic mode is supported for video creation"
            assert self.config.geometry is not None
            self.model = self.config.get_model()
            device = torch.device(0)
            print(f"Loading model from {base_path / 'checkpoints' / 'last.ckpt'}. This may take a while.")
            print("If the model is large, it may appear that the GUI is frozen.")
            checkpoints = torch.load(base_path / "checkpoints" / "last.ckpt", map_location="cpu")
            print("Finished loading model")
            self.model.load_state_dict(checkpoints["model"])
            del checkpoints
            self.model = self.model.to(device)
            if self.config.geometry.timesteps is not None:
                self.timesteps = self.config.geometry.timesteps
                if isinstance(self.timesteps, torch.Tensor):
                    self.timesteps = self.timesteps.cpu().numpy().tolist()
                elif isinstance(self.timesteps, np.ndarray):
                    self.timesteps = self.timesteps.tolist()
                self.num_frames = len(list(set(self.timesteps)))
                self.timesteps = sorted(list(set(self.timesteps)))
                if max(self.timesteps) > 1:
                    self.timesteps = np.array(self.timesteps) / max(self.timesteps)
            else:
                self.num_frames = len(self.config.geometry.angles)
                self.timesteps = np.linspace(0, 1, self.num_frames)
                
            self.height, self.width = self.config.geometry.nVoxel[0], self.config.geometry.nVoxel[1]
        else:
            self.height, self.width = 256, 256
            self.num_frames = 100
            self.timesteps = np.linspace(0, 1, self.num_frames)

        self.current_time_point = 0
        self.difference = False
        self.difference_volume = None
        self.volume = None
        self.default_binning_factor = 5
        self.base_volume_shape = (self.height, self.width, self.width)
        self.layer = self.viewer.add_image(np.zeros(self.base_volume_shape), name='3D Volume', scale=[1, 1, 1], blending='additive', rendering='iso', opacity=0.25)

        self.updated_state = {
            "x_min": 0,
            "x_max": self.width,
            "y_min": 0,
            "y_max": self.height,
            "z_min": 0,
            "z_max": self.height,
            "binning_factor": self.default_binning_factor-1,
            "difference": self.difference,
            "self.current_time_point": self.current_time_point
        }
        self.updated_state_diff = {
            "x_min": -1,  # Just set this to something wrong so that it updates
            "x_max": self.width,
            "y_min": -1,
            "y_max": self.height,
            "z_min": -1,
            "z_max": self.height,
            "binning_factor": self.default_binning_factor-1,
        }
        self.control_widget = QWidget()
        self.control_layout = QVBoxLayout()
        self.control_widget.setLayout(self.control_layout)

        self.time_label = QLabel(f'Time Point:')
        self.time_hbox = QHBoxLayout()
        self.time_hbox.addWidget(self.time_label)
        self.time_slider = QSlider()
        self.time_slider.setOrientation(1)
        self.time_slider.setRange(0, self.num_frames - 1)
        self.time_slider.setValue(0)
        self.time_slider.valueChanged.connect(self.update_time)
        self.time_hbox.addWidget(self.time_slider)

        self.time_input = QLineEdit(str(self.current_time_point))
        self.time_input.setValidator(QIntValidator(0, self.num_frames - 1))
        self.time_input.textChanged.connect(self.update_time_input)
        self.time_hbox.addWidget(self.time_input)
        self.control_layout.addLayout(self.time_hbox)

        self.binning_hbox = QHBoxLayout()
        self.binning_label = QLabel('Binning Factor:')
        self.binning_hbox.addWidget(self.binning_label)
        # self.binning_slider = QSlider()
        # self.binning_slider.setOrientation(1)
        # self.binning_slider.setRange(1, 30)
        # self.binning_slider.setValue(self.default_binning_factor)
        # self.binning_slider.valueChanged.connect(self.update_binning)
        # self.binning_hbox.addWidget(self.binning_slider)

        self.binning_input = QLineEdit(str(self.default_binning_factor))
        self.binning_input.setValidator(QDoubleValidator(0.1, 100, 2))
        self.binning_input.textChanged.connect(self.set_not_updated)
        self.binning_hbox.addWidget(self.binning_input)
        self.control_layout.addLayout(self.binning_hbox)

        self.speed_hbox = QHBoxLayout()
        self.speed_label = QLabel('Frames to skip:')
        self.speed_hbox.addWidget(self.speed_label)
        self.speed_slider = QSlider()
        self.speed_slider.setOrientation(1)
        self.speed_slider.setRange(1, len(self.timesteps)-1)
        self.speed_slider.setValue(1)
        self.speed_slider.valueChanged.connect(self.update_speed)
        self.speed_hbox.addWidget(self.speed_slider)

        self.speed_input = QLineEdit('1')
        self.speed_input.setValidator(QIntValidator(1, len(self.timesteps)-1))
        self.speed_input.textChanged.connect(self.update_speed_input)
        self.speed_hbox.addWidget(self.speed_input)
        self.control_layout.addLayout(self.speed_hbox)

        self.z_min_label_from, self.z_min_input, self.z_min_label_to, self.z_max_input = self.add_crop_inputs('Crop Z', 0, self.height, (0, self.height), self.control_layout)
        self.y_min_label_from, self.y_min_input, self.y_min_label_to, self.y_max_input = self.add_crop_inputs('Crop Y', 0, self.width, (0, self.width), self.control_layout)
        self.x_min_label_from, self.x_min_input, self.x_min_label_to, self.x_max_input = self.add_crop_inputs('Crop X', 0, self.width, (0, self.width), self.control_layout)

        difference_hbox = QHBoxLayout()
        self.difference_button = QPushButton('Difference: False')
        self.difference_button.clicked.connect(self.update_difference)
        difference_hbox.addWidget(self.difference_button)
        self.control_layout.addLayout(difference_hbox)

        self.control_layout.addSpacing(20)
        hbox_layout = QHBoxLayout()
        self.update_button = QPushButton('Updated')
        self.update_button.clicked.connect(self.update_volume)
        hbox_layout.addWidget(self.update_button)
        next_button = QPushButton('Next')
        next_button.clicked.connect(self.next_volume)
        hbox_layout.addWidget(next_button)
        self.control_layout.addLayout(hbox_layout)

        self.control_layout.addSpacing(20)

        export_hbox = QHBoxLayout()
        self.num_frames_to_export_label = QLabel('Number of volumes to export:')
        export_hbox.addWidget(self.num_frames_to_export_label)
        self.num_frames_to_export_slider = QSlider()
        self.num_frames_to_export_slider.setOrientation(1)
        self.num_frames_to_export_slider.setRange(1, len(self.timesteps))
        self.num_frames_to_export_slider.setValue(1)
        self.num_frames_to_export_slider.valueChanged.connect(self.update_num_frames_to_export)
        export_hbox.addWidget(self.num_frames_to_export_slider)

        self.num_frames_to_export_input = QLineEdit('1')
        self.num_frames_to_export_input.setValidator(QIntValidator(1, len(self.timesteps)))
        self.num_frames_to_export_input.textChanged.connect(self.update_num_frames_to_export_input)
        export_hbox.addWidget(self.num_frames_to_export_input)
        self.control_layout.addLayout(export_hbox)

        avg_hbox = QHBoxLayout()
        self.num_frames_to_avg_label = QLabel('Frame averaging:')
        avg_hbox.addWidget(self.num_frames_to_avg_label)
        self.num_frames_to_avg_slider = QSlider()
        self.num_frames_to_avg_slider.setOrientation(1)
        self.num_frames_to_avg_slider.setRange(1, 100)
        self.num_frames_to_avg_slider.setValue(1)
        self.num_frames_to_avg_slider.valueChanged.connect(self.update_num_frames_to_avg)
        avg_hbox.addWidget(self.num_frames_to_avg_slider)

        self.num_frames_to_avg_input = QLineEdit('1')
        self.num_frames_to_avg_input.setValidator(QIntValidator(1, 100))
        self.num_frames_to_avg_input.textChanged.connect(self.update_num_frames_to_avg_input)
        avg_hbox.addWidget(self.num_frames_to_avg_input)
        self.control_layout.addLayout(avg_hbox)

        # Add export format selection
        self.export_format_hbox = QHBoxLayout()
        self.export_format_label = QLabel('Export format:')
        self.export_format_hbox.addWidget(self.export_format_label)
        self.export_format = QComboBox()
        self.export_format.addItems(['TIFF', 'NPY'])
        self.export_format_hbox.addWidget(self.export_format)
        self.control_layout.addLayout(self.export_format_hbox)

        export_button = QPushButton('Export volumes')
        export_button.clicked.connect(self.export_volumes)
        self.control_layout.addWidget(export_button)

        self.viewer.window.add_dock_widget(self.control_widget, name='Controls', area='right')

        self.update_volume()

        self.viewer.camera.center = np.array(self.base_volume_shape) / 2
        napari.run()
        
    def add_crop_inputs(self, name, min_val, max_val, default_val, layout):
        label_from = QLabel(f'{name} From:')
        input_from = QLineEdit(str(default_val[0]))
        input_from.setValidator(QIntValidator(min_val, max_val))
        label_to = QLabel('To:')
        input_to = QLineEdit(str(default_val[1]))
        input_to.setValidator(QIntValidator(min_val, max_val))
        label_max = QLabel(f'({max_val})')
        hbox_layout = QHBoxLayout()
        hbox_layout.addWidget(label_from)
        hbox_layout.addWidget(input_from)
        hbox_layout.addWidget(label_to)
        hbox_layout.addWidget(input_to)
        hbox_layout.addWidget(label_max)
        input_from.textChanged.connect(self.set_not_updated)
        input_to.textChanged.connect(self.set_not_updated)
        
        layout.addLayout(hbox_layout)
        return label_from, input_from, label_to, input_to
    
    def load_image(self, time_point, binning_factor):
        z_min_val = self.z_min_input.text()
        z_min_val = int(z_min_val) if z_min_val else 0
        z_max_val = self.z_max_input.text()
        z_max_val = int(z_max_val) if z_max_val else self.height
        y_min_val = self.y_min_input.text()
        y_min_val = int(y_min_val) if y_min_val else 0
        y_max_val = self.y_max_input.text()
        y_max_val = int(y_max_val) if y_max_val else self.width
        x_min_val = self.x_min_input.text()
        x_min_val = int(x_min_val) if x_min_val else 0
        x_max_val = self.x_max_input.text()
        x_max_val = int(x_max_val) if x_max_val else self.width
        if (z_min_val == self.updated_state["z_min"]
            and z_max_val == self.updated_state["z_max"]
            and y_min_val == self.updated_state["y_min"]
            and y_max_val == self.updated_state["y_max"]
            and x_min_val == self.updated_state["x_min"]
            and x_max_val == self.updated_state["x_max"]
            and binning_factor == self.updated_state["binning_factor"]
            and self.difference == self.updated_state["difference"]
            and time_point == self.updated_state["self.current_time_point"]):
            return self.layer.data

        with torch.no_grad():
            if self.difference is True:
                if (self.difference_volume is None 
                    or z_min_val != self.updated_state_diff["z_min"]
                    or z_max_val != self.updated_state_diff["z_max"]
                    or y_min_val != self.updated_state_diff["y_min"]
                    or y_max_val != self.updated_state_diff["y_max"]
                    or x_min_val != self.updated_state_diff["x_min"]
                    or x_max_val != self.updated_state_diff["x_max"]
                    or binning_factor != self.updated_state_diff["binning_factor"]):
                    self.difference_volume = self.create_volume(
                        time=0.0, 
                        x_min_val=x_min_val, 
                        x_max_val=x_max_val,
                        y_min_val=y_min_val,
                        y_max_val=y_max_val,
                        z_min_val=z_min_val,
                        z_max_val=z_max_val,
                        binning_factor=binning_factor,
                        desc="Updating difference volume")
                    self.updated_state_diff["z_min"] = z_min_val
                    self.updated_state_diff["z_max"] = z_max_val
                    self.updated_state_diff["y_min"] = y_min_val
                    self.updated_state_diff["y_max"] = y_max_val
                    self.updated_state_diff["x_min"] = x_min_val
                    self.updated_state_diff["x_max"] = x_max_val
                    self.updated_state_diff["binning_factor"] = binning_factor
            if (self.volume is None 
                or z_min_val != self.updated_state["z_min"]
                or z_max_val != self.updated_state["z_max"]
                or y_min_val != self.updated_state["y_min"]
                or y_max_val != self.updated_state["y_max"]
                or x_min_val != self.updated_state["x_min"]
                or x_max_val != self.updated_state["x_max"]
                or binning_factor != self.updated_state["binning_factor"]
                or time_point != self.updated_state["self.current_time_point"]):
                self.volume = self.create_volume(
                    time=self.timesteps[time_point],
                    x_min_val=x_min_val, 
                    x_max_val=x_max_val,
                    y_min_val=y_min_val,
                    y_max_val=y_max_val,
                    z_min_val=z_min_val,
                    z_max_val=z_max_val,
                    binning_factor=binning_factor,
                    desc="Updating volume"
                )
            if self.difference is True:
                volume = self.volume - self.difference_volume
            else:
                volume = self.volume
        self.updated_state["z_min"] = z_min_val
        self.updated_state["z_max"] = z_max_val
        self.updated_state["y_min"] = y_min_val
        self.updated_state["y_max"] = y_max_val
        self.updated_state["x_min"] = x_min_val
        self.updated_state["x_max"] = x_max_val
        self.updated_state["binning_factor"] = binning_factor
        self.updated_state["difference"] = self.difference
        self.updated_state["self.current_time_point"] = time_point
        return volume

    def create_volume(self, time, x_min_val, x_max_val, y_min_val, y_max_val, z_min_val, z_max_val, binning_factor, desc):
        if self.dummy:
            size = int(self.height // binning_factor)
            blobs = data.binary_blobs(length=size, volume_fraction=0.1, n_dim=3).astype(np.float32)
            z_min = int(z_min_val // binning_factor)
            z_max = int(z_max_val // binning_factor)
            y_min = int(y_min_val // binning_factor)
            y_max = int(y_max_val // binning_factor)
            x_min = int(x_min_val // binning_factor)
            x_max = int(x_max_val // binning_factor)
            return blobs[z_min:z_max, y_min:y_max, x_min:x_max]
        else:
            nVoxels = self.config.geometry.nVoxel
            rm = self.config.sample_outside
            nVoxels = [nVoxels[0], nVoxels[1]+2*rm, nVoxels[2]+2*rm]
            start_x = (x_min_val - rm) / nVoxels[2]
            end_x = (x_max_val - rm) / nVoxels[2]
            x_w = int((x_max_val - x_min_val) // binning_factor)
            start_y = (y_min_val - rm) / nVoxels[1]
            end_y = (y_max_val - rm) / nVoxels[1]
            y_w = int((y_max_val - y_min_val) // binning_factor)
            start_z = (z_min_val - rm) / nVoxels[0]
            end_z = (z_max_val - rm) / nVoxels[0]
            z_h = int((z_max_val - z_min_val) // binning_factor)
            output = torch.zeros((z_h, y_w, x_w), device="cpu", dtype=torch.float32)
            output = output.flatten()
            z_lin = torch.linspace(start_z, end_z, steps=z_h, dtype=torch.float32, device="cpu")
            y_lin = torch.linspace(start_y, end_y, steps=y_w, dtype=torch.float32, device="cpu")
            x_lin = torch.linspace(start_x, end_x, steps=x_w, dtype=torch.float32, device="cpu")

            batch_size = 1_000_000
            total_points = z_h * y_w * x_w
            indices = torch.arange(total_points, dtype=torch.int64)
            batches = torch.split(indices, batch_size)
            for batch in tqdm(batches, desc=desc):
                z_indices = batch // (y_w * x_w)
                y_indices = (batch % (y_w * x_w)) // x_w
                x_indices = batch % x_w
                z = z_lin[z_indices]
                y = y_lin[y_indices]
                x = x_lin[x_indices]
                grid = torch.stack((z, y, x), dim=1).cuda()
                batch_output = self.model(grid, time).flatten().float().detach().cpu()
                output.view(-1)[batch] = batch_output
            output = output.view((z_h, y_w, x_w))
            return output.numpy()

    def update_difference(self):
        self.difference = not self.difference
        self.difference_button.setText(f"Difference: {self.difference}")
        self.set_not_updated()

    def update_volume(self):
        self.update_button.setText("Updating...")
        binning_factor = float(self.binning_input.text().replace(',', '.'))
        new_data = self.load_image(self.current_time_point, binning_factor)
        self.layer.data = new_data
        self.layer.scale = [binning_factor, binning_factor, binning_factor]
        self.update_button.setText("Updated")

    def update_time(self, value):
        self.current_time_point = value
        self.time_input.setText(str(self.current_time_point))
        self.set_not_updated()

    def next_volume(self):
        self.update_time((self.current_time_point + int(self.speed_input.text())) % self.num_frames)
        self.update_volume()

    def update_time_input(self):
        value = int(self.time_input.text())
        self.time_slider.setValue(value)
        self.update_time(value)

    def update_binning(self, value):
        self.binning_input.setText(str(value))
        self.set_not_updated()

    def update_speed(self, value):
        self.speed_input.setText(str(value))
        self.set_not_updated()
        
    def update_speed_input(self):
        value = int(self.speed_input.text())
        self.speed_slider.setValue(value)
        self.set_not_updated()
        
    def update_num_frames_to_export(self, value):
        self.num_frames_to_export_input.setText(str(value))
        self.set_not_updated()
        
    def update_num_frames_to_export_input(self):
        value = int(self.num_frames_to_export_input.text())
        self.num_frames_to_export_slider.setValue(value)
        self.set_not_updated()
        
    def update_num_frames_to_avg(self, value):
        self.num_frames_to_avg_input.setText(str(value))
        self.set_not_updated()
        
    def update_num_frames_to_avg_input(self):
        value = int(self.num_frames_to_avg_input.text())
        self.num_frames_to_avg_slider.setValue(value)
        self.set_not_updated()
        
    def set_not_updated(self):
        try:
            if (self.difference == self.updated_state["difference"]
                and self.current_time_point == self.updated_state["self.current_time_point"]
                and float(self.binning_input.text().replace(',', '.')) == self.updated_state["binning_factor"]
                and int(self.z_min_input.text()) == self.updated_state["z_min"]
                and int(self.z_max_input.text()) == self.updated_state["z_max"]
                and int(self.y_min_input.text()) == self.updated_state["y_min"]
                and int(self.y_max_input.text()) == self.updated_state["y_max"]
                and int(self.x_min_input.text()) == self.updated_state["x_min"]
                and int(self.x_max_input.text()) == self.updated_state["x_max"]):
                self.update_button.setText('Updated')
            else:
                self.update_button.setText('Update (click to update)')
        except ValueError:
            self.update_button.setText('Update (click to update)')

    def export_volumes(self):
        binning_factor = float(self.binning_input.text().replace(',', '.'))
        num_export_frames = int(self.num_frames_to_export_input.text())
        save_dir = QFileDialog.getExistingDirectory(caption="Select Directory")
        if not save_dir:
            return
        prev_difference = self.difference
        if self.difference is True:
            self.difference = False
        for i in tqdm(range(num_export_frames), desc="Exporting volumes"):
            volume = None
            for j in tqdm(range(int(self.num_frames_to_avg_input.text())), desc="Averaging timesteps"):
                time_point = (self.current_time_point + i * int(self.speed_input.text()) + j) % self.num_frames
                volume_data = self.load_image(time_point, binning_factor).astype(np.float32)
                if volume is None:
                    volume = volume_data / int(self.num_frames_to_avg_input.text())
                else:
                    volume += volume_data / int(self.num_frames_to_avg_input.text())
            self.layer.data = volume_data
            self.layer.scale = [binning_factor, binning_factor, binning_factor]
            binning_factor_str = str(binning_factor).replace('.', '_')
            file_path = f"{save_dir}/volume_{time_point}_binning_factor_{binning_factor_str}"
            if self.export_format.currentText() == 'TIFF':
                tif.imsave(f"{file_path}.tiff", volume_data)
                print(f"Saved volume {time_point} to {file_path}.tiff")
            elif self.export_format.currentText() == 'NPY':
                np.save(f"{file_path}.npy", volume_data)
                print(f"Saved volume {time_point} to {file_path}.npy")
        self.difference = prev_difference


if __name__ == '__main__':
    GUI(dummy=False)
