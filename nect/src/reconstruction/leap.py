import torch
import torch.nn as nn
from leaptorch import Projector


class FBP(nn.Module):
    def __init__(self, proj):
        super().__init__()
        # initialize projector and load parameters
        self.projector = proj

    def filter_project(self, sino):
        return self.projector.filter_projections(sino)

    def forward(self, sino):
        # input sino: batch_size x projection_angles x detector_row(1) x detector_column
        batch, nviews, nrows, ncols = sino.shape
        sino_filtered = torch.zeros_like(sino)
        for n in range(batch):
            sino_filtered[n, :, :, :] = self.filter_project(sino[n, :, :, :])
        img = self.projector(sino_filtered.contiguous())
        # img = img_.clone() ########### need to fix it
        img[img < 0] = 0
        return img


def fbp(sinogram, theta, proj: Projector = None, *args, **kwargs):
    proj.update_phi(torch.from_numpy(theta).float())
    fbp = FBP(proj.proj)
    return fbp(torch.from_numpy(sinogram).unsqueeze(0).to(proj.device)).detach().cpu()
