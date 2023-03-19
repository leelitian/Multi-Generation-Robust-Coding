import torch
import math
import torch.nn as nn

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, metrics='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.lmbda = lmbda
        self.metrics = metrics

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        if self.metrics == 'mse':
            out["dist"] = self.mse(output["x_hat"], target)
            out["loss"] = self.lmbda * 255 ** 2 * out["dist"] + out["bpp_loss"]
        elif self.metrics == 'rl':
            out["dist"] = self.mse(output["x_hat"], target)
            out["rl"] = self.mse(output["x1"], target)
            out["loss"] = self.lmbda * 255 ** 2 * (out["dist"] + out["rl"]) + out["bpp_loss"]

        return out