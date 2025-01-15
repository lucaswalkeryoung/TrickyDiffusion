# --------------------------------------------------------------------------------------------------
# ---------------------------------------- Tricky Diffusion ----------------------------------------
# --------------------------------------------------------------------------------------------------
import torch.nn.utils.parametrizations as params
import torch.nn as networks

import torch
import math

from Utils.Init import init
from Utils.Timestamp import Timestamp


# --------------------------------------------------------------------------------------------------
# ------------------------------------- CLASS :: U-Net Encoder -------------------------------------
# --------------------------------------------------------------------------------------------------
class BottleneckBlock(networks.Module):

    # ------------------------------------------------------------------------------------------
    # ------------------------------- CONSTRUCTOR :: Constructor -------------------------------
    # ------------------------------------------------------------------------------------------
    def __init__(self, channels: int, inner: None | networks.Module = None) -> None:
        super().__init__()

        self.conv1 = networks.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv1 = params.weight_norm(self.conv1)
        self.norm1 = networks.GroupNorm(32, channels)
        self.conv2 = networks.ConvTranspose2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = params.weight_norm(self.conv2)
        self.norm2 = networks.GroupNorm(32, channels)
        self.swish = networks.SiLU()

        self.inner = inner or networks.Identity()

        init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        residual = x

        x = self.swish(self.norm1(self.conv1(x)))
        x = self.inner(x)
        x = self.swish(self.norm2(self.conv2(x)))

        return x + (residual / math.sqrt(2))