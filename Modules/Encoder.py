# --------------------------------------------------------------------------------------------------
# ---------------------------------------- Tricky Diffusion ----------------------------------------
# --------------------------------------------------------------------------------------------------
import torch.nn.utils.parametrizations as params
import torch.nn as networks

import torch

from Utils.Init import init
from Utils.Timestamp import Timestamp


# --------------------------------------------------------------------------------------------------
# ------------------------------------ CLASS :: Downscale Block ------------------------------------
# --------------------------------------------------------------------------------------------------
class DownscaleBlock(networks.Module):

    def __init__(self, i_channels: int, o_channels: int) -> None:
        super().__init__()

        self.scale = networks.Conv2d(i_channels, o_channels, kernel_size=3, stride=2, padding=1)
        self.scale = params.weight_norm(self.scale)
        self.shift = networks.GroupNorm(32, o_channels)
        self.swish = networks.SiLU()

        init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swish(self.shift(self.scale(x)))


# --------------------------------------------------------------------------------------------------
# ---------------------------------- CLASS :: Concatenation Block ----------------------------------
# --------------------------------------------------------------------------------------------------
class ConcatenationBlock(networks.Module):

    def __init__(self, a_channels: int, b_channels: int, o_channels: int) -> None:
        super().__init__()

        i_channels = a_channels + b_channels
        self.merge = networks.Conv2d(i_channels, o_channels, kernel_size=3, padding=1)
        self.merge = params.weight_norm(self.merge)
        self.shift = networks.GroupNorm(32, o_channels)
        self.swish = networks.SiLU()

        init(self)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.swish(self.shift(self.merge(torch.cat([a, b], 1))))


# --------------------------------------------------------------------------------------------------
# ------------------------------- CLASS :: Residual Connection Block -------------------------------
# --------------------------------------------------------------------------------------------------
class ResidualBlock(networks.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.conv1 = networks.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv1 = params.weight_norm(self.conv1)
        self.conv2 = networks.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = params.weight_norm(self.conv2)
        self.norm2 = networks.GroupNorm(32, channels)
        self.norm1 = networks.GroupNorm(32, channels)
        self.swish = networks.SiLU()

        init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        residual = x

        x = self.swish(self.norm1(self.conv1(x)))
        x = self.swish(self.norm2(self.conv2(x)))

        return x + residual


# --------------------------------------------------------------------------------------------------
# ------------------------------------- CLASS :: Decoder Block -------------------------------------
# --------------------------------------------------------------------------------------------------
class EncoderEntry(networks.Module):

    def __init__(self, i_channels: int, o_channels: int, t_channels: int) -> None:
        super().__init__()

        self.embed = ConcatenationBlock(i_channels, t_channels, o_channels)
        self.block = ResidualBlock(o_channels)
        self.noise = networks.Dropout(0.00)

        init(self)

    def forward(self, t: Timestamp, x: torch.Tensor) -> torch.Tensor:

        x = self.embed(x, t.get(x))
        x = self.block(x)
        x = self.noise(x)

        return x


# --------------------------------------------------------------------------------------------------
# ------------------------------------- CLASS :: Decoder Block -------------------------------------
# --------------------------------------------------------------------------------------------------
class EncoderBlock(networks.Module):

    def __init__(self, i_channels: int, o_channels: int, t_channels: int) -> None:
        super().__init__()

        self.scale = DownscaleBlock(i_channels, o_channels)
        self.embed = ConcatenationBlock(o_channels, t_channels, o_channels)
        self.block = ResidualBlock(o_channels)
        self.noise = networks.Dropout(0.00)

        init(self)

    def forward(self, t: Timestamp, x: torch.Tensor) -> torch.Tensor:

        x = self.scale(x)
        x = self.embed(x, t.get(x))
        x = self.block(x)
        x = self.noise(x)

        return x