# --------------------------------------------------------------------------------------------------
# ---------------------------------------- Tricky Diffusion ----------------------------------------
# --------------------------------------------------------------------------------------------------
import torch.nn.utils.parametrizations as params
import torch.nn as networks

import torch

from Utils.Timestamp import Timestamp


# --------------------------------------------------------------------------------------------------
# ------------------------------------ CLASS :: Downscale Block ------------------------------------
# --------------------------------------------------------------------------------------------------
class DownscaleBlock(networks.Module):

    def __init__(self, i_channels: int, o_channels: int) -> None:
        super().__init__()

        self.scale = networks.Conv2d(i_channels, o_channels, kernel_size=3, stride=2)
        self.scale = params.weight_norm(self.scale)
        self.shift = networks.GroupNorm(32, o_channels)
        self.swish = networks.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swish(self.shift(self.merge(x)))


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
        self.norm1 = networks.GroupNorm(32, channels)
        self.conv2 = networks.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv1 = params.weight_norm(self.conv2)
        self.norm1 = networks.GroupNorm(32, channels)
        self.swish = networks.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.swish(self.norm1(self.conv1(x)))
        x = self.swish(self.norm2(self.conv2(x)))

        return x


# --------------------------------------------------------------------------------------------------
# ------------------------------------- CLASS :: Decoder Block -------------------------------------
# --------------------------------------------------------------------------------------------------
class EncoderEntry(networks.Module):

    def __init__(self, i_channels: int, o_channels: int, t_channels: int) -> None:
        super().__init__()

        self.embed = ConcatenationBlock(i_channels, t_channels, o_channels)
        self.block = ResidualBlock(o_channels)
        self.noise = networks.Dropout(0.05)

    def forward(self, t: Timestamp, x: torch.Tensor) -> torch.Tensor:

        x = self.embed(x, t.stamp_like(x))
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
        self.embed = ConcatenationBlock(i_channels, t_channels, o_channels)
        self.merge = ConcatenationBlock(i_channels, i_channels, o_channels)
        self.block = ResidualBlock(o_channels)
        self.noise = networks.Dropout(0.05)

    def forward(self, t: Timestamp, x: torch.Tensor) -> torch.Tensor:

        x = self.scale(x)
        x = self.embed(x, t.stamp_like(x))
        x = self.merge(x, torch.zeros_like(x))
        x = self.block(x)
        x = self.noise(x)

        return x