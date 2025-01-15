# --------------------------------------------------------------------------------------------------
# ---------------------------------------- Tricky Diffusion ----------------------------------------
# --------------------------------------------------------------------------------------------------
import torch.nn.utils.parametrizations as params
import torch.nn as networks

import torch

from Utils.Init import init
from Utils.Timestamp import Timestamp


# --------------------------------------------------------------------------------------------------
# ------------------------------------- CLASS :: Upscale Block -------------------------------------
# --------------------------------------------------------------------------------------------------
class UpscaleBlock(networks.Module):

    def __init__(self, i_channels: int, o_channels: int) -> None:
        super().__init__()

        self.scale = networks.ConvTranspose2d(i_channels, o_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
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

        self.merge = networks.ConvTranspose2d(i_channels, o_channels, kernel_size=3, padding=1)
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

        self.conv1 = networks.ConvTranspose2d(channels, channels, kernel_size=3, padding=1)
        self.conv1 = params.weight_norm(self.conv1)
        self.norm1 = networks.GroupNorm(32, channels)
        self.conv2 = networks.ConvTranspose2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = params.weight_norm(self.conv2)
        self.norm2 = networks.GroupNorm(32, channels)
        self.swish = networks.SiLU()

        init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.swish(self.norm1(self.conv1(x)))
        x = self.swish(self.norm2(self.conv2(x)))

        return x


# --------------------------------------------------------------------------------------------------
# ---------------------------------- CLASS :: Final Decoder Block ----------------------------------
# --------------------------------------------------------------------------------------------------
class DecoderFinal(networks.Module):

    def __init__(self, i_channels: int, o_channels: int, t_channels: int) -> None:
        super().__init__()

        self.embed = ConcatenationBlock(i_channels, t_channels, i_channels)
        self.block = ResidualBlock(i_channels)
        self.final = networks.ConvTranspose2d(i_channels, o_channels, kernel_size=3, padding=1)
        self.final = params.weight_norm(self.final)

        init(self)

    def forward(self, t: Timestamp, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        x = self.embed(x, t.get(x))
        x = self.block(x)
        x = self.final(x)

        return x

# --------------------------------------------------------------------------------------------------
# ------------------------------------- CLASS :: Decoder Block -------------------------------------
# --------------------------------------------------------------------------------------------------
class DecoderBlock(networks.Module):

    def __init__(self, i_channels: int, o_channels: int, t_channels: int) -> None:
        super().__init__()

        self.embed = ConcatenationBlock(i_channels, t_channels, i_channels)
        self.merge = ConcatenationBlock(i_channels, i_channels, i_channels)
        self.block = ResidualBlock(i_channels)
        self.scale = UpscaleBlock(i_channels, o_channels)
        self.noise = networks.Dropout(0.05)

    def forward(self, t: Timestamp, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        x = self.embed(x, t.get(x))
        x = self.block(x)
        x = self.merge(x, y)
        x = self.scale(x)
        x = self.noise(x)

        return x