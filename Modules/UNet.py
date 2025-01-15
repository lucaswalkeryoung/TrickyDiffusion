# --------------------------------------------------------------------------------------------------
# ---------------------------------------- Tricky Diffusion ----------------------------------------
# --------------------------------------------------------------------------------------------------
from . Bottleneck import BottleneckBlock
from . Encoder import EncoderBlock
from . Encoder import EncoderEntry
from . Decoder import DecoderBlock
from . Decoder import DecoderFinal

from Utils.Timestamp import Timestamp
from Utils.Init import init

import torch.nn as networks
import torch


# --------------------------------------------------------------------------------------------------
# ----------------------------------------- CLASS :: U-Net -----------------------------------------
# --------------------------------------------------------------------------------------------------
class UNet(networks.Module):

    def __init__(self, t_channels: int) -> None:
        super().__init__()

        self.encoder1 = EncoderEntry(3,   32, t_channels)
        self.encoder2 = EncoderBlock(32,  64, t_channels)
        self.encoder3 = EncoderBlock(64, 128, t_channels)

        self.bottleneck = BottleneckBlock(128, None)
        self.bottleneck = BottleneckBlock(128, self.bottleneck)
        self.bottleneck = BottleneckBlock(128, self.bottleneck)

        self.decoder3 = DecoderBlock(128, 64, t_channels)
        self.decoder2 = DecoderBlock(64,  32, t_channels)
        self.decoder1 = DecoderFinal(32,   3, t_channels)

    def forward(self, t: Timestamp, x0: torch.Tensor) -> torch.Tensor:

        x1 = self.encoder1(t, x0)
        x2 = self.encoder2(t, x1)
        x3 = self.encoder3(t, x2)

        y3 = self.bottleneck(x3)

        y2 = self.decoder3(t, x3, y3)
        y1 = self.decoder2(t, x2, y2)
        y0 = self.decoder1(t, x1, y1)

        return y0

