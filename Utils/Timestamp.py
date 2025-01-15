# --------------------------------------------------------------------------------------------------
# ---------------------------------------- Tricky Diffusion ----------------------------------------
# --------------------------------------------------------------------------------------------------
import torch.nn as networks
import torch
import math


# --------------------------------------------------------------------------------------------------
# ----------------------------------- CLASS :: Timestamp Handler -----------------------------------
# --------------------------------------------------------------------------------------------------
class Timestamp(object):

    def __init__(self, t: torch.Tensor, channels: int = 32) -> None:

        t = t.unsqueeze(1)

        indices = torch.arange(0, channels, 2, dtype=torch.float32, device=t.device)
        scaling = math.log(10000.0 / channels)

        frequencies = indices * -scaling

        sinusoidal = torch.zeros(t.size(0), channels, device=t.device) # all components
        sinusoidal[:, 0::2] = torch.sin(t * frequencies)  # Sin components
        sinusoidal[:, 1::2] = torch.cos(t * frequencies)  # Cos components

        self.sinusoidal = sinusoidal

    def get(self, target: torch.Tensor) -> torch.Tensor:

        _, _, height, width = target.shape
        return self.sinusoidal.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)