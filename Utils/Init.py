# --------------------------------------------------------------------------------------------------
# ---------------------------------------- Tricky Diffusion ----------------------------------------
# --------------------------------------------------------------------------------------------------
import torch.nn as networks
import torch.nn.init as initialization
import torch


# --------------------------------------------------------------------------------------------------
# ----------------------------- FUNCTION :: Initialize a Neural Module -----------------------------
# --------------------------------------------------------------------------------------------------
def init(module: networks.Module) -> networks.Module:

    for layer in module.modules():

        if isinstance(layer, networks.ConvTranspose2d):
            initialization.kaiming_uniform_(layer.weight)
            initialization.zeros_(layer.bias)

        if isinstance(layer, networks.Conv2d):
            initialization.kaiming_uniform_(layer.weight)
            initialization.zeros_(layer.bias)

    return module