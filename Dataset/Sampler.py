# --------------------------------------------------------------------------------------------------
# ---------------------------------------- Tricky Diffusion ----------------------------------------
# --------------------------------------------------------------------------------------------------
import torchvision.transforms as transforms
import torch.utils.data as datatools

import random
import torch
import typing


# --------------------------------------------------------------------------------------------------
# ---------------------------------- CLASS :: Custom Image Loader ----------------------------------
# --------------------------------------------------------------------------------------------------
class Sampler(datatools.Dataset):

    def __iter__(self) -> typing.Iterator[int]:
        return iter(random.sample(range(59198), 59198))

