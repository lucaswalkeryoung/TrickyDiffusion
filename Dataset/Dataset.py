# --------------------------------------------------------------------------------------------------
# ---------------------------------------- Tricky Diffusion ----------------------------------------
# --------------------------------------------------------------------------------------------------
import torchvision.transforms as transforms
import torch.utils.data as datatools

import random
import torch
import pathlib

from PIL import Image


# --------------------------------------------------------------------------------------------------
# ---------------------------------- CLASS :: Custom Data Sampler ----------------------------------
# --------------------------------------------------------------------------------------------------
class Dataset(datatools.Dataset):

    def __init__(self) -> None:
        super(Dataset, self).__init__()

        self.images = list((pathlib.Path() / 'Data').rglob('*.jpg'))

        self.transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index: int, T: int = 500) -> tuple[torch.Tensor, torch.Tensor, int]:

        source = self.transforms(Image.open(random.choice(self.images)))
        target = torch.randn_like(source)

        t = torch.tensor(random.randint(1, T), dtype=torch.float32)

        alpha = torch.cos((t / T) * (torch.pi / 2)) ** 2
        source_alpha = torch.sqrt(alpha)
        target_alpha = torch.sqrt(1 - alpha)

        noised  = target * target_alpha
        noised += source * source_alpha

        return noised, target, t