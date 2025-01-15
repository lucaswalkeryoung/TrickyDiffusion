# --------------------------------------------------------------------------------------------------
# ---------------------------------------- Tricky Diffusion ----------------------------------------
# --------------------------------------------------------------------------------------------------
import torchvision.transforms as transforms
import torch.utils.data as datatools
import torch
import uuid
import os
import shutil

from Utils.Timestamp import Timestamp
from Dataset.Sampler import Sampler
from Dataset.Dataset import Dataset
from Modules.UNet import UNet

from PIL import Image


# --------------------------------------------------------------------------------------------------
# ------------------------------------------- Save Image -------------------------------------------
# --------------------------------------------------------------------------------------------------
count = 0

def save(original: torch.Tensor, noisy: torch.Tensor, stamp: torch.Tensor) -> None:

    with torch.no_grad():

        for i in reversed(range(1, int(stamp[0].item()) + 1)):
            t = Timestamp(torch.tensor([i]).to(stamp.device), channels=T_CHANNELS)
            noisy = noisy - unet(t, noisy).detach()

    o = transforms.ToPILImage()(original[0])
    r = transforms.ToPILImage()(noisy[0])

    output = Image.new('RGB', (256, 128), color='white')
    output.paste(o, (0,   0))
    output.paste(r, (128, 0))

    global count
    count += 1

    output.save(f"./Output/{count:05}.png")

shutil.rmtree("./Output")
os.mkdir("./Output")


# --------------------------------------------------------------------------------------------------
# ----------------------------------- Environment and Utilities ------------------------------------
# --------------------------------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")

BATCH_SIZE = 16
EPOCHS = 10
T_CHANNELS = 32

unet = UNet(T_CHANNELS).to(DEVICE).train()
optimizer = torch.optim.Adam(unet.parameters(), lr=1e-5)

sampler = Sampler()
payload = Dataset()
dataset = datatools.DataLoader(dataset=payload, sampler=sampler, batch_size=BATCH_SIZE)


# --------------------------------------------------------------------------------------------------
# --------------------------------------- Main Training Loop ---------------------------------------
# --------------------------------------------------------------------------------------------------
for eid in range(EPOCHS):
    for bid, (original, noise, noisy, stamp) in enumerate(dataset):

        noise = noise.to(DEVICE)
        noisy = noisy.to(DEVICE)
        stamp = stamp.to(DEVICE)

        optimizer.zero_grad()

        t = Timestamp(stamp, channels=T_CHANNELS)

        prediction = unet(t, noisy)
        loss = torch.nn.functional.mse_loss(prediction, noise)

        loss.backward()
        optimizer.step()

        if bid %  10 == 0:
            unet = unet.eval()
            print(f"MSE Loss: {loss.item()}")

        if bid % 100 == 0:
            unet = unet.eval()
            save(original[0:1], noisy[0:1], stamp[0:1])
            unet = unet.train()
