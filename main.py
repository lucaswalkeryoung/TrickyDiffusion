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

def save(t_o: torch.Tensor, t_n: torch.Tensor, t_t: torch.Tensor, t_p: torch.Tensor) -> None:

    convert = transforms.ToPILImage()

    t_original   = t_o[0] # clean image
    t_noise      = t_n[0] # noisy image
    t_noisy      = t_t[0] # original noise
    t_prediction = t_p[0] # predicted noise

    t_reconstruction = t_noisy - t_prediction

    original = convert(t_original)
    reconstruction = convert(t_reconstruction)
    noise = convert(t_noise)
    prediction = convert(t_prediction)

    output = Image.new('RGB', (256, 256), color='white')
    output.paste(original, (0, 0))
    output.paste(reconstruction, (128, 0))
    output.paste(noise, (0, 128))
    output.paste(prediction, (128, 128))

    global count
    count += 1
    output.save(f'./Output/{count:05}-{uuid.uuid4()}.png')


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
    for bid, (original, noise, noisy, step) in enumerate(dataset):

        original = original.to(DEVICE)
        noise    = noise.to(DEVICE)
        noisy    = noisy.to(DEVICE)
        step     = step.to(DEVICE)

        optimizer.zero_grad()

        t = Timestamp(step, channels=T_CHANNELS)

        prediction = unet(t, noisy)
        loss = torch.nn.functional.mse_loss(prediction, noise)

        loss.backward()
        optimizer.step()

        if bid % 10 == 0:
            print(f"MSE Loss: {loss.item()}")
            save(original, noise, noisy, prediction)
