# --------------------------------------------------------------------------------------------------
# ---------------------------------------- Tricky Diffusion ----------------------------------------
# --------------------------------------------------------------------------------------------------
import torch.utils.data as datatools
import torch

from Utils.Timestamp import Timestamp
from Dataset.Sampler import Sampler
from Dataset.Dataset import Dataset

from Modules.UNet import UNet


# --------------------------------------------------------------------------------------------------
# ----------------------------------- Environment and Utilities ------------------------------------
# --------------------------------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")

BATCH_SIZE = 8
EPOCHS = 1
T_CHANNELS = 32

unet = UNet(T_CHANNELS)
optimizer = torch.optim.Adam(unet.parameters(), lr=1e-5)

sampler = Sampler()
payload = Dataset()
dataset = datatools.DataLoader(dataset=payload, sampler=sampler, batch_size=BATCH_SIZE)


# --------------------------------------------------------------------------------------------------
# --------------------------------------- Main Training Loop ---------------------------------------
# --------------------------------------------------------------------------------------------------
for eid in range(EPOCHS):
    for bid, (source, target, stamps) in enumerate(dataset):

        t = Timestamp(stamps, channels=T_CHANNELS)
