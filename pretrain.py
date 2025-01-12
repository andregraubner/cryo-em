from segformer3d import SegFormer3D
from data import CryoETDataset, PretrainingCryoETDataset
import torch
from torch.utils.data import Dataset, DataLoader
from utils import create_submission
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import numpy as np
import pandas as pd
from evaluate import inference
from monai.transforms import (
    EnsureChannelFirstd,
    LoadImage,
    LoadImaged,
    Orientationd,
    Rand3DElasticd,
    RandAffined,
    Spacingd,
)
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
    RandSpatialCropD,
    RandGaussianNoiseD,
    RandRotate90D,
    RandSpatialCropSamplesD,
    RandFlipD
)
from monai.networks.nets import SwinUNETR
import wandb
from monai.networks.nets import UNet
from monai.losses import DiceLoss
import transformers
import monai
from vit import Loss, aug_rand, rot_rand, SSLHead

wandb.init(project="cryo-em-pretrain")

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = PretrainingCryoETDataset(
    "/scratch2/andregr/cryo-em/data/dump/*/*.mrc",
    crop_size=(96,96,96),
    epoch_length=10000,
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    prefetch_factor=2,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
)

model = torch.nn.DataParallel(UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=4,
)).to(device)

#model = torch.nn.DataParallel(SSLHead(dim=192)).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 0, len(dataloader) * 50)
#loss_function = Loss(batch_size=8)

# gradscaler
scaler = torch.amp.GradScaler("cuda")

samples_seen = 0
# Example iteration
for epoch in range(10000):
    model.train()
    for batch in tqdm(dataloader):

        optimizer.zero_grad()

        x = batch["tomograms"].to(device)

        # Normalize
        mean = x.mean(dim=(1, 2, 3, 4), keepdim=True)  # Mean across channels, height, width
        std = x.std(dim=(1, 2, 3, 4), keepdim=True)   # Standard deviation across channels, height, width
        x = (x - mean) / (std + 1e-6)

        """
        with torch.no_grad():
            x1, rot1 = rot_rand(x)
            x2, rot2 = rot_rand(x)
            x1_augment = aug_rand(x1)
            x2_augment = aug_rand(x2)

        with torch.autocast(device_type="cuda"):
            rot1_p, contrastive1_p, rec_x1 = model(x1_augment)
            rot2_p, contrastive2_p, rec_x2 = model(x2_augment)
            rot_p = torch.cat([rot1_p, rot2_p], dim=0)
            rots = torch.cat([rot1, rot2], dim=0)
            imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
            imgs = torch.cat([x1, x2], dim=0)
            loss, losses_tasks = loss_function(rot_p, rots, contrastive1_p, contrastive2_p, imgs_recon, imgs)

            #loss.backward()
            #optimizer.step()
            #scheduler.step()
        """

        with torch.autocast(device_type="cuda"):
            noise = torch.randn_like(x) * 0.2
            x_noisy = x + noise
            out = model(x_noisy)
            loss = F.mse_loss(out, x) 

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        samples_seen += x.shape[0]

        wandb.log({
            "loss": loss.item(),
            "lr": scheduler.get_last_lr()[0]
        }, step=samples_seen)

    torch.save(model.state_dict(), "pretrained.pth")