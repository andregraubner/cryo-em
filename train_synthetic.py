from segformer3d import SegFormer3D
from data import SyntheticCryoETDataset, CryoETDataset
import torch
from torch.utils.data import Dataset, DataLoader
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

wandb.init(project="cryo-em")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.nn.DataParallel(UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=7,
    channels=(16, 32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2, 2),
    num_res_units=2,
)).to(device)

# Create dataset
dataset = SyntheticCryoETDataset(
    path="/scratch2/andregr/cryo-em/data/10441/TS_*/",
    crop_size=(64,64,64),
    epoch_length=10000,
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    prefetch_factor=2,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
class_weights = [1,1,0,2,1,2,1]
class_weights = torch.tensor(class_weights).float().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 0, len(dataloader) * 10)

dice_loss = DiceLoss(to_onehot_y=True, softmax=True)

train_transforms = Compose([
    RandRotate90D(keys=["tomograms", "labels"], prob=0.5, spatial_axes=(1, 2)),
    RandFlipD(keys=["tomograms", "labels"], prob=0.5, spatial_axis=0),
    RandFlipD(keys=["tomograms", "labels"], prob=0.5, spatial_axis=1),
    RandFlipD(keys=["tomograms", "labels"], prob=0.5, spatial_axis=2),
    #RandGaussianNoiseD(keys=["tomograms"], prob=0.5, mean=0.0, std=0.1)
    ])
train_transforms.set_random_state(seed=123)

# Example iteration
for epoch in range(10):
    model.train()
    for batch in tqdm(dataloader):

        optimizer.zero_grad()

        batch["tomograms"] = (batch["tomograms"] - batch["tomograms"].mean()) / batch["tomograms"].std()
        batch = train_transforms(batch)

        tomograms = batch["tomograms"][:,None].to(device)
        labels = batch["labels"][:,None].long().to(device)

        out = model(tomograms)

        loss = dice_loss(out, labels) 

        loss.backward()
        optimizer.step()
        scheduler.step()

        wandb.log({
            "loss": loss.item(),
            "lr": scheduler.get_last_lr()[0]
        })
        
    if epoch % 1 == 0:
        score = inference(model, "TS_99_9")
        wandb.log({"score": score})
        torch.save(model.state_dict(), "synthetic_model.pth")

score = inference(model, "TS_99_9")
wandb.log({"score": score})

torch.save(model.state_dict(), "synthetic_model.pth")