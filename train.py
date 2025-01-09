from segformer3d import SegFormer3D
from data import CryoETDataset
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
from monai.networks.nets import SwinUNETR, UNETR
import wandb
from monai.networks.nets import UNet
from monai.losses import DiceLoss, DiceFocalLoss
import transformers
import monai

wandb.init(project="cryo-em")

device = "cuda" if torch.cuda.is_available() else "cpu"

weights = torch.load("synthetic_model.pth")

"""
model = torch.nn.DataParallel(UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=7,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=4,
)).to(device)
"""

model = torch.nn.DataParallel(UNETR(img_size=(64,64,64), in_channels=1, out_channels=7, feature_size=32, norm_name='batch', spatial_dims=3)).to(device)

#model = torch.nn.DataParallel(SwinUNETR(img_size=(64,64,64), in_channels=1, out_channels=7, num_heads=(3, 6, 12, 24), feature_size=24)).to(device)

#model.load_state_dict(weights)

# Create dataset
dataset = CryoETDataset(
    "/scratch2/andregr/cryo-em/data/preprocessed/tensors/*.npy",
    crop_size=(64,64,64),
    epoch_length=10000,
    run_ids=["TS_5_4", "TS_6_4", "TS_6_6", "TS_69_2", "TS_73_6", "TS_86_3"]#, "TS_99_9"]
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

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 0, len(dataloader) * 100)

dice_loss = DiceLoss(to_onehot_y=True, softmax=True, batch=True)

train_transforms = Compose([
    RandRotate90D(keys=["tomograms", "labels"], prob=0.5, spatial_axes=(1, 2)),
    RandFlipD(keys=["tomograms", "labels"], prob=0.5, spatial_axis=0),
    RandFlipD(keys=["tomograms", "labels"], prob=0.5, spatial_axis=1),
    RandFlipD(keys=["tomograms", "labels"], prob=0.5, spatial_axis=2),
    RandGaussianNoiseD(keys=["tomograms"], prob=0.5, mean=0.0, std=0.1)
])
train_transforms.set_random_state(seed=123)

samples_seen = 0
# Example iteration
for epoch in range(1000):
    model.train()
    for batch in tqdm(dataloader):

        optimizer.zero_grad()

        batch["tomograms"] = (batch["tomograms"] - batch["tomograms"].mean()) / batch["tomograms"].std()
        batch = train_transforms(batch)

        tomograms = batch["tomograms"][:,None].to(device)
        labels = batch["labels"][:,None].long().to(device)

        out = model(tomograms)

        loss = dice_loss(out, labels) #+ 0.1 * F.cross_entropy(out, labels[:,0], weight=class_weights)

        loss.backward()
        optimizer.step()
        scheduler.step()
        samples_seen += tomograms.shape[0]

        wandb.log({
            "loss": loss.item(),
            "lr": scheduler.get_last_lr()[0],
            "samples": samples_seen
        })
        
    if epoch % 1 == 0:
        score = inference(model, "TS_99_9")
        wandb.log({"score": score})
        torch.save(model.state_dict(), "model.pth")

score = inference(model, "TS_99_9")
wandb.log({"score": score})

torch.save(model.state_dict(), "model.pth")