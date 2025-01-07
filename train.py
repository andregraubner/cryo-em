from segformer3d import SegFormer3D
from data import CryoETDataset
import torch
from torch.utils.data import Dataset, DataLoader
from utils import jaccard_loss, create_submission
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

criterion = DiceLoss(to_onehot_y=True, softmax=True)

#model = torch.nn.DataParallel(SegFormer3D(in_channels=1, num_classes=7)).to(device)

#model = torch.nn.DataParallel(SwinUNETR(
#    #img_size=(64, 64, 64),
#    in_channels=1,
#    out_channels=7,
#    feature_size=48,
#    use_checkpoint=True,
#)).to(device)

# Create dataset
dataset = CryoETDataset(
    "/scratch2/andregr/cryo-em/data/preprocessed/tensors/*.npy",
    run_ids=["TS_5_4", "TS_6_4", "TS_6_6", "TS_69_2", "TS_73_6", "TS_86_3"]
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)
class_weights = [1,1,0,2,1,2,1]
class_weights = torch.tensor(class_weights).float().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 0, len(dataloader) * 60)

#criterion = DiceLoss(to_onehot_y=True, softmax=True, weight=class_weights)

train_transforms = Compose([
    #RandSpatialCropSamplesD(keys=["tomograms", "labels"], roi_size=(128,128,128), num_samples=1, random_size=False),
    RandRotate90D(keys=["tomograms", "labels"], prob=0.5, spatial_axes=(1, 2)),
    RandFlipD(keys=["tomograms", "labels"], prob=0.5, spatial_axis=0),
    RandFlipD(keys=["tomograms", "labels"], prob=0.5, spatial_axis=1),
    RandFlipD(keys=["tomograms", "labels"], prob=0.5, spatial_axis=2),
    #RandSpatialCropD(keys=["tomograms", "labels"], roi_size=(64,64,64), random_size=False),
    RandGaussianNoiseD(keys=["tomograms"], prob=0.5, mean=0.0, std=0.1)
    ])
train_transforms.set_random_state(seed=123)

# Example iteration
for epoch in range(60):
    model.train()
    for batch in tqdm(dataloader):

        optimizer.zero_grad()

        batch["tomograms"] = (batch["tomograms"] - batch["tomograms"].mean()) / batch["tomograms"].std()
        batch = train_transforms(batch)

        #tomograms = torch.cat([b["tomograms"][:,None] for b in batch]).to(device)
        #labels = torch.cat([b["labels"].long() for b in batch]).to(device)
        tomograms = batch["tomograms"][:,None].to(device)
        labels = batch["labels"][:,None].long().to(device)

        out = model(tomograms)

        #loss = jaccard_loss(out, labels, eps=1e-3)
        #loss = F.cross_entropy(out, labels)#, weight=class_weights)
        loss = criterion(out, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        wandb.log({
            "loss": loss.item(),
            "lr": scheduler.get_last_lr()[0]
        })
        

    if epoch % 20 == 0:

        outputs = out.argmax(1)[0].cpu().numpy()

        #save_image(make_grid(labels[0,::5,None,:,:].float(), normalize=True), "labels.jpg")
        #save_image(make_grid(torch.tensor(outputs[::5,None,:,:]).float(), normalize=True), "preds.jpg")

        score = inference(model, "TS_99_9")
        wandb.log({"score": score})

score = inference(model, "TS_99_9")
wandb.log({"score": score})