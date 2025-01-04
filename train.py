from segformer3d import SegFormer3D
from data import CryoETDataset
import torch
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SegFormer3D(in_channels=1).to(device)

base_path = "/scratch2/andregr/cryo-em/data/10441"

# Create dataset
dataset = CryoETDataset(base_path)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Example iteration
for batch in dataloader:
    print(batch["tomograms"].shape, batch["labels"].shape)
    tomograms = batch["tomograms"][:,None,:128,:128,:128].to(device) # TODO: move slicing to dataset

    out = model(tomograms)
    break