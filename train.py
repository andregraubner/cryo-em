from segformer3d import SegFormer3D
from data import CryoETDataset
import torch
from torch.utils.data import Dataset, DataLoader
from utils import jaccard_loss
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.nn.DataParallel(SegFormer3D(in_channels=1, num_classes=7)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Create dataset
dataset = CryoETDataset("/scratch2/andregr/cryo-em/data/preprocessed/*.npy")

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)
#class_weights = dataset.class_weights.to(device)

# Example iteration
for epoch in range(10):
    for batch in tqdm(dataloader):

        optimizer.zero_grad()
        tomograms = batch["tomograms"][:,None].to(device)
        labels = batch["labels"].long().to(device)

        tomograms = (tomograms - tomograms.mean()) / tomograms.std()

        out = model(tomograms)[:,:,1:-1, 1:-1, 1:-1]

        #loss = jaccard_loss(out, labels, eps=1e-3)
        loss = F.cross_entropy(out, labels)#, weight=class_weights)

        loss.backward()
        optimizer.step()

        print(loss.item())

    save_image(make_grid(labels[0,::10,None,:,:].float()), "labels.jpg")
    save_image(make_grid(out.argmax(1)[0,::10,None,:,:].float()), "preds.jpg")