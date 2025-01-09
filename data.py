import zarr
import mrcfile
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import glob
import torch
import numpy as np
from einops import rearrange
import copick
import random

class CryoETDataset(Dataset):
    def __init__(self, path, crop_size, epoch_length=1000, run_ids=[]):

        self.data = glob.glob(path)
        self.crop_size = crop_size
        self.epoch_length = epoch_length
        root = copick.from_file("data/config.json")

        self.tomograms = [(run.name, run.voxel_spacings[0].get_tomograms("denoised")[0].numpy()) for run in root.runs if run.name in run_ids]

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):
            
        idx = random.randint(0, len(self.tomograms) - 1)
        run_name, tomogram = self.tomograms[idx]
        i, j, k = [random.randint(0, tomogram.shape[i] - self.crop_size[i]) for i in range(3)]

        labels = np.load("data/preprocessed/labels/" + run_name + ".npy")

        tomogram = tomogram[i:i+self.crop_size[0], j:j+self.crop_size[1], k:k+self.crop_size[2]]
        labels = labels[i:i+self.crop_size[0], j:j+self.crop_size[1], k:k+self.crop_size[2]]

        return {
            'tomograms': tomogram,
            'labels': labels,
        }

class SyntheticCryoETDataset(Dataset):
    def __init__(self, path, crop_size, epoch_length=1000):

        self.crop_size = crop_size
        self.epoch_length = epoch_length
        self.data_shape = (200, 630, 630)

        self.runs = glob.glob(path)
        self.tomograms = []
        for p in self.runs:
            tomogram_path = p + "Reconstructions/VoxelSpacing10.000/Tomograms/100"
            tomogram_path = glob.glob(tomogram_path + "/*.mrc")[0]
            self.tomograms.append(tomogram_path) 

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):
            
        idx = random.randint(0, len(self.tomograms) - 1)

        path = self.tomograms[idx]
        run_name = path.split("/")[-6]
        
        i, j, k = [random.randint(0, self.data_shape[i] - self.crop_size[i]) for i in range(3)]

        # Get tomogram crop
        tomogram = mrcfile.mmap(path, mode='r').data[i:i+self.crop_size[0], j:j+self.crop_size[1], k:k+self.crop_size[2]].copy()
        tomogram = torch.from_numpy(tomogram).float()
        labels = np.load("data/preprocessed/synthetic_labels/" + run_name + ".npy", mmap_mode='r')[i:i+self.crop_size[0], j:j+self.crop_size[1], k:k+self.crop_size[2]].copy()
        labels = torch.from_numpy(labels).long()

        return {
            'tomograms': tomogram,
            'labels': labels,
        }