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
    def __init__(self, path, crop_size, run_ids=[]):

        self.data = glob.glob(path)
        self.crop_size = crop_size
        root = copick.from_file("data/config.json")

        self.tomograms = [(run.name, run.voxel_spacings[0].get_tomograms("denoised")[0].numpy()) for run in root.runs if run.name in run_ids]

    def __len__(self):
        return 5000

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