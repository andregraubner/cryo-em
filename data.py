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

class CryoETDataset(Dataset):
    def __init__(self, path):

        self.data = glob.glob(path)

        #self.class_weights = torch.zeros(7)
        #for labels in self.labels:
        #    self.class_weights += labels.sum(axis=(1,2,3))
        #self.class_weights /= self.class_weights.sum()
        #self.class_weights = 1 / self.class_weights

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
            
        data = np.load(self.data[idx], allow_pickle=True)

        return {
            'tomograms': data.item()["tomogram"],
            'labels': data.item()["labels"],
        }
