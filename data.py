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
    def __init__(self, path, run_ids=[]):

        self.data = glob.glob(path)

        # filter for files where path basename starts with any of the run_ids
        self.data = [d for d in self.data if any([d.split("/")[-1].startswith(run_id) for run_id in run_ids])]

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
