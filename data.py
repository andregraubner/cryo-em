import zarr
import mrcfile
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import glob
import torch
import numpy as np

class CryoETDataset(Dataset):
    def __init__(self, base_path, ts_range=range(13), slice_range=range(100, 107)):
        self.base_path = Path(base_path)
        self.samples = []
        
        # Collect all paired tomogram and annotation files
        for ts_idx in ts_range:
            ts_name = f"TS_{ts_idx}"
            recon_path = self.base_path / ts_name / "Reconstructions/VoxelSpacing10.000"
            
            # Path to tomogram
            tomo_path = recon_path / "Tomograms" / "100" / f"{ts_name}.mrc"
            annotations = []

            anno_dirs = [recon_path / "Annotations" / str(idx) for idx in slice_range]
            anno_files = [list(anno_dir.glob("*.mrc"))[0] for anno_dir in anno_dirs]

            self.samples.append({
                "tomogram": tomo_path,
                "annotations": anno_files
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load tomogram
        with mrcfile.open(sample['tomogram']) as mrc:
            tomogram = torch.from_numpy(mrc.data.astype('float32'))
            
        # Load annotation
        labels = []
        for annotation in sample['annotations']:
            with mrcfile.open(annotation) as mrc:
                annotation = torch.from_numpy(mrc.data.astype('float32'))
            
            labels.append(annotation)

        labels = torch.stack(labels, dim=0)
            
        return {
            'tomograms': tomogram,
            'labels': labels
        }

base_path = "/scratch2/andregr/cryo-em/data/10441"
    
# Create dataset
dataset = CryoETDataset(base_path)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Example iteration
for batch in dataloader:
    print(batch)
    break