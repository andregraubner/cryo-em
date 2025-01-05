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
import xarray as xr
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import copick
import os
import shutil

# First, rename json files to appease copick
source_dir = '/scratch2/andregr/cryo-em/data/train/overlay'
destination_dir = '/scratch2/andregr/cryo-em/data/'

for root, _, files in os.walk(source_dir):
    for file in files:
        if file.endswith(".json"):
            old_path = os.path.join(root, file)
            if old_path.find("curation_0_") != -1:
                continue
            new_path = os.path.join(root, "curation_0_" + file)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")

root = copick.from_file("config.json")
objects = [obj for obj in root.pickable_objects if obj.is_particle]

# Now, create segmentation masks for each run
for run in tqdm(root.runs):

    tomogram = root.runs[0].voxel_spacings[0].get_tomograms("denoised")[0].numpy()
    tomogram = (tomogram - 2.06502e-07) / 5.5327368e-11 # normalize tomogram

    labels = np.zeros_like(tomogram)
    z, y, x = np.ogrid[:labels.shape[0], :labels.shape[1], :labels.shape[2]]
    
    for obj in objects:
        picks = root.runs[0].get_picks(object_name=obj.name)[0]
        for point in picks.points:
            dist = np.sqrt((z - point.location.z / 10)**2 + (y - point.location.y / 10)**2 + (x - point.location.x / 10)**2)
            sphere_mask = dist <= obj.radius / 10
            labels[sphere_mask] = obj.label

    # Rearrange into patches
    tomogram_patches = rearrange(
        tomogram[2:-2],
        '(d pd) (h ph) (w pw) -> (d h w) pd ph pw',
        pd=90, ph=90, pw=90
    )
    
    label_patches = rearrange(
        labels[2:-2],
        '(d pd) (h ph) (w pw) -> (d h w) pd ph pw',
        pd=90, ph=90, pw=90
    )
    
    # Convert to list of patches
    tomogram_patches = list(tomogram_patches)
    label_patches = list(label_patches)

    for idx, (tomogram, labels) in enumerate(zip(tomogram_patches, label_patches)):
        data = {
            "tomogram": tomogram,
            "labels": labels,
        }
        np.save(f"preprocessed/{run.name}_{idx}.npy", data)