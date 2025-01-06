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
import pandas as pd

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

    tomogram = run.voxel_spacings[0].get_tomograms("denoised")[0].numpy()

    labels = np.zeros_like(tomogram)
    z, y, x = np.ogrid[:labels.shape[0], :labels.shape[1], :labels.shape[2]]

    annotations = []
    for obj in objects:
        picks = run.get_picks(object_name=obj.name)[0]
        for point in picks.points:
            dist = np.sqrt((z - point.location.z / 10)**2 + (y - point.location.y / 10)**2 + (x - point.location.x / 10)**2)
            sphere_mask = dist <= obj.radius / 10
            labels[sphere_mask] = obj.label

            # Append data for this point
            annotations.append({
                "experiment": run.name,
                "particle_type": obj.name,
                "x": point.location.x,
                "y": point.location.y,
                "z": point.location.z
            })

    annotations = pd.DataFrame(annotations)
    annotations.to_csv(f"preprocessed/annotations/{run.name}.csv")

    # Rearrange into patches
    tomogram_patches = rearrange(
        tomogram[28:-28, 27:-27, 27:-27],
        '(d pd) (h ph) (w pw) -> (d h w) pd ph pw',
        pd=64, ph=64, pw=64
    )
    
    label_patches = rearrange(
        labels[28:-28, 27:-27, 27:-27],
        '(d pd) (h ph) (w pw) -> (d h w) pd ph pw',
        pd=64, ph=64, pw=64
    )
    
    # Convert to list of patches
    tomogram_patches = list(tomogram_patches)
    label_patches = list(label_patches)

    for idx, (tomogram, labels) in enumerate(zip(tomogram_patches, label_patches)):
        data = {
            "tomogram": tomogram,
            "labels": labels,
        }
        np.save(f"preprocessed/tensors/{run.name}_{idx}.npy", data)