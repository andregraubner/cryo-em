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

# Create segmentation masks for each synthetic run
synthetic_runs = glob.glob("/scratch2/andregr/cryo-em/data/10441/TS_*/",)
for p in synthetic_runs:
    print(p)
    labels = np.zeros(shape=(200, 630, 630))
    for i in range(1, 7):
        label_path = glob.glob(p + f"/Reconstructions/VoxelSpacing10.000/Annotations/10{i}/*.mrc")[0]
        labels += mrcfile.open(label_path).data * i

    # Get run name
    run_name = p.split("/")[-2]
    np.save(f'preprocessed/synthetic_labels/{run_name}.npy', labels.astype(np.uint8))

quit()
# Now, create segmentation masks for each real run
root = copick.from_file("config.json")
objects = [obj for obj in root.pickable_objects if obj.is_particle]

for run in tqdm(root.runs):

    tomogram = run.voxel_spacings[0].get_tomograms("denoised")[0].numpy()

    labels = np.zeros_like(tomogram, dtype=np.uint8)
    z, y, x = np.ogrid[:labels.shape[0], :labels.shape[1], :labels.shape[2]]

    annotations = []
    for obj in objects:
        picks = run.get_picks(object_name=obj.name)[0]
        for point in picks.points:
            dist = np.sqrt((z - point.location.z / 10)**2 + (y - point.location.y / 10)**2 + (x - point.location.x / 10)**2)
            sphere_mask = dist <= obj.radius / 10
            labels[sphere_mask] = int(obj.label)

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

    np.save(f'preprocessed/labels/{run.name}.npy', labels)