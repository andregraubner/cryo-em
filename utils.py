import torch
from torch.autograd import Variable
from torch import nn
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import label, center_of_mass
import pandas as pd
import cc3d
import matplotlib.pyplot as plt
from tqdm import tqdm

label_to_particle_type = {
    1: "apo-ferritin",
    2: "beta-amylase",
    3: "beta-galactosidase",
    4: "ribosome",
    5: "thyroglobulin",
    6: "virus-like-particle"
}

#faster version using cuda
#https://github.com/kornia/kornia/blob/9ccae8c297a00a35d811b5a6e4f468a1d54d17f4/kornia/contrib/connected_components.py#L7
#https://stackoverflow.com/questions/46840707/efficiently-find-centroid-of-labelled-image-regions
def find_connected_component(probability, max_radius=100):
    device = probability.device
    probability = probability.detach()
    num_particle_type = 6
    D, H, W = probability.shape
    mask = F.one_hot(probability, num_classes=7).bool().permute(3,0,1,2)[1:]

    # allocate the output tensors for labels
    out = (torch.arange(D * H * W, device=device, dtype=torch.float32)+1).reshape(1, D, H, W)
    out = out.repeat(num_particle_type, 1, 1, 1)
    out[~mask] = 0

    out = out.reshape(num_particle_type, 1, D, H, W)
    mask = mask.reshape(num_particle_type, 1, D, H, W)
    for _ in range(max_radius):
        out = F.max_pool3d(out, kernel_size=3, stride=1, padding=1)
        out = torch.mul(out, mask)  # mask using element-wise multiplication
    out = out.reshape(num_particle_type, D, H, W)
    out = out.short()
    
    component=[]
    
    for i in range(num_particle_type):
        u, inverse = torch.unique(out[i], sorted=True, return_inverse=True)

        out[i] = inverse
        #component.append(inverse)
    #component = torch.stack(component)

    return out
    
def find_centroid(component):
    device = component.device
    num_particle_type, D, H, W = component.shape
    count = component.flatten(1).max(-1)[0]+1
    cumcount = torch.zeros(num_particle_type+1, dtype=torch.int32, device=device)
    cumcount[1:] = torch.cumsum(count,0)
    component = component+cumcount[:-1].reshape(num_particle_type,1,1,1)

    gridz = torch.arange(0, D, device=device).reshape(1,D,1,1).expand(num_particle_type,-1,H,W)
    gridy = torch.arange(0, H, device=device).reshape(1,1,H,1).expand(num_particle_type,D,-1,W)
    gridx = torch.arange(0, W, device=device).reshape(1,1,1,W).expand(num_particle_type,D,H,-1)
    n  = torch.bincount(component.flatten())
    nx = torch.bincount(component.flatten(),weights=gridx.flatten())
    ny = torch.bincount(component.flatten(),weights=gridy.flatten())
    nz = torch.bincount(component.flatten(),weights=gridz.flatten())

    x=nx/n
    y=ny/n
    z=nz/n
    zyx = torch.stack([z,y,x],1).float()
    zyx = torch.split(zyx, count.tolist(), dim=0)
    centroid = [zzyyxx[1:] for zzyyxx in zyx]
    return centroid 

def create_submission(preds, experiment):

    with torch.no_grad():
        components = find_connected_component(preds)
        centroids = find_centroid(components)

    submission_data = []

    id_counter = 0
    for label_id, centroids in enumerate(centroids):
        label_id += 1  # Skip class 0 (background)
        particle_type = label_to_particle_type.get(label_id, "unknown")
        for centroid in centroids:
            submission_data.append({
                "id": id_counter,
                "experiment": experiment,
                "particle_type": particle_type,
                "x": centroid[2].item() * 10,  # Centroid format is (z, y, x)
                "y": centroid[1].item() * 10,
                "z": centroid[0].item() * 10
            })
            id_counter += 1 

    # Convert to DataFrame
    submission_df = pd.DataFrame(submission_data)
    print(submission_df)
    return submission_df