from segformer3d import SegFormer3D
from data import CryoETDataset
import torch
from torch.utils.data import Dataset, DataLoader
from utils import create_submission
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import numpy as np
import pandas as pd
import copick
from sklearn.metrics import fbeta_score
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy import ndimage

import torch
import torch.nn.functional as F

def gaussian_kernel_3d(kernel_size, sigma):
    # Create a 3D grid of (x, y, z) coordinates
    coords = torch.meshgrid(
        [torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2] * 3
    )
    x, y, z = coords[0], coords[1], coords[2]

    # Compute the squared distance from the center of the kernel
    distance_squared = x**2 + y**2 + z**2

    # Compute the Gaussian function for each point in the grid
    kernel = torch.exp(-distance_squared / (2 * sigma**2))

    # Normalize the kernel so that the sum of all elements equals 1
    kernel = kernel / kernel.sum()
    return kernel

def inference(model, run_name):

    model.eval()

    root = copick.from_file("data/config.json")
    objects = [obj for obj in root.pickable_objects if obj.is_particle]
    tomogram = root.get_run(run_name).voxel_spacings[0].get_tomograms("denoised")[0].numpy()

    tomogram = torch.tensor(tomogram, device="cuda").float()
    tomogram = (tomogram - tomogram.mean()) / tomogram.std()
    tomogram = F.pad(tomogram, (5, 5, 5, 5), mode="reflect")

    preds = torch.zeros(size=(7, 184, 640, 640), device="cuda")
    counts = torch.zeros_like(preds) 
    with torch.inference_mode():
        for z in tqdm(range(0, 184 - 64 + 1, 32)):
            for y in tqdm(range(0, 640 - 64 + 1, 32)):
                for x in range(0, 640 - 64 + 1, 32):
                    inputs = tomogram[None,None,z:z+64,y:y+64,x:x+64]
                    out = model(inputs)[0]
                    preds[:, z:z+64,y:y+64,x:x+64] += out
                    counts[z:z+64,y:y+64,x:x+64] += 1

        preds /= counts

        kernel = gaussian_kernel_3d(11, 1)[None, None].cuda()
        preds = F.conv3d(preds[:,None], kernel, padding="same")[:,0]

        preds = preds.argmax(0)
        preds = preds[:,5:-5, 5:-5]

    plt.imsave("preds.png", preds[90].cpu().numpy())

    result = create_submission(preds, run_name)

    annotations = pd.read_csv(f"data/preprocessed/annotations/{run_name}.csv")

    s = score(annotations, result, distance_multiplier=0.5, beta=4)
    print(s)
    return s

class ParticipantVisibleError(Exception):
    pass

def compute_metrics(reference_points, reference_radius, candidate_points):
    num_reference_particles = len(reference_points)
    num_candidate_particles = len(candidate_points)

    if len(reference_points) == 0:
        return 0, num_candidate_particles, 0

    if len(candidate_points) == 0:
        return 0, 0, num_reference_particles

    ref_tree = KDTree(reference_points)
    candidate_tree = KDTree(candidate_points)
    raw_matches = candidate_tree.query_ball_tree(ref_tree, r=reference_radius)
    matches_within_threshold = []
    for match in raw_matches:
        matches_within_threshold.extend(match)
    # Prevent submitting multiple matches per particle.
    # This won't be be strictly correct in the (extremely rare) case where true particles
    # are very close to each other.
    matches_within_threshold = set(matches_within_threshold)
    tp = int(len(matches_within_threshold))
    fp = int(num_candidate_particles - tp)
    fn = int(num_reference_particles - tp)
    return tp, fp, fn

def score(
        solution: pd.DataFrame,
        submission: pd.DataFrame,
        distance_multiplier: float,
        beta: int) -> float:
    '''
    F_beta
      - a true positive occurs when
         - (a) the predicted location is within a threshold of the particle radius, and
         - (b) the correct `particle_type` is specified
      - raw results (TP, FP, FN) are aggregated across all experiments for each particle type
      - f_beta is calculated for each particle type
      - individual f_beta scores are weighted by particle type for final score
    '''

    particle_radius = {
        'apo-ferritin': 60,
        'beta-amylase': 65,
        'beta-galactosidase': 90,
        'ribosome': 150,
        'thyroglobulin': 130,
        'virus-like-particle': 135,
    }

    weights = {
        'apo-ferritin': 1,
        'beta-amylase': 0,
        'beta-galactosidase': 2,
        'ribosome': 1,
        'thyroglobulin': 2,
        'virus-like-particle': 1,
    }

    particle_radius = {k: v * distance_multiplier for k, v in particle_radius.items()}

    # Filter submission to only contain experiments found in the solution split
    split_experiments = set(solution['experiment'].unique())
    submission = submission.loc[submission['experiment'].isin(split_experiments)]

    # Only allow known particle types
    if not set(submission['particle_type'].unique()).issubset(set(weights.keys())):
        raise ParticipantVisibleError('Unrecognized `particle_type`.')

    assert solution.duplicated(subset=['experiment', 'x', 'y', 'z']).sum() == 0
    assert particle_radius.keys() == weights.keys()

    results = {}
    for particle_type in solution['particle_type'].unique():
        results[particle_type] = {
            'total_tp': 0,
            'total_fp': 0,
            'total_fn': 0,
        }

    for experiment in split_experiments:
        for particle_type in solution['particle_type'].unique():
            reference_radius = particle_radius[particle_type]
            select = (solution['experiment'] == experiment) & (solution['particle_type'] == particle_type)
            reference_points = solution.loc[select, ['x', 'y', 'z']].values

            select = (submission['experiment'] == experiment) & (submission['particle_type'] == particle_type)
            candidate_points = submission.loc[select, ['x', 'y', 'z']].values

            if len(reference_points) == 0:
                reference_points = np.array([])
                reference_radius = 1

            if len(candidate_points) == 0:
                candidate_points = np.array([])

            tp, fp, fn = compute_metrics(reference_points, reference_radius, candidate_points)

            results[particle_type]['total_tp'] += tp
            results[particle_type]['total_fp'] += fp
            results[particle_type]['total_fn'] += fn

    print(results)

    aggregate_fbeta = 0.0
    for particle_type, totals in results.items():
        tp = totals['total_tp']
        fp = totals['total_fp']
        fn = totals['total_fn']

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        fbeta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (precision + recall) > 0 else 0.0
        aggregate_fbeta += fbeta * weights.get(particle_type, 1.0)

    if weights:
        aggregate_fbeta = aggregate_fbeta / sum(weights.values())
    else:
        aggregate_fbeta = aggregate_fbeta / len(results)
    return aggregate_fbeta