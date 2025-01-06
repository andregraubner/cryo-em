import torch
from torch.autograd import Variable
from torch import nn
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import label, center_of_mass
import pandas as pd

label_to_particle_type = {
    1: "apo-ferritin",
    2: "beta-amylase",
    3: "beta-galactosidase",
    4: "ribosome",
    5: "thyroglobulin",
    6: "virus-like-particle"
}

def create_submission(preds, experiment):

    centroids_by_class = {}

    for class_label in range(1, 7):  # Skip class 0 (background)
        # Create a binary mask for the current class
        class_mask = (preds == class_label)

        # Label connected components for the current class
        labeled_volume, num_features = label(class_mask)

        # Compute centroids for connected components
        centroids = center_of_mass(class_mask, labeled_volume, range(1, num_features + 1))

        # Store centroids in the dictionary
        centroids_by_class[class_label] = centroids

    submission_data = []

    id_counter = 0
    for label_id, centroids in centroids_by_class.items():
        particle_type = label_to_particle_type.get(label_id, "unknown")
        for centroid in centroids:
            submission_data.append({
                "id": id_counter,
                "experiment": experiment,
                "particle_type": particle_type,
                "x": centroid[2] * 10,  # Centroid format is (z, y, x)
                "y": centroid[1] * 10,
                "z": centroid[0] * 10
            })
            id_counter += 1

    # Convert to DataFrame
    submission_df = pd.DataFrame(submission_data)
    return submission_df

def jaccard_loss(logits, true, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Modified to handle n-dimensional inputs.
    
    Args:
        true: a tensor of shape [B, *spatial_dims] or [B, 1, *spatial_dims].
        logits: a tensor of shape [B, C, *spatial_dims]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    
    # Handle arbitrary spatial dimensions
    spatial_dims = true.shape[2:] if true.shape[1] == 1 else true.shape[1:]
    
    # Reshape true to [B, -1] to use torch.eye
    true_flat = true.reshape(true.shape[0], -1)
    if true.shape[1] == 1:
        true_flat = true_flat.squeeze(1)
    
    # Convert to one-hot encoding
    true_1_hot = torch.eye(num_classes, device=true.device)[true_flat]
    
    # Reshape back to match logits shape
    new_shape = (true.shape[0], num_classes) + spatial_dims
    true_1_hot = true_1_hot.reshape(new_shape)
    
    probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    
    # Calculate dims for reduction
    dims = (0,) + tuple(range(2, len(true_1_hot.shape)))
    
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    
    return (1 - jacc_loss)