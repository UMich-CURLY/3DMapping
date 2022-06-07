import numpy as np
#import open3d as o3d
import time
import torch
import torch.nn as nn
import random

from MotionSC import MotionSC
from SSCNet_full import SSCNet_full
from LMSCNet_SS import LMSCNet_SS
from SSCNet import SSCNet

frequencies_cartesian = np.asarray([
    4166593275,
    42309744,
    8550180,
    478193,
    905663,
    2801091,
    6452733,
    229316930,
    112863867,
    29816894,
    13839655,
    15581458,
    221821,
    0,
    7931550,
    467989,
    3354,
    9201043,
    61011,
    3796746,
    3217865,
    215372,
    79669695
    ])

LABEL_COLORS = np.array([
    (255, 255, 255), # None
    (70, 70, 70),    # Building
    (100, 40, 40),   # Fences
    (55, 90, 80),    # Other
    (255, 255, 0),   # Pedestrian
    (153, 153, 153), # Pole
    (0, 0, 255),  # RoadLines
    (0, 0, 255),  # Road
    (255, 255, 255),  # Sidewalk
    (0, 155, 0),  # Vegetation
    (255, 0, 0),     # Vehicle
    (102, 102, 156), # Wall
    (220, 220, 0),   # TrafficSign
    (70, 130, 180),  # Sky
    (255, 255, 255),     # Ground
    (150, 100, 100), # Bridge
    (230, 150, 140), # RailTrack
    (180, 165, 180), # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160), # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),   # Water
    (145, 170, 100), # Terrain
]) / 255.0 # normalize each channel [0-1] since is what Open3D uses


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# Intersection, union for one frame
def iou_one_frame(pred, target, n_classes=23):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = np.zeros(n_classes)
    union = np.zeros(n_classes)

    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection[cls] = (pred_inds[target_inds]).long().sum().item()  # Cast to long to prevent overflows
        union[cls] = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection[cls]
    return intersection, union


def resample_free_space(flattened_preds, flattened_output):
    # Occupied indices
    occupied_indices = (flattened_output != 0).nonzero().view(-1)
    num_occupied = occupied_indices.shape[0]
    # Select 2 N free inds randomly, without replacement
    free_indices = (flattened_output == 0).nonzero()
    free_indices = free_indices[np.random.choice(free_indices.shape[0], int(2 * num_occupied), replace=False)].view(-1)
    # Combine and resample predictions, outputs
    all_indices = torch.cat((occupied_indices, free_indices)).long()
    return flattened_preds[all_indices], flattened_output[all_indices]

"""
def visualize_set(model, dataloader, carla_ds, cylindrical, min_thresh=0.75):
    model.eval()
    vis = None
    geometry = None
    with torch.no_grad():
        running_loss = 0.0
        counter = 0
        num_correct = 0
        num_total = 0
        for input_data, output, counts in dataloader:
            input_data = torch.tensor(input_data).to(carla_ds.device)
            preds = model(input_data)

            probs = nn.functional.softmax(preds, dim=4)
            vis, geometry = visualize_preds(probs[0].detach().cpu().numpy(), 
                            np.asarray(carla_ds._eval_param['min_bound']), 
                            np.asarray(carla_ds._eval_param['max_bound']),
                            cylindrical=cylindrical, vis=vis, geometry=geometry, min_thresh=min_thresh)
"""

def get_model(model_name, num_classes, voxel_sizes, coor_ranges, grid_dim, device):
    # Model parameters
    resample_free = False
    if model_name == "MotionSC":
        B = 8
        T = 10
        model = MotionSC(voxel_sizes, coor_ranges, grid_dim, T=T, device=device, num_classes=num_classes)
        decayRate = 0.96
    elif model_name == "LMSC":
        B = 4
        T = 1
        decayRate = 0.98
        model = LMSCNet_SS(num_classes, grid_dim, frequencies_cartesian).to(device)
    elif model_name == "SSC":
        B = 4
        T = 1
        decayRate = 1.00
        resample_free = True
        lr = 0.001
        model = SSCNet(num_classes).to(device)
    elif model_name == "SSC_Full":
        B = 4
        T = 1
        decayRate = 1.00
        resample_free = True
        lr = 0.001
        model = SSCNet_full(num_classes).to(device)
    else:
        print("Please choose either MotionSC, LMSC, or SSC / SSC_Full. Thank you.")
        exit()
    model.weights_init()
    return model, B, T, decayRate, resample_free
