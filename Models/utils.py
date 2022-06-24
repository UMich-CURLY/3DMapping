import numpy as np
import open3d as o3d
import time
import torch
import torch.nn as nn
import random
import os

from MotionSC import MotionSC
from SSCNet_full import SSCNet_full
from LMSCNet_SS import LMSCNet_SS
from SSCNet import SSCNet

from ptflops import get_model_complexity_info

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


def visualize_preds(probs, min_dim, max_dim, num_samples=20, vis=None, geometry=None, cylindrical=True, display_time=1, min_thresh=0.75):
    preds = np.argmax(probs, axis=3)
    max_probs = np.amax(probs, axis=3)
    intervals = (max_dim - min_dim) / preds.shape
    
    
    x = np.linspace(min_dim[0], max_dim[0], num=preds.shape[0]) + intervals[0] / 2
    y = np.linspace(min_dim[1], max_dim[1], num=preds.shape[1]) + intervals[1] / 2
    z = np.linspace(min_dim[2], max_dim[2], num=preds.shape[2]) + intervals[2] / 2
    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")

    valid_cells = max_probs > min_thresh 
    valid_x = xv[valid_cells]
    valid_y = yv[valid_cells]
    valid_z = zv[valid_cells]
    labels = preds[valid_cells]

    valid_points = np.stack((valid_x, valid_y, valid_z)).T
    non_free = labels != 0
    valid_points = valid_points[non_free, :]
    labels = labels[non_free]

    # Fill in voxels
    N, __ = valid_points.shape
    new_points = np.random.uniform((valid_points - intervals / 2).reshape(N, 3, 1),
                                   (valid_points + intervals / 2).reshape(N, 3, 1),
                                   (N, 3, num_samples))
    new_labels = np.zeros((N, num_samples), dtype=np.uint32)
    labels = (new_labels + labels.reshape(-1, 1)).reshape(-1)
    valid_points = np.transpose(new_points, (0, 2, 1)).reshape(-1, 3)

    if cylindrical:
        x = (valid_points[:, 0] * np.cos(valid_points[:, 1])).reshape(-1, 1)
        y = (valid_points[:, 0] * np.sin(valid_points[:, 1])).reshape(-1, 1)
        points = np.hstack((x, y, valid_points[:, 2:]))
    else:
        points = valid_points

    # swap axes
    new_points = np.zeros(points.shape)
    new_points[:, 0] = points[:, 1]
    new_points[:, 1] = points[:, 0]
    new_points[:, 2] = points[:, 2]
    points = new_points

    print(points.shape, labels.shape)

    int_color = LABEL_COLORS[labels]
    point_list = o3d.geometry.PointCloud()
    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)
    
    if not vis:  
        vis = o3d.visualization.Visualizer()
        vis.create_window(
        window_name='Segmented Scene',
        width=960,
        height=540,
        left=480,
        top=270)
        vis.get_render_option().background_color = [0.0, 0.0, 0.0]
        vis.get_render_option().point_size = 5
        vis.get_render_option().show_coordinate_frame = True
        geometry = o3d.geometry.PointCloud(point_list)
        vis.add_geometry(geometry)
    else:
        geometry.points = point_list.points
        geometry.colors = point_list.colors
        vis.update_geometry(geometry)
    
    for i in range(display_time):
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.005)
        
    return vis, geometry


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# Intersection, union for one frame
def iou_one_frame(pred, target, n_classes=23):
    pred = pred.view(-1).detach().cpu().numpy()
    target = target.view(-1).detach().cpu().numpy()
    intersection = np.zeros(n_classes)
    union = np.zeros(n_classes)

    for cls in range(n_classes):
        intersection[cls] = np.sum((pred == cls) & (target == cls))
        union[cls] = np.sum((pred == cls) | (target == cls))
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


def visualize_set(model, dataloader, carla_ds, cylindrical, min_thresh=0.75):
    model.eval()
    vis = None
    geometry = None
    with torch.no_grad():
        for input_data, output, counts in dataloader:
            input_data = torch.tensor(input_data).to(carla_ds.device)
            preds = model(input_data)

            probs = nn.functional.softmax(preds, dim=4)
            vis, geometry = visualize_preds(probs[0].detach().cpu().numpy(), 
                            np.asarray(carla_ds._eval_param['min_bound']), 
                            np.asarray(carla_ds._eval_param['max_bound']),
                            cylindrical=cylindrical, vis=vis, geometry=geometry, min_thresh=min_thresh)


def get_model(model_name, num_classes, voxel_sizes, coor_ranges, grid_dim, device, T=16):
    # Model parameters
    resample_free = False
    if model_name == "MotionSC":
        B = 16
        T = T
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


def measure_geom(dataloader_test, model, device):
    TP = 0.0
    FP = 0.0
    TN = 0.0
    FN = 0.0
    with torch.no_grad():
        for input_data, output, counts in dataloader_test:
            input_data = torch.tensor(input_data).to(device)
            output = torch.tensor(output).to(device)
            counts = torch.tensor(counts).to(device)

            # Predictions
            preds = model(input_data)
            counts = counts.view(-1)
            output = output.view(-1).long()
            preds = preds.contiguous().view(-1, preds.shape[4])
            probs = nn.functional.softmax(preds, dim=1)
            preds = torch.argmax(probs, dim=1)
            # Criterion requires input (NxC), output (N) dimension
            mask = counts > 0
            output_masked = output[mask].cpu().numpy()
            output_masked[output_masked != 0] = 1
            preds_masked = preds[mask].cpu().numpy()
            preds_masked[preds_masked != 0] = 1

            TP += np.sum((preds_masked == 1) & (output_masked == 1))
            FP += np.sum((preds_masked == 1) & (output_masked == 0))
            TN += np.sum((preds_masked == 0) & (output_masked == 0))
            FN += np.sum((preds_masked == 0) & (output_masked == 1))

    precision = 100 * TP / (TP + FP)
    recall = 100 * TP / (TP + FN)
    iou = 100 * TP / (TP + FP + FN)

    print("Completeness")
    print("precision:", precision)
    print("recall:", recall)
    print("iou:", iou)

    return precision, recall, iou


def measure_accuracy(dataloader_test, model, device):
    num_correct = 0.0
    num_total = 0.0
    with torch.no_grad():
        for input_data, output, counts in dataloader_test:
            input_data = torch.tensor(input_data).to(device)
            output = torch.tensor(output).to(device)
            counts = torch.tensor(counts).to(device)

            # Predictions
            preds = model(input_data)
            counts = counts.view(-1)
            output = output.view(-1).long()
            preds = preds.contiguous().view(-1, preds.shape[4])
            probs = nn.functional.softmax(preds, dim=1)
            preds = torch.argmax(probs, dim=1)
            # Criterion requires input (NxC), output (N) dimension
            mask = counts > 0
            output_masked = output[mask]
            preds_masked = preds[mask]

            # I, U for a frame
            num_correct += np.sum(output_masked.cpu().numpy() == preds_masked.cpu().numpy())
            num_total += preds_masked.shape[0]

    accuracy = num_correct/num_total
    print("Accuracy", )

    return accuracy


def measure_miou(dataloader_test, model, device, num_classes):
    all_intersections = np.zeros(num_classes)
    all_unions = np.zeros(num_classes) + 1e-6
    with torch.no_grad():
        for input_data, output, counts in dataloader_test:
            input_data = torch.tensor(input_data).to(device)
            output = torch.tensor(output).to(device)
            counts = torch.tensor(counts).to(device)

            # Predictions
            preds = model(input_data)
            counts = counts.view(-1)
            output = output.view(-1).long()
            preds = preds.contiguous().view(-1, preds.shape[4])
            probs = nn.functional.softmax(preds, dim=1)
            preds = torch.argmax(probs, dim=1)

            # Criterion requires input (NxC), output (N) dimension
            mask = counts > 0
            output_masked = output[mask]
            preds_masked = preds[mask]

            # I, U for a frame
            intersection, union = iou_one_frame(preds_masked, output_masked, n_classes=num_classes)
            all_intersections += intersection
            all_unions += union

    iou = all_intersections / all_unions
    print("Semantic IoU Per Class")
    for i in range(num_classes):
        print(100 * iou[i])

    return iou


def measure_inference_time(dataloader_test, model, device):
    total_time = 0.0
    repetitions = 0
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        for input_data, output, counts in dataloader_test:
            input_data = torch.tensor(input_data).to(device)
            starter.record()
            preds = model(input_data)
            ender.record()
            torch.cuda.synchronize()
            total_time += starter.elapsed_time(ender)
            repetitions += 1
            if repetitions >= 100:
                print("Inference Time ms:", total_time / repetitions)
                break
    return total_time/repetitions


def save_preds(test_ds, model, model_name, device):
    with torch.no_grad():
        for idx in range(test_ds.__len__()):
            # File path
            point_path = test_ds._velodyne_list[idx]
            paths = point_path.split("/")
            save_dir = os.path.join(*paths[:-2], model_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            fpath = os.path.join(save_dir, paths[-1].split(".")[0] + ".label")
            # Input data
            current_horizon, output, counts = test_ds.__getitem__(idx)
            input_data = torch.tensor(current_horizon).to(device)
            input_data = torch.unsqueeze(input_data, 0)
            # Label predictions
            preds = model(input_data)
            preds = nn.functional.softmax(preds, dim=4)
            preds = torch.argmax(preds, dim=4)
            preds = torch.squeeze(preds, dim=0)
            # Save data
            preds = preds.detach().cpu().numpy()
            preds.astype('uint32').tofile(fpath)


def count_parameters(model, T):
    macs, params = get_model_complexity_info(model, (T, 128, 128, 8), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    model_size = sum(p.numel() for p in model.parameters())
    print("Parameters:", model_size)