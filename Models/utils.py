import numpy as np
import open3d as o3d
import time
import torch
import torch.nn as nn
import random

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
    (157, 234, 50),  # RoadLines
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


def visualize_preds(probs, min_dim, max_dim, num_samples=10, vis=None, geometry=None, cylindrical=True, display_time=1, min_thresh=0.75):
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