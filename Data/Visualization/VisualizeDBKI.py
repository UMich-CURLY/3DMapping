import glob
import os
import sys
import argparse
import time
from datetime import datetime
import random
import numpy as np
import copy
from matplotlib import cm
import open3d as o3d

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
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
    (0, 0, 0),     # Ground
    (150, 100, 100), # Bridge
    (230, 150, 140), # RailTrack
    (180, 165, 180), # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160), # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),   # Water
    (145, 170, 100), # Terrain
]) / 255.0 # normalize each channel [0-1] since is what Open3D uses

def add_open3d_axis(vis):
    """Add a small 3D axis on Open3D Visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)


def main(args):
    try:
        load_dir = args["load_dir"]
        point_list = o3d.geometry.PointCloud()

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name='Segmented Scene',
            width=960,
            height=540,
            left=480,
            top=270)
        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 3
        vis.get_render_option().show_coordinate_frame = True

        frame = 0
        while True:
            # Load points
            frame_str = load_dir + str(frame).zfill(6)
            labels = np.fromfile(frame_str + ".label", dtype=np.uint32)
            points = np.fromfile(frame_str + ".bin", dtype=np.float32).reshape(-1, 3)
            non_free = labels != 0
            points = points[non_free, :]
            new_points = np.zeros(points.shape)
            new_points[:, 0] = points[:, 1]
            new_points[:, 1] = points[:, 0]
            new_points[:, 2] = points[:, 2]
            points = new_points
            labels = labels[non_free]

            # Fill in voxels
            voxel_resolution = 0.4
            N, __ = points.shape
            num_samples = 50
            new_points = np.reshape(points, (N, 3, 1))
            new_points = np.random.uniform(new_points - voxel_resolution/2, new_points + voxel_resolution/2, (N, 3, num_samples))
            new_labels = np.zeros((N, num_samples), dtype=np.uint32)
            new_labels = new_labels + labels.reshape(-1, 1)

            points = np.transpose(new_points, axes=(0, 2, 1)).reshape(-1, 3)
            labels = new_labels.reshape(-1)

            print(points.shape, labels.shape)

            int_color = LABEL_COLORS[labels]
            point_list.points = o3d.utility.Vector3dVector(points)
            point_list.colors = o3d.utility.Vector3dVector(int_color)

            if frame == 0:
                vis.add_geometry(point_list)

            # Update vis
            vis.update_geometry(point_list)

            # Sleep in a loop
            for i in range(5000):
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.005)

            frame += 1

    finally:
        vis.destroy_window()


if __name__ == "__main__":
    args = {
        "load_dir": "/home/tigeriv/Data/Carla/Data/Scenes/Town01_Heavy/dbki/evaluation/all/"
    }

    try:
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')