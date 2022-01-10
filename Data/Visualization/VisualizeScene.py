import open3d as o3d
import time
import numpy as np
import os
import json
import pdb
from PIL import Image
import psutil

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


# vectorized point generation
def gen_points(counts, labels, min_dim, max_dim, num_samples, vis, cylindrical=True):
    """
        min_dim     -
        max_dim     -
        steps       1x3
        c
    """
    intervals = (max_dim - min_dim) / counts.shape
    x = np.linspace(min_dim[0], max_dim[0], num=counts.shape[0]) + intervals[0] / 2
    y = np.linspace(min_dim[1], max_dim[1], num=counts.shape[1]) + intervals[1] / 2
    z = np.linspace(min_dim[2], max_dim[2], num=counts.shape[2]) + intervals[2] / 2
    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")

    valid_cells = counts >= 1
    valid_x = xv[valid_cells]
    valid_y = yv[valid_cells]
    valid_z = zv[valid_cells]
    labels = labels[valid_cells]

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

    return point_list


# test code
def main():
    vis = o3d.visualization.Visualizer()
    parent_dir = "../Scenes/Town03_Heavy/"
    cylindrical = False
    num_samples = 100  # samples per cell
    try:
        if cylindrical:
            load_dir = parent_dir + "cylindrical/evaluation/"
        else:
            load_dir = parent_dir + "cartesian/evaluation/"

        # Load params
        with open(load_dir + "params.json") as f:
            params = json.load(f)
            grid_shape = [params["num_channels"]] + list(params["grid_size"])
            grid_shape = [int(i) for i in grid_shape][1:]
            min_dim = params["min_bound"]
            max_dim = params["max_bound"]
        
        vis.create_window(
            window_name='Segmented Scene',
            width=960,
            height=540,
            left=480,
            top=270)
        vis.get_render_option().background_color = [0.0, 0.0, 0.0]
        vis.get_render_option().point_size = 5
        vis.get_render_option().show_coordinate_frame = True

        # Load frames
        first_frame = True
        frame = 0
        geometry = None
        im = None
        while True:
            print("frame:", frame)

            counts = np.fromfile(load_dir + str(frame).zfill(6) + ".bin", dtype="float32").reshape(grid_shape)
            labels = np.fromfile(load_dir + str(frame).zfill(6) + ".label", dtype="uint32").reshape(grid_shape)

            point_list = gen_points(counts, labels, np.array(min_dim), np.array(max_dim), int(num_samples), vis, cylindrical=cylindrical)
            
            if first_frame:
                geometry = o3d.geometry.PointCloud(point_list)
                vis.add_geometry(geometry)
                first_frame = False
            else:
                geometry.points = point_list.points
                geometry.colors = point_list.colors

            # Display Scene
            vis.update_geometry( geometry)

            # Close BEV
            if im:
                im.close()
                for proc in psutil.process_iter():
                    # check whether the process name matches
                    if proc.name() == "eog":
                        proc.kill()
            # Show next BEV
            im = Image.open(load_dir + "../bev/" + str(frame).zfill(6) + ".jpg")
            w, h = im.size
            im = im.resize((int(w * 0.25), int(h * 0.25)))
            im.show()

            for i in range(5000):
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.005)

            frame += 1
    
    finally:
        vis.destroy_window()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
