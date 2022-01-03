import open3d as o3d
import time
import numpy as np
import os
import json
import pdb

LABEL_COLORS = np.array([
    (0, 0, 0), # None
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


# vectorized point generation
def gen_points(load_dir, frame):
    """
        min_dim     -
        max_dim     -
        steps       1x3
        c
    """
    labels = np.fromfile(load_dir + "labels/" + str(frame).zfill(6) + ".label", dtype=np.uint32)
    pc = np.fromfile(load_dir + "velodyne/" + str(frame).zfill(6) + ".bin", dtype=np.float32).reshape(-1, 4)[:, :3]
    pose = np.loadtxt(load_dir + "poses.txt", dtype=np.float32).reshape(-1, 12)[int(frame), :].reshape(3, 4)
    # labels = np.load("/home/tigeriv/Data/Carla/Data/Scenes/03/raw/labels0/" + str(frame) + ".npy")
    # pc = np.load("/home/tigeriv/Data/Carla/Data/Scenes/03/raw/velodyne0/" + str(frame) + ".npy")
    # pose = np.load("/home/tigeriv/Data/Carla/Data/Scenes/03/raw/pose0/" + str(frame) + ".npy")

    # pc = np.append(pc, np.ones((pc.shape[0], 1)), axis=1)
    # pc = np.dot(pose, pc.T).T
    # pc = pc[:, :-1]
    # print(pc.shape)

    pc = np.dot(pose[:3, :3], pc.T).T + pose[:3, 3]
    print(pc.shape)
    
    # add to point lists
    point_list = o3d.geometry.PointCloud()     
    point_list.points = o3d.utility.Vector3dVector(pc)
    point_list.colors = o3d.utility.Vector3dVector(LABEL_COLORS[labels])
    return point_list

# test code

def main():
    vis = o3d.visualization.Visualizer()
    try: 
        load_dir = "../Scenes/04/cartesian/"
        sensor = 0
        vis.create_window(
            window_name='Segmented Scene',
            width=960,
            height=540,
            left=480,
            top=270)
        vis.get_render_option().background_color = [0.0, 0.0, 0.0]
        vis.get_render_option().point_size = 3

        # Load frames
        frame = 0
        point_list = gen_points(load_dir, frame)
        geometry = o3d.geometry.PointCloud(point_list)
        vis.add_geometry(geometry)
        while True:
            print("frame:", frame)

            new_list = gen_points(load_dir, frame)
            point_list = point_list + new_list
            # point_list = gen_points(load_dir, frame)
            
            geometry.points = point_list.points
            geometry.colors = point_list.colors

            vis.update_geometry(geometry)
            
            for i in range(10):
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
