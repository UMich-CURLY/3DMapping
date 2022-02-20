import open3d as o3d
import time
import numpy as np
import os
import json
import pdb
from functools import partial


def remove_static(flow):
    mask1a = (flow[:,0] != 0)
    mask1b =  (flow[:,1] != 0)
    mask1c = (flow[:,2] != 0)

    mask1 = np.all([mask1a, mask1b, mask1c],axis=0)

    return mask1


def gen_points(load_dir, frame, vis, calc_flow=False):
    """
        min_dim     -
        max_dim     -
        steps       1x3
        c
    """

    pc = np.fromfile(load_dir + "velodyne/" + str(frame).zfill(6) + ".bin", dtype=np.float32).reshape(-1, 4)[:, :3]
    to_world = np.loadtxt(load_dir + "poses.txt", dtype=np.float32).reshape(-1, 12)[int(frame), :].reshape(3, 4)
    flow = np.fromfile(load_dir + "predictions/" + str(frame).zfill(6) + ".bin", dtype=np.float32).reshape(-1, 3)

    mask = remove_static(flow)
    pc = pc[mask,:]

    if calc_flow:
        flow = flow[mask,:]
        flow = flow * 0.1
        pc = pc + flow

    pc = np.dot(to_world[:3, :3], pc.T).T + to_world[:3, 3]

    # add to point lists
    point_list = o3d.geometry.PointCloud()     
    point_list.points = o3d.utility.Vector3dVector(pc)
    cl = np.zeros_like(pc)

    # Current frame is green, next frame is blue
    if calc_flow:
        cl[:,0] = 0
        cl[:,1] = 1 # green
        cl[:,2] = 0
    else:
        cl[:,0] = 0
        cl[:,1] = 0
        cl[:,2] = 1 # blue
        
    point_list.colors = o3d.utility.Vector3dVector(cl)
    return point_list

def update_pc(vis):
    
    # Gloval vars
    global load_dir
    global frame
    global geometry

    frame +=1

    print("current frame:", frame)

    point_list = gen_points(load_dir, frame+1, vis, calc_flow=False) + gen_points(load_dir, frame, vis, calc_flow=True)
    geometry.points = point_list.points
    geometry.colors = point_list.colors
    vis.update_geometry(geometry)


def main():

    # Gloval vars
    global load_dir
    global frame
    global geometry

    vis = o3d.visualization.VisualizerWithKeyCallback()
    try: 
        load_dir = "/home/jason/Data/Carla/Test_Cartesian/Test/Town10_Heavy/cartesian/"
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
        point_list = gen_points(load_dir, frame+1, vis, calc_flow=False) + gen_points(load_dir, frame, vis, calc_flow=True)
        geometry = o3d.geometry.PointCloud(point_list)
        vis.add_geometry(geometry)

        while True:
            vis.register_key_callback(ord("A"), partial(update_pc))
            vis.poll_events()
            vis.update_renderer()
    
    finally:
        vis.destroy_window()

if __name__ == "__main__":
    try:
        print("press 'A' to advance frame")
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
