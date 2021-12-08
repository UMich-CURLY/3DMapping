import os
import pdb
import glob
import numpy as np
from numpy.lib.function_base import trim_zeros
import json

from ShapeContainer import ShapeContainer

# New
def initialize_grid(grid_size=np.array([100., 100., 10.]),
        min_bound=np.array([0, -1.0*np.pi, 0], dtype=np.float32),
        max_bound=np.array([20, 1.0*np.pi, 10], dtype=np.float32),
        num_channels=25,
        coordinates="cylindrical"):
        
    cylinder_mat = ShapeContainer(grid_size, min_bound, max_bound, num_channels, coordinates)
    return cylinder_mat

# Original
def load_times(seq_dir, t_start, t_end, dt, save_dir):
    """
    Loads times from reading time file name numbers
    """
    times = []

    t_from = open(seq_dir + "times0.txt", "r")
    for t_stamp in t_from.readlines():
        t_frame = t_stamp.split(", ")[0]
        if int(t_frame) < t_start:
            continue
        if int(t_frame) >= t_end:
            continue
        t_frame = (float(t_frame) - t_start) * dt
        times.append(t_frame)
    t_from.close()

    times = np.array(times)
    np.savetxt(save_dir + 'times.txt', times)

    return times

# Original
def get_frame_str(t_frame, t_start, t_end):
    if int(t_frame) < t_start:
        return None
    if int(t_frame) >= t_end:
        return None
    t_frame = int(t_frame) - t_start
    return str(t_frame).zfill(6)

def load_poses(sensors, save_dir, seq_dir, t_start, t_end):
    """
    Loads poses from sensors, sorts poses, and computes transformation matrix from 
    all poses to pose 0 (ego sensor)
    """
    # Original view
    poses = {}
    sorted_poses_all = {}
    inv_first = None 
    for sensor in sensors:
        poses[sensor] = {}
        # Get poses
        for pose_file in os.listdir(seq_dir + "pose" + str(sensor)):
            t_frame = pose_file.split(".")[0]
            frame_str = get_frame_str(t_frame, t_start, t_end)
            if frame_str:
                pose = np.load(seq_dir + "pose" + str(sensor) + "/" + pose_file)
                poses[sensor][frame_str] = pose

        # Sort poses
        sorted_poses = [poses[sensor][fr] for fr in sorted(poses[sensor].keys())]
        # Make first pose origin
        if sensor == 0:
            inv_first = np.linalg.inv(sorted_poses[0])
        for i in range(len(sorted_poses)):
            sorted_poses[i] = np.matmul(inv_first, sorted_poses[i])
            sorted_poses[i] = sorted_poses[i].reshape(-1)[:12]

        sorted_poses = np.array(sorted_poses)
        if sensor == 0:
            np.savetxt(save_dir + '/poses.txt', sorted_poses)
        sorted_poses_all[sensor] = sorted_poses

    return poses, sorted_poses_all, inv_first

# Labels for ego sensor
def save_labels(view_num, query, save_dir, seq_dir, t_start, t_end):
    to_folder = save_dir + "labels"
    if query:
        to_folder += "_query"
    to_folder += "/"
    
    if not os.path.exists(to_folder):
        os.mkdir(to_folder)
    
    for label_file in os.listdir(seq_dir + "labels" + str(view_num)):
        t_frame = label_file.split(".")[0]
        frame_str = get_frame_str(t_frame, t_start, t_end)
        if frame_str:
            label = np.load(seq_dir + "labels" + str(view_num) + "/" + label_file)
            label.astype('uint32').tofile(to_folder + frame_str + ".label")

# Labels
def save_points(view_num, query, save_dir, seq_dir, t_start, t_end):
    to_folder_points = save_dir + "velodyne"
    to_folder_flow = save_dir + "predictions"
    if query:
        to_folder_points += "_query"
        to_folder_flow += "_query"
    to_folder_points += "/"
    to_folder_flow += "/"
    
    if not os.path.exists(to_folder_points):
        os.mkdir(to_folder_points)
    if not os.path.exists(to_folder_flow):
        os.mkdir(to_folder_flow)
    
    for point_file in os.listdir(seq_dir + "velodyne" + str(view_num)):
        t_frame = point_file.split(".")[0]
        frame_str = get_frame_str(t_frame, t_start, t_end)
        if frame_str:
            points = np.load(seq_dir + "velodyne" + str(view_num) + "/" + point_file) # Point cloud
            instances = np.load(seq_dir + "instances" + str(view_num) + "/" + point_file) # Instances per point
            velocities = np.load(seq_dir + "velocities" + str(view_num) + "/" + point_file) # Per-instance velocity
            flow = np.zeros(points.shape, dtype=np.float32) # Actual movement of things
            for row in velocities:
                ind = int(row[0])
                velocity = row[1:]
                flow[instances == ind] = velocity
 
            points = np.c_[points, np.zeros(points.shape[0])] # Dummy intensity
            points.astype('float32').tofile(to_folder_points + frame_str + ".bin")
            flow.astype('float32').tofile(to_folder_flow + frame_str + ".bin")

# Helper to for creating data used in voxel grid
def get_info(sensor, frame_str, seq_dir):
    points = np.load(seq_dir + "velodyne" + str(sensor) + "/" + frame_str + ".npy") # Point cloud
    instances = np.load(seq_dir + "instances" + str(sensor) + "/" + frame_str + ".npy") # Instances per point
    velocities = np.load(seq_dir + "velocities" + str(sensor) + "/" + frame_str + ".npy") # Per-instance velocity
    flow = np.zeros(points.shape, dtype=np.float32) # Actual movement of things
    for row in velocities:
        ind = int(row[0])
        velocity = row[1:]
        flow[instances == ind] = velocity
    label = np.load(seq_dir + "labels" + str(sensor) + "/" + frame_str + ".npy")
    return [points, instances, flow, label]

def get_inv_transforms(sensors, seq_dir, t_start, t_end):
    # Initial transforms (other lidar to ego sensor)
    inv_transforms = {} 
    for pose_file in os.listdir(seq_dir + "pose0"):
        t_frame = pose_file.split(".")[0]
        frame_str = get_frame_str(t_frame, t_start, t_end)
        if frame_str:
            ego_pose = np.load(seq_dir + "pose0/" + pose_file)
            for sensor in sensors:
                sensor_pose = np.load(seq_dir + "pose" + str(sensor) + "/" + pose_file)
                inv_sensor = np.linalg.inv(sensor_pose)
                inv_transforms[sensor] = np.matmul(ego_pose, inv_sensor)
    return inv_transforms

# Add points to cylinder grid
def add_points(points, labels, grid):
    pointlabels = np.hstack((points, labels))
    grid[pointlabels] = grid[pointlabels] + 1
    return grid

# Add points along ray to grid
def ray_trace(point, label, sample_spacing):
    vec_norm = np.linalg.norm(point)
    vec_angle = point / vec_norm
    # Iterate from sample inwards
    dists = np.arange(vec_norm, 0, -sample_spacing)
    new_points = np.reshape(dists, (-1, 1)) * np.reshape(vec_angle, (1, 3))
    labels = [0] * new_points.shape[0]
    # End Point is label, free points all 0
    labels[0] = label
    return new_points, np.asarray(labels).reshape(-1, 1)

def main():
    """
    Initialize settings and data structures
    """
    t_start = 5
    t_end = 195
    dt = 0.1
    seq_dir = "../Scenes/02/raw/"
    save_dir = "../Scenes/02/cylindrical/"
    free_res = 1.5
    
    # Parameters for container: cylindrical
    grid_size = np.array([100., 100., 10.])
    min_bound = np.array([0, -1.0*np.pi, -3], dtype=np.float32)
    max_bound = np.array([40, 1.0*np.pi, 2], dtype=np.float32)
    num_channels = 25
    coordinates = "cylindrical"
    
    # Parameters for container: cartesian
#    grid_size = np.array([100., 100., 10.])
#    min_bound = np.array([-20, -20, -2.5], dtype=np.float32)
#    max_bound = np.array([20, 20, 2.5], dtype=np.float32)
#    num_channels = 25
#    coordinates = "cartesian"

    # Initialize grid
    voxel_grid = initialize_grid(grid_size=grid_size,
        min_bound=min_bound,
        max_bound=max_bound,
        num_channels=num_channels,
        coordinates=coordinates)
        
    # Save params
    params = {  "grid_size": grid_size.tolist(),
                "min_bound": min_bound.tolist(),
                "max_bound": max_bound.tolist(),
                "num_channels": num_channels,
                "coordinates": coordinates
    }

    # Load sensors data
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    sensors = glob.glob(seq_dir + "velodyne*")
    sensors = sorted([int(sensor.split("velodyne")[1]) for sensor in sensors])
    try:
        ego_sensor = sensors[1]
    except IndexError:
        print("Error: less than one sensor found, exiting...")
        return


    # Input data
    times = load_times(seq_dir, t_start, t_end, dt, save_dir)
    poses, sorted_poses_all, inv_first = load_poses(sensors, save_dir, seq_dir, t_start, t_end)
    save_labels(0, False, save_dir, seq_dir, t_start, t_end)
    save_points(0, False, save_dir, seq_dir, t_start, t_end)

    # Get transforms from lidars to ego sensor
    inv_transforms = get_inv_transforms(sensors, seq_dir, t_start, t_end)

    # Whether any measurements were found
    if not os.path.exists(os.path.join(save_dir, "evaluation")):
        os.mkdir(os.path.join(save_dir, "evaluation"))
    
    with open(os.path.join(save_dir, "evaluation/params.json"), "w") as f:
        json.dump(params, f)

    # Loop over frames
    for i in range(t_start, t_end):
        voxel_grid.reset_grid()
        t_frame = str(i)
        frame_str = get_frame_str(t_frame, t_start, t_end)

        if frame_str:
            for sensor in sensors:
                [points, instances, flow, label] = get_info(sensor, t_frame, seq_dir) # Get info for sensor at frame
                for i in range(points.shape[0]):
                    temp_points, temp_labels = ray_trace(points[i, :], label[i], free_res)
                    transformed_points = np.matmul(temp_points, inv_transforms[sensor][:3, :3]) # Convert points to ego frame
                    transformed_points = transformed_points + inv_transforms[sensor][:3, 3]
                    voxel_grid = add_points(transformed_points, temp_labels, voxel_grid)
                    
            # Save volume
            voxels = voxel_grid.get_voxels()
            voxels = np.transpose(voxels, axes=(3, 0, 1, 2))
            print(voxels.shape)
            voxels.astype('float32').tofile(os.path.join(save_dir, "evaluation/",  frame_str + ".bin"))


if __name__ == '__main__':
    main()
