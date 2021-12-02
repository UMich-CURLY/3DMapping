import os
import pdb
import glob
import numpy as np
from numpy.lib.function_base import trim_zeros

from CylinderContainer import CylinderContainer
from Utils import LABEL_COLORS

# New
def initialize_grid(grid_lengths = np.array([40.0, 40.0, 4.0]), 
                    voxel_resolution = 0.2,
                    num_classes = 25):
    # Load grid settings
    grid_lengths = np.array([40.0, 40.0, 4.0]) # xyz dims
    grid_dim = 2 * grid_lengths / voxel_resolution
    grid_dim = np.array([
        int(grid_dim[0]), int(grid_dim[1]), int(grid_dim[2]), num_classes]) # 0 is free
    print(grid_dim)

    # Cylinder cells
    cells_per_dim = grid_dim[0:3]

    min_bound = np.array([-grid_lengths[0], -2.0*np.pi, -grid_lengths[2]])
    max_bound = np.array([grid_lengths[0], 2.0*np.pi, grid_lengths[2]])
    default_voxel = np.array([0]*num_classes, dtype=np.uint8)

    cylinder_mat = CylinderContainer(   cells_per_dim, min_bound, max_bound, 
                                        default_voxel_val=default_voxel)
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
    print(times)
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
            values, counts = np.unique(label, return_counts=True)
            print(values, counts)

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
    
    grid[pointlabels] += 1

# Add points along ray to grid
def ray_trace(point, label, grid, sample_spacing):
    vec_norm = np.linalg.norm(point)
    vec_angle = point / vec_norm
    # Iterate from lidar outwards to preserve even spacing between points
    dists = np.arange(0, vec_norm, sample_spacing)
    new_points = np.reshape(dists, (-1, 1)) * np.reshape(vec_angle, (1, 3))
    labels = [0] * new_points.shape[0]
    # End Point is label, free points all 0
    labels[0] = label
    add_points(new_points, labels, grid)

def main():
    """
    Initialize settings and data structures
    """
    t_start = 16650
    t_end = 17000
    dt = 0.1
    seq_dir = "02/"
    save_dir = "02_processed/"

    # Initialize grid
    grid_lengths = np.array([40.0, 40.0, 4.0])
    voxel_resolution = 0.2
    num_classes = 25
    voxel_grid = initialize_grid(grid_lengths, voxel_resolution, num_classes)

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

    times = load_times(seq_dir, t_start, t_end, dt, save_dir)

    poses, sorted_poses_all, inv_first = \
        load_poses(sensors, save_dir, seq_dir, t_start, t_end)

    save_labels(0, False, save_dir, seq_dir, t_start, t_end)

    save_points(0, False, save_dir, seq_dir, t_start, t_end)

    # Get transforms from lidars to ego sensor
    inv_transforms = get_inv_transforms(sensors, seq_dir, t_start, t_end)

    # Whether any measurements were found
    if not os.path.exists(save_dir + "evaluation"):
        os.mkdir(save_dir + "evaluation")

    # Generate voxel grid from data
    x = np.arange(-grid_lengths[0], grid_lengths[0], voxel_resolution) + voxel_resolution/2
    y = np.arange(-grid_lengths[1], grid_lengths[1], voxel_resolution) + voxel_resolution/2
    z = np.arange(-grid_lengths[2], grid_lengths[2], voxel_resolution) + voxel_resolution/2
    xv, yv, zv = np.meshgrid(x, y, z)

    for pose_file in os.listdir(seq_dir + "pose0"):
        t_frame = pose_file.split(".")[0]
        frame_str = get_frame_str(t_frame, t_start, t_end)

        if frame_str:
            for sensor in sensors:
                [points, instances, flow, label] = get_info(sensor, t_frame, seq_dir) # Get info for sensor at frame
                transformed_points = np.matmul(points, inv_transforms[sensor][:3, :3]) # Convert points to ego frame
                transformed_points = transformed_points + inv_transforms[sensor][3, :3]
                for i in range(transformed_points.shape[0]):
                    ray_trace(transformed_points[i, :], label[i], voxel_grid, voxel_resolution)
            # Save
            valid_cells = np.sum(voxel_grid, axis=3) > 0
            labels = np.argmax(voxel_grid, axis=3)
            values, counts = np.unique(labels[valid_cells], return_counts=True)
            print(frame_str, values, counts)
            valid_x = xv[valid_cells]
            valid_y = yv[valid_cells]
            valid_z = zv[valid_cells]
            valid_points = np.stack((valid_x, valid_y, valid_z)).T
            valid_labels = labels[valid_cells]
            valid_points.astype('float32').tofile(save_dir + "/evaluation/" + frame_str + ".bin")
            valid_labels.astype('uint32').tofile(save_dir + "/evaluation/" + frame_str + ".label")


if __name__ == '__main__':
    main()