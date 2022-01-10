import os
import pdb
import glob
import numpy as np
from numpy.lib.function_base import trim_zeros
import json
import copy
from shutil import copyfile

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


def get_inv_transforms(sensors, seq_dir, t_frame):
    # Initial transforms (other lidar to ego sensor)
    inv_transforms = {}
    pose_file = t_frame + ".npy"
    ego_pose = np.load(seq_dir + "pose0/" + pose_file)
    for sensor in sensors:
        to_world = np.load(seq_dir + "pose" + str(sensor) + "/" + pose_file)
        to_ego = np.linalg.inv(ego_pose)
        inv_transforms[sensor] = np.matmul(to_ego, to_world)
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


# Add points along ray to grid
def ray_trace_batch(points, labels, sample_spacing):
    points = points[labels != 0]
    labels = labels[labels != 0]
    # Compute samples using array broadcasting
    vec_norms = np.reshape(np.linalg.norm(points, axis=1), (-1, 1))
    vec_angles = points / vec_norms
    difs = np.reshape(np.arange(0.0, 100.0, sample_spacing), (1, -1, 1))
    difs = np.reshape(vec_angles, (-1, 1, 3)) * difs
    new_samples = np.reshape(points, (-1, 1, 3)) - difs

    # Create labels
    new_labels = np.zeros((new_samples.shape[0], new_samples.shape[1]))
    new_labels[:, 0] = labels
    new_labels = new_labels.reshape(-1)

    # Remove points with dist < 0
    vec_dists = new_samples / np.reshape(vec_angles, (-1, 1, 3))
    first_pts = vec_dists[:, 0, 0]
    good_samples = np.reshape(new_samples[vec_dists[:, :, 0] > 0], (-1, 3))
    good_labels = new_labels[vec_dists[:, :, 0].reshape(-1) > 0]

    return good_samples, np.reshape(good_labels, (-1, 1))


def make_grid(voxel_grid, t_frame, sensors, inv_transforms, free_res, seq_dir):
    voxel_grid.reset_grid()
    for sensor in sensors:
        [points, instances, flow, label] = get_info(sensor, t_frame, seq_dir)  # Get info for sensor at frame
        temp_points, temp_labels = ray_trace_batch(points, label, free_res)
        transformed_points = np.dot(inv_transforms[sensor][:3, :3], temp_points.T).T + inv_transforms[sensor][:3, 3]
        # transformed_points = np.matmul(temp_points, inv_transforms[sensor][:3, :3])  # Convert points to ego frame
        # transformed_points = transformed_points + inv_transforms[sensor][:3, 3]
        voxel_grid = add_points(transformed_points, temp_labels, voxel_grid)
    return voxel_grid


def data_from_grid(voxel_grid, min_num=1):
    # Save volume
    voxels = voxel_grid.get_voxels()
    min_bound = voxel_grid.min_bound
    max_bound = voxel_grid.max_bound
    intervals = voxel_grid.intervals

    x = np.linspace(min_bound[0], max_bound[0], num=voxels.shape[0]) + intervals[0] / 2
    y = np.linspace(min_bound[1], max_bound[1], num=voxels.shape[1]) + intervals[1] / 2
    z = np.linspace(min_bound[2], max_bound[2], num=voxels.shape[2]) + intervals[2] / 2
    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")

    valid_cells = np.sum(voxels, axis=3) >= min_num
    valid_x = xv[valid_cells]
    valid_y = yv[valid_cells]
    valid_z = zv[valid_cells]
    valid_points = np.stack((valid_x, valid_y, valid_z)).T

    valid_labels = np.argmax(voxels[valid_cells], axis=1)

    return valid_points, valid_labels


def copy_bev(t_start, t_end, seq_dir, save_dir):
    if not os.path.exists(os.path.join(save_dir, "bev")):
        os.mkdir(os.path.join(save_dir, "bev"))
    for i in range(t_start, t_end):
        t_frame = str(i)
        frame_str = get_frame_str(t_frame, t_start, t_end)
        if frame_str:
            copyfile(os.path.join(seq_dir, "bev", t_frame + ".jpg"), os.path.join(save_dir, "bev", frame_str + ".jpg"))


def main():
    """
    Initialize settings and data structures
    """
    t_start = 65
    t_end = 265
    dt = 0.1
    parent_dir = "../Scenes/Town01_Heavy/"
    seq_dir = parent_dir + "raw/"
    save_dir = parent_dir + "dbki/"
    free_res = 1.5
    min_num = 1
    
    # Parameters for container: cylindrical
#     grid_size = np.array([100., 100., 10.])
#     min_bound = np.array([0, -1.0*np.pi, 0], dtype=np.float32)
#     max_bound = np.array([20, 1.0*np.pi, 10], dtype=np.float32)
#     num_channels = 25
#     coordinates = "cylindrical"
    
    # Parameters for container: cartesian
    grid_size = np.array([600., 600., 50.])
    min_bound = np.array([-30, -30, -2.5], dtype=np.float32)
    max_bound = np.array([30, 30, 2.5], dtype=np.float32)
    num_channels = 25
    coordinates = "cartesian"

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

    sensors = glob.glob(seq_dir + "/velodyne*")
    sensors = sorted([int(sensor.split("velodyne")[1]) for sensor in sensors])
    try:
        ego_sensor = sensors[0]
    except IndexError:
        print("Error: less than one sensor found, exiting...")
        return


    # Input data
    times = load_times(seq_dir, t_start, t_end, dt, save_dir)
    poses, sorted_poses_all, inv_first = load_poses(sensors, save_dir, seq_dir, t_start, t_end)
    save_labels(0, False, save_dir, seq_dir, t_start, t_end)
    save_points(0, False, save_dir, seq_dir, t_start, t_end)

    # Whether any measurements were found
    if not os.path.exists(os.path.join(save_dir, "evaluation")):
        os.mkdir(os.path.join(save_dir, "evaluation"))
    if not os.path.exists(os.path.join(save_dir, "evaluation", "completion")):
        os.mkdir(os.path.join(save_dir, "evaluation", "completion"))
        os.mkdir(os.path.join(save_dir, "evaluation", "completion", "occluded"))
        os.mkdir(os.path.join(save_dir, "evaluation", "completion", "visible"))
    if not os.path.exists(os.path.join(save_dir, "evaluation", "accuracy")):
        os.mkdir(os.path.join(save_dir, "evaluation", "accuracy"))
    if not os.path.exists(os.path.join(save_dir, "evaluation", "all")):
        os.mkdir(os.path.join(save_dir, "evaluation", "all"))

    copy_bev(t_start, t_end, seq_dir, save_dir)
        
    with open(save_dir + "/evaluation/params.json", "w") as f:
        json.dump(params, f)

    # Loop over frames
    for i in range(t_start, t_end):
        t_frame = str(i)
        frame_str = get_frame_str(t_frame, t_start, t_end)

        if frame_str:
            # Get transforms from lidars to ego sensor
            inv_transforms = get_inv_transforms(sensors, seq_dir, t_frame)

            all_grid = make_grid(voxel_grid, t_frame, sensors, inv_transforms, free_res, seq_dir)
            all_points, all_labels = data_from_grid(all_grid, min_num=min_num)
            print("All:", np.unique(all_labels, return_counts=True))

            ego_grid = make_grid(voxel_grid, t_frame, [sensors[0]], inv_transforms, free_res, seq_dir)
            ego_points, ego_labels = data_from_grid(ego_grid, min_num=min_num)

            temp_ego_points = []
            temp_ego_labels = []
            occ_points = []
            occ_labels = []
            ego_point_set = set(tuple(ego_points[j, :]) for j in range(ego_labels.shape[0]))
            for j in range(all_labels.shape[0]):
                if tuple(all_points[j, :]) not in ego_point_set:
                    occ_points.append(all_points[j, :])
                    occ_labels.append(all_labels[j])
                # V2
                else:
                    temp_ego_points.append(all_points[j, :])
                    temp_ego_labels.append(all_labels[j])
            ego_points = np.array(temp_ego_points)
            ego_labels = np.array(temp_ego_labels)

            occ_points = np.array(occ_points)
            occ_labels = np.array(occ_labels)
            print("Ego:", np.unique(ego_labels, return_counts=True))
            print("Occluded:", np.unique(occ_labels, return_counts=True))

            occ_points.astype('float32').tofile(save_dir + "/evaluation/completion/occluded/" + frame_str + ".bin")
            occ_labels.astype('uint32').tofile(save_dir + "/evaluation/completion/occluded/" + frame_str + ".label")
            ego_points.astype('float32').tofile(save_dir + "/evaluation/completion/visible/" + frame_str + ".bin")
            ego_labels.astype('uint32').tofile(save_dir + "/evaluation/completion/visible/" + frame_str + ".label")
            ego_points.astype('float32').tofile(save_dir + "/evaluation/accuracy/" + frame_str + ".bin")
            ego_labels.astype('uint32').tofile(save_dir + "/evaluation/accuracy/" + frame_str + ".label")
            all_points.astype('float32').tofile(save_dir + "/evaluation/all/" + frame_str + ".bin")
            all_labels.astype('uint32').tofile(save_dir + "/evaluation/all/" + frame_str + ".label")


if __name__ == '__main__':
    main()
