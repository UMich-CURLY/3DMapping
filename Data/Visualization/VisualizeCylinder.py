import open3d as o3d
import time
import numpy as np
import os
import json

LABEL_COLORS = np.array([
    (0, 0, 0), # None
    (70, 70, 70),    # Building
    (100, 40, 40),   # Fences
    (55, 90, 80),    # Other
    (255, 255, 0),   # Pedestrian
    (153, 153, 153), # Pole
    (157, 234, 50),  # RoadLines
    (0, 0, 50),  # Road
    (0, 0, 50),  # Sidewalk
    (107, 142, 35),  # Vegetation
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

# container should be a C x R x T x Z numpy array containing class probabilities that sum to 1 over the channel dimension
# min dim should be the lower bound on the axis (e.g. [0, 0, 0])
# max dim should be the upper bound on the axis (e.g. [25, 2*pi, 50]) for a full cylinder of radius 25 and height 50
def vis_cyl(container, min_dim, max_dim, num_samples): 
    # Create visualizer 
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

    # compute steps
    r_step = (max_dim[0] - min_dim[0])/container.shape[0]
    t_step = (max_dim[1] - min_dim[1])/container.shape[1]
    z_step = (max_dim[2] - min_dim[2])/container.shape[2]

    points_raw = []
    colors_raw = []

    for i in range(0, num_samples):
        # pick a random cell in the container
        cell = [np.random.randint(0, i) for i in container.shape]

        # determine the cylindrical bounds on that cell
        r_bound = [min_dim[0] + r_step * cell[1], min_dim[0] + r_step * (cell[1] + 1)]
        t_bound = [min_dim[1] + t_step * cell[2], min_dim[1] + t_step * (cell[2] + 1)]
        z_bound = [min_dim[2] + z_step * cell[3], min_dim[2] + z_step * (cell[3] + 1)]

        # sample a random cylindrical coordinate from the cell 
        r = np.random.uniform(r_bound[0], r_bound[1])
        t = np.random.uniform(t_bound[0], t_bound[1])
        z = np.random.uniform(z_bound[0], z_bound[1])

        # choose a class based on class probabilities 
        p = np.array(container[:, cell[1], cell[2], cell[3]]) # distribution over classes
        c = np.random.choice(container.shape[0], 1, p=p/p.sum())[0]

        # convert coordinate to cartesian 
        x = r * np.cos(t)
        y = r * np.sin(t)

        # add to point lists
        points_raw.append([x, y, z])
        try:
            colors_raw.append(list(LABEL_COLORS[c]))
        except:
            c = 0
            colors_raw.append(list(LABEL_COLORS[c]))
    
    points = np.array(points_raw)
    colors = np.array(colors_raw)

    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(colors)

    # Add to visualizer
    vis.add_geometry(point_list)

    # Graphics loop
    while(True):
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.005)


# test code
if __name__ == "__main__":
    load_dir = "../Scenes/03/03_processed/evaluation/"
    # Load params
    with open(load_dir + "params.json") as f:
        params = json.load(f)
        grid_shape = [params["num_channels"]] + list(params["grid_size"])
        grid_shape = [int(i) for i in grid_shape]
        min_dim = params["min_bound"]
        max_dim = params["max_bound"]
    # Load frames
    for frame in os.listdir(load_dir):
        if not frame.endswith("bin"):
            continue
        c = np.fromfile(load_dir + frame, dtype="float32").reshape(grid_shape)
        c[0, :, :, :] += 1e-6
        c = c / np.sum(c, axis=0)
        vis_cyl(c, min_dim, max_dim, 1000000)
