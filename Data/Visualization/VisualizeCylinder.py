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

# vectorized point generation
def gen_points(container, min_dim, max_dim,  num_samples, vis):
    """
        min_dim     -
        max_dim     -
        steps       1x3
        c
    """

    # r x t x z x c
    # c x r x t x z
    start_time = time.time()
    # Generate random cell indices num_samplesx3
    cell = np.random.randint([0, 0, 0], high=container.shape[1:], size=(num_samples, 3))
    
    # compute steps
    steps = (max_dim - min_dim)/container.shape[1:] 

    bound_low = min_dim + steps * cell
    bound_high = min_dim + steps * (cell+1)

    # sample a random cylindrical coordinate from the cell, num_samplex3
    cyl_coords = np.random.uniform(bound_low, bound_high, (num_samples, 3))
    
    #  num_samples x C
    p = container[:, cell[:, 0], cell[:, 1], cell[:, 2]].T
    original_num_samples = num_samples
    # Remove free points
    mask = (p[:, 0] != 1)
    p = p[mask]
    cyl_coords = cyl_coords[mask]
    cell = cell[mask]
    num_samples = len(p)
    print("free sampled points:", (original_num_samples - num_samples) / original_num_samples)

    # choose a class based on class probabilities 
    prob_sum        = np.cumsum(p, axis=1) # num_cyl_cells x classes 
    rand_samples    = np.random.rand(num_samples, 1)
    label          = (rand_samples < prob_sum).argmax(axis=1) # num_samples x 1
    
    truelabel = (label[:] < 23)
    label = label[truelabel]
        
    # convert coordinate to cartesian 
    x = (cyl_coords[:, 0] * np.cos(cyl_coords[:, 1])).reshape(-1, 1)
    y = (cyl_coords[:, 0] * np.sin(cyl_coords[:, 1])).reshape(-1, 1)
    z = (cyl_coords[:, 2]).reshape(-1, 1)
    #pdb.set_trace()
    y *= -1
    points_cart = np.hstack((x, y, z))
    
    # add to point lists
    point_list = o3d.geometry.PointCloud()     
    point_list.points = o3d.utility.Vector3dVector(points_cart)
    point_list.colors = o3d.utility.Vector3dVector(LABEL_COLORS[label])

    print("gen_points: Total runtime ", time.time()-start_time)
    return point_list

# test code

def main():
    vis = o3d.visualization.Visualizer()
    try: 
        load_dir = "../Generation/Scenes/01/01_processed/evaluation/"      
        # Load params
        with open(load_dir + "params.json") as f:
            params = json.load(f)
            grid_shape = [params["num_channels"]] + list(params["grid_size"])
            grid_shape = [int(i) for i in grid_shape]
            min_dim = params["min_bound"]
            max_dim = params["max_bound"]
        
        vis.create_window(
            window_name='Segmented Scene',
            width=960,
            height=540,
            left=480,
            top=270)
        vis.get_render_option().background_color = [0.0, 0.0, 0.0]
        vis.get_render_option().point_size = 3
        vis.get_render_option().show_coordinate_frame = True

        # Load frames
        frame = 0
        
        while True:
            print("frame:", frame)

            c = np.fromfile(load_dir + str(frame).zfill(6) + ".bin", dtype="float32").reshape(grid_shape)
            c[0, :, :, :] += 1e-6
            c = c / np.sum(c, axis=0)
            
            point_list = gen_points(c, np.array(min_dim), np.array(max_dim), 10000000, vis)

            vis.add_geometry(point_list)
            if frame < 2:
                for i in range(50):
                    vis.poll_events()
                    vis.update_renderer()
                    time.sleep(0.005)
            else:
                vis.poll_events()
                vis.update_renderer()
            
            vis.remove_geometry(point_list)

            frame += 1
    
    finally:
        vis.destroy_window()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')