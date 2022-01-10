import open3d as o3d
import time
import numpy as np
import os
import json
import pdb
from PIL import Image
import psutil

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
    print(container.shape)
    cell = np.random.randint([0, 0, 0], high=container.shape[1:], size=(num_samples, 3))
    
    #  num_samples x C
    p = container[:, cell[:, 0], cell[:, 1], cell[:, 2]].T
    original_num_samples = num_samples
    # Remove free points
    mask = (p[:, 0] != 1)
    p = p[mask]
    cell = cell[mask]
    num_samples = len(p)
    print("free sampled points:", (original_num_samples - num_samples) / original_num_samples)
    
    # compute steps
    steps = (max_dim - min_dim)/container.shape[1:]

    bound_low = min_dim + steps * cell
    bound_high = min_dim + steps * (cell+1)

    # sample a random cylindrical coordinate from the cell, num_samplex3
    cyl_coords = np.random.uniform(bound_low, bound_high, (num_samples, 3))
    
    # choose a class based on class probabilities 
    prob_sum        = np.cumsum(p, axis=1) # num_cyl_cells x classes 
    rand_samples    = np.random.rand(num_samples, 1)
    label          = (rand_samples < prob_sum).argmax(axis=1) # num_samples x 1
    
    truelabel = (label[:] <= 23)
    label = label[truelabel]
        
    # convert coordinate to cartesian
    # x = cyl_coords[:, 0].reshape(-1, 1)
    # y = cyl_coords[:, 1].reshape(-1, 1)
    x = (cyl_coords[:, 0] * np.cos(cyl_coords[:, 1])).reshape(-1, 1)
    y = (cyl_coords[:, 0] * np.sin(cyl_coords[:, 1])).reshape(-1, 1)
    z = (cyl_coords[:, 2]).reshape(-1, 1)
    #pdb.set_trace()
    points_cart = np.hstack((x, y, z))

    # Rotate
    new_points = np.zeros(points_cart.shape)
    new_points[:, 0] = points_cart[:, 1]
    new_points[:, 1] = points_cart[:, 0]
    new_points[:, 2] = points_cart[:, 2]
    
    # add to point lists
    point_list = o3d.geometry.PointCloud()     
    point_list.points = o3d.utility.Vector3dVector(new_points)
    point_list.colors = o3d.utility.Vector3dVector(LABEL_COLORS[label])

    print("gen_points: Total runtime ", time.time()-start_time)
    return point_list


# test code
def main():
    vis = o3d.visualization.Visualizer()
    try: 
        load_dir = "../Scenes/Town06_Light/cylindrical/evaluation/"
        num_samples = 5e7
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
        vis.get_render_option().point_size = 5
        vis.get_render_option().show_coordinate_frame = True

        # Load frames
        first_frame = True
        frame = 0
        geometry = None
        im = None
        while True:
            print("frame:", frame)

            c = np.fromfile(load_dir + str(frame).zfill(6) + ".bin", dtype="float32").reshape(grid_shape)
            c[0, :, :, :] += 1e-6
            c = c / np.sum(c, axis=0)

            point_list = gen_points(c, np.array(min_dim), np.array(max_dim), int(num_samples), vis)
            
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
