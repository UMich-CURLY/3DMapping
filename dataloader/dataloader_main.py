import os
import pdb
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import trace
from numpy.random.mtrand import rand
from mpl_toolkits.mplot3d import Axes3D

from cylinder_container import CylinderContainer
from trace import trace_cells
from utils import cat_idx_dict, Voxel

"""
Driver file for loading data into cylindrical coordinate voxels
"""

def load_dummy_point_cloud(min_bound, max_bound, num_points):
    rand_radius = np.random.uniform(    min_bound[0], max_bound[0], 
                                        num_points).reshape((num_points, 1))
    rand_azimuth = np.random.uniform(   min_bound[1], max_bound[1], 
                                        num_points).reshape((num_points, 1))
    rand_height = np.random.uniform(    min_bound[2], max_bound[2], 
                                        num_points).reshape((num_points, 1))

    random_points = np.hstack((rand_radius, rand_azimuth, rand_height))

    return random_points

def plot_points(points, fig=None, marker='o', color=(1, 0, 0)):
    ax = []
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.gca()

    ax.scatter3D(points[:,0], points[:,1], points[:,2], marker=marker, color=color)
    # ax.scatter(points[:,0], points[:,1], marker='o', color=color)
    return fig

def main():
    # TODO: Run simulation here

    # Initializes cylinder container with 1000 cells and default range settings
    # 
    # Note: Not making container size scale with the number of
    #       occupied cells because we want to ray trace in empty cells too
    num_points = 10
    num_cells = 10

    min_bound = np.array([0, -2.0*np.pi, 0])
    max_bound = np.array([20, 2.0*np.pi, 20])

    default_voxel = Voxel()
    cells = CylinderContainer(  num_cells, min_bound, max_bound, 
                                default_voxel_val=default_voxel)

    random_points = cells.polar2cart(load_dummy_point_cloud(min_bound, max_bound, num_points))
    
    # fig = plot_points(random_points)

    # Visualize points mapped to centroid
    voxel_centers = cells.get_voxel_centers(random_points)

    # fig = plot_points(voxel_centers, fig, 'x', (0, 0, 1))

    # Center of coordinate system 
    start = np.array([0, 0, 10])
    for i in range(0, len(random_points)-1):
        curr = random_points[i, :]
        pdb.set_trace()
        points = trace_cells(cells, start, curr, cat_idx_dict['bg'], cat_idx_dict['other'])
        fig = plot_points(voxel_centers, fig, 'x', (0, 0, 1))

    plt.show()




if __name__ == '__main__':
    main()