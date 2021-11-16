import os
import numpy as np
import yaml

"""
    Cylinder Container takes a cartesian xyz coordinate and 
    allows users to access which cell it is in for cylindrical coordinates

    The cell volume can be configured using the cell size ctor arg
    The total volume of the cylinder cells can be set using the grid size ctor arg

    Source Code Referenced From: 
    https://github.com/xinge008/Cylinder3D/blob/cc2116c8cb99b8c23c186691f344b1d95ee0d04f/dataloader/dataset_nuscenes.py#L99-L111
"""
class CylinderContainer:
    def __init__(self, grid_size,
        max_bound=np.array([2*np.pi, 50, np.pi/2.0], dtype=np.float32),
        min_bound=np.array([0, 0, -np.pi/2.0], dtype=np.float32)):
        """
        Constructor that creates the cylinder coordinate container

        :param grid_size: The number of cells that we want to divide the grid into

        """
        self.grid_size = grid_size
        self.max_bound = max_bound
        self.min_bound = min_bound

        # Class variables to be set by reset_grid
        self.intervals = None
        self.voxels = None

        self.reset_grid()

    def reset_grid(self):
        """
        Recomputes voxel grid and intializes all values to 0

        Condition:  Requires that grid_size, max_bound, and min_bound be set prior to 
                    calling function
        """
        crop_range = self.max_bound - self.min_bound
        self.intervals = crop_range / (self.grid_size - 1)

        if (self.intervals == 0).any(): 
            print("Error zero interval detected...")
            return

        # Initialize voxel grid with float32
        self.voxels = np.array([0]*self.grid_size, dtype=np.float32)

        print("Initialized voxel grid with {num_cells} cells".format(
            num_cells=self.grid_size))


    def __len__(self):
        return self.grid_size

    def __getitem__(self, input_xyz):
        """
        Returns the voxel centroid that the cartesian coordinate falls in

        :param xyz: nx3 np array where rows are points and cols are x,y,z
        """
        return self.voxels[self.grid_ind(input_xyz)]

    def __setitem__(self, input_xyz, input_value):
        """
        Sets the voxel to the input cell (cylindrical coordinates)

        :param input_xyz: nx3 np array where rows are points and cols are x,y,z
        """
        self.voxels[self.grid_ind(input_xyz)] = input_value

    def grid_ind(self, input_xyz):
        xyz_pol = self.cart2polar(input_xyz)

        grid_ind = (np.floor((np.clip(xyz_pol, self.min_bound, self.max_bound) 
                    - self.min_bound) / self.intervals)).astype(np.int)
        return grid_ind

    def get_voxel_centers(self, input_xyz):
        """
        Return voxel centers corresponding to each input xyz cartesian coordinate

        :param input_xyz: nx3 np array where rows are points and cols are x,y,z
        """
        input_xyz_size = np.shape(input_xyz)[0]
        voxel_position = np.zeros(input_xyz_size, dtype=np.float32)
        dim_array = np.ones(len(input_xyz_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(input_xyz_size) * self.intervals.reshape(dim_array) \
                            + self.min_bound.reshape(dim_array)
        voxel_position = self.polar2cat(voxel_position)
        
        # Center data on each voxel centroid for cylindrical coordinate PTnet
        voxel_centers = (self.grid_ind(input_xyz).astype(np.float32) + 0.5) * \
                        self.intervals + self.min_bound
        return voxel_centers

    def cart2polar(self, input_xyz):
        rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
        phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
        return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


    def polar2cat(self, input_xyz_polar):
        x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
        y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
        return np.stack((x, y, input_xyz_polar[2]), axis=0)


