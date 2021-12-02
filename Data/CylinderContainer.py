import os
import pdb
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
        min_bound=np.array([0, -2.0*np.pi, 0], dtype=np.float32),
        max_bound=np.array([20, 2.0*np.pi, 10], dtype=np.float32),
        default_voxel_val= 0 ):
        """
        Constructor that creates the cylinder coordinate container

        :param grid_size: 1x3 np array that represents the number of cells in the
                          radius, azimuth, and height dimensions
        :param max_bound: [max radial distance, max_azimuth, max_height]
        :param min_bound: [min radial distance, min_azimuth, min_height]
        :param default_voxel_val: default object to initialize for each voxel cell
        """
        self.grid_size = grid_size
        self.num_classes = len(default_voxel_val)
        self.max_bound = max_bound
        self.min_bound = min_bound

        # Class variables to be set by reset_grid
        self.intervals = None
        self.voxels = None

        self.reset_grid(default_voxel_val)

    def reset_grid(self, default_voxel_val):
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
        self.voxels = np.array([default_voxel_val]*np.prod(self.grid_size)).reshape(
                tuple(np.append(self.grid_size, [self.num_classes]))
        )

        print("Initialized voxel grid with {num_cells} cells".format(
            num_cells=np.prod(self.grid_size)))


    def __len__(self):
        return self.grid_size

    def get_voxels(self):
        """
        Returns an instance of the voxels grid
        """
        return self.voxels

    def __getitem__(self, input_xyzl):
        """
        Returns the voxel centroid that the cartesian coordinate falls in

        :param input_xyzl:  nx4 np array where rows are points and cols are x,y,z 
                            and last col is semantic label idx
        :return: nx1 np array where rows are points and col is value at each point
        """
        
        # Reshape coordinates for 2d indexing
        input_idxl = self.grid_ind(input_xyzl)

        return self.voxels[ input_idxl[:, 0],
                            input_idxl[:, 1],
                            input_idxl[:, 2],
                            input_idxl[:, 3] ]

    def __setitem__(self, input_xyzl, input_value):
        """
        Sets the voxel to the input cell (cylindrical coordinates)

        :param input_xyzl:  nx4 np array where rows are points and cols are x,y,z 
                            and last col is semantic label idx
        :param input_value: scalar value for how much to increment cell by
        """
        # Reshape coordinates for 2d indexing
        input_idxl   = self.grid_ind(input_xyzl)

        self.voxels[input_idxl[:,0],
                    input_idxl[:, 1],
                    input_idxl[:, 2],
                    input_idxl[:, 3]] = input_value


    def grid_ind(self, input_xyzl):
        """
        Returns index of each cartesian coordinate in grid

        :param input_xyz:   nx4 np array where rows are points and cols are x,y,z 
                            and last col is semantic label idx
        :return:    nx4 np array where rows are points and cols are x,y,z 
                    and last col is semantic label idx
        """
        input_xyzl  = input_xyzl.reshape(-1, 4)
        input_xyz   = input_xyzl[:, 0:3]
        labels      = input_xyzl[:, 3].reshape(-1, 1)
        xyz_pol = self.cart2polar(input_xyz)

        valid_input_mask= np.all(
            (xyz_pol < self.max_bound) & (xyz_pol >= self.min_bound), axis=1)
        valid_xyz_pol   = xyz_pol[valid_input_mask]
        valid_labels    = labels[valid_input_mask]

        grid_ind = (np.floor((valid_xyz_pol
                    - self.min_bound) / self.intervals)).astype(np.int)
        grid_ind = np.hstack( (grid_ind, valid_labels) )
        return grid_ind

    def get_voxel_centers(self, input_xyzl):
        """
        Return voxel centers corresponding to each input xyz cartesian coordinate

        :param input_xyzl:  nx4 np array where rows are points and cols are x,y,z 
                            and last col is semantic label idx

        :return:    nx4 np array where rows are points and cols are x,y,z 
                    and last col is semantic label idx
        """
        
        # Center data on each voxel centroid for cylindrical coordinate PTnet
        valid_idxl  = self.grid_ind(input_xyzl)
        valid_idx   = valid_idxl[:, 0:3]
        valid_labels= valid_idxl[:, 3].reshape(-1, 1)

        valid_idx  = ( (valid_idx+0.5) * self.intervals ) + self.min_bound
        voxel_centers = np.hstack( (valid_idx, valid_labels))

        return self.polar2cart(voxel_centers)

    def cart2polar(self, input_xyz):
        """
        Converts cartesian to polar coordinates

        :param input_xyz_polar: nx3 or 4 np array where rows are points and cols are x,y,z 
                                and last col is semantic label idx

        :return:    size of input np array where rows are points and cols are r,theta,z, 
                    label (optional)
        """
        rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2).reshape(-1, 1)
        phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0]).reshape(-1, 1)

        return np.hstack((rho, phi, input_xyz[:, 2:]))


    def polar2cart(self, input_xyz_polar):
        """
        Converts polar to cartesian coordinates

        :param input_xyz_polar: nx3 or 4 np array where rows are points and cols 
                                are r,theta,z

        :return:    nx3 or 4 np array where rows are points and cols are 
                    x,y,z,label (optional)
        """
        x = (input_xyz_polar[:, 0] * np.cos(input_xyz_polar[:, 1])).reshape(-1, 1)
        y = (input_xyz_polar[:, 0] * np.sin(input_xyz_polar[:, 1])).reshape(-1, 1)

        return np.hstack((x, y, input_xyz_polar[:, 2:]))


