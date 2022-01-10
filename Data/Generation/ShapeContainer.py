import os
import numpy as np

"""
    Generic Volumetric container
"""
class ShapeContainer:
    def __init__(self, grid_size,
        min_bound=np.array([0, -1.0*np.pi, 0], dtype=np.float32),
        max_bound=np.array([20, 1.0*np.pi, 10], dtype=np.float32),
        num_channels=25,
        coordinates="cylindrical"):
        """
        Constructor that creates the cylinder volume container

        :param grid_size: 1x3 np array that represents the number of cells in each dimension
        :param max_bound: [max in 3 dimensions]
        :param min_bound: [min in 3 dimensions]
        :param num_channels: number of semantic channels
        """
        self.coordinates = coordinates
        
        self.grid_size = grid_size
        self.num_classes = num_channels
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
        self.intervals = crop_range / self.grid_size

        if (self.intervals == 0).any(): 
            print("Error zero interval detected...")
            return
        # Initialize voxel grid with float32
        self.voxels = np.zeros(list(self.grid_size.astype(np.uint32)) + [self.num_classes])

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
        input_idxl = self.grid_ind(input_xyzl).astype(int)

        return self.voxels[ list(input_idxl[:, 0]),
                            list(input_idxl[:, 1]),
                            list(input_idxl[:, 2]),
                            list(input_idxl[:, 3])
                        ]

    def __setitem__(self, input_xyzl, input_value):
        """
        Sets the voxel to the input cell (cylindrical coordinates)

        :param input_xyzl:  nx4 np array where rows are points and cols are x,y,z 
                            and last col is semantic label idx
        :param input_value: scalar value for how much to increment cell by
        """
        # Reshape coordinates for 2d indexing
        input_idxl = self.grid_ind(input_xyzl).astype(int)

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
        
        xyz_pol = self.cart2grid(input_xyz)

        valid_input_mask= np.all(
            (xyz_pol < self.max_bound) & (xyz_pol >= self.min_bound), axis=1)
        valid_xyz_pol   = xyz_pol[valid_input_mask]
        valid_labels    = labels[valid_input_mask]

        grid_ind = (np.floor((valid_xyz_pol
                    - self.min_bound) / self.intervals)).astype(np.int)
        # Clip due to edge cases
        maxes = np.reshape(self.grid_size - 1, (1, 3))
        mins = np.zeros_like(maxes)
        grid_ind = np.clip(grid_ind, mins, maxes)
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

        return self.grid2cart(voxel_centers)

    def cart2grid(self, input_xyz):
        """
        Converts cartesian to grid's coordinates system

        :param input_xyz_polar: nx3 or 4 np array where rows are points and cols are x,y,z 
                                and last col is semantic label idx

        :return:    size of input np array where rows are points and cols are r,theta,z, 
                    label (optional)
        """
        if self.coordinates == "cylindrical":
            rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2).reshape(-1, 1)
            phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0]).reshape(-1, 1)

            return np.hstack((rho, phi, input_xyz[:, 2:]))
        else:
            return input_xyz


    def grid2cart(self, input_xyz_polar):
        """
        Converts grid to cartesian coordinates

        :param input_xyz_polar: nx3 or 4 np array where rows are points and cols 
                                are r,theta,z

        :return:    nx3 or 4 np array where rows are points and cols are 
                    x,y,z,label (optional)
        """
        if self.coordinates == "cylindrical":
            x = (input_xyz_polar[:, 0] * np.cos(input_xyz_polar[:, 1])).reshape(-1, 1)
            y = (input_xyz_polar[:, 0] * np.sin(input_xyz_polar[:, 1])).reshape(-1, 1)

            return np.hstack((x, y, input_xyz_polar[:, 2:]))
        else:
            return input_xyz_polar


