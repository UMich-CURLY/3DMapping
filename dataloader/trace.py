import os
import pdb
import numpy as np
from cylinder_container import CylinderContainer
from utils import Voxel

"""
Utility functions for ray tracing through cylinder coordinates

"""

def trace_cells(container, start, end, empty_cat_idx, semantic_cat_idx):
    """
    Updates container by ray tracing through cells and assigning semantic category 
    probablities for each

    :param container:   container object to ray trace through, must override setter 
                        & getter
    :param start:       nx3 numpy array with starting xyz coordinate 
    :param end:         nx3 numpy array with ending xyz coordinate 
    :empty_cat:         number of semantic classes x 1 np array of how much to increase
                        probablity of semantic class corresponding to empty cells
    :semantic_cat:      number of semantic classes x 1 np array of how much to increase
                        probablity of semantic class corresponding to hit cell            
    
    Questions: how much to increase probability by for each semantic class
    """
    dir_vec = ((end - start) / np.linalg.norm(end - start))

    points = np.array([])
    curr = start
    next_dir = dir_vec

    while np.allclose(next_dir, dir_vec):
        next = (curr + dir_vec)
        next_dir = (end-next)/np.linalg.norm(end-next)

        if np.allclose(next_dir, dir_vec):
            container[curr][0].set_class(empty_cat_idx)
        else:
            container[curr][0].set_class(semantic_cat_idx)

        points = curr if len(points)==0 else np.vstack( (points, curr))
        curr = next

    return points



