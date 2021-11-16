import os
import pdb
import time

from cylinder_container import CylinderContainer

"""
Driver file for loading data into cylindrical coordinate voxels
"""
def main():
    # TODO: Run simulation here
    while(1):
        # Initializes cylinder container with 1000 cells and default range settings
        # 
        # Note: Not making container size scale with the number of
        #       occupied cells because we want to ray trace in empty cells too
        cells = CylinderContainer(1000)

if __name__ == '__main__':
    main()