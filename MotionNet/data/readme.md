Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved. The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages. As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.

## Usage
The directory containing the nuScenes data should be like this:
```
|-- path to nuScenes root directory
|   |-- maps
|   |-- samples
|   |-- sweeps
|   |-- v1.0-trainval
```

On my computer it took about 26 hours to generate all the training data, and about 8 and 5 hours to generate the validation and test data, respectively. The major reasons could be that, extracting the LiDAR data and their associated annotations is slow, and generating the ground-truth prediction is not efficient. Also, the IO speed could be a bottleneck.
As a future work, we will speed up this process with multiprocessing.
The size of the generated training data is about 26.5G. For validation and testing data their sizes are 2.5G and 6.5G, respectively.

## Data Structure:
For each generated BEV map, it is actually a 3D binary volume. To reduce the disk space for data storage, we organize the data in sparse format. Specifically, we only store the indices to the occupied voxels in each data file. When loading the data, we reassemble the volume (i.e., BEV map) from these indices to produce the dense BEV map. In this way we are able to reduce about 20x more hard disk space compared to storing naive dense BEV map.

Also, for the other ground-truth data files, we also store them in the sparse format.
