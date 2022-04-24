import os
import numpy as np
# from utils import laserscan
import yaml
from torch.utils.data import Dataset
import torch
import spconv
import math

config_file = os.path.join('semantic-kitti.yaml')
kitti_config = yaml.safe_load(open(config_file, 'r'))
remapdict = kitti_config["learning_map"]
# print(kitti_config['content'])
# print(remapdict)




SPLIT_SEQUENCES = {
    "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
    "valid": ["08"],
    "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]
}




class KittiDataset(Dataset):
    """Kitti Dataset for 3D mapping project
    
    Access to the processed data, including evaluation labels predictions velodyne poses times
    """

    def __init__(self, directory,
            device='cuda',
            num_frames=4,
            voxelize_input=True,
            binary_counts=False,
            random_flips=False,
            remap=False,
            split='train'
            ):
        
        self.voxelize_input = voxelize_input
        self.binary_counts = binary_counts
        self._directory = os.path.join(directory, 'sequences')
        self._num_frames = num_frames
        self.device = device
        self.random_flips = random_flips
        self.remap = remap
        self.split = split

        self._velodyne_list = []
        self._label_list = []
        self._pred_list = []
        self._eval_labels = []
        self._eval_counts = []
        self._frames_list = []
        self._timestamps = []
        self._poses = [] 

        self._num_frames_scene = []

        self._seqs = SPLIT_SEQUENCES[self.split]

        for seq in self._seqs:
            velodyne_dir = os.path.join(self._directory, seq, 'velodyne')
            label_dir = os.path.join(self._directory, seq, 'labels')
            eval_dir = os.path.join(self._directory, seq, 'evaluation')
            self._num_frames_scene.append(len(os.listdir(velodyne_dir)))
            frames_list = [os.path.splitext(filename)[0] for filename in sorted(os.listdir(velodyne_dir))]
            self._velodyne_list.extend([os.path.join(velodyne_dir, str(frame).zfill(6)+'.bin') for frame in frames_list])
            self._label_list.extend([os.path.join(label_dir, str(frame).zfill(6)+'.label') for frame in frames_list])
        
        print(np.sum(self._num_frames_scene))

        self._cum_num_frames = np.cumsum(np.array(self._num_frames_scene) - self._num_frames + 1)

    
    
    # Use all frames, if there is no data then zero pad
    def __len__(self):
        return sum(self._num_frames_scene)

    def __getitem__(self, idx):
        # -1 indicates no data
        # the final index is the output
        idx_range = self.find_horizon(idx)
         
        if self.voxelize_input:
            current_horizon = np.zeros((idx_range.shape[0], int(self.grid_dims[0]), int(self.grid_dims[1]), int(self.grid_dims[2])), dtype=np.float32)
        else:
            current_horizon = []
        t_i = 0
        for i in idx_range:
            if i == -1: # Zero pad
                points = np.zeros((1, 3), dtype=np.float32)
                
            else:
                points = np.fromfile(self._velodyne_list[i],dtype=np.float32).reshape(-1,4)[:, :3]
            if self.voxelize_input:
                current_horizon = self.points_to_voxels(current_horizon, points, t_i)
            else:
                current_horizon.append(points)
            t_i += 1
        
        output = np.fromfile(self._eval_labels[idx_range[-1]],dtype=np.uint32).reshape(self._eval_size).astype(np.uint8)
        counts = np.fromfile(self._eval_counts[idx_range[-1]],dtype=np.float32).reshape(self._eval_size)
        
        if self.voxelize_input and self.random_flips:
            # X flip
            if np.random.randint(2):
                output = np.flip(output, axis=0)
                counts = np.flip(counts, axis=0)
                current_horizon = np.flip(current_horizon, axis=1) # Because there is a time dimension
            # Y Flip
            if np.random.randint(2):
                output = np.flip(output, axis=1)
                counts = np.flip(counts, axis=1)
                current_horizon = np.flip(current_horizon, axis=2) # Because there is a time dimension
                
        if self.remap:
            output = LABELS_REMAP[output].astype(np.uint8)

        return current_horizon, output, counts
        
        # no enough frames
    
    def find_horizon(self, idx):
        end_idx = idx
        idx_range = np.arange(idx-self._num_frames, idx)+1
        diffs = np.asarray([int(self._frames_list[end_idx]) - int(self._frames_list[i]) for i in idx_range])
        good_difs = -1 * (np.arange(-self._num_frames, 0) + 1)
        
        idx_range[good_difs != diffs] = -1

        return idx_range


ds = KittiDataset('/media/sde1/kitti/')
len(ds)

