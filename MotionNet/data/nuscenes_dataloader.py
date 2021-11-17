# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved. The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages. As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.


from torch.utils.data import Dataset
import numpy as np
import os
import warnings
from multiprocessing import Manager
from data.data_utils import classify_speed_level


class TrainDatasetMultiSeq(Dataset):
    def __init__(self, dataset_root=None, future_frame_skip=0, voxel_size=(0.25, 0.25, 0.4),
                 area_extents=np.array([[-32., 32.], [-32., 32.], [-3., 2.]]),  num_past_frames=5,
                 num_future_frames=20, num_category=5, cache_size=10000):
        """
        This dataloader loads multiple sequences for a keyframe, for computing the spatio-temporal consistency losses

        dataset_root: Data path to the preprocessed sparse nuScenes data (for training)
        future_frame_skip: Specify to skip how many future frames
        voxel_size: The lattice resolution. Should be consistent with the preprocessed data
        area_extents: The area extents of the processed LiDAR data. Should be consistent with the preprocessed data
        num_past_frames: The number of past frames within a BEV sequence
        num_future_frames: The number of future frames within a BEV sequence. Default: 20
        num_category: The number of object categories (including the background)
        cache_size: The cache size for storing parts of data in the memory (for reducing the IO cost)
        """
        if dataset_root is None:
            raise ValueError("The dataset root is None. Should specify its value.")

        self.dataset_root = dataset_root
        print("data root:", dataset_root)

        seq_dirs = [os.path.join(self.dataset_root, d) for d in os.listdir(self.dataset_root)
                    if os.path.isdir(os.path.join(self.dataset_root, d))]

        self.seq_dirs = seq_dirs
        self.num_sample_seqs = len(self.seq_dirs)
        if self.num_sample_seqs != 17065:
            warnings.warn(">> The size of training dataset is not 17065.\n")

        self.future_frame_skip = future_frame_skip
        self.voxel_size = voxel_size
        self.area_extents = area_extents
        self.num_category = num_category
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames

        manager = Manager()
        self.cache = manager.dict()
        self.cache_size = cache_size

    def __len__(self):
        return self.num_sample_seqs

    def __getitem__(self, idx):
        if idx in self.cache:
            gt_dict_list = self.cache[idx]
        else:
            seq_dir = self.seq_dirs[idx]
            gt_file_paths = [os.path.join(seq_dir, f)for f in os.listdir(seq_dir)
                             if os.path.isfile(os.path.join(seq_dir, f))]
            num_gt_files = len(gt_file_paths)

            gt_dict_list = list()
            for f in range(num_gt_files):  # process the files, starting from 0.npy to 1.npy, etc
                gt_file_path = gt_file_paths[f]
                gt_data_handle = np.load(gt_file_path, allow_pickle=True)
                gt_dict = gt_data_handle.item()
                gt_dict_list.append(gt_dict)

            if len(self.cache) < self.cache_size:
                self.cache[idx] = gt_dict_list

        padded_voxel_points_list = list()
        all_disp_field_gt_list = list()
        all_valid_pixel_maps_list = list()
        non_empty_map_list = list()
        pixel_cat_map_list = list()
        trans_matrices_list = list()
        pixel_instance_map_list = list()

        for gt_dict in gt_dict_list:
            dims = gt_dict['3d_dimension']
            num_future_pcs = gt_dict['num_future_pcs']
            pixel_indices = gt_dict['pixel_indices']
            trans_matrices = gt_dict['trans_matrices']

            sparse_disp_field_gt = gt_dict['disp_field']
            all_disp_field_gt = np.zeros((num_future_pcs, dims[0], dims[1], 2), dtype=np.float32)
            all_disp_field_gt[:, pixel_indices[:, 0], pixel_indices[:, 1], :] = sparse_disp_field_gt[:]

            sparse_valid_pixel_maps = gt_dict['valid_pixel_map']
            all_valid_pixel_maps = np.zeros((num_future_pcs, dims[0], dims[1]), dtype=np.float32)
            all_valid_pixel_maps[:, pixel_indices[:, 0], pixel_indices[:, 1]] = sparse_valid_pixel_maps[:]

            sparse_pixel_cat_maps = gt_dict['pixel_cat_map']
            pixel_cat_map = np.zeros((dims[0], dims[1], self.num_category), dtype=np.float32)
            pixel_cat_map[pixel_indices[:, 0], pixel_indices[:, 1], :] = sparse_pixel_cat_maps[:]

            sparse_pixel_instance_maps = gt_dict['pixel_instance_ids']
            pixel_instance_map = np.zeros((dims[0], dims[1]), dtype=np.uint8)
            pixel_instance_map[pixel_indices[:, 0], pixel_indices[:, 1]] = sparse_pixel_instance_maps[:]

            non_empty_map = np.zeros((dims[0], dims[1]), dtype=np.float32)
            non_empty_map[pixel_indices[:, 0], pixel_indices[:, 1]] = 1.0

            padded_voxel_points = list()
            num_past_pcs = self.num_past_frames
            for i in range(num_past_pcs):
                indices = gt_dict['voxel_indices_' + str(i)]
                curr_voxels = np.zeros(dims, dtype=np.bool)
                curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
                padded_voxel_points.append(curr_voxels)
            padded_voxel_points = np.stack(padded_voxel_points, 0).astype(np.float32)

            padded_voxel_points_list.append(padded_voxel_points[-5:])
            all_disp_field_gt_list.append(all_disp_field_gt)
            all_valid_pixel_maps_list.append(all_valid_pixel_maps)
            non_empty_map_list.append(non_empty_map)
            pixel_cat_map_list.append(pixel_cat_map)
            trans_matrices_list.append(trans_matrices)
            pixel_instance_map_list.append(pixel_instance_map)

        padded_voxel_points_list = np.stack(padded_voxel_points_list, 0)
        all_disp_field_gt_list = np.stack(all_disp_field_gt_list, 0)
        all_valid_pixel_maps_list = np.stack(all_valid_pixel_maps_list, 0)
        non_empty_map_list = np.stack(non_empty_map_list, 0)
        pixel_cat_map_list = np.stack(pixel_cat_map_list, 0)
        trans_matrices_list = np.stack(trans_matrices_list, 0)
        pixel_instance_map_list = np.stack(pixel_instance_map_list, 0)


        # Classify speed-level (ie, state estimation: static or moving)
        adj_seq_num = all_disp_field_gt_list.shape[0]
        motion_cat_list = list()

        for i in range(adj_seq_num):
            tmp_motion_cat = classify_speed_level(all_disp_field_gt_list[i], total_future_sweeps=20,
                                                  future_frame_skip=self.future_frame_skip)
            motion_cat_list.append(tmp_motion_cat)
        motion_cat_list = np.stack(motion_cat_list, axis=0)

        return padded_voxel_points_list, all_disp_field_gt_list, all_valid_pixel_maps_list, non_empty_map_list, \
            pixel_cat_map_list, trans_matrices_list, motion_cat_list, pixel_instance_map_list, \
            self.num_past_frames, self.num_future_frames


class DatasetSingleSeq(Dataset):
    def __init__(self, dataset_root=None, split='train', future_frame_skip=0, voxel_size=(0.25, 0.25, 0.4),
                 area_extents=np.array([[-32., 32.], [-32., 32.], [-3., 2.]]), num_category=5, cache_size=8000):
        """
        This dataloader loads single sequence for a keyframe, and is not designed for computing the
         spatio-temporal consistency losses. It supports train, val and test splits.

        dataset_root: Data path to the preprocessed sparse nuScenes data (for training)
        split: [train/val/test]
        future_frame_skip: Specify to skip how many future frames
        voxel_size: The lattice resolution. Should be consistent with the preprocessed data
        area_extents: The area extents of the processed LiDAR data. Should be consistent with the preprocessed data
        num_category: The number of object categories (including the background)
        cache_size: The cache size for storing parts of data in the memory (for reducing the IO cost)
        """
        if dataset_root is None:
            raise ValueError("The {} dataset root is None. Should specify its value.".format(split))

        self.dataset_root = dataset_root
        print("data root:", dataset_root)

        seq_dirs = [os.path.join(self.dataset_root, d) for d in os.listdir(self.dataset_root)
                    if os.path.isfile(os.path.join(self.dataset_root, d))]
        # seq_files = [os.path.join(seq_dir, f) for seq_dir in seq_dirs for f in os.listdir(seq_dir)
        #              if os.path.isfile(os.path.join(seq_dir, f))]

        self.seq_files = seq_dirs
        self.num_sample_seqs = len(self.seq_files)
        print("The number of {} sequences: {}".format(split, self.num_sample_seqs))

        # For training, the size of dataset should be 17065 * 2; for validation: 1719; for testing: 4309
        if split == 'train' and self.num_sample_seqs != 17065 * 2:
            warnings.warn(">> The size of training dataset is not 17065 * 2.\n")
        elif split == 'val' and self.num_sample_seqs != 1719:
            warnings.warn(">> The size of validation dataset is not 1719.\n")
        elif split == 'test' and self.num_sample_seqs != 4309:
            warnings.warn('>> The size of test dataset is not 4309.\n')

        self.split = split
        self.voxel_size = voxel_size
        self.area_extents = area_extents
        self.num_category = num_category
        self.future_frame_skip = future_frame_skip

        manager = Manager()
        self.cache = manager.dict()
        self.cache_size = cache_size if split == 'train' else 0

    def __len__(self):
        return self.num_sample_seqs

    def __getitem__(self, idx):
        if idx in self.cache:
            gt_dict = self.cache[idx]
        else:
            seq_file = self.seq_files[idx]
            gt_data_handle = np.load(seq_file, allow_pickle=True)
            gt_dict = gt_data_handle.item()

            if len(self.cache) < self.cache_size:
                self.cache[idx] = gt_dict

        dims = gt_dict['3d_dimension']
        num_future_pcs = gt_dict['num_future_pcs']
        num_past_pcs = gt_dict['num_past_pcs']
        pixel_indices = gt_dict['pixel_indices']

        sparse_disp_field_gt = gt_dict['disp_field']
        all_disp_field_gt = np.zeros((num_future_pcs, dims[0], dims[1], 2), dtype=np.float32)
        all_disp_field_gt[:, pixel_indices[:, 0], pixel_indices[:, 1], :] = sparse_disp_field_gt[:]

        sparse_valid_pixel_maps = gt_dict['valid_pixel_map']
        all_valid_pixel_maps = np.zeros((num_future_pcs, dims[0], dims[1]), dtype=np.float32)
        all_valid_pixel_maps[:, pixel_indices[:, 0], pixel_indices[:, 1]] = sparse_valid_pixel_maps[:]

        sparse_pixel_cat_maps = gt_dict['pixel_cat_map']
        pixel_cat_map = np.zeros((dims[0], dims[1], self.num_category), dtype=np.float32)
        pixel_cat_map[pixel_indices[:, 0], pixel_indices[:, 1], :] = sparse_pixel_cat_maps[:]

        non_empty_map = np.zeros((dims[0], dims[1]), dtype=np.float32)
        non_empty_map[pixel_indices[:, 0], pixel_indices[:, 1]] = 1.0

        padded_voxel_points = list()
        for i in range(num_past_pcs):
            indices = gt_dict['voxel_indices_' + str(i)]
            curr_voxels = np.zeros(dims, dtype=np.bool)
            curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
            padded_voxel_points.append(curr_voxels)
        padded_voxel_points = np.stack(padded_voxel_points, 0).astype(np.float32)

        # Classify speed-level (ie, state estimation: static or moving)
        motion_state_gt = classify_speed_level(all_disp_field_gt, total_future_sweeps=20,
                                              future_frame_skip=self.future_frame_skip)

        return padded_voxel_points, all_disp_field_gt, all_valid_pixel_maps, \
            non_empty_map, pixel_cat_map, num_past_pcs, num_future_pcs, motion_state_gt


if __name__ == "__main__":
    data_nuscenes = TrainDatasetMultiSeq(dataset_root='/media/pwu/62316788-a8e6-423c-9ed3-303ebb3ab2de/pwu/temporal_data/train')
    a = data_nuscenes[0]



