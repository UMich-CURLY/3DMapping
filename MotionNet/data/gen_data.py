# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved. The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages. As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.


from nuscenes.nuscenes import NuScenes
import os
from nuscenes.utils.data_classes import LidarPointCloud
import numpy as np
import argparse
from data.data_utils import voxelize_occupy, gen_2d_grid_gt


def check_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    return folder_name


parser = argparse.ArgumentParser()
parser.add_argument('-r', '--root', default='/media/pwu/Data/3D_data/nuscene/all_nuscene', type=str, help='Root path to nuScenes dataset')
parser.add_argument('-s', '--split', default='train', type=str, help='The data split [train/val/test]')
parser.add_argument('-p', '--savepath', default='/media/pwu/62316788-a8e6-423c-9ed3-303ebb3ab2de/pwu/temporal_data/train', type=str, help='Directory for saving the generated data')
args = parser.parse_args()

nusc = NuScenes(version='v1.0-trainval', dataroot=args.root, verbose=True)
print("Total number of scenes:", len(nusc.scene))

class_map = {'vehicle.car': 1, 'vehicle.bus.rigid': 1, 'vehicle.bus.bendy': 1, 'human.pedestrian': 2,
             'vehicle.bicycle': 3}  # background: 0, other: 4


if args.split == 'train':
    num_keyframe_skipped = 0  # The number of keyframes we will skip when dumping the data
    nsweeps_back = 30  # Number of frames back to the history (including the current timestamp)
    nsweeps_forward = 20  # Number of frames into the future (does not include the current timestamp)
    skip_frame = 0  # The number of frames skipped for the adjacent sequence
    num_adj_seqs = 2  # number of adjacent sequences, among which the time gap is \delta t
else:
    num_keyframe_skipped = 1
    nsweeps_back = 25  # Setting this to 30 (for training) or 25 (for testing) allows conducting ablation studies on frame numbers
    nsweeps_forward = 20
    skip_frame = 0
    num_adj_seqs = 1


# The specifications for BEV maps
voxel_size = (0.25, 0.25, 0.4)
area_extents = np.array([[-32., 32.], [-32., 32.], [-3., 2.]])
past_frame_skip = 3  # when generating the BEV maps, how many history frames need to be skipped
future_frame_skip = 0  # when generating the BEV maps, how many future frames need to be skipped
num_past_frames_for_bev_seq = 5  # the number of past frames for BEV map sequence


scenes = np.load('data/split.npy', allow_pickle=True).item().get(args.split)
print("Split: {}, which contains {} scenes.".format(args.split, len(scenes)))

# ---------------------- Extract the scenes, and then pre-process them into BEV maps ----------------------
def gen_data():
    res_scenes = list()
    for s in scenes:
        s_id = s.split('_')[1]
        res_scenes.append(int(s_id))

    for scene_idx in res_scenes:
        curr_scene = nusc.scene[scene_idx]

        first_sample_token = curr_scene['first_sample_token']
        curr_sample = nusc.get('sample', first_sample_token)
        curr_sample_data = nusc.get('sample_data', curr_sample['data']['LIDAR_TOP'])

        save_data_dict_list = list()  # for storing consecutive sequences; the data consists of timestamps, points, etc
        save_box_dict_list = list()  # for storing box annotations in consecutive sequences
        save_instance_token_list = list()
        adj_seq_cnt = 0
        save_seq_cnt = 0  # only used for save data file name


        # Iterate each sample data
        print("Processing scene {} ...".format(scene_idx))
        while curr_sample_data['next'] != '':

            # Get the synchronized point clouds
            all_pc, all_times, trans_matrices = \
                LidarPointCloud.from_file_multisweep_bf_sample_data(nusc, curr_sample_data,
                                                                    return_trans_matrix=True,
                                                                    nsweeps_back=nsweeps_back,
                                                                    nsweeps_forward=nsweeps_forward)
            # Store point cloud of each sweep
            pc = all_pc.points
            _, sort_idx = np.unique(all_times, return_index=True)
            unique_times = all_times[np.sort(sort_idx)]  # Preserve the item order in unique_times
            num_sweeps = len(unique_times)

            # Make sure we have sufficient past and future sweeps
            if num_sweeps != (nsweeps_back + nsweeps_forward):

                # Skip some keyframes if necessary
                flag = False
                for _ in range(num_keyframe_skipped + 1):
                    if curr_sample['next'] != '':
                        curr_sample = nusc.get('sample', curr_sample['next'])
                    else:
                        flag = True
                        break

                if flag:  # No more keyframes
                    break
                else:
                    curr_sample_data = nusc.get('sample_data', curr_sample['data']['LIDAR_TOP'])

                # Reset
                adj_seq_cnt = 0
                save_data_dict_list = list()
                save_box_dict_list = list()
                save_instance_token_list = list()
                continue

            # Prepare data dictionary for the next step (ie, generating BEV maps)
            save_data_dict = dict()
            box_data_dict = dict()  # for remapping the instance ids, according to class_map
            curr_token_list = list()

            for tid in range(num_sweeps):
                _time = unique_times[tid]
                points_idx = np.where(all_times == _time)[0]
                _pc = pc[:, points_idx]
                save_data_dict['pc_' + str(tid)] = _pc

            save_data_dict['times'] = unique_times
            save_data_dict['num_sweeps'] = num_sweeps
            save_data_dict['trans_matrices'] = trans_matrices

            # Get the synchronized bounding boxes
            # First, we need to iterate all the instances, and then retrieve their corresponding bounding boxes
            num_instances = 0  # The number of instances within this sample
            corresponding_sample_token = curr_sample_data['sample_token']
            corresponding_sample_rec = nusc.get('sample', corresponding_sample_token)

            for ann_token in corresponding_sample_rec['anns']:
                ann_rec = nusc.get('sample_annotation', ann_token)
                category_name = ann_rec['category_name']
                instance_token = ann_rec['instance_token']

                flag = False
                for c, v in class_map.items():
                    if category_name.startswith(c):
                        box_data_dict['category_' + instance_token] = v
                        flag = True
                        break
                if not flag:
                    box_data_dict['category_' + instance_token] = 4  # Other category

                instance_boxes, instance_all_times, _, _ = LidarPointCloud. \
                    get_instance_boxes_multisweep_sample_data(nusc, curr_sample_data,
                                                              instance_token,
                                                              nsweeps_back=nsweeps_back,
                                                              nsweeps_forward=nsweeps_forward)

                assert np.array_equal(unique_times, instance_all_times), "The sweep and instance times are inconsistent!"
                assert num_sweeps == len(instance_boxes), "The number of instance boxes does not match that of sweeps!"

                # Each row corresponds to a box annotation; the column consists of box center, box size, and quaternion
                box_data = np.zeros((len(instance_boxes), 3 + 3 + 4), dtype=np.float32)
                box_data.fill(np.nan)
                for r, box in enumerate(instance_boxes):
                    if box is not None:
                        row = np.concatenate([box.center, box.wlh, box.orientation.elements])
                        box_data[r] = row[:]

                # Save the box data for current instance
                box_data_dict['instance_boxes_' + instance_token] = box_data
                num_instances += 1

                curr_token_list.append(instance_token)

            save_data_dict['num_instances'] = num_instances
            save_data_dict_list.append(save_data_dict)
            save_box_dict_list.append(box_data_dict)
            save_instance_token_list.append(curr_token_list)

            # Update the counter and save the data if desired (But here we do not want to
            # save the data to disk since it would cost about 2TB space)
            adj_seq_cnt += 1
            if adj_seq_cnt == num_adj_seqs:

                # First, we need to reorganize the instance tokens (ids)
                num_instance_token_list = len(save_instance_token_list)
                if num_instance_token_list > 1:
                    common_tokens = set(save_instance_token_list[0]).intersection(save_instance_token_list[1])

                    for l in range(2, num_instance_token_list):
                        common_tokens = common_tokens.intersection(save_instance_token_list[l])

                    for l in range(num_instance_token_list):
                        exclusive_tokens = set(save_instance_token_list[l]).difference(common_tokens)

                        # we store the common instances first, then store the remaining instances
                        curr_save_data_dict = save_data_dict_list[l]
                        curr_save_box_dict = save_box_dict_list[l]
                        counter = 0
                        for token in common_tokens:
                            box_info = curr_save_box_dict['instance_boxes_' + token]
                            box_cat = curr_save_box_dict['category_' + token]

                            curr_save_data_dict['instance_boxes_' + str(counter)] = box_info
                            curr_save_data_dict['category_' + str(counter)] = box_cat

                            counter += 1

                        for token in exclusive_tokens:
                            box_info = curr_save_box_dict['instance_boxes_' + token]
                            box_cat = curr_save_box_dict['category_' + token]

                            curr_save_data_dict['instance_boxes_' + str(counter)] = box_info
                            curr_save_data_dict['category_' + str(counter)] = box_cat

                            counter += 1

                        assert counter == curr_save_data_dict['num_instances'], "The number of instances is inconsistent."

                        save_data_dict_list[l] = curr_save_data_dict
                else:
                    curr_save_box_dict = save_box_dict_list[0]
                    curr_save_data_dict = save_data_dict_list[0]
                    for index, token in enumerate(save_instance_token_list[0]):
                        box_info = curr_save_box_dict['instance_boxes_' + token]
                        box_cat = curr_save_box_dict['category_' + token]

                        curr_save_data_dict['instance_boxes_' + str(index)] = box_info
                        curr_save_data_dict['category_' + str(index)] = box_cat

                    save_data_dict_list[0] = curr_save_data_dict

                # ------------------------ Now we generate dense BEV maps ------------------------
                for seq_idx, seq_data_dict in enumerate(save_data_dict_list):
                    dense_bev_data = convert_to_dense_bev(seq_data_dict)
                    sparse_bev_data = convert_to_sparse_bev(dense_bev_data)

                    # save the data
                    save_directory = check_folder(os.path.join(args.savepath, str(scene_idx) + '_' + str(save_seq_cnt)))
                    save_file_name = os.path.join(save_directory, str(seq_idx) + '.npy')
                    np.save(save_file_name, arr=sparse_bev_data)

                    print("  >> Finish sample: {}, sequence {}".format(save_seq_cnt, seq_idx))
                # --------------------------------------------------------------------------------

                save_seq_cnt += 1
                adj_seq_cnt = 0
                save_data_dict_list = list()
                save_box_dict_list = list()
                save_instance_token_list = list()

                # Skip some keyframes if necessary
                flag = False
                for _ in range(num_keyframe_skipped + 1):
                    if curr_sample['next'] != '':
                        curr_sample = nusc.get('sample', curr_sample['next'])
                    else:
                        flag = True
                        break

                if flag:  # No more keyframes
                    break
                else:
                    curr_sample_data = nusc.get('sample_data', curr_sample['data']['LIDAR_TOP'])
            else:
                flag = False
                for _ in range(skip_frame + 1):
                    if curr_sample_data['next'] != '':
                        curr_sample_data = nusc.get('sample_data', curr_sample_data['next'])
                    else:
                        flag = True
                        break

                if flag:  # No more sample frames
                    break


# ---------------------- Convert the raw data into (dense) BEV maps ----------------------
def convert_to_dense_bev(data_dict):
    num_sweeps = data_dict['num_sweeps']
    times = data_dict['times']
    trans_matrices = data_dict['trans_matrices']

    num_past_sweeps = len(np.where(times >= 0)[0])
    num_future_sweeps = len(np.where(times < 0)[0])
    assert num_past_sweeps + num_future_sweeps == num_sweeps, "The number of sweeps is incorrect!"

    # Load point cloud
    pc_list = []

    for i in range(num_sweeps):
        pc = data_dict['pc_' + str(i)]
        pc_list.append(pc.T)

    # Reorder the pc, and skip sample frames if wanted
    # Currently the past frames in pc_list are stored in the following order [current, current + 1, current + 2, ...]
    # Therefore, we would like to reorder the frames
    tmp_pc_list_1 = pc_list[0:num_past_sweeps:(past_frame_skip + 1)]
    tmp_pc_list_1 = tmp_pc_list_1[::-1]
    tmp_pc_list_2 = pc_list[(num_past_sweeps + future_frame_skip)::(future_frame_skip + 1)]
    pc_list = tmp_pc_list_1 + tmp_pc_list_2  # now the order is: [past frames -> current frame -> future frames]

    num_past_pcs = len(tmp_pc_list_1)
    num_future_pcs = len(tmp_pc_list_2)

    # Discretize the input point clouds, and compute the ground-truth displacement vectors
    # The following two variables contain the information for the
    # compact representation of binary voxels, as described in the paper
    voxel_indices_list = list()
    padded_voxel_points_list = list()

    past_pcs_idx = list(range(num_past_pcs))
    past_pcs_idx = past_pcs_idx[-num_past_frames_for_bev_seq:]  # we typically use 5 past frames (including the current one)
    for i in past_pcs_idx:
        res, voxel_indices = voxelize_occupy(pc_list[i], voxel_size=voxel_size, extents=area_extents, return_indices=True)
        voxel_indices_list.append(voxel_indices)
        padded_voxel_points_list.append(res)

    # Compile the batch of voxels, so that they can be fed into the network.
    # Note that, the padded_voxel_points in this script will only be used for sanity check.
    padded_voxel_points = np.stack(padded_voxel_points_list, axis=0).astype(np.bool)

    # Finally, generate the ground-truth displacement field
    # - all_disp_field_gt: the ground-truth displacement vectors for each grid cell
    # - all_valid_pixel_maps: the masking map for valid pixels, used for loss computation
    # - non_empty_map: the mask which represents the non-empty grid cells, used for loss computation
    # - pixel_cat_map: the map specifying the category for each non-empty grid cell
    # - pixel_indices: the indices of non-empty grid cells, used to generate sparse BEV maps
    # - pixel_instance_map: the map specifying the instance id for each grid cell, used for loss computation
    all_disp_field_gt, all_valid_pixel_maps, non_empty_map, pixel_cat_map, pixel_indices, pixel_instance_map \
        = gen_2d_grid_gt(data_dict, grid_size=voxel_size[0:2], extents=area_extents,
                         frame_skip=future_frame_skip, return_instance_map=True)

    return voxel_indices_list, padded_voxel_points, pixel_indices, pixel_instance_map, all_disp_field_gt,\
        all_valid_pixel_maps, non_empty_map, pixel_cat_map, num_past_frames_for_bev_seq, num_future_pcs, trans_matrices


# ---------------------- Convert the dense BEV data into sparse format ----------------------
# This will significantly reduce the space used for data storage
def convert_to_sparse_bev(dense_bev_data):
    save_voxel_indices_list, save_voxel_points, save_pixel_indices, save_pixel_instance_maps, \
        save_disp_field_gt, save_valid_pixel_maps, save_non_empty_maps, save_pixel_cat_maps, \
        save_num_past_pcs, save_num_future_pcs, save_trans_matrices = dense_bev_data

    save_valid_pixel_maps = save_valid_pixel_maps.astype(np.bool)
    save_voxel_dims = save_voxel_points.shape[1:]
    num_categories = save_pixel_cat_maps.shape[-1]

    sparse_disp_field_gt = save_disp_field_gt[:, save_pixel_indices[:, 0], save_pixel_indices[:, 1], :]
    sparse_valid_pixel_maps = save_valid_pixel_maps[:, save_pixel_indices[:, 0], save_pixel_indices[:, 1]]
    sparse_pixel_cat_maps = save_pixel_cat_maps[save_pixel_indices[:, 0], save_pixel_indices[:, 1]]
    sparse_pixel_instance_maps = save_pixel_instance_maps[save_pixel_indices[:, 0], save_pixel_indices[:, 1]]

    save_data_dict = dict()
    for i in range(len(save_voxel_indices_list)):
        save_data_dict['voxel_indices_' + str(i)] = save_voxel_indices_list[i].astype(np.int32)

    save_data_dict['disp_field'] = sparse_disp_field_gt
    save_data_dict['valid_pixel_map'] = sparse_valid_pixel_maps
    save_data_dict['pixel_cat_map'] = sparse_pixel_cat_maps
    save_data_dict['num_past_pcs'] = save_num_past_pcs
    save_data_dict['num_future_pcs'] = save_num_future_pcs
    save_data_dict['trans_matrices'] = save_trans_matrices
    save_data_dict['3d_dimension'] = save_voxel_dims
    save_data_dict['pixel_indices'] = save_pixel_indices
    save_data_dict['pixel_instance_ids'] = sparse_pixel_instance_maps

    # -------------------------------- Sanity Check --------------------------------
    dims = save_non_empty_maps.shape

    test_disp_field_gt = np.zeros((save_num_future_pcs, dims[0], dims[1], 2), dtype=np.float32)
    test_disp_field_gt[:, save_pixel_indices[:, 0], save_pixel_indices[:, 1], :] = sparse_disp_field_gt[:]
    assert np.all(test_disp_field_gt == save_disp_field_gt), "Error: Mismatch"

    test_valid_pixel_maps = np.zeros((save_num_future_pcs, dims[0], dims[1]), dtype=np.bool)
    test_valid_pixel_maps[:, save_pixel_indices[:, 0], save_pixel_indices[:, 1]] = sparse_valid_pixel_maps[:]
    assert np.all(test_valid_pixel_maps == save_valid_pixel_maps), "Error: Mismatch"

    test_pixel_cat_maps = np.zeros((dims[0], dims[1], num_categories), dtype=np.float32)
    test_pixel_cat_maps[save_pixel_indices[:, 0], save_pixel_indices[:, 1], :] = sparse_pixel_cat_maps[:]
    assert np.all(test_pixel_cat_maps == save_pixel_cat_maps), "Error: Mismatch"

    test_non_empty_map = np.zeros((dims[0], dims[1]), dtype=np.float32)
    test_non_empty_map[save_pixel_indices[:, 0], save_pixel_indices[:, 1]] = 1.0
    assert np.all(test_non_empty_map == save_non_empty_maps), "Error: Mismatch"

    test_pixel_instance_map = np.zeros((dims[0], dims[1]), dtype=np.uint8)
    test_pixel_instance_map[save_pixel_indices[:, 0], save_pixel_indices[:, 1]] = sparse_pixel_instance_maps[:]
    assert np.all(test_pixel_instance_map == save_pixel_instance_maps), "Error: Mismatch"

    for i in range(len(save_voxel_indices_list)):
        indices = save_data_dict['voxel_indices_' + str(i)]
        curr_voxels = np.zeros(save_voxel_dims, dtype=np.bool)
        curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
        assert np.all(curr_voxels == save_voxel_points[i]), "Error: Mismatch"

    return save_data_dict


if __name__ == "__main__":
    gen_data()

