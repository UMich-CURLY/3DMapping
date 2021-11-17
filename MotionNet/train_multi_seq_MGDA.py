# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved. The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages. As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.


#################################################################################
#                       Note: The code requires PyTorch 1.1                     #
#################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import numpy as np
import time
import sys
import os
from shutil import copytree, copy
from model import MotionNetMGDA, FeatEncoder
from data.nuscenes_dataloader import TrainDatasetMultiSeq
from min_norm_solvers import MinNormSolver


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


pred_adj_frame_distance = True  # Whether to predict the relative offset between frames

height_feat_size = 13  # The size along the height dimension
cell_category_num = 5  # The number of object categories (including the background)

out_seq_len = 20  # The number of future frames we are going to predict
trans_matrix_idx = 1  # Among N transformation matrices (N=2 in our experiment), which matrix is used for alignment (see paper)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', default=None, type=str, help='The path to the preprocessed sparse BEV training data')
parser.add_argument('--resume', default='', type=str, help='The path to the saved model that is loaded to resume training')
parser.add_argument('--batch', default=8, type=int, help='Batch size')
parser.add_argument('--nepoch', default=70, type=int, help='Number of epochs')
parser.add_argument('--nworker', default=4, type=int, help='Number of workers')

parser.add_argument('--reg_weight_bg_tc', default=0.1, type=float, help='Weight of background temporal consistency term')
parser.add_argument('--reg_weight_fg_tc', default=2.5, type=float, help='Weight of instance temporal consistency')
parser.add_argument('--reg_weight_sc', default=15.0, type=float, help='Weight of spatial consistency term')
parser.add_argument('--reg_weight_cls', default=2.0, type=float, help='The extra weight for grid cell classification term')

parser.add_argument('--use_bg_tc', action='store_true', help='Whether to use background temporal consistency loss')
parser.add_argument('--use_fg_tc', action='store_true', help='Whether to use foreground loss in st.')
parser.add_argument('--use_sc', action='store_true', help='Whether to use spatial consistency loss')

parser.add_argument('--nn_sampling', action='store_true', help='Whether to use nearest neighbor sampling in bg_tc loss')
parser.add_argument('--log', action='store_true', help='Whether to log')
parser.add_argument('--logpath', default='', help='The path to the output log file')

args = parser.parse_args()
print(args)

need_log = args.log
BATCH_SIZE = args.batch
num_epochs = args.nepoch
num_workers = args.nworker

reg_weight_bg_tc = args.reg_weight_bg_tc  # The weight of background temporal consistency term
reg_weight_fg_tc = args.reg_weight_fg_tc  # The weight of foreground temporal consistency term
reg_weight_sc = args.reg_weight_sc  # The weight of spatial consistency term
reg_weight_cls = args.reg_weight_cls  # The weight for grid cell classification term

use_bg_temporal_consistency = args.use_bg_tc
use_fg_temporal_consistency = args.use_fg_tc
use_spatial_consistency = args.use_sc

use_nn_sampling = args.nn_sampling


def main():
    start_epoch = 1
    # Whether to log the training information
    if need_log:
        logger_root = args.logpath if args.logpath != '' else 'logs'
        time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        if args.resume == '':
            model_save_path = check_folder(logger_root)
            model_save_path = check_folder(os.path.join(model_save_path, 'train_multi_seq'))
            model_save_path = check_folder(os.path.join(model_save_path, time_stamp))

            log_file_name = os.path.join(model_save_path, 'log.txt')
            saver = open(log_file_name, "w")
            saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
            saver.flush()

            # Logging the details for this experiment
            saver.write("command line: {}\n".format(" ".join(sys.argv[0:])))
            saver.write(args.__repr__() + "\n\n")
            saver.flush()

            # Copy the code files as logs
            copytree('nuscenes-devkit', os.path.join(model_save_path, 'nuscenes-devkit'))
            copytree('data', os.path.join(model_save_path, 'data'))
            python_files = [f for f in os.listdir('.') if f.endswith('.py')]
            for f in python_files:
                copy(f, model_save_path)
        else:
            model_save_path = args.resume  # eg, "logs/train_multi_seq/1234-56-78-11-22-33"

            log_file_name = os.path.join(model_save_path, 'log.txt')
            saver = open(log_file_name, "a")
            saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
            saver.flush()

            # Logging the details for this experiment
            saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
            saver.write(args.__repr__() + "\n\n")
            saver.flush()

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    voxel_size = (0.25, 0.25, 0.4)
    area_extents = np.array([[-32., 32.], [-32., 32.], [-3., 2.]])

    trainset = TrainDatasetMultiSeq(dataset_root=args.data, future_frame_skip=0, voxel_size=voxel_size,
                                    area_extents=area_extents, num_category=cell_category_num)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    print("Training dataset size:", len(trainset))

    model_encoder = FeatEncoder(height_feat_size=height_feat_size)
    model_head = MotionNetMGDA(out_seq_len=out_seq_len, motion_category_num=2)
    model_encoder = nn.DataParallel(model_encoder)
    model_encoder = model_encoder.to(device)
    model_head = nn.DataParallel(model_head)
    model_head = model_head.to(device)

    criterion = nn.SmoothL1Loss(reduction='none')

    encoder_optimizer = optim.Adam(model_encoder.parameters(), lr=0.002)
    head_optimizer = optim.Adam(model_head.parameters(), lr=0.002)
    encoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_optimizer, milestones=[20, 40, 50, 60], gamma=0.5)
    head_scheduler = torch.optim.lr_scheduler.MultiStepLR(head_optimizer, milestones=[20, 40, 50, 60], gamma=0.5)

    if args.resume != '':
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        model_encoder.load_state_dict(checkpoint['encoder_state_dict'])
        model_head.load_state_dict(checkpoint['head_state_dict'])

        encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        head_optimizer.load_state_dict(checkpoint['head_optimizer_state_dict'])

        encoder_scheduler.load_state_dict(checkpoint['encoder_scheduler_state_dict'])
        head_scheduler.load_state_dict(checkpoint['head_scheduler_state_dict'])

        print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))

    for epoch in range(start_epoch, num_epochs + 1):
        lr = encoder_optimizer.param_groups[0]['lr']
        print("Epoch {}, learning rate {}".format(epoch, lr))

        if need_log:
            saver.write("epoch: {}, lr: {}\t".format(epoch, lr))
            saver.flush()

        encoder_scheduler.step()
        head_scheduler.step()
        model_encoder.train()
        model_head.train()

        models = [model_encoder, model_head]
        optimizers = [encoder_optimizer, head_optimizer]

        loss_disp, loss_class, loss_motion, loss_bg_tc, loss_sc, loss_fg_tc \
            = train(models, criterion, trainloader, optimizers, device, epoch)

        if need_log:
            saver.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(loss_disp, loss_class, loss_motion, loss_bg_tc,
                                                          loss_fg_tc, loss_sc))
            saver.flush()

        # save model
        if need_log and (epoch % 5 == 0 or epoch == num_epochs or epoch == 1 or epoch > 40):
            save_dict = {'epoch': epoch,
                         'encoder_state_dict': model_encoder.state_dict(),
                         'head_state_dict': model_head.state_dict(),
                         'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                         'head_optimizer_state_dict': head_optimizer.state_dict(),
                         'encoder_scheduler_state_dict': encoder_scheduler.state_dict(),
                         'head_scheduler_state_dict': head_scheduler.state_dict(),
                         'loss': loss_disp.avg}
            torch.save(save_dict, os.path.join(model_save_path, 'epoch_' + str(epoch) + '.pth'))

    if need_log:
        saver.close()


def train(models, criterion, trainloader, optimizers, device, epoch):
    running_loss_bg_tc = AverageMeter('bg_tc', ':.7f')  # background temporal consistency error
    running_loss_fg_tc = AverageMeter('fg_tc', ':.7f')  # foreground temporal consistency error
    running_loss_sc = AverageMeter('sc', ':.7f')  # spatial consistency error
    running_loss_disp = AverageMeter('Disp', ':.6f')  # for motion prediction error
    running_loss_class = AverageMeter('Obj_Cls', ':.6f')  # for cell classification error
    running_loss_motion = AverageMeter('Motion_Cls', ':.6f')  # for state estimation error

    encoder = models[0]
    pred_head = models[1]

    for i, data in enumerate(trainloader, 0):
        padded_voxel_points, all_disp_field_gt, all_valid_pixel_maps, non_empty_map, pixel_cat_map_gt, \
            trans_matrices, motion_gt, pixel_instance_map, num_past_frames, num_future_frames = data

        # Move to GPU/CPU
        padded_voxel_points = padded_voxel_points.view(-1, num_past_frames[0].item(), 256, 256, height_feat_size)
        padded_voxel_points = padded_voxel_points.to(device)

        # Make prediction
        # -- Prepare for computing coefficients of loss terms
        with torch.no_grad():
            shared_feats = encoder(padded_voxel_points)

        # Compute loss coefficients
        shared_feats_tensor = shared_feats.clone().detach().requires_grad_(True)
        disp_pred_tensor, class_pred_tensor, motion_pred_tensor = pred_head(shared_feats_tensor)
        scale = compute_loss_coeff(optimizers, device, num_future_frames[0].item(), all_disp_field_gt,
                                   all_valid_pixel_maps, pixel_cat_map_gt, disp_pred_tensor, criterion,
                                   non_empty_map, class_pred_tensor, motion_gt, motion_pred_tensor, shared_feats_tensor)

        # Forward prediction
        shared_feats = encoder(padded_voxel_points)
        disp_pred, class_pred, motion_pred = pred_head(shared_feats)

        # Compute and back-propagate the losses
        loss_disp, loss_class, loss_motion, loss_bg_tc, loss_sc, loss_fg_tc = \
            compute_and_bp_loss(optimizers, device, num_future_frames[0].item(), all_disp_field_gt, all_valid_pixel_maps,
                                pixel_cat_map_gt, disp_pred, criterion, non_empty_map, class_pred, motion_gt,
                                motion_pred, trans_matrices, pixel_instance_map, scale)

        if not all((loss_disp, loss_class, loss_motion)):
            print("{}, \t{}, \tat epoch {}, \titerations {} [empty occupy map]".
                  format(running_loss_disp, running_loss_class, epoch, i))
            continue

        running_loss_bg_tc.update(loss_bg_tc)
        running_loss_fg_tc.update(loss_fg_tc)
        running_loss_sc.update(loss_sc)
        running_loss_disp.update(loss_disp)
        running_loss_class.update(loss_class)
        running_loss_motion.update(loss_motion)
        print("[{}/{}]\t{}, \t{}, \t{}, \t{}, \t{}, \t{}".
              format(epoch, i, running_loss_disp, running_loss_class, running_loss_motion, running_loss_bg_tc,
                     running_loss_sc, running_loss_fg_tc))

    return running_loss_disp, running_loss_class, running_loss_motion, running_loss_bg_tc, \
        running_loss_sc, running_loss_fg_tc


# Compute the loss coefficients adaptively
def compute_loss_coeff(optimizers, device, future_frames_num, all_disp_field_gt, all_valid_pixel_maps, pixel_cat_map_gt,
                       disp_pred, criterion, non_empty_map, class_pred, motion_gt, motion_pred, shared_feats_tensor):
    encoder_optimizer = optimizers[0]
    head_optimizer = optimizers[1]
    encoder_optimizer.zero_grad()
    head_optimizer.zero_grad()

    grads = {}

    # Compute the displacement loss
    all_disp_field_gt = all_disp_field_gt.view(-1, future_frames_num, 256, 256, 2)
    gt = all_disp_field_gt[:, -future_frames_num:, ...].contiguous()
    gt = gt.view(-1, gt.size(2), gt.size(3), gt.size(4))
    gt = gt.permute(0, 3, 1, 2).to(device)

    all_valid_pixel_maps = all_valid_pixel_maps.view(-1, future_frames_num, 256, 256)
    valid_pixel_maps = all_valid_pixel_maps[:, -future_frames_num:, ...].contiguous()
    valid_pixel_maps = valid_pixel_maps.view(-1, valid_pixel_maps.size(2), valid_pixel_maps.size(3))
    valid_pixel_maps = torch.unsqueeze(valid_pixel_maps, 1)
    valid_pixel_maps = valid_pixel_maps.to(device)

    valid_pixel_num = torch.nonzero(valid_pixel_maps).size(0)
    if valid_pixel_num == 0:
        return [None] * 3

    # ---------------------------------------------------------------------
    # -- Generate the displacement w.r.t. the keyframe
    if pred_adj_frame_distance:
        disp_pred = disp_pred.view(-1, future_frames_num, disp_pred.size(-3), disp_pred.size(-2), disp_pred.size(-1))

        for c in range(1, disp_pred.size(1)):
            disp_pred[:, c, ...] = disp_pred[:, c, ...] + disp_pred[:, c - 1, ...]
        disp_pred = disp_pred.view(-1, disp_pred.size(-3), disp_pred.size(-2), disp_pred.size(-1))

    # ---------------------------------------------------------------------
    # -- Compute the gated displacement loss
    pixel_cat_map_gt = pixel_cat_map_gt.view(-1, 256, 256, cell_category_num)

    pixel_cat_map_gt_numpy = pixel_cat_map_gt.numpy()
    pixel_cat_map_gt_numpy = np.argmax(pixel_cat_map_gt_numpy, axis=-1) + 1
    cat_weight_map = np.zeros_like(pixel_cat_map_gt_numpy, dtype=np.float32)
    weight_vector = [0.005, 1.0, 1.0, 1.0, 1.0]  # [bg, car & bus, ped, bike, other]
    for k in range(5):
        mask = pixel_cat_map_gt_numpy == (k + 1)
        cat_weight_map[mask] = weight_vector[k]

    cat_weight_map = cat_weight_map[:, np.newaxis, np.newaxis, ...]  # (batch, 1, 1, h, w)
    cat_weight_map = torch.from_numpy(cat_weight_map).to(device)
    map_shape = cat_weight_map.size()

    loss_disp = criterion(gt * valid_pixel_maps, disp_pred * valid_pixel_maps)
    loss_disp = loss_disp.view(map_shape[0], -1, map_shape[-3], map_shape[-2], map_shape[-1])
    loss_disp = torch.sum(loss_disp * cat_weight_map) / valid_pixel_num

    encoder_optimizer.zero_grad()
    head_optimizer.zero_grad()

    loss_disp.backward(retain_graph=True)  # this operation is expensive
    grads[0] = []
    grads[0].append(shared_feats_tensor.grad.data.clone().detach())
    shared_feats_tensor.grad.data.zero_()

    # ---------------------------------------------------------------------
    # -- Compute the classification loss
    non_empty_map = non_empty_map.view(-1, 256, 256)
    non_empty_map = non_empty_map.to(device)
    pixel_cat_map_gt = pixel_cat_map_gt.permute(0, 3, 1, 2).to(device)

    log_softmax_probs = F.log_softmax(class_pred, dim=1)

    map_shape = cat_weight_map.size()
    cat_weight_map = cat_weight_map.view(map_shape[0], map_shape[-2], map_shape[-1])  # (bs, h, w)
    loss_class = torch.sum(- pixel_cat_map_gt * log_softmax_probs, dim=1) * cat_weight_map
    loss_class = torch.sum(loss_class * non_empty_map) / torch.nonzero(non_empty_map).size(0)

    encoder_optimizer.zero_grad()
    head_optimizer.zero_grad()

    loss_class.backward(retain_graph=True)
    grads[1] = []
    grads[1].append(shared_feats_tensor.grad.data.clone().detach())
    shared_feats_tensor.grad.data.zero_()

    # ---------------------------------------------------------------------
    # -- Compute the speed level classification loss
    motion_gt = motion_gt.view(-1, 256, 256, 2)
    motion_gt_numpy = motion_gt.numpy()
    motion_gt = motion_gt.permute(0, 3, 1, 2).to(device)
    log_softmax_motion_pred = F.log_softmax(motion_pred, dim=1)

    motion_gt_numpy = np.argmax(motion_gt_numpy, axis=-1) + 1
    motion_weight_map = np.zeros_like(motion_gt_numpy, dtype=np.float32)
    weight_vector = [0.005, 1.0]
    for k in range(2):
        mask = motion_gt_numpy == (k + 1)
        motion_weight_map[mask] = weight_vector[k]

    motion_weight_map = torch.from_numpy(motion_weight_map).to(device)
    loss_speed = torch.sum(- motion_gt * log_softmax_motion_pred, dim=1) * motion_weight_map

    loss_motion = torch.sum(loss_speed * non_empty_map) / torch.nonzero(non_empty_map).size(0)

    encoder_optimizer.zero_grad()
    head_optimizer.zero_grad()

    loss_motion.backward(retain_graph=True)
    grads[2] = []
    grads[2].append(shared_feats_tensor.grad.data.clone().detach())
    shared_feats_tensor.grad.data.zero_()

    # ---------------------------------------------------------------------
    # -- Frank-Wolfe iteration to compute scales.
    scale = np.zeros(3, dtype=np.float32)
    sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in range(3)])
    for i in range(3):
        scale[i] = float(sol[i])

    return scale


# Compute and back-propagate the loss
def compute_and_bp_loss(optimizers, device, future_frames_num, all_disp_field_gt, all_valid_pixel_maps, pixel_cat_map_gt,
                        disp_pred, criterion, non_empty_map, class_pred, motion_gt, motion_pred, trans_matrices,
                        pixel_instance_map, scale):
    encoder_optimizer = optimizers[0]
    head_optimizer = optimizers[1]
    encoder_optimizer.zero_grad()
    head_optimizer.zero_grad()

    # Compute the displacement loss
    all_disp_field_gt = all_disp_field_gt.view(-1, future_frames_num, 256, 256, 2)
    gt = all_disp_field_gt[:, -future_frames_num:, ...].contiguous()
    gt = gt.view(-1, gt.size(2), gt.size(3), gt.size(4))
    gt = gt.permute(0, 3, 1, 2).to(device)

    all_valid_pixel_maps = all_valid_pixel_maps.view(-1, future_frames_num, 256, 256)
    valid_pixel_maps = all_valid_pixel_maps[:, -future_frames_num:, ...].contiguous()
    valid_pixel_maps = valid_pixel_maps.view(-1, valid_pixel_maps.size(2), valid_pixel_maps.size(3))
    valid_pixel_maps = torch.unsqueeze(valid_pixel_maps, 1)
    valid_pixel_maps = valid_pixel_maps.to(device)

    valid_pixel_num = torch.nonzero(valid_pixel_maps).size(0)
    if valid_pixel_num == 0:
        return [None] * 6

    # ---------------------------------------------------------------------
    # -- Generate the displacement w.r.t. the keyframe
    if pred_adj_frame_distance:
        disp_pred = disp_pred.view(-1, future_frames_num, disp_pred.size(-3), disp_pred.size(-2), disp_pred.size(-1))

        # Compute temporal consistency loss
        if use_bg_temporal_consistency:
            bg_tc_loss = background_temporal_consistency_loss(disp_pred, pixel_cat_map_gt, non_empty_map, trans_matrices)

        if use_fg_temporal_consistency or use_spatial_consistency:
            instance_spatio_temp_loss, instance_spatial_loss_value, instance_temporal_loss_value \
                = instance_spatial_temporal_consistency_loss(disp_pred, pixel_instance_map)

        for c in range(1, disp_pred.size(1)):
            disp_pred[:, c, ...] = disp_pred[:, c, ...] + disp_pred[:, c - 1, ...]
        disp_pred = disp_pred.view(-1, disp_pred.size(-3), disp_pred.size(-2), disp_pred.size(-1))

    # ---------------------------------------------------------------------
    # -- Compute the masked displacement loss
    pixel_cat_map_gt = pixel_cat_map_gt.view(-1, 256, 256, cell_category_num)

    # Note: have also tried focal loss, but did not observe noticeable improvement
    pixel_cat_map_gt_numpy = pixel_cat_map_gt.numpy()
    pixel_cat_map_gt_numpy = np.argmax(pixel_cat_map_gt_numpy, axis=-1) + 1
    cat_weight_map = np.zeros_like(pixel_cat_map_gt_numpy, dtype=np.float32)
    weight_vector = [0.005, 1.0, 1.0, 1.0, 1.0]  # [bg, car & bus, ped, bike, other]
    for k in range(len(weight_vector)):
        mask = pixel_cat_map_gt_numpy == (k + 1)
        cat_weight_map[mask] = weight_vector[k]

    cat_weight_map = cat_weight_map[:, np.newaxis, np.newaxis, ...]  # (batch, 1, 1, h, w)
    cat_weight_map = torch.from_numpy(cat_weight_map).to(device)
    map_shape = cat_weight_map.size()

    loss_disp = criterion(gt * valid_pixel_maps, disp_pred * valid_pixel_maps)
    loss_disp = loss_disp.view(map_shape[0], -1, map_shape[-3], map_shape[-2], map_shape[-1])
    loss_disp = torch.sum(loss_disp * cat_weight_map) / valid_pixel_num

    # ---------------------------------------------------------------------
    # -- Compute the grid cell classification loss
    non_empty_map = non_empty_map.view(-1, 256, 256)
    non_empty_map = non_empty_map.to(device)
    pixel_cat_map_gt = pixel_cat_map_gt.permute(0, 3, 1, 2).to(device)

    log_softmax_probs = F.log_softmax(class_pred, dim=1)

    map_shape = cat_weight_map.size()
    cat_weight_map = cat_weight_map.view(map_shape[0], map_shape[-2], map_shape[-1])  # (bs, h, w)
    loss_class = torch.sum(- pixel_cat_map_gt * log_softmax_probs, dim=1) * cat_weight_map
    loss_class = torch.sum(loss_class * non_empty_map) / torch.nonzero(non_empty_map).size(0)

    # ---------------------------------------------------------------------
    # -- Compute the speed level classification loss
    motion_gt = motion_gt.view(-1, 256, 256, 2)
    motion_gt_numpy = motion_gt.numpy()
    motion_gt = motion_gt.permute(0, 3, 1, 2).to(device)
    log_softmax_motion_pred = F.log_softmax(motion_pred, dim=1)

    motion_gt_numpy = np.argmax(motion_gt_numpy, axis=-1) + 1
    motion_weight_map = np.zeros_like(motion_gt_numpy, dtype=np.float32)
    weight_vector = [0.005, 1.0]  # [static, moving]
    for k in range(len(weight_vector)):
        mask = motion_gt_numpy == (k + 1)
        motion_weight_map[mask] = weight_vector[k]

    motion_weight_map = torch.from_numpy(motion_weight_map).to(device)
    loss_speed = torch.sum(- motion_gt * log_softmax_motion_pred, dim=1) * motion_weight_map
    loss_motion = torch.sum(loss_speed * non_empty_map) / torch.nonzero(non_empty_map).size(0)

    # ---------------------------------------------------------------------
    # -- Sum up all the losses
    if use_bg_temporal_consistency and (use_fg_temporal_consistency or use_spatial_consistency):
        loss = scale[0] * loss_disp + reg_weight_cls * scale[1] * loss_class + scale[2] * loss_motion + \
               reg_weight_bg_tc * bg_tc_loss + instance_spatio_temp_loss
    elif use_bg_temporal_consistency:
        loss = scale[0] * loss_disp + reg_weight_cls * scale[1] * loss_class + scale[2] * loss_motion + \
               reg_weight_bg_tc * bg_tc_loss
    elif use_spatial_consistency or use_fg_temporal_consistency:
        loss = scale[0] * loss_disp + reg_weight_cls * scale[1] * loss_class + scale[2] * loss_motion + \
               instance_spatio_temp_loss
    else:
        loss = scale[0] * loss_disp + reg_weight_cls * scale[1] * loss_class + scale[2] * loss_motion
    loss.backward()
    encoder_optimizer.step()
    head_optimizer.step()

    if use_bg_temporal_consistency:
        bg_tc_loss_value = bg_tc_loss.item()
    else:
        bg_tc_loss_value = -1

    if use_spatial_consistency or use_fg_temporal_consistency:
        sc_loss_value = instance_spatial_loss_value
        fg_tc_loss_value = instance_temporal_loss_value
    else:
        sc_loss_value = -1
        fg_tc_loss_value = -1

    return loss_disp.item(), loss_class.item(), loss_motion.item(), bg_tc_loss_value, \
        sc_loss_value, fg_tc_loss_value


def background_temporal_consistency_loss(disp_pred, pixel_cat_map_gt, non_empty_map, trans_matrices):
    """
    disp_pred: Should be relative displacement between adjacent frames. shape (batch * 2, sweep_num, 2, h, w)
    pixel_cat_map_gt: Shape (batch, 2, h, w, cat_num)
    non_empty_map: Shape (batch, 2, h, w)
    trans_matrices: Shape (batch, 2, sweep_num, 4, 4)
    """
    criterion = nn.SmoothL1Loss(reduction='sum')

    non_empty_map_numpy = non_empty_map.numpy()
    pixel_cat_maps = pixel_cat_map_gt.numpy()
    max_prob = np.amax(pixel_cat_maps, axis=-1)
    filter_mask = max_prob == 1.0
    pixel_cat_maps = np.argmax(pixel_cat_maps, axis=-1) + 1  # category starts from 1 (background), etc
    pixel_cat_maps = (pixel_cat_maps * non_empty_map_numpy * filter_mask)  # (batch, 2, h, w)

    trans_matrices = trans_matrices.numpy()
    device = disp_pred.device

    pred_shape = disp_pred.size()
    disp_pred = disp_pred.view(-1, 2, pred_shape[1], pred_shape[2], pred_shape[3], pred_shape[4])

    seq_1_pred = disp_pred[:, 0]  # (batch, sweep_num, 2, h, w)
    seq_2_pred = disp_pred[:, 1]

    seq_1_absolute_pred_list = list()
    seq_2_absolute_pred_list = list()

    seq_1_absolute_pred_list.append(seq_1_pred[:, 1])
    for i in range(2, pred_shape[1]):
        seq_1_absolute_pred_list.append(seq_1_pred[:, i] + seq_1_absolute_pred_list[i - 2])

    seq_2_absolute_pred_list.append(seq_2_pred[:, 0])
    for i in range(1, pred_shape[1] - 1):
        seq_2_absolute_pred_list.append(seq_2_pred[:, i] + seq_2_absolute_pred_list[i - 1])

    # ----------------- Compute the consistency loss -----------------
    # Compute the transformation matrices
    # First, transform the coordinate
    transformed_disp_pred_list = list()

    trans_matrix_global = trans_matrices[:, 1]  # (batch, sweep_num, 4, 4)
    trans_matrix_global = trans_matrix_global[:, trans_matrix_idx, 0:3]  # (batch, 3, 4)  # <---
    trans_matrix_global = trans_matrix_global[:, :, (0, 1, 3)]  # (batch, 3, 3)
    trans_matrix_global[:, 2] = np.array([0.0, 0.0, 1.0])

    # --- Move pixel coord to global and rescale; then rotate; then move back to local pixel coord
    translate_to_global = np.array([[1.0, 0.0, -120.0], [0.0, 1.0, -120.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    scale_global = np.array([[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    trans_global = scale_global @ translate_to_global
    inv_trans_global = np.linalg.inv(trans_global)

    trans_global = np.expand_dims(trans_global, axis=0)
    inv_trans_global = np.expand_dims(inv_trans_global, axis=0)
    trans_matrix_total = inv_trans_global @ trans_matrix_global @ trans_global

    # --- Generate grid transformation matrix, so as to use Pytorch affine_grid and grid_sample function
    w, h = pred_shape[-2], pred_shape[-1]
    resize_m = np.array([
        [2 / w, 0.0, -1],
        [0.0, 2 / h, -1],
        [0.0, 0.0, 1]
    ], dtype=np.float32)
    inverse_m = np.linalg.inv(resize_m)
    resize_m = np.expand_dims(resize_m, axis=0)
    inverse_m = np.expand_dims(inverse_m, axis=0)

    grid_trans_matrix = resize_m @ trans_matrix_total @ inverse_m  # (batch, 3, 3)
    grid_trans_matrix = grid_trans_matrix[:, :2].astype(np.float32)
    grid_trans_matrix = torch.from_numpy(grid_trans_matrix)

    # --- For displacement field
    trans_matrix_translation_global = np.eye(trans_matrix_total.shape[1])
    trans_matrix_translation_global = np.expand_dims(trans_matrix_translation_global, axis=0)
    trans_matrix_translation_global = np.repeat(trans_matrix_translation_global, grid_trans_matrix.shape[0], axis=0)
    trans_matrix_translation_global[:, :, 2] = trans_matrix_global[:, :, 2]  # only translation
    trans_matrix_translation_total = inv_trans_global @ trans_matrix_translation_global @ trans_global

    grid_trans_matrix_disp = resize_m @ trans_matrix_translation_total @ inverse_m
    grid_trans_matrix_disp = grid_trans_matrix_disp[:, :2].astype(np.float32)
    grid_trans_matrix_disp = torch.from_numpy(grid_trans_matrix_disp).to(device)

    disp_rotate_matrix = trans_matrix_global[:, 0:2, 0:2].astype(np.float32)  # (batch, 2, 2)
    disp_rotate_matrix = torch.from_numpy(disp_rotate_matrix).to(device)

    for i in range(len(seq_1_absolute_pred_list)):

        # --- Start transformation for displacement field
        curr_pred = seq_1_absolute_pred_list[i]  # (batch, 2, h, w)

        # First, rotation
        curr_pred = curr_pred.permute(0, 2, 3, 1).contiguous()  # (batch, h, w, 2)
        curr_pred = curr_pred.view(-1, h * w, 2)
        curr_pred = torch.bmm(curr_pred, disp_rotate_matrix)
        curr_pred = curr_pred.view(-1, h, w, 2)
        curr_pred = curr_pred.permute(0, 3, 1, 2).contiguous()  # (batch, 2, h, w)

        # Next, translation
        curr_pred = curr_pred.permute(0, 1, 3, 2).contiguous()  # swap x and y axis
        curr_pred = torch.flip(curr_pred, dims=[2])

        grid = F.affine_grid(grid_trans_matrix_disp, curr_pred.size())
        if use_nn_sampling:
            curr_pred = F.grid_sample(curr_pred, grid, mode='nearest')
        else:
            curr_pred = F.grid_sample(curr_pred, grid)

        curr_pred = torch.flip(curr_pred, dims=[2])
        curr_pred = curr_pred.permute(0, 1, 3, 2).contiguous()

        transformed_disp_pred_list.append(curr_pred)

    # --- Start transformation for category map
    pixel_cat_map = pixel_cat_maps[:, 0]  # (batch, h, w)
    pixel_cat_map = torch.from_numpy(pixel_cat_map.astype(np.float32))
    pixel_cat_map = pixel_cat_map[:, None, :, :]  # (batch, 1, h, w)
    trans_pixel_cat_map = pixel_cat_map.permute(0, 1, 3, 2)  # (batch, 1, h, w), swap x and y axis
    trans_pixel_cat_map = torch.flip(trans_pixel_cat_map, dims=[2])

    grid = F.affine_grid(grid_trans_matrix, pixel_cat_map.size())
    trans_pixel_cat_map = F.grid_sample(trans_pixel_cat_map, grid, mode='nearest')

    trans_pixel_cat_map = torch.flip(trans_pixel_cat_map, dims=[2])
    trans_pixel_cat_map = trans_pixel_cat_map.permute(0, 1, 3, 2)

    # --- Compute the loss, using smooth l1 loss
    adj_pixel_cat_map = pixel_cat_maps[:, 1]
    adj_pixel_cat_map = torch.from_numpy(adj_pixel_cat_map.astype(np.float32))
    adj_pixel_cat_map = torch.unsqueeze(adj_pixel_cat_map, dim=1)

    mask_common = trans_pixel_cat_map == adj_pixel_cat_map
    mask_common = mask_common.float()
    non_empty_map_gpu = non_empty_map.to(device)
    non_empty_map_gpu = non_empty_map_gpu[:, 1:2, :, :]  # select the second sequence, keep dim
    mask_common = mask_common.to(device)
    mask_common = mask_common * non_empty_map_gpu

    loss_list = list()
    for i in range(len(seq_1_absolute_pred_list)):
        trans_seq_1_pred = transformed_disp_pred_list[i]  # (batch, 2, h, w)
        seq_2_pred = seq_2_absolute_pred_list[i]  # (batch, 2, h, w)

        trans_seq_1_pred = trans_seq_1_pred * mask_common
        seq_2_pred = seq_2_pred * mask_common

        num_non_empty_cells = torch.nonzero(mask_common).size(0)
        if num_non_empty_cells != 0:
            loss = criterion(trans_seq_1_pred, seq_2_pred) / num_non_empty_cells
            loss_list.append(loss)

    res_loss = torch.mean(torch.stack(loss_list, 0))

    return res_loss


# We name it instance spatial-temporal consistency loss because it involves each instance
def instance_spatial_temporal_consistency_loss(disp_pred, pixel_instance_map):
    device = disp_pred.device

    pred_shape = disp_pred.size()
    disp_pred = disp_pred.view(-1, 2, pred_shape[1], pred_shape[2], pred_shape[3], pred_shape[4])

    seq_1_pred = disp_pred[:, 0]  # (batch, sweep_num, 2, h, w)
    seq_2_pred = disp_pred[:, 1]

    pixel_instance_map = pixel_instance_map.numpy()
    batch = pixel_instance_map.shape[0]

    spatial_loss = 0.0
    temporal_loss = 0.0
    counter = 0
    criterion = nn.SmoothL1Loss()

    for i in range(batch):
        curr_batch_instance_maps = pixel_instance_map[i]
        seq_1_instance_map = curr_batch_instance_maps[0]
        seq_2_instance_map = curr_batch_instance_maps[1]

        seq_1_instance_ids = np.unique(seq_1_instance_map)
        seq_2_instance_ids = np.unique(seq_2_instance_map)

        common_instance_ids = np.intersect1d(seq_1_instance_ids, seq_2_instance_ids, assume_unique=True)

        seq_1_batch_pred = seq_1_pred[i]  # (sweep_num, 2, h, w)
        seq_2_batch_pred = seq_2_pred[i]

        for h in common_instance_ids:
            if h == 0:  # do not consider the background instance
                continue

            seq_1_mask = np.where(seq_1_instance_map == h)
            seq_1_idx_x = torch.from_numpy(seq_1_mask[0]).to(device)
            seq_1_idx_y = torch.from_numpy(seq_1_mask[1]).to(device)
            seq_1_selected_cells = seq_1_batch_pred[:, :, seq_1_idx_x, seq_1_idx_y]

            seq_2_mask = np.where(seq_2_instance_map == h)
            seq_2_idx_x = torch.from_numpy(seq_2_mask[0]).to(device)
            seq_2_idx_y = torch.from_numpy(seq_2_mask[1]).to(device)
            seq_2_selected_cells = seq_2_batch_pred[:, :, seq_2_idx_x, seq_2_idx_y]

            seq_1_selected_cell_num = seq_1_selected_cells.size(2)
            seq_2_selected_cell_num = seq_2_selected_cells.size(2)

            # for spatial loss
            if use_spatial_consistency:
                tmp_seq_1 = 0
                if seq_1_selected_cell_num > 1:
                    tmp_seq_1 = criterion(seq_1_selected_cells[:, :, :-1], seq_1_selected_cells[:, :, 1:])

                tmp_seq_2 = 0
                if seq_2_selected_cell_num > 1:
                    tmp_seq_2 = criterion(seq_2_selected_cells[:, :, :-1], seq_2_selected_cells[:, :, 1:])

                spatial_loss += tmp_seq_1 + tmp_seq_2

            if use_fg_temporal_consistency:
                seq_1_mean = torch.mean(seq_1_selected_cells, dim=2)
                seq_2_mean = torch.mean(seq_2_selected_cells, dim=2)
                temporal_loss += criterion(seq_1_mean, seq_2_mean)

            counter += 1

    if counter != 0:
        spatial_loss = spatial_loss / counter
        temporal_loss = temporal_loss / counter

    total_loss = reg_weight_sc * spatial_loss + reg_weight_fg_tc * temporal_loss

    spatial_loss_value = 0 if type(spatial_loss) == float else spatial_loss.item()
    temporal_loss_value = 0 if type(temporal_loss) == float else temporal_loss.item()

    return total_loss, spatial_loss_value, temporal_loss_value


if __name__ == "__main__":
    main()
