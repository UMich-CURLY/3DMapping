# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved. The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages. As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import sys
import argparse
import os
from shutil import copytree, copy
from model import MotionNet
from data.nuscenes_dataloader import DatasetSingleSeq


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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


out_seq_len = 20  # The number of future frames we are going to predict
height_feat_size = 13  # The size along the height dimension
cell_category_num = 5  # The number of object categories (including the background)

pred_adj_frame_distance = True  # Whether to predict the relative offset between frames
use_weighted_loss = True  # Whether to set different weights for different grid cell categories for loss computation

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', default=None, type=str, help='The path to the preprocessed sparse BEV training data')
parser.add_argument('--resume', default='', type=str, help='The path to the saved model that is loaded to resume training')
parser.add_argument('--batch', default=8, type=int, help='Batch size')
parser.add_argument('--nepoch', default=45, type=int, help='Number of epochs')
parser.add_argument('--nworker', default=4, type=int, help='Number of workers')
parser.add_argument('--log', action='store_true', help='Whether to log')
parser.add_argument('--logpath', default='', help='The path to the output log file')

args = parser.parse_args()
print(args)

num_epochs = args.nepoch
need_log = args.log
BATCH_SIZE = args.batch
num_workers = args.nworker


def main():
    start_epoch = 1
    # Whether to log the training information
    if need_log:
        logger_root = args.logpath if args.logpath != '' else 'logs'
        time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        if args.resume == '':
            model_save_path = check_folder(logger_root)
            model_save_path = check_folder(os.path.join(model_save_path, 'train_single_seq'))
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

    trainset = DatasetSingleSeq(dataset_root=args.data, split='train', future_frame_skip=0,
                                voxel_size=voxel_size, area_extents=area_extents, num_category=cell_category_num)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    print("Training dataset size:", len(trainset))

    model = MotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
    model = nn.DataParallel(model)
    model = model.to(device)

    if use_weighted_loss:
        criterion = nn.SmoothL1Loss(reduction='none')
    else:
        criterion = nn.SmoothL1Loss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.5)

    if args.resume != '':
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))

    for epoch in range(start_epoch, num_epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        print("Epoch {}, learning rate {}".format(epoch, lr))

        if need_log:
            saver.write("epoch: {}, lr: {}\t".format(epoch, lr))
            saver.flush()

        scheduler.step()
        model.train()

        loss_disp, loss_class, loss_motion = train(model, criterion, trainloader, optimizer, device, epoch)

        if need_log:
            saver.write("{}\t{}\t{}\n".format(loss_disp, loss_class, loss_motion))
            saver.flush()

        # save model
        if need_log and (epoch % 5 == 0 or epoch == num_epochs or epoch == 1 or epoch >= 20):
            save_dict = {'epoch': epoch,
                         'model_state_dict': model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'scheduler_state_dict': scheduler.state_dict(),
                         'loss': loss_disp.avg}
            torch.save(save_dict, os.path.join(model_save_path, 'epoch_' + str(epoch) + '.pth'))

    if need_log:
        saver.close()


def train(model, criterion, trainloader, optimizer, device, epoch):
    running_loss_disp = AverageMeter('Disp', ':.6f')  # for motion prediction error
    running_loss_class = AverageMeter('Obj_Cls', ':.6f')  # for cell classification error
    running_loss_motion = AverageMeter('Motion_Cls', ':.6f')  # for state estimation error

    for i, data in enumerate(trainloader, 0):
        padded_voxel_points, all_disp_field_gt, all_valid_pixel_maps, non_empty_map, pixel_cat_map_gt, \
            past_steps, future_steps, motion_gt = data

        # Move to GPU/CPU
        padded_voxel_points = padded_voxel_points.to(device)

        # Make prediction
        disp_pred, class_pred, motion_pred = model(padded_voxel_points)

        # Compute and back-propagate the losses
        loss_disp, loss_class, loss_motion = \
            compute_and_bp_loss(optimizer, device, future_steps[0].item(), all_disp_field_gt, all_valid_pixel_maps,
                                pixel_cat_map_gt, disp_pred, criterion, non_empty_map, class_pred, motion_gt, motion_pred)

        if not all((loss_disp, loss_class, loss_motion)):
            print("{}, \t{}, \tat epoch {}, \titerations {} [empty occupy map]".
                  format(running_loss_disp, running_loss_class, epoch, i))
            continue

        running_loss_disp.update(loss_disp)
        running_loss_class.update(loss_class)
        running_loss_motion.update(loss_motion)
        print("{}, \t{}, \t{}, \tat epoch {}, \titerations {}".
              format(running_loss_disp, running_loss_class, running_loss_motion, epoch, i))

    return running_loss_disp, running_loss_class, running_loss_motion


# Compute and back-propagate the loss
def compute_and_bp_loss(optimizer, device, future_frames_num, all_disp_field_gt, all_valid_pixel_maps, pixel_cat_map_gt,
                        disp_pred, criterion, non_empty_map, class_pred, motion_gt, motion_pred):
    optimizer.zero_grad()

    # Compute the displacement loss
    gt = all_disp_field_gt[:, -future_frames_num:, ...].contiguous()
    gt = gt.view(-1, gt.size(2), gt.size(3), gt.size(4))
    gt = gt.permute(0, 3, 1, 2).to(device)

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
    # -- Compute the masked displacement loss
    if use_weighted_loss:  # Note: have also tried focal loss, but did not observe noticeable improvement
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
    else:
        loss_disp = criterion(gt * valid_pixel_maps, disp_pred * valid_pixel_maps) / valid_pixel_num

    # ---------------------------------------------------------------------
    # -- Compute the grid cell classification loss
    non_empty_map = non_empty_map.view(-1, 256, 256)
    non_empty_map = non_empty_map.to(device)
    pixel_cat_map_gt = pixel_cat_map_gt.permute(0, 3, 1, 2).to(device)

    log_softmax_probs = F.log_softmax(class_pred, dim=1)

    if use_weighted_loss:
        map_shape = cat_weight_map.size()
        cat_weight_map = cat_weight_map.view(map_shape[0], map_shape[-2], map_shape[-1])  # (bs, h, w)
        loss_class = torch.sum(- pixel_cat_map_gt * log_softmax_probs, dim=1) * cat_weight_map
    else:
        loss_class = torch.sum(- pixel_cat_map_gt * log_softmax_probs, dim=1)
    loss_class = torch.sum(loss_class * non_empty_map) / torch.nonzero(non_empty_map).size(0)

    # ---------------------------------------------------------------------
    # -- Compute the speed level classification loss
    motion_gt_numpy = motion_gt.numpy()
    motion_gt = motion_gt.permute(0, 3, 1, 2).to(device)
    log_softmax_motion_pred = F.log_softmax(motion_pred, dim=1)

    if use_weighted_loss:
        motion_gt_numpy = np.argmax(motion_gt_numpy, axis=-1) + 1
        motion_weight_map = np.zeros_like(motion_gt_numpy, dtype=np.float32)
        weight_vector = [0.005, 1.0]
        for k in range(2):
            mask = motion_gt_numpy == (k + 1)
            motion_weight_map[mask] = weight_vector[k]

        motion_weight_map = torch.from_numpy(motion_weight_map).to(device)
        loss_speed = torch.sum(- motion_gt * log_softmax_motion_pred, dim=1) * motion_weight_map
    else:
        loss_speed = torch.sum(- motion_gt * log_softmax_motion_pred, dim=1)
    loss_motion = torch.sum(loss_speed * non_empty_map) / torch.nonzero(non_empty_map).size(0)

    # ---------------------------------------------------------------------
    # -- Sum up all the losses
    loss = loss_disp + loss_class + loss_motion
    loss.backward()
    optimizer.step()

    return loss_disp.item(), loss_class.item(), loss_motion.item()


if __name__ == "__main__":
    main()
