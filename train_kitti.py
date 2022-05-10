import torch
import sys
# from torch._C import LongStorageBase

sys.path.append("./Models")
from Models.utils import *
from Data.dataset import *
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import math
from tqdm import tqdm
import numpy as np
import random
import argparse
import os
import json

import time
import numpy as np
import os
import json
import pdb
from PIL import Image

from Data.kitti_dataset import KittiDataset

from torch.utils.tensorboard import SummaryWriter

from Models.MotionSC import MotionSC
from Models.SSCNet_full import SSCNet_full
from Models.LMSCNet_SS import LMSCNet_SS
from Models.SSCNet import SSCNet


class_frequencies = np.array([5.41773033e+09, 1.57835390e+07, 1.25136000e+05, 1.18809000e+05,
                                6.46799000e+05, 8.21951000e+05, 2.62978000e+05, 2.83696000e+05,
                                2.04750000e+05, 6.16887030e+07, 4.50296100e+06, 4.48836500e+07,
                                2.26992300e+06, 5.68402180e+07, 1.57196520e+07, 1.58442623e+08,
                                2.06162300e+06, 3.69705220e+07, 1.15198800e+06, 3.34146000e+05])

def get_class_weights(freq):
    '''
    Cless weights being 1/log(fc) (https://arxiv.org/pdf/2008.10559.pdf)
    '''
    epsilon_w = 0.001  # eps to avoid zero division
    weights = torch.from_numpy(1 / np.log(freq + epsilon_w))

    return weights

# TODO: you may change these parameters if needed
# PARAMETERS
seed = 42
x_dim = 256
y_dim = 256
z_dim = 32
model_name = "MotionSC"
num_workers = 16
train_dir = "/media/jingyu/Jingyu-Data/dataset"
val_dir = "/media/jingyu/Jingyu-Data/dataset"
cylindrical = False
epoch_num = 100
remap = True
num_classes = 20


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weights = get_class_weights(class_frequencies).to(torch.float32)
criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=255, reduction='mean').to(device=device)
# criterion = nn.CrossEntropyLoss(weight=weights.to(device))

coor_ranges = [0,-25.6,-2] + [51.2,25.6,4.4]
voxel_sizes = [abs(coor_ranges[3] - coor_ranges[0]) / x_dim,
              abs(coor_ranges[4] - coor_ranges[1]) / y_dim,
              abs(coor_ranges[5] - coor_ranges[2]) / z_dim] # since BEV


lr = 0.001
BETA1 = 0.9
BETA2 = 0.999
model, B, T, decayRate, resample_free = get_model(model_name, num_classes, voxel_sizes, coor_ranges, [x_dim, y_dim, z_dim], device)
model_name += "_" + str(num_classes)
print("Running:", model_name)


# Data Loaders
carla_ds = KittiDataset(directory=train_dir, device=device, num_frames=T, random_flips=True, remap=remap, split='train')
dataloader = DataLoader(carla_ds, batch_size=B, shuffle=True, collate_fn=carla_ds.collate_fn, num_workers=num_workers)
val_ds = KittiDataset(directory=val_dir, device=device, num_frames=T, remap=remap, split='valid')
dataloader_val = DataLoader(val_ds, batch_size=B, shuffle=True, collate_fn=val_ds.collate_fn, num_workers=num_workers)
# test_ds = CarlaDataset(directory=val_dir, device=device, num_frames=T, cylindrical=cylindrical, remap=remap)
# dataloader_test = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=test_ds.collate_fn, num_workers=num_workers)


writer = SummaryWriter("./Models/Runs/" + model_name)
save_dir = "./Models/Weights/" + model_name

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

if device == "cuda":
    torch.cuda.empty_cache()

setup_seed(seed)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(BETA1, BETA2))
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

train_count = 0
for epoch in range(epoch_num):
    # Training
    model.train()
    for input_data, output, counts in dataloader:
        optimizer.zero_grad()
        input_data = torch.from_numpy(np.array(input_data)).to(device)
        output = torch.from_numpy(np.array(output)).to(device)
        counts = torch.from_numpy(np.array(counts)).to(device)
        preds = model(input_data)
         
        counts = counts.view(-1)
        output = output.view(-1).long()
        preds = preds.contiguous().view(-1, preds.shape[4])

        # Criterion requires input (NxC), output (N) dimension
        mask = counts == 0
        output_masked = output[mask]
        preds_masked = preds[mask]

        new_mask = counts == 1
        output[new_mask] = 255
        if resample_free:
            preds_masked, output_masked = resample_free_space(preds_masked, output_masked)

        # loss = criterion(preds_masked, output_masked)
        loss = criterion(preds, output)
        loss.backward()
        optimizer.step()
        
        # Accuracy
        with torch.no_grad():
            probs = nn.functional.softmax(preds_masked, dim=1)
            # preds_masked = np.argmax(probs.detach().cpu().numpy(), axis=1)
            # outputs_np = output_masked.detach().cpu().numpy()
            # accuracy = np.sum(preds_masked == outputs_np) / outputs_np.shape[0]

            preds_masked = torch.argmax(probs.detach(), dim=1)
            accuracy = torch.sum(preds_masked == output_masked.detach()) / output_masked.shape[0]
            # num_correct += torch.sum(preds_masked == output_masked)
            # num_total += output_masked.shape[0]
            
        # Record
        writer.add_scalar(model_name + '/Loss/Train', loss.item(), train_count)
        writer.add_scalar(model_name + '/Accuracy/Train', accuracy, train_count)
            
        train_count += input_data.shape[0]
        
    # Save model, decreaser learning rate
    my_lr_scheduler.step()
    torch.save(model.state_dict(), os.path.join(save_dir, "Epoch" + str(epoch) + ".pt"))

    # Validation
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        counter = 0
        num_correct = 0
        num_total = 0
        all_intersections = np.zeros(num_classes)
        all_unions = np.zeros(num_classes) + 1e-6 # SMOOTHING

        for input_data, output, counts in dataloader_val:
            optimizer.zero_grad()
            input_data = torch.from_numpy(np.array(input_data)).to(device)
            output = torch.from_numpy(np.array(output)).to(device)
            counts = torch.from_numpy(np.array(counts)).to(device)
            preds = model(input_data)

            counts = counts.view(-1)
            output = output.view(-1).long()
            preds = preds.contiguous().view(-1, preds.shape[4])

            # Criterion requires input (NxC), output (N) dimension
            mask = counts == 0
            output_masked = output[mask]
            preds_masked = preds[mask]
            loss = criterion(preds_masked, output_masked)

            running_loss += loss.item()
            counter += input_data.shape[0]

            # Accuracy
            probs = nn.functional.softmax(preds_masked, dim=1)
            
            # preds_masked = np.argmax(probs.detach().cpu().numpy(), axis=1)
            # outputs_np = output_masked.detach().cpu().numpy()
            # num_correct += np.sum(preds_masked == outputs_np)
            # num_total += outputs_np.shape[0]

            # Optimzied validation speed
            preds_masked = torch.argmax(probs.detach(), dim=1)
            num_correct += torch.sum(preds_masked == output_masked)
            num_total += output_masked.shape[0]

            # intersection, union = iou_one_frame(torch.tensor(preds_masked), torch.tensor(output_masked), n_classes=num_classes)
            intersection, union = iou_one_frame(preds_masked, output_masked, n_classes=num_classes)
            all_intersections += intersection
            all_unions += union
        
        print(f'Eppoch Num: {epoch} ------ average val loss: {running_loss/counter}')
        print(f'Eppoch Num: {epoch} ------ average val accuracy: {num_correct/num_total}')
        print(f'Eppoch Num: {epoch} ------ val miou: {np.mean(all_intersections / all_unions)}')
        writer.add_scalar(model_name + '/Loss/Val', running_loss/counter, epoch)
        writer.add_scalar(model_name + '/Accuracy/Val', num_correct/num_total, epoch)
        writer.add_scalar(model_name + '/mIoU/Val', np.mean(all_intersections / all_unions), epoch)
    
writer.close()