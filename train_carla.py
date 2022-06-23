#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import sys
from torch._C import LongStorageBase

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
import psutil

from torch.utils.tensorboard import SummaryWriter

from Models.MotionSC import MotionSC
from Models.SSCNet_full import SSCNet_full
from Models.LMSCNet_SS import LMSCNet_SS
from Models.SSCNet import SSCNet


# PARAMETERS
seed = 42
x_dim = 128
y_dim = 128
z_dim = 8

T = 1
binary_counts = True
transform_pose = True

model_name = "MotionSC"
num_workers = 16
train_dir = "./Data/Scenes/Cartesian/Train"
val_dir = "./Data/Scenes/Cartesian/Val"
cylindrical = False
epoch_num = 500
remap = True

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if remap:
    num_classes = 11
    frequncies_mapped = np.zeros(num_classes)
    for cls in range(23):
        frequncies_mapped[LABELS_REMAP[cls]] += frequencies_cartesian[cls]
    frequencies_cartesian = frequncies_mapped
else:
    num_classes = 23

# Weights
epsilon_w = 0.001  # eps to avoid zero division
weights = torch.from_numpy(1 / np.log(frequencies_cartesian + epsilon_w)).to(torch.float32)
criterion = nn.CrossEntropyLoss(weight=weights.to(device))

# Grid Parameters
carla_ds = CarlaDataset(directory=train_dir, device=device, num_frames=1, cylindrical=cylindrical)
coor_ranges = carla_ds._eval_param['min_bound'] + carla_ds._eval_param['max_bound']
voxel_sizes = [abs(coor_ranges[3] - coor_ranges[0]) / x_dim,
              abs(coor_ranges[4] - coor_ranges[1]) / y_dim,
              abs(coor_ranges[5] - coor_ranges[2]) / z_dim] # since BEV

# Load model
lr = 0.001
BETA1 = 0.9
BETA2 = 0.999
model, B, __, decayRate, resample_free = get_model(model_name, num_classes, voxel_sizes, coor_ranges, [x_dim, y_dim, z_dim], device, T=T)
model_name += "_" + str(num_classes) + "_T" + str(T)
if binary_counts:
    model_name += "B"
print("Running:", model_name)

# Data Loaders
carla_ds = CarlaDataset(directory=train_dir, device=device, num_frames=T, cylindrical=cylindrical, random_flips=True, remap=remap, binary_counts=binary_counts, transform_pose=transform_pose)
dataloader = DataLoader(carla_ds, batch_size=B, shuffle=True, collate_fn=carla_ds.collate_fn, num_workers=num_workers)
val_ds = CarlaDataset(directory=val_dir, device=device, num_frames=T, cylindrical=cylindrical, remap=remap, binary_counts=binary_counts, transform_pose=transform_pose)
dataloader_val = DataLoader(val_ds, batch_size=B, shuffle=True, collate_fn=val_ds.collate_fn, num_workers=num_workers)
test_ds = CarlaDataset(directory=val_dir, device=device, num_frames=T, cylindrical=cylindrical, remap=remap, binary_counts=binary_counts, transform_pose=transform_pose)
dataloader_test = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=test_ds.collate_fn, num_workers=num_workers)


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
        input_data = torch.tensor(input_data).to(device)
        output = torch.tensor(output).to(device)
        counts = torch.tensor(counts).to(device)
        preds = model(input_data)
         
        counts = counts.view(-1)
        output = output.view(-1).long()
        preds = preds.contiguous().view(-1, preds.shape[4])

        # Criterion requires input (NxC), output (N) dimension
        mask = counts > 0
        output_masked = output[mask]
        preds_masked = preds[mask]
        if resample_free:
            preds_masked, output_masked = resample_free_space(preds_masked, output_masked)

        loss = criterion(preds_masked, output_masked)
        loss.backward()
        optimizer.step()
        
        # Accuracy
        with torch.no_grad():
            probs = nn.functional.softmax(preds_masked, dim=1)
            preds_masked = np.argmax(probs.detach().cpu().numpy(), axis=1)
            outputs_np = output_masked.detach().cpu().numpy()
            accuracy = np.sum(preds_masked == outputs_np) / outputs_np.shape[0]
            
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
            input_data = torch.tensor(input_data).to(device)
            output = torch.tensor(output).to(device)
            counts = torch.tensor(counts).to(device)
            preds = model(input_data)

            counts = counts.view(-1)
            output = output.view(-1).long()
            preds = preds.contiguous().view(-1, preds.shape[4])

            # Criterion requires input (NxC), output (N) dimension
            mask = counts > 0
            output_masked = output[mask]
            preds_masked = preds[mask]
            loss = criterion(preds_masked, output_masked)

            running_loss += loss.item()
            counter += input_data.shape[0]

            # Accuracy
            probs = nn.functional.softmax(preds_masked, dim=1)
            preds_masked = np.argmax(probs.detach().cpu().numpy(), axis=1)
            outputs_np = output_masked.detach().cpu().numpy()
            num_correct += np.sum(preds_masked == outputs_np)
            num_total += outputs_np.shape[0]

            intersection, union = iou_one_frame(torch.tensor(preds_masked), torch.tensor(output_masked), n_classes=num_classes)
            all_intersections += intersection
            all_unions += union
        
        print(f'Eppoch Num: {epoch} ------ average val loss: {running_loss/counter}')
        print(f'Eppoch Num: {epoch} ------ average val accuracy: {num_correct/num_total}')
        print(f'Eppoch Num: {epoch} ------ val miou: {np.mean(all_intersections / all_unions)}')
        writer.add_scalar(model_name + '/Loss/Val', running_loss/counter, epoch)
        writer.add_scalar(model_name + '/Accuracy/Val', num_correct/num_total, epoch)
        writer.add_scalar(model_name + '/mIoU/Val', np.mean(all_intersections / all_unions), epoch)
    
writer.close()
