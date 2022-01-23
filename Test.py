#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import sys
from torch._C import LongStorageBase

sys.path.append("./Models")
from Models.utils import *
from Data.dataset import CarlaDataset
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
from Models.LMSCNet_SS import LMSCNet_SS
from Models.SSCNet import SSCNet


# Put parameters here
seed = 42
x_dim = 128
y_dim = 128
z_dim = 8
model_name = "MotionSC"
num_workers = 24
train_dir = "./Data/Scenes/Cartesian/Train"
val_dir = "./Data/Scenes/Cartesian/Test"
cylindrical = False
epoch_num = 500
MODEL_PATH = "./Models/Weights/MotionSC/Epoch3.pt"

VISUALIZE = False
MEASURE_INFERENCE = False
MEASURE_MIOU = True
remap = True

if remap:
    num_classes = 12
else:
    num_classes = 23

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data loaders
test_ds = CarlaDataset(directory=val_dir, device=device, num_frames=1, cylindrical=cylindrical, remap=remap)
dataloader_test = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=test_ds.collate_fn, num_workers=num_workers)

# Grid parameters
coor_ranges = test_ds._eval_param['min_bound'] + test_ds._eval_param['max_bound']
voxel_sizes = [abs(coor_ranges[3] - coor_ranges[0]) / x_dim,
              abs(coor_ranges[4] - coor_ranges[1]) / y_dim,
              abs(coor_ranges[5] - coor_ranges[2]) / z_dim] # since BEV

# Model parameters
lr = 0.001
if model_name == "MotionSC":
    B = 16
    T = 16
    model = MotionSC(voxel_sizes, coor_ranges, [x_dim, y_dim, z_dim], T=T, device=device)
    decayRate = 0.96
elif model_name == "LMSC":
    B = 4
    T = 1
    decayRate = 0.98
    model = LMSCNet_SS(num_classes, [x_dim, y_dim, z_dim], frequencies_cartesian).to(device)
elif model_name == "SSC":
    B = 4
    T = 1
    decayRate = 1.00
    model = SSCNet(num_classes).to(device)

# Data loaders
test_ds = CarlaDataset(directory=val_dir, device=device, num_frames=T, cylindrical=cylindrical)
dataloader_test = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=test_ds.collate_fn, num_workers=num_workers)


writer = SummaryWriter("./Models/Runs/" + model_name)
save_dir = "./Models/Weights/" + model_name

if device == "cuda":
    torch.cuda.empty_cache()


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
setup_seed(seed)


# Intersection, union for one frame
def iou_one_frame(pred, target, n_classes=23):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = np.zeros(n_classes)
    union = np.zeros(n_classes)

    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection[cls] = (pred_inds[target_inds]).long().sum().item()  # Cast to long to prevent overflows
        union[cls] = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection[cls]
    return intersection, union


if MODEL_PATH:
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

if VISUALIZE:
    visualize_set(model, dataloader_test, test_ds, cylindrical, min_thresh=0.75)

if MEASURE_INFERENCE:
    total_time = 0.0
    repetitions = 0
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        for input_data, output, counts in dataloader_test:
            input_data = torch.tensor(input_data).to(test_ds.device)
            starter.record()
            preds = model(input_data)
            ender.record()
            torch.cuda.synchronize()
            total_time += starter.elapsed_time(ender)
            repetitions += 1
            if repetitions >= 100:
                print(total_time / repetitions)
                break

if MEASURE_MIOU:
    all_intersections = np.zeros(num_classes)
    all_unions = np.zeros(num_classes)
    with torch.no_grad():
        for input_data, output, counts in dataloader_test:
            input_data = torch.tensor(input_data).to(device)
            output = torch.tensor(output).to(device)
            counts = torch.tensor(counts).to(device)

            # Predictions
            preds = model(input_data)
            counts = counts.view(-1)
            output = output.view(-1).long()
            preds = preds.contiguous().view(-1, preds.shape[4])
            probs = nn.functional.softmax(preds, dim=1)
            preds = torch.argmax(probs, dim=1)
            # Criterion requires input (NxC), output (N) dimension
            mask = counts > 0
            output_masked = output[mask]
            preds_masked = preds[mask]

            # I, U for a frame
            intersection, union = iou_one_frame(preds_masked, output_masked, n_classes=23)
            all_intersections += intersection
            all_unions += union

            print(all_intersections/all_unions)









