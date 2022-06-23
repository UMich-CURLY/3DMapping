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

from ptflops import get_model_complexity_info

from torch.utils.tensorboard import SummaryWriter


# Put parameters here
seed = 42
x_dim = 128
y_dim = 128
z_dim = 8
model_name = "MotionSC"
num_workers = 16

T = 1
binary_counts = True

train_dir = "./Data/Scenes/Cartesian/Train"
val_dir = "./Data/Scenes/Cartesian/Test"
cylindrical = False
MODEL_PATH = "./Models/Weights/MotionSC_11_T1B/Epoch10.pt"
remap = True

print(MODEL_PATH)

# Which task to perform
VISUALIZE = False
MEASURE_INFERENCE = True
MEASURE_MIOU = True
MEASURE_ACCURACY = True
SAVE_PREDS = False
MEASURE_GEOMETRY = True
MEASURE_SIZE = True

if remap:
    num_classes = 11
else:
    num_classes = 23

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data loaders
test_ds = CarlaDataset(directory=val_dir, device=device, num_frames=T, cylindrical=cylindrical, remap=remap, binary_counts=binary_counts)
dataloader_test = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=test_ds.collate_fn, num_workers=num_workers)

# Grid parameters
coor_ranges = test_ds._eval_param['min_bound'] + test_ds._eval_param['max_bound']
voxel_sizes = [abs(coor_ranges[3] - coor_ranges[0]) / x_dim,
              abs(coor_ranges[4] - coor_ranges[1]) / y_dim,
              abs(coor_ranges[5] - coor_ranges[2]) / z_dim] # since BEV

# Load model
model, B, T, decayRate, resample_free = get_model(model_name, num_classes, voxel_sizes, coor_ranges, [x_dim, y_dim, z_dim], device, T=T)
model_name += "_" + str(num_classes)
print("Running:", model_name)

# Data loaders
test_ds = CarlaDataset(directory=val_dir, device=device, num_frames=T, cylindrical=cylindrical, remap=remap, binary_counts=binary_counts)
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
    pred = pred.view(-1).detach().cpu().numpy()
    target = target.view(-1).detach().cpu().numpy()
    intersection = np.zeros(n_classes)
    union = np.zeros(n_classes)

    for cls in range(n_classes):
        intersection[cls] = np.sum((pred == cls) & (target == cls))
        union[cls] = np.sum((pred == cls) | (target == cls))
    return intersection, union


def count_parameters(model, T):
    macs, params = get_model_complexity_info(model, (T, 128, 128, 8), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    model_size = sum(p.numel() for p in model.parameters())
    print("Parameters:", model_size)


if MODEL_PATH:
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

if VISUALIZE:
    visualize_set(model, dataloader_test, test_ds, cylindrical, min_thresh=0.75)

if MEASURE_SIZE:
    count_parameters(model, T)

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
                print("Inference Time ms:", total_time / repetitions)
                break

if MEASURE_MIOU:
    all_intersections = np.zeros(num_classes)
    all_unions = np.zeros(num_classes) + 1e-6
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
            intersection, union = iou_one_frame(preds_masked, output_masked, n_classes=num_classes)
            all_intersections += intersection
            all_unions += union

    print(model_name)
    iou = all_intersections/all_unions
    print("IoU")
    for i in range(num_classes):
        print(100 * iou[i])

if MEASURE_ACCURACY:
    num_correct = 0.0
    num_total = 0.0
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
            num_correct += np.sum(output_masked.cpu().numpy() == preds_masked.cpu().numpy())
            num_total += preds_masked.shape[0]

    print("Accuracy", num_correct/num_total)


if MEASURE_GEOMETRY:
    TP = 0.0
    FP = 0.0
    TN = 0.0
    FN = 0.0
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
            output_masked = output[mask].cpu().numpy()
            output_masked[output_masked != 0] = 1
            preds_masked = preds[mask].cpu().numpy()
            preds_masked[preds_masked != 0] = 1

            TP += np.sum((preds_masked == 1) & (output_masked == 1))
            FP += np.sum((preds_masked == 1) & (output_masked == 0))
            TN += np.sum((preds_masked == 0) & (output_masked == 0))
            FN += np.sum((preds_masked == 0) & (output_masked == 1))

    print("Completeness")
    print("precision:", 100 * TP / (TP + FP))
    print("recall:", 100 * TP / (TP + FN))
    print("iou:", 100 * TP / (TP + FP + FN))


if SAVE_PREDS:
    with torch.no_grad():
        for idx in range(test_ds.__len__()):
            # File path
            point_path = test_ds._velodyne_list[idx]
            paths = point_path.split("/")
            save_dir = os.path.join(*paths[:-2], model_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            fpath = os.path.join(save_dir, paths[-1].split(".")[0] + ".label")
            # Input data
            current_horizon, output, counts = test_ds.__getitem__(idx)
            input_data = torch.tensor(current_horizon).to(device)
            input_data = torch.unsqueeze(input_data, 0)
            # Label predictions
            preds = model(input_data)
            preds = nn.functional.softmax(preds, dim=4)
            preds = torch.argmax(preds, dim=4)
            preds = torch.squeeze(preds, dim=0)
            # Save data
            preds = preds.detach().cpu().numpy()
            preds.astype('uint32').tofile(fpath)











