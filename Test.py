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
model_name = "LMSC"
num_workers = 8
num_classes = 23
coordinate_type = "Cartesian"
cylindrical = coordinate_type == "Cylindrical"
epoch_num = 500
MODEL_PATH = "./Models/Weights/LMSC/Epoch5.pt"
VISUALIZE = True

# Data directories
train_dir = "./Data/Scenes/{ctype}/Train".format(ctype=coordinate_type)
val_dir = "./Data/Scenes/{ctype}/Val".format(ctype=coordinate_type)
test_dir = "./Data/Scenes/{ctype}/Test".format(ctype=coordinate_type)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Weights
epsilon_w = 0.001  # eps to avoid zero division
weights = torch.from_numpy(1 / np.log(frequencies_cartesian + epsilon_w))
criterion = nn.CrossEntropyLoss(weight=weights.to(device))

# Data loaders
test_ds = CarlaDataset(directory=val_dir, device=device, num_frames=1, cylindrical=cylindrical)
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
test_ds = CarlaDataset(directory=test_dir, device=device, num_frames=T, cylindrical=cylindrical)
dataloader_test = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=test_ds.collate_fn, num_workers=num_workers)

writer = SummaryWriter("./Models/Runs/" + model_name)
weights_dir = "./Models/Weights/" + model_name
save_dir = os.path.join("./Models/Predictions/", model_name, coordinate_type)

if not os.path.exists(weights_dir):
    print("Weights directory to load form does not exist, exiting...")
    sys.exit()
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if device == "cuda":
    torch.cuda.empty_cache()


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
setup_seed(seed)


if MODEL_PATH:
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    num_correct = 0
    num_total = 0
    # Make predictions
    with torch.no_grad():
        scene_idx = 0
        for input_data, output, counts in dataloader_test:
            input_data = torch.tensor(input_data).to(device)
            output = torch.tensor(output).to(device)
            counts = torch.tensor(counts).to(device)
            preds = model(input_data)

            # Save predictions to file
            preds_np = preds.detach().cpu().numpy()
            labels = np.argmax(preds_np, axis=3)
            frame_str = test_ds.get_item_scene_name(scene_idx)
            labels.astype('uint32').tofile(os.path.join(save_dir, frame_str + ".label"))

            counts = counts.view(-1)
            output = output.view(-1).long()
            preds = preds.contiguous().view(-1, preds.shape[4])

            # Criterion requires input (NxC), output (N) dimension
            mask = counts > 0
            output_masked = output[mask]
            preds_masked = preds[mask]

            # loss = criterion(preds_masked, output_masked)

            # Accuracy
            probs = nn.functional.softmax(preds_masked, dim=1)
            preds_masked = np.argmax(probs.detach().cpu().numpy(), axis=1)
            outputs_np = output_masked.detach().cpu().numpy()
            num_correct += np.sum(preds_masked == outputs_np)
            num_total += outputs_np.shape[0]

            writer.add_scalar(model_name + '/Accuracy/Test', num_correct/num_total, scene_idx)

            # Increment scene
            scene_idx += 1

    print("Test Accuracy: ", num_correct / num_total)

if VISUALIZE:
    visualize_set(model, dataloader_test, test_ds, cylindrical, min_thresh=0.9)






