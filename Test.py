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
num_classes = 23
train_dir = "./Data/Scenes/Cartesian/Train"
val_dir = "./Data/Scenes/Cartesian/Val"
cylindrical = False
epoch_num = 500
MODEL_PATH = "./Models/Weights/MotionSC/Epoch4.pt"
VISUALIZE = True

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Weights
epsilon_w = 0.001  # eps to avoid zero division
weights = torch.from_numpy(1 / np.log(ratios_cartesian + epsilon_w))
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
    model = LMSCNet_SS(num_classes, [x_dim, y_dim, z_dim], ratios_cartesian).to(device)
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


if MODEL_PATH:
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
if VISUALIZE:
    visualize_set(model, dataloader_test, test_ds, cylindrical, min_thresh=0.9)






