#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys

sys.path.append("./Models")
from Models.utils import *
from Data.dataset import CarlaDataset
from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter


# Put parameters here
seed = 42
x_dim = 128
y_dim = 128
z_dim = 8
model_name = "SSC_Full"
num_workers = 16
B = 1
T = 1
binary_counts = True
transform_pose = True

train_dir = "./Data/Scenes/Cartesian/Train"
val_dir = "./Data/Scenes/Cartesian/Test"
cylindrical = False
MODEL_PATH = "./Models/Weights/SSC_Full_11_T1B/Epoch19.pt"
remap = True

print(MODEL_PATH)

# Which task to perform
VISUALIZE = False
MEASURE_INFERENCE = True
MEASURE_MIOU = True
MEASURE_ACCURACY = True
SAVE_PREDS = True
MEASURE_GEOMETRY = True
MEASURE_SIZE = True

if remap:
    num_classes = 11
else:
    num_classes = 23

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data loaders
test_ds = CarlaDataset(directory=val_dir, device=device, num_frames=T, cylindrical=cylindrical, remap=remap, binary_counts=binary_counts, transform_pose=transform_pose)
dataloader_test = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=test_ds.collate_fn, num_workers=num_workers)

# Grid parameters
coor_ranges = test_ds._eval_param['min_bound'] + test_ds._eval_param['max_bound']
voxel_sizes = [abs(coor_ranges[3] - coor_ranges[0]) / x_dim,
              abs(coor_ranges[4] - coor_ranges[1]) / y_dim,
              abs(coor_ranges[5] - coor_ranges[2]) / z_dim] # since BEV

# Load model
model, __, __, decayRate, resample_free = get_model(model_name, num_classes, voxel_sizes, coor_ranges, [x_dim, y_dim, z_dim], device, T=T)
model_name += "_" + str(num_classes) + "_T" + str(T)
print("Running:", model_name)

# Data loaders
test_ds = CarlaDataset(directory=val_dir, device=device, num_frames=T, cylindrical=cylindrical, remap=remap, binary_counts=binary_counts, transform_pose=transform_pose)
dataloader_test = DataLoader(test_ds, batch_size=B, shuffle=False, collate_fn=test_ds.collate_fn, num_workers=num_workers)


writer = SummaryWriter("./Models/Runs/" + model_name)
save_dir = "./Models/Weights/" + model_name

if device == "cuda":
    torch.cuda.empty_cache()

setup_seed(seed)

if MODEL_PATH:
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

if VISUALIZE:
    visualize_set(model, dataloader_test, test_ds, cylindrical, min_thresh=0.75)

if MEASURE_SIZE:
    count_parameters(model, T)

if MEASURE_INFERENCE:
    avg_inf_ms = measure_inference_time(dataloader_test, model, device)

if MEASURE_MIOU:
    semantic_iou = measure_miou(dataloader_test, model, device, num_classes)

if MEASURE_ACCURACY:
    accuracy = measure_geom(dataloader_test, model, device)

if MEASURE_GEOMETRY:
    precision, recall, iou = measure_geom(dataloader_test, model, device)

if SAVE_PREDS:
    save_preds(test_ds, model, model_name, device)
