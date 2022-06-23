from cv2 import split
import torch
import sys
# from torch._C import LongStorageBase
# torch.cuda.set_device(1) 
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



# TODO: you may change these parameters if needed
# PARAMETERS
seed = 42
x_dim = 256
y_dim = 256
z_dim = 32
model_name = "MotionSC"
num_workers = 0
train_dir = "./Data/kitti"
val_dir = "./Data/kitti"
cylindrical = False
epoch_num = 100
remap = True
num_classes = 20

T = 1
binary_counts = True
transform_pose = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# weights = get_class_weights(class_frequencies).to(torch.float32)
# criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=255, reduction='mean').to(device=device)
# criterion = nn.CrossEntropyLoss(weight=weights.to(device))

coor_ranges = [0,-25.6,-2] + [51.2,25.6,4.4]
voxel_sizes = [abs(coor_ranges[3] - coor_ranges[0]) / x_dim,
              abs(coor_ranges[4] - coor_ranges[1]) / y_dim,
              abs(coor_ranges[5] - coor_ranges[2]) / z_dim] # since BEV


lr = 0.001
BETA1 = 0.9
BETA2 = 0.999
model, __, __, decayRate, resample_free = get_model(model_name, num_classes, voxel_sizes, coor_ranges, [x_dim, y_dim, z_dim], device, T=T)
B = 1
model_name += "_" + str(num_classes)
print("Running:", model_name)


# Data Loaders
carla_ds = KittiDataset(directory=train_dir, device=device, num_frames=T, random_flips=True, remap=remap, split='train', binary_counts=binary_counts, transform_pose=transform_pose)
dataloader = DataLoader(carla_ds, batch_size=B, shuffle=True, collate_fn=carla_ds.collate_fn, num_workers=num_workers)
val_ds = KittiDataset(directory=val_dir, device=device, num_frames=T, remap=remap, split='valid', binary_counts=binary_counts, transform_pose=transform_pose)
dataloader_val = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=val_ds.collate_fn, num_workers=num_workers)

inv_lut_table = val_ds.get_inv_remap_lut()

MODEL_PATH = './Models/Weights/MotionSC_20_KITTI__T1B/Epoch22.pt'
model.load_state_dict(torch.load(MODEL_PATH))

model.eval()

# count = 0
# with torch.no_grad():
#     for input_data, output, counts in dataloader_val:
#         input_data = torch.tensor(input_data).to(device)
#         preds = model(input_data)
#         preds = preds.contiguous().view(-1, preds.shape[4])
#         probs = nn.functional.softmax(preds, dim=1)
#         preds = torch.argmax(probs.detach(), dim=1).cpu().numpy()
#         preds = inv_lut_table[preds].astype(np.uint16)
#         filename = str(count).zfill(6)
#         preds.tofile(f'./preds/{filename}.label')
#         print(f'Predict No. {count}')
#         count += 1

test_ds = KittiDataset(directory=val_dir, device=device, num_frames=T, remap=remap, split='test', binary_counts=binary_counts, transform_pose=transform_pose, get_gt=False)
dataloader_test = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers)

check_path = './Data/kitti'
with torch.no_grad():
    for input_data, file_name in dataloader_test:
        file_name = file_name[0]
        sequence = os.path.dirname(file_name).split('/')[-2]
        frame, extension = os.path.splitext(os.path.basename(file_name))
        if int(frame) % 5 != 0:
            continue
        if os.path.isfile(os.path.join(check_path, 'sequences', sequence, 'voxels',  f'{frame}.bin')) == False:
            print(os.path.join(check_path, 'sequences', sequence, 'voxels',  f'{frame}.bin'))
            continue
        input_data = torch.tensor(input_data).to(device)
        preds = model(input_data)
        preds = preds.contiguous().view(-1, preds.shape[4])
        probs = nn.functional.softmax(preds, dim=1)
        output_np = torch.argmax(probs.detach(), dim=1).cpu().numpy()
        output_remapped = inv_lut_table[output_np].astype(np.uint16)
        # print(os.path.basename(file_name))
        
        
        out_filename = os.path.join('./prediction_T1', 'sequences', sequence, 'predictions', frame + '.label')
        if not os.path.exists(os.path.dirname(out_filename)):
            os.makedirs(os.path.dirname(out_filename))
        output_remapped.tofile(out_filename)
        # print(file_name)
