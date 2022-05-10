import torch.nn.functional as F
import torch.nn as nn
import torch
from STPN import *
from LMSCNet.LMSCNet.models.LMSCNet import SegmentationHead


class MotionSC(nn.Module):
    def __init__(self, voxel_sizes, coor_ranges, grid_dims, device="cuda", T=8, binary=False, num_classes=23):
        super().__init__()
        self.device = device
        self.T = T
        self.min_bound = torch.tensor(coor_ranges[:3], device=device)
        self.max_bound = torch.tensor(coor_ranges[3:], device=device)
        self.voxel_sizes = torch.tensor(voxel_sizes, device=device)
        self.coor_ranges = coor_ranges
        self.grid_dims = grid_dims
        self.segmentation_head = SegmentationHead(1, 8, num_classes, [1, 2, 3]).to(device)
        self.stpn = STPN(height_feat_size=self.grid_dims[2], cell_feat_size=grid_dims[2], T=T).to(device)
        self.binary = binary

    def forward(self, x_in):
        # After grid is constructed, pass through model
        x = torch.permute(x_in, (0, 1, 4, 2, 3))
        x = self.stpn(x) # Output is B x Z x H x W, Input is B x T x Z x H x W
                
        x_out = self.segmentation_head(x) # Output is B x C x Z x H x W
        
        return x_out.permute(0, 3, 4, 2, 1)

    def weights_initializer(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def weights_init(self):
        self.apply(self.weights_initializer)