import torch.nn.functional as F
import torch.nn as nn
import torch
import math

from PointPillar import *
from spconv.pytorch.utils import PointToVoxel
from STPN import *


'''
    Converts Nx3+ points from (x, y, z) to (r, t, z)
'''
def euc_to_cyl(points):
    cyl_points = torch.clone(points)
    cyl_points[:, 0] = torch.norm(points[:, :2], dim=1)
    cyl_points[:, 1] = torch.atan2(points[:, 1], points[:, 0]) + math.pi
    return cyl_points


class FiniteHorizon(nn.Module):
    def __init__(self, voxel_sizes, coor_ranges, point_dim=30, grid_dims=(96, 192, 10),
                 z_latent_dim=20, max_num_voxels=10000, out_dim=26, device="cuda",
                 max_num_points_per_voxel=5, T=5, num_filters=(64, 128), hidden_dim=100):
        super(FiniteHorizon, self).__init__()
        self.T = T
        self.out_dim = out_dim
        self.r_dim, self.t_dim, self.z_dim = grid_dims
        self.z_latent_dim = z_latent_dim
        self.enc_dim = num_filters[-1]
        self.device = device

        self.stpn = STPN(height_feat_size=self.enc_dim, cell_feat_size=int(self.z_latent_dim * self.z_dim)).to(device)
        
        # Feed points to gen to voxelize points
        self.gen = PointToVoxel(vsize_xyz=voxel_sizes,
                           coors_range_xyz=coor_ranges,
                           num_point_features=point_dim,
                           max_num_voxels=max_num_voxels,
                           max_num_points_per_voxel=max_num_points_per_voxel,
                           device=device)

        # Learn pillars from voxelized points
        self.encoder = PillarFeatureNet(num_input_features=point_dim, num_filters=num_filters, with_distance=True, 
                                   voxel_size=voxel_sizes, pc_range=coor_ranges).to(device)

        self.segmentation_head = nn.Sequential(
            nn.Conv1d(z_latent_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, out_dim, 1)
        ).to(device)


    def forward(self, point_lists):
        B = len(point_lists)
        bevs = torch.zeros(B, self.T, self.r_dim, self.t_dim, self.enc_dim, device=self.device)
        for b_i in range(B):
            for t_i in range(self.T):
                cyl_pc = euc_to_cyl(point_lists[b_i][t_i])
                vox_feats, coords, num_voxels = self.gen(cyl_pc)
                bevs[b_i, t_i, coords[:, 0].long(), coords[:, 1].long(), :] = self.encoder(vox_feats, num_voxels, coords)
                
        # Spatio temporal network
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)
        latent_stack = self.stpn(bevs) # B, C, H, W
        
        # Flatten z_dim, H(r), W(theta) 
        latent_cells = latent_stack.view(B, self.z_latent_dim, -1) 
        
        # Semantic Predictions
        preds = self.segmentation_head(latent_cells).view(B, self.out_dim, self.z_dim, self.r_dim, self.t_dim)

        return preds.permute(0, 3, 4, 2, 1)

