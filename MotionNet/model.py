# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved. The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages. As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.

import torch.nn.functional as F
import torch.nn as nn
import torch


class CellClassification(nn.Module):
    def __init__(self, category_num=5):
        super(CellClassification, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, category_num, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x


class StateEstimation(nn.Module):
    def __init__(self, motion_category_num=2):
        super(StateEstimation, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, motion_category_num, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x


class MotionPrediction(nn.Module):
    def __init__(self, seq_len):
        super(MotionPrediction, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 2 * seq_len, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x


class Conv3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(Conv3D, self).__init__()
        self.conv3d = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn3d = nn.BatchNorm3d(out_channel)

    def forward(self, x):
        # input x: (batch, seq, c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, c, seq_len, h, w)
        x = F.relu(self.bn3d(self.conv3d(x)))
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, seq_len, c, h, w)

        return x


class STPN(nn.Module):
    def __init__(self, height_feat_size=13):
        super(STPN, self).__init__()
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        self.conv3d_1 = Conv3D(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3d_2 = Conv3D(128, 128, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv6_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        self.bn4_1 = nn.BatchNorm2d(512)
        self.bn4_2 = nn.BatchNorm2d(512)

        self.bn5_1 = nn.BatchNorm2d(256)
        self.bn5_2 = nn.BatchNorm2d(256)

        self.bn6_1 = nn.BatchNorm2d(128)
        self.bn6_2 = nn.BatchNorm2d(128)

        self.bn7_1 = nn.BatchNorm2d(64)
        self.bn7_2 = nn.BatchNorm2d(64)

        self.bn8_1 = nn.BatchNorm2d(32)
        self.bn8_2 = nn.BatchNorm2d(32)

    def forward(self, x):
        batch, seq, z, h, w = x.size()

        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x)))
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))


        # -------------------------------- Encoder Path --------------------------------
        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_1 = self.conv3d_1(x_1)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w)

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_2 = self.conv3d_2(x_2)
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h, w), seq = 1

        # -- STC block 3
        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))
        x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))

        # -- STC block 4
        x_4 = F.relu(self.bn4_1(self.conv4_1(x_3)))
        x_4 = F.relu(self.bn4_2(self.conv4_2(x_4)))


        # -------------------------------- Decoder Path --------------------------------
        x_5 = F.relu(self.bn5_1(self.conv5_1(torch.cat((F.interpolate(x_4, scale_factor=(2, 2)), x_3), dim=1))))
        x_5 = F.relu(self.bn5_2(self.conv5_2(x_5)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = F.adaptive_max_pool3d(x_2, (1, None, None))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()

        x_6 = F.relu(self.bn6_1(self.conv6_1(torch.cat((F.interpolate(x_5, scale_factor=(2, 2)), x_2), dim=1))))
        x_6 = F.relu(self.bn6_2(self.conv6_2(x_6)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = F.adaptive_max_pool3d(x_1, (1, None, None))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()

        x_7 = F.relu(self.bn7_1(self.conv7_1(torch.cat((F.interpolate(x_6, scale_factor=(2, 2)), x_1), dim=1))))
        x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))

        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = F.adaptive_max_pool3d(x, (1, None, None))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous()

        x_8 = F.relu(self.bn8_1(self.conv8_1(torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x), dim=1))))
        res_x = F.relu(self.bn8_2(self.conv8_2(x_8)))

        return res_x


class MotionNet(nn.Module):
    def __init__(self, out_seq_len=20, motion_category_num=2, height_feat_size=13):
        super(MotionNet, self).__init__()
        self.out_seq_len = out_seq_len

        self.cell_classify = CellClassification()
        self.motion_pred = MotionPrediction(seq_len=self.out_seq_len)
        self.state_classify = StateEstimation(motion_category_num=motion_category_num)
        self.stpn = STPN(height_feat_size=height_feat_size)


    def forward(self, bevs):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)

        # Backbone network
        x = self.stpn(bevs)

        # Cell Classification head
        cell_class_pred = self.cell_classify(x)

        # Motion State Classification head
        state_class_pred = self.state_classify(x)

        # Motion Displacement prediction
        disp = self.motion_pred(x)
        disp = disp.view(-1, 2, x.size(-2), x.size(-1))

        return disp, cell_class_pred, state_class_pred


# For MGDA loss computation
class FeatEncoder(nn.Module):
    def __init__(self, height_feat_size=13):
        super(FeatEncoder, self).__init__()
        self.stpn = STPN(height_feat_size=height_feat_size)

    def forward(self, bevs):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)
        x = self.stpn(bevs)

        return x


class MotionNetMGDA(nn.Module):
    def __init__(self, out_seq_len=20, motion_category_num=2):
        super(MotionNetMGDA, self).__init__()
        self.out_seq_len = out_seq_len

        self.cell_classify = CellClassification()
        self.motion_pred = MotionPrediction(seq_len=self.out_seq_len)
        self.state_classify = StateEstimation(motion_category_num=motion_category_num)

    def forward(self, stpn_out):
        # Cell Classification head
        cell_class_pred = self.cell_classify(stpn_out)

        # Motion State Classification head
        state_class_pred = self.state_classify(stpn_out)

        # Motion Displacement prediction
        disp = self.motion_pred(stpn_out)
        disp = disp.view(-1, 2, stpn_out.size(-2), stpn_out.size(-1))

        return disp, cell_class_pred, state_class_pred
