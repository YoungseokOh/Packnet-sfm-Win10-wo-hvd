# Copyright 2020 Toyota Research Institute.  All rights reserved.

# Adapted from monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/layers.py

from __future__ import absolute_import, division, print_function

import torch.nn as nn
import torch.nn.functional as F
import torch


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        # ori
        # self.conv = Conv3x3(in_channels, out_channels)
        # revise
        self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 3)
        # ReLU
        self.nonlin = nn.ReLU(inplace=True)
        # ELU
        # self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()
        # use_refl
        # if use_refl:
        #     self.pad = nn.ReflectionPad2d(1)
        # else:
        # zero padding only
        self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


class fSEModule(nn.Module):
    def __init__(self, high_feature_channel, low_feature_channels, output_channel=None):
        super(fSEModule, self).__init__()
        in_channel = high_feature_channel + low_feature_channels
        out_channel = high_feature_channel
        if output_channel is not None:
            out_channel = output_channel
        reduction = 16
        channel = in_channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Original fully connected layer
        # self.fc = nn.Sequential(
        #     nn.Linear(channel, channel // reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(channel // reduction, channel, bias=False)
        # )

        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1)
        )

        self.sigmoid = nn.Sigmoid()
        self.conv_se = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        # self.conv1x1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1)

    def forward(self, high_features, low_features):
        features = [upsample(high_features)]
        features += low_features
        features = torch.cat(features, 1)
        # New code
        y = self.avg_pool(features)
        y = self.fc(y)
        y = self.sigmoid(y)
        # Not support expand_as
        features = features * y

        # Original code
        # b, c, _, _ = features.size()
        # y = self.avg_pool(features).view(b, c)

        # y = self.fc(y).view(b, c, 1, 1)
        # y = self.sigmoid(y)
        # # Not support expand_as
        # features = features * y
        # Original code
        # features = features * y.expand_as(features)

        return self.relu(self.conv_se(features))