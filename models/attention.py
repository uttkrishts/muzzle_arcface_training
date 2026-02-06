"""
CBAM: Convolutional Block Attention Module
Paper: https://arxiv.org/abs/1807.06521

Combines Channel Attention and Spatial Attention to help the model
focus on "what" (channels) and "where" (spatial locations) are important.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention: "What" to focus on.
    Uses both MaxPool and AvgPool for richer channel statistics.
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # Avg-pool branch
        avg_out = self.fc(self.avg_pool(x).view(b, c))

        # Max-pool branch
        max_out = self.fc(self.max_pool(x).view(b, c))

        # Combine and apply sigmoid
        out = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out


class SpatialAttention(nn.Module):
    """
    Spatial Attention: "Where" to focus on.
    Uses channel-wise max and avg pooling, then a 7x7 conv.
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate and apply conv
        combined = torch.cat([avg_out, max_out], dim=1)
        out = torch.sigmoid(self.conv(combined))
        return x * out


class CBAM(nn.Module):
    """
    CBAM: Channel Attention + Spatial Attention (Sequential).
    """

    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
