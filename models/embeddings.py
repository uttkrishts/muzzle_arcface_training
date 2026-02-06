import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from models.attention import CBAM


class MuzzleEmbeddingNet(nn.Module):
    """
    CNN embedding network for cattle muzzle biometrics.
    Outputs L2-normalized embeddings suitable for metric learning.

    Now includes CBAM (Channel + Spatial Attention) for improved focus
    on discriminative muzzle regions.
    """

    def __init__(self, embedding_dim=256, pretrained=True, use_attention=True):
        super().__init__()
        self.use_attention = use_attention

        # Load ResNet-50 backbone
        backbone = models.resnet50(pretrained=pretrained)

        # Extract layers before global pooling
        # We need to insert attention BEFORE the final avgpool
        self.conv_layers = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,  # Output: (B, 2048, H, W)
        )

        # CBAM Attention (applied to 2048-channel feature maps)
        if use_attention:
            self.attention = CBAM(in_channels=2048, reduction=16)
        else:
            self.attention = nn.Identity()

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Projection head
        self.embedding = nn.Sequential(
            nn.Linear(2048, embedding_dim), nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x):
        """
        Input:  x -> (B, 3, H, W)
        Output: embeddings -> (B, embedding_dim), L2 normalized
        """

        # Feature extraction
        x = self.conv_layers(x)  # (B, 2048, H/32, W/32)

        # Attention (CBAM)
        x = self.attention(x)  # (B, 2048, H/32, W/32) - weighted

        # Global pooling
        x = self.global_pool(x)  # (B, 2048, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 2048)

        # Projection
        x = self.embedding(x)  # (B, embedding_dim)

        # L2 normalization (CRITICAL)
        x = F.normalize(x, p=2, dim=1)

        return x
