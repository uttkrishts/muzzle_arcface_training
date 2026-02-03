import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MuzzleEmbeddingNet(nn.Module):
    """
    CNN embedding network for cattle muzzle biometrics.
    Outputs L2-normalized embeddings suitable for metric learning.
    """

    def __init__(self, embedding_dim=256, pretrained=True):
        super().__init__()

        # Load ResNet-50 backbone
        backbone = models.resnet50(pretrained=pretrained)

        # Remove the classifier head
        self.backbone = nn.Sequential(
            *list(backbone.children())[:-1]  # removes fc layer
        )

        # Projection head
        self.embedding = nn.Sequential(
            nn.Linear(2048, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x):
        """
        Input:  x -> (B, 3, 224, 224)
        Output: embeddings -> (B, embedding_dim), L2 normalized
        """

        # Feature extraction
        x = self.backbone(x)              # (B, 2048, 1, 1)
        x = x.view(x.size(0), -1)         # (B, 2048)

        # Projection
        x = self.embedding(x)             # (B, embedding_dim)

        # L2 normalization (CRITICAL)
        x = F.normalize(x, p=2, dim=1)

        return x
