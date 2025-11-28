"""Abstract base class for CNN backbones."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseBackbone(ABC, nn.Module):
    """
    Base class for CNN feature extraction backbones.

    Subclasses must define `feature_dim` and implement `forward()`.
    """

    feature_dim: int  # Output feature dimension

    def __init__(self):
        super().__init__()
        self._frozen = False

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Feature tensor of shape (batch, feature_dim)
        """
        pass

    def freeze(self) -> 'BaseBackbone':
        """Freeze all parameters (disable gradient computation)."""
        for param in self.parameters():
            param.requires_grad = False
        self._frozen = True
        return self

    def unfreeze(self) -> 'BaseBackbone':
        """Unfreeze all parameters (enable gradient computation)."""
        for param in self.parameters():
            param.requires_grad = True
        self._frozen = False
        return self

    @property
    def is_frozen(self) -> bool:
        """Check if backbone is frozen."""
        return self._frozen


class GlobalPool(nn.Module):
    """Global pooling layer for feature extraction."""

    def __init__(self, pool_type: str = "avg"):
        """
        Initialize global pooling.

        Args:
            pool_type: 'avg' for average pooling, 'max' for max pooling
        """
        super().__init__()
        if pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pool_type == "max":
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise ValueError(f"Unknown pool type: {pool_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return torch.flatten(x, 1)
