"""Backbone implementations for image feature extraction."""

import torch
import torch.nn as nn
from torchvision import models

from ..registry import Registry
from .base import BaseBackbone, GlobalPool


class BackboneRegistry(Registry):
    """Registry for CNN backbones."""
    _registry = {}


@BackboneRegistry.register("resnet50")
class ResNet50Backbone(BaseBackbone):
    """ResNet-50 backbone for feature extraction."""

    feature_dim = 2048

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        net = models.resnet50(weights=weights)
        # Remove final FC layer, keep everything up to avgpool
        self.model = nn.Sequential(*list(net.children())[:-1])
        self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return torch.flatten(x, 1)


@BackboneRegistry.register("resnet101")
class ResNet101Backbone(BaseBackbone):
    """ResNet-101 backbone for feature extraction."""

    feature_dim = 2048

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet101_Weights.DEFAULT if pretrained else None
        net = models.resnet101(weights=weights)
        self.model = nn.Sequential(*list(net.children())[:-1])
        self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return torch.flatten(x, 1)


@BackboneRegistry.register("resnet18")
class ResNet18Backbone(BaseBackbone):
    """ResNet-18 backbone (smaller, faster)."""

    feature_dim = 512

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        self.model = nn.Sequential(*list(net.children())[:-1])
        self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return torch.flatten(x, 1)


@BackboneRegistry.register("vgg16")
class VGG16Backbone(BaseBackbone):
    """VGG-16 backbone for feature extraction."""

    feature_dim = 512

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        net = models.vgg16(weights=weights)
        self.model = nn.Sequential(
            *list(net.features.children()),
            GlobalPool("avg")
        )
        self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@BackboneRegistry.register("vgg19")
class VGG19Backbone(BaseBackbone):
    """VGG-19 backbone for feature extraction."""

    feature_dim = 512

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.VGG19_Weights.DEFAULT if pretrained else None
        net = models.vgg19(weights=weights)
        self.model = nn.Sequential(
            *list(net.features.children()),
            GlobalPool("avg")
        )
        self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@BackboneRegistry.register("densenet121")
class DenseNet121Backbone(BaseBackbone):
    """DenseNet-121 backbone for feature extraction."""

    feature_dim = 1024

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        net = models.densenet121(weights=weights)
        self.features = net.features
        self.pool = GlobalPool("avg")
        self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.relu(x)
        x = self.pool(x)
        return x


@BackboneRegistry.register("efficientnet_b0")
class EfficientNetB0Backbone(BaseBackbone):
    """EfficientNet-B0 backbone for feature extraction."""

    feature_dim = 1280

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        self.features = net.features
        self.pool = GlobalPool("avg")
        self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return x


@BackboneRegistry.register("efficientnet_b4")
class EfficientNetB4Backbone(BaseBackbone):
    """EfficientNet-B4 backbone for feature extraction."""

    feature_dim = 1792

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        net = models.efficientnet_b4(weights=weights)
        self.features = net.features
        self.pool = GlobalPool("avg")
        self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return x


@BackboneRegistry.register("convnext_tiny")
class ConvNextTinyBackbone(BaseBackbone):
    """ConvNeXt-Tiny backbone for feature extraction."""

    feature_dim = 768

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        net = models.convnext_tiny(weights=weights)
        self.features = net.features
        self.pool = GlobalPool("avg")
        self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return x


@BackboneRegistry.register("mobilenet_v3")
class MobileNetV3Backbone(BaseBackbone):
    """MobileNet-V3 Large backbone (lightweight, fast)."""

    feature_dim = 960

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        net = models.mobilenet_v3_large(weights=weights)
        self.features = net.features
        self.pool = GlobalPool("avg")
        self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return x


def get_backbone(name: str, **kwargs) -> BaseBackbone:
    """
    Factory function to get a backbone by name.

    Args:
        name: Backbone name
        **kwargs: Additional arguments for the backbone

    Returns:
        Backbone instance
    """
    return BackboneRegistry.create(name, **kwargs)
