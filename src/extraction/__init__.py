"""Extensible image feature extraction pipeline."""

from .base import BaseBackbone
from .backbones import BackboneRegistry
from .extractor import FeatureExtractor

__all__ = [
    "BaseBackbone",
    "BackboneRegistry",
    "FeatureExtractor",
]
