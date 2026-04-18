"""Feature extractor using multiple CNN backbones."""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from .backbones import BackboneRegistry, get_backbone
from .base import BaseBackbone


@dataclass
class ExtractionConfig:
    """Configuration for feature extraction."""
    backbones: list[str] = field(default_factory=lambda: ["vgg16"])
    img_size: int = 224
    batch_size: int = 16
    num_workers: int = 2
    pooling: str = "avg"
    parallel: bool = False  # Run multiple backbones in parallel


class ImageDataset(Dataset):
    """Dataset for loading images from file paths."""

    def __init__(
        self,
        image_paths: list[str],
        transform=None
    ):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            print(f"  [WARN] Skipping unreadable image: {img_path}")
            img = Image.new("RGB", (224, 224), 0)

        if self.transform is not None:
            img = self.transform(img)

        return img, filename


class FeatureExtractor:
    """
    Extensible multi-backbone feature extractor.

    Uses registry pattern to support any registered backbone.
    Concatenates features from multiple backbones.
    """

    def __init__(self, config: ExtractionConfig):
        """
        Initialize the feature extractor.

        Args:
            config: Extraction configuration
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize backbones from registry
        self.backbones: list[BaseBackbone] = []
        print(f"Initializing feature extractor on {self.device}...")

        for name in config.backbones:
            backbone = get_backbone(name, pretrained=True)
            backbone = backbone.to(self.device).eval()
            self.backbones.append(backbone)
            print(f"  Loaded {name}: {backbone.feature_dim} features")

        # Total feature dimension
        self.feature_dim = sum(b.feature_dim for b in self.backbones)
        print(f"  Total feature dimension: {self.feature_dim}")

        # Image transforms (ImageNet normalization)
        self.transform = T.Compose([
            T.Resize((config.img_size, config.img_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    @torch.no_grad()
    def extract_features(
        self,
        image_paths: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """
        Extract features from images using all backbones.

        Args:
            image_paths: List of image file paths

        Returns:
            Tuple of (features array, filenames list)
        """
        dataset = ImageDataset(image_paths, transform=self.transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

        all_features = []
        all_filenames = []

        for imgs, filenames in tqdm(dataloader, desc="Extracting features"):
            imgs = imgs.to(self.device, non_blocking=True)

            # Extract from each backbone (parallel or sequential)
            if self.config.parallel and len(self.backbones) > 1:
                batch_features = self._extract_parallel(imgs)
            else:
                batch_features = self._extract_sequential(imgs)

            # Concatenate backbone features
            combined = np.concatenate(batch_features, axis=1)
            all_features.append(combined)
            all_filenames.extend(list(filenames))

        # Stack all batches
        X = np.vstack(all_features)

        return X, all_filenames

    def _extract_sequential(self, imgs: torch.Tensor) -> list[np.ndarray]:
        """Extract features sequentially from each backbone."""
        batch_features = []
        for backbone in self.backbones:
            features = backbone(imgs)
            batch_features.append(features.cpu().numpy())
        return batch_features

    def _extract_parallel(self, imgs: torch.Tensor) -> list[np.ndarray]:
        """Extract features in parallel from multiple backbones."""
        def run_backbone(backbone: BaseBackbone) -> np.ndarray:
            features = backbone(imgs)
            return features.cpu().numpy()

        with ThreadPoolExecutor(max_workers=len(self.backbones)) as executor:
            futures = [executor.submit(run_backbone, bb) for bb in self.backbones]
            batch_features = [f.result() for f in futures]

        return batch_features

    def get_feature_names(self) -> list[str]:
        """Get feature names for each dimension."""
        names = []
        for backbone in self.backbones:
            backbone_name = type(backbone).__name__.replace("Backbone", "").lower()
            for i in range(backbone.feature_dim):
                names.append(f"{backbone_name}_{i}")
        return names

    @classmethod
    def list_available_backbones(cls) -> list[str]:
        """List all available backbone names."""
        return BackboneRegistry.list_available()


def extract_image_features(
    image_paths: list[str],
    backbones: list[str] = None,
    img_size: int = 224,
    batch_size: int = 16,
    cache_path: str | None = None,
    parallel: bool = False
) -> tuple[np.ndarray, list[str]]:
    """
    Convenience function to extract image features.

    Args:
        image_paths: List of image file paths
        backbones: List of backbone names (default: vgg16)
        img_size: Image size for processing
        batch_size: Batch size for extraction
        cache_path: Optional path to cache features
        parallel: Run multiple backbones in parallel

    Returns:
        Tuple of (features array, filenames list)
    """
    # Check cache
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data['X'], list(data['filenames'])

    # Create config
    config = ExtractionConfig(
        backbones=backbones or ["vgg16"],
        img_size=img_size,
        batch_size=batch_size,
        parallel=parallel
    )

    # Extract
    extractor = FeatureExtractor(config)
    X, filenames = extractor.extract_features(image_paths)

    # Cache if path provided
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, X=X, filenames=filenames)
        print(f"Cached features to {cache_path}")

    return X, filenames
