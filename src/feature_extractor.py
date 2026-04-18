"""CNN feature extraction using dual backbone (ResNet50 + VGG16)."""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms as T
from tqdm import tqdm

from .config import Config, ensure_dir


class GlobalPool(nn.Module):
    """Global pooling layer for feature extraction."""

    def __init__(self, pool_type: str = "avg"):
        super().__init__()
        if pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return torch.flatten(x, 1)


class MicrographDataset(Dataset):
    """Dataset for loading micrograph images."""

    def __init__(
        self,
        image_paths: list[str],
        labels_df: pd.DataFrame = None,
        label_columns: list[str] = None,
        transform=None
    ):
        """
        Initialize the dataset.

        Args:
            image_paths: List of image file paths
            labels_df: Optional DataFrame with labels
            label_columns: Optional list of label column names
            transform: Image transforms
        """
        self.image_paths = image_paths
        self.labels_df = labels_df
        self.label_columns = label_columns or []
        self.transform = transform

        # Build filename to index mapping if labels provided
        self._label_map = {}
        if labels_df is not None and 'row_id' in labels_df.columns:
            for idx, row in labels_df.iterrows():
                self._label_map[row['row_id']] = idx

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, np.ndarray, str]:
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)

        # Load and transform image
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # Get labels if available
        labels = np.zeros(len(self.label_columns), dtype=np.float32)
        if self.labels_df is not None and self.label_columns:
            # Extract row_id from filename (format: {row_id}_{column}_{index}.ext)
            parts = filename.rsplit('_', 2)
            if len(parts) >= 1:
                row_id = parts[0]
                if row_id in self._label_map:
                    row_idx = self._label_map[row_id]
                    row = self.labels_df.iloc[row_idx]
                    for i, col in enumerate(self.label_columns):
                        if col in row:
                            val = row[col]
                            labels[i] = float(val) if val is not None else 0.0

        return img, labels, filename


def build_feature_backbone(
    backbone_name: str,
    device: torch.device
) -> tuple[nn.Module, int]:
    """
    Build a pretrained CNN backbone for feature extraction.

    Args:
        backbone_name: "resnet50" or "vgg16"
        device: Torch device

    Returns:
        Tuple of (backbone model, feature dimension)
    """
    if backbone_name == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove fc, keep convs + avgpool
        modules = list(net.children())[:-1]
        feature_dim = net.fc.in_features
        backbone = nn.Sequential(*modules)

    elif backbone_name == "vgg16":
        net = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        backbone = nn.Sequential(
            *list(net.features.children()),
            GlobalPool("avg")
        )
        feature_dim = 512

    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    # Freeze parameters
    for p in backbone.parameters():
        p.requires_grad = False

    return backbone.eval().to(device), feature_dim


class FeatureExtractor:
    """Dual CNN feature extractor using ResNet50 + VGG16."""

    def __init__(self, config: Config):
        """
        Initialize the feature extractor.

        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set random seeds
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)

        # Build backbones
        print(f"Initializing feature extractor on {self.device}...")
        self.resnet_backbone, self.resnet_dim = build_feature_backbone(
            "resnet50", self.device
        )
        self.vgg_backbone, self.vgg_dim = build_feature_backbone(
            "vgg16", self.device
        )

        self.feature_dim = self.resnet_dim + self.vgg_dim
        print(f"✓ Feature dimension: {self.feature_dim} "
              f"(ResNet: {self.resnet_dim}, VGG: {self.vgg_dim})")

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
        image_paths: list[str],
        labels_df: pd.DataFrame = None,
        label_columns: list[str] = None
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Extract features from images using dual CNN backbone.

        Args:
            image_paths: List of image file paths
            labels_df: Optional DataFrame with labels
            label_columns: Optional list of label column names

        Returns:
            Tuple of (features, labels, filenames)
        """
        dataset = MicrographDataset(
            image_paths=image_paths,
            labels_df=labels_df,
            label_columns=label_columns,
            transform=self.transform
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

        res_feats, vgg_feats, all_labels, all_names = [], [], [], []

        for imgs, labels, filenames in tqdm(dataloader, desc="Extracting features"):
            imgs = imgs.to(self.device, non_blocking=True)

            # ResNet features
            f_res = self.resnet_backbone(imgs)
            f_res = torch.flatten(f_res, 1)  # (B, 2048)

            # VGG features
            f_vgg = self.vgg_backbone(imgs)  # (B, 512)

            res_feats.append(f_res.cpu().numpy())
            vgg_feats.append(f_vgg.cpu().numpy())
            all_labels.append(labels.numpy())
            all_names.extend(list(filenames))

        # Stack all features
        X_res = np.vstack(res_feats)
        X_vgg = np.vstack(vgg_feats)
        X = np.concatenate([X_res, X_vgg], axis=1)
        Y = np.vstack(all_labels)

        return X, Y, all_names

    def load_or_extract_features(
        self,
        image_paths: list[str],
        labels_df: pd.DataFrame = None,
        label_columns: list[str] = None,
        cache_path: str = None
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Load cached features or extract new ones.

        Args:
            image_paths: List of image file paths
            labels_df: Optional DataFrame with labels
            label_columns: Optional list of label column names
            cache_path: Optional path to cache file

        Returns:
            Tuple of (features, labels, filenames)
        """
        cache_path = cache_path or self.config.feature_cache

        if os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            return data['X'], data['Y'], list(data['names'])

        print("Extracting features...")
        X, Y, names = self.extract_features(
            image_paths, labels_df, label_columns
        )

        # Cache features
        ensure_dir(os.path.dirname(cache_path))
        np.savez(cache_path, X=X, Y=Y, names=names)
        print(f"✓ Cached features to {cache_path}")

        return X, Y, names
