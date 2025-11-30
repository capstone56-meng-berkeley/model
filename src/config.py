"""Configuration loader for the model pipeline."""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dotenv import load_dotenv


@dataclass
class LocalConfig:
    """Configuration for local data source."""
    images_dir: str = "data/images"
    labels_csv: str = "data/labels.csv"


@dataclass
class GoogleDriveConfig:
    """Configuration for Google Drive data source."""
    row_id_column: str = "A"
    image_columns: List[str] = field(default_factory=lambda: ["D", "E", "F"])
    column_types: Dict[str, str] = field(default_factory=lambda: {
        "D": "image",
        "E": "image",
        "F": "folder"
    })


@dataclass
class MissingDataConfig:
    """Configuration for missing data handling."""
    column_drop_threshold: float = 0.95  # Drop column if >95% missing
    row_fill_threshold: float = 0.10     # Fill if <10% missing
    numeric_fill_strategy: str = "mean"  # mean, median, zero
    categorical_fill_strategy: str = "mode"  # mode, unknown
    mid_range_strategy: str = "fill"  # drop_rows, fill, flag


@dataclass
class ScalingConfig:
    """Configuration for feature scaling."""
    method: str = "standard"  # standard, minmax, robust, none
    enabled: bool = True


@dataclass
class EncodingConfig:
    """Configuration for feature encoding."""
    categorical: str = "onehot"  # onehot, label
    text: str = "tfidf"  # tfidf, skip
    unique_string: str = "label"  # label, skip
    max_categories: int = 50


@dataclass
class PreprocessingConfig:
    """Configuration for tabular data preprocessing."""
    missing_data: MissingDataConfig = field(default_factory=MissingDataConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)


@dataclass
class FeaturesConfig:
    """Configuration for feature and label columns."""
    feature_columns: List[str] = field(default_factory=list)
    label_columns: List[str] = field(default_factory=list)
    column_types: Dict[str, str] = field(default_factory=dict)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)


@dataclass
class ExtractionConfig:
    """Configuration for image feature extraction."""
    backbones: List[str] = field(default_factory=lambda: ["vgg16"])
    pooling: str = "avg"
    parallel: bool = False  # Run multiple backbones in parallel


@dataclass
class Config:
    """Main configuration for the model pipeline."""
    # Data source: "drive" or "local"
    data_source: str = "drive"

    # Local data config
    local: LocalConfig = field(default_factory=LocalConfig)

    # Temp directory for downloaded images
    temp_dir: str = "data/temp_images"

    # Feature cache path
    feature_cache: str = "data/feature_cache.npz"

    # Model output directory
    model_dir: str = "models"

    # Training parameters
    random_seed: int = 42
    img_size: int = 224
    batch_size: int = 16
    num_workers: int = 2

    # Google Drive config
    google_drive: GoogleDriveConfig = field(default_factory=GoogleDriveConfig)

    # Features config (tabular)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)

    # Image feature extraction config
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)

    # Environment variables (loaded from .env)
    sheet_id: Optional[str] = None
    worksheet_name: str = "Sheet1"
    credentials_path: str = "credentials.json"
    token_path: str = "token.json"

    @property
    def is_drive_mode(self) -> bool:
        """Check if using Google Drive as data source."""
        return self.data_source == "drive"

    @property
    def is_local_mode(self) -> bool:
        """Check if using local data source."""
        return self.data_source == "local"

    @property
    def images_dir(self) -> str:
        """Get the images directory based on data source."""
        if self.is_drive_mode:
            return self.temp_dir
        return self.local.images_dir


def load_config(config_path: str = "config.json", env_path: str = ".env") -> Config:
    """
    Load configuration from JSON file and environment variables.

    Args:
        config_path: Path to config.json file
        env_path: Path to .env file

    Returns:
        Config object with all settings
    """
    # Load environment variables
    load_dotenv(env_path)

    # Load JSON config
    config_data = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_data = json.load(f)

    # Parse nested configs
    local_data = config_data.get("local", {})
    local_config = LocalConfig(
        images_dir=local_data.get("images_dir", "data/images"),
        labels_csv=local_data.get("labels_csv", "data/labels.csv")
    )

    drive_data = config_data.get("google_drive", {})
    drive_config = GoogleDriveConfig(
        row_id_column=drive_data.get("row_id_column", "A"),
        image_columns=drive_data.get("image_columns", ["D", "E", "F"]),
        column_types=drive_data.get("column_types", {"D": "image", "E": "image", "F": "folder"})
    )

    # Parse features config with preprocessing
    features_data = config_data.get("features", {})

    # Parse preprocessing config
    preproc_data = features_data.get("preprocessing", {})

    missing_data = preproc_data.get("missing_data", {})
    missing_config = MissingDataConfig(
        column_drop_threshold=missing_data.get("column_drop_threshold", 0.95),
        row_fill_threshold=missing_data.get("row_fill_threshold", 0.10),
        numeric_fill_strategy=missing_data.get("numeric_fill_strategy", "mean"),
        categorical_fill_strategy=missing_data.get("categorical_fill_strategy", "mode"),
        mid_range_strategy=missing_data.get("mid_range_strategy", "fill")
    )

    scaling_data = preproc_data.get("scaling", {})
    scaling_config = ScalingConfig(
        method=scaling_data.get("method", "standard"),
        enabled=scaling_data.get("enabled", True)
    )

    encoding_data = preproc_data.get("encoding", {})
    encoding_config = EncodingConfig(
        categorical=encoding_data.get("categorical", "onehot"),
        text=encoding_data.get("text", "tfidf"),
        unique_string=encoding_data.get("unique_string", "label"),
        max_categories=encoding_data.get("max_categories", 50)
    )

    preproc_config = PreprocessingConfig(
        missing_data=missing_config,
        scaling=scaling_config,
        encoding=encoding_config
    )

    features_config = FeaturesConfig(
        feature_columns=features_data.get("feature_columns", []),
        label_columns=features_data.get("label_columns", []),
        column_types=features_data.get("column_types", {}),
        preprocessing=preproc_config
    )

    # Parse extraction config
    extraction_data = config_data.get("extraction", {})
    extraction_config = ExtractionConfig(
        backbones=extraction_data.get("backbones", ["vgg16"]),
        pooling=extraction_data.get("pooling", "avg"),
        parallel=extraction_data.get("parallel", False)
    )

    # Build main config
    config = Config(
        data_source=config_data.get("data_source", "drive"),
        local=local_config,
        temp_dir=config_data.get("temp_dir", "data/temp_images"),
        feature_cache=config_data.get("feature_cache", "data/feature_cache.npz"),
        model_dir=config_data.get("model_dir", "models"),
        random_seed=config_data.get("random_seed", 42),
        img_size=config_data.get("img_size", 224),
        batch_size=config_data.get("batch_size", 16),
        num_workers=config_data.get("num_workers", 2),
        google_drive=drive_config,
        features=features_config,
        extraction=extraction_config,
        # Environment variables
        sheet_id=os.getenv("GOOGLE_SHEET_ID"),
        worksheet_name=os.getenv("GOOGLE_WORKSHEET_NAME", "Sheet1"),
        credentials_path=os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json"),
        token_path=os.getenv("GOOGLE_TOKEN_PATH", "token.json"),
    )

    return config


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
