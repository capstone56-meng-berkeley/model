"""Unified data loader supporting both Drive and local data sources."""

import os

import pandas as pd

from .config import Config
from .drive_client import GoogleDriveClient
from .feature_loader import FeatureLoader
from .image_downloader import ImageDownloader
from .sheet_reader import SheetReader
from .sheets_client import GoogleSheetsClient


class DataLoader:
    """
    Unified data loader for the model pipeline.

    Supports two modes:
    - drive: Download images from Google Drive via Google Sheet references
    - local: Load images from local directory with CSV labels
    """

    def __init__(self, config: Config):
        """
        Initialize the data loader.

        Args:
            config: Configuration object
        """
        self.config = config
        self._drive_client: GoogleDriveClient | None = None
        self._sheets_client: GoogleSheetsClient | None = None

    @property
    def drive_client(self) -> GoogleDriveClient:
        """Lazy-initialize Drive client."""
        if self._drive_client is None:
            self._drive_client = GoogleDriveClient(
                credentials_path=self.config.credentials_path,
                token_path=self.config.token_path
            )
        return self._drive_client

    @property
    def sheets_client(self) -> GoogleSheetsClient:
        """Lazy-initialize Sheets client."""
        if self._sheets_client is None:
            self._sheets_client = GoogleSheetsClient(self.drive_client.creds)
        return self._sheets_client

    def load_data(self) -> tuple[list[str], pd.DataFrame]:
        """
        Load images and labels based on configured data source.

        Returns:
            Tuple of (image_paths, labels_df)
        """
        if self.config.is_drive_mode:
            return self._load_from_drive()
        else:
            return self._load_from_local()

    def _load_from_drive(self) -> tuple[list[str], pd.DataFrame]:
        """
        Load data from Google Drive.

        Returns:
            Tuple of (image_paths, labels_df)
        """
        if not self.config.sheet_id:
            raise ValueError(
                "GOOGLE_SHEET_ID not set. Please configure in .env file."
            )

        print("=== Loading Data from Google Drive ===")

        # Initialize sheet reader
        sheet_reader = SheetReader(
            sheets_client=self.sheets_client,
            spreadsheet_id=self.config.sheet_id,
            worksheet_name=self.config.worksheet_name
        )

        # Read image references from sheet
        print("Reading image references from sheet...")
        sheet_data = sheet_reader.read_image_references(
            row_id_column=self.config.google_drive.row_id_column,
            image_columns=self.config.google_drive.image_columns,
            column_types=self.config.google_drive.column_types
        )

        print(f"Found {len(sheet_data.image_references)} image references "
              f"from {len(sheet_data.row_ids)} rows")

        # Download images
        image_downloader = ImageDownloader(
            drive_client=self.drive_client,
            temp_dir=self.config.temp_dir
        )

        image_paths = image_downloader.download_all(sheet_data.image_references)

        # Load features/labels if configured
        labels_df = pd.DataFrame()
        if (self.config.features.feature_columns or
                self.config.features.label_columns):
            feature_loader = FeatureLoader(
                sheets_client=self.sheets_client,
                spreadsheet_id=self.config.sheet_id,
                worksheet_name=self.config.worksheet_name
            )

            labels_df = feature_loader.load_features(
                row_id_column=self.config.google_drive.row_id_column,
                feature_columns=self.config.features.feature_columns,
                label_columns=self.config.features.label_columns
            )

        print(f"✓ Loaded {len(image_paths)} images")
        return image_paths, labels_df

    def _load_from_local(self) -> tuple[list[str], pd.DataFrame]:
        """
        Load data from local directory.

        Returns:
            Tuple of (image_paths, labels_df)
        """
        print("=== Loading Data from Local Directory ===")

        images_dir = self.config.local.images_dir
        labels_csv = self.config.local.labels_csv

        # Check paths exist
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        # Get all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        image_paths = []

        for filename in sorted(os.listdir(images_dir)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                image_paths.append(os.path.join(images_dir, filename))

        print(f"Found {len(image_paths)} images in {images_dir}")

        # Load labels CSV if exists
        labels_df = pd.DataFrame()
        if os.path.exists(labels_csv):
            labels_df = pd.read_csv(labels_csv)
            print(f"Loaded labels from {labels_csv}: {labels_df.shape}")
        else:
            print(f"No labels CSV found at {labels_csv}")

        print(f"✓ Loaded {len(image_paths)} images")
        return image_paths, labels_df

    def get_images_dir(self) -> str:
        """Get the directory containing images."""
        return self.config.images_dir

    def clear_temp_data(self) -> None:
        """Clear downloaded temp data (Drive mode only)."""
        if self.config.is_drive_mode and os.path.exists(self.config.temp_dir):
            image_downloader = ImageDownloader(
                drive_client=self.drive_client,
                temp_dir=self.config.temp_dir
            )
            image_downloader.clear_temp_dir()
