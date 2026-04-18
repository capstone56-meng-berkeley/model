"""Image downloader for fetching images from Google Drive."""

import os

from .config import ensure_dir
from .drive_client import GoogleDriveClient
from .sheet_reader import ImageReference

# Supported image extensions
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']


class ImageDownloader:
    """Downloads images from Google Drive to local temp folder."""

    def __init__(
        self,
        drive_client: GoogleDriveClient,
        temp_dir: str
    ):
        """
        Initialize the image downloader.

        Args:
            drive_client: GoogleDriveClient instance
            temp_dir: Local directory to save downloaded images
        """
        self.drive_client = drive_client
        self.temp_dir = temp_dir
        ensure_dir(temp_dir)

    def download_all(
        self,
        image_references: list[ImageReference]
    ) -> list[str]:
        """
        Download all images from the given references.

        Args:
            image_references: List of ImageReference objects

        Returns:
            List of local file paths for downloaded images
        """
        downloaded_files = []
        total = len(image_references)

        print(f"Downloading images from {total} references...")

        for idx, ref in enumerate(image_references, start=1):
            print(f"[{idx}/{total}] Processing {ref.row_id} column {ref.column}...")

            if ref.source_type == "folder":
                # List and download all images in the folder
                files = self._download_folder(ref)
                downloaded_files.extend(files)
            else:
                # Download single image
                file_path = self._download_image(ref, index=0)
                if file_path:
                    downloaded_files.append(file_path)

        print(f"✓ Downloaded {len(downloaded_files)} images")
        return downloaded_files

    def _download_image(
        self,
        ref: ImageReference,
        index: int = 0,
        original_name: str | None = None
    ) -> str | None:
        """
        Download a single image.

        Args:
            ref: ImageReference with drive_id
            index: Index for naming (used when multiple images per ref)
            original_name: Original filename from Drive (for extension)

        Returns:
            Local file path or None if download failed
        """
        try:
            # Get file metadata to determine extension
            if original_name:
                ext = os.path.splitext(original_name)[1].lower()
            else:
                metadata = self.drive_client.get_file_metadata(ref.drive_id)
                original_name = metadata.get('name', 'image.png')
                ext = os.path.splitext(original_name)[1].lower()

            # Default to .png if no extension
            if not ext:
                ext = '.png'

            # Build filename: {row_id}_{column}_{index}.{ext}
            filename = f"{ref.row_id}_{ref.column}_{index}{ext}"
            destination = os.path.join(self.temp_dir, filename)

            # Download the file
            self.drive_client.download_file(ref.drive_id, destination)

            return destination

        except Exception as e:
            print(f"  ✗ Failed to download {ref.drive_id}: {e}")
            return None

    def _download_folder(self, ref: ImageReference) -> list[str]:
        """
        Download all images from a folder.

        Args:
            ref: ImageReference with folder drive_id

        Returns:
            List of local file paths
        """
        downloaded = []

        try:
            # List all image files in the folder
            files = self.drive_client.list_files_in_folder(
                ref.drive_id,
                file_extensions=IMAGE_EXTENSIONS
            )

            if not files:
                print(f"  No images found in folder {ref.drive_id}")
                return []

            print(f"  Found {len(files)} images in folder")

            for idx, file_info in enumerate(files):
                file_id = file_info['id']
                file_name = file_info['name']

                # Create a new reference for this specific image
                img_ref = ImageReference(
                    row_id=ref.row_id,
                    column=ref.column,
                    drive_id=file_id,
                    source_type="image"
                )

                file_path = self._download_image(img_ref, index=idx, original_name=file_name)
                if file_path:
                    downloaded.append(file_path)

        except Exception as e:
            print(f"  ✗ Failed to process folder {ref.drive_id}: {e}")

        return downloaded

    def clear_temp_dir(self) -> None:
        """Remove all files from the temp directory."""
        if os.path.exists(self.temp_dir):
            for filename in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"✓ Cleared temp directory: {self.temp_dir}")
