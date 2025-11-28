"""Sheet reader for extracting Drive IDs from Google Sheets."""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .sheets_client import GoogleSheetsClient

# Regex patterns for extracting Drive IDs

# IMAGE function pattern: =IMAGE("https://drive.google.com/uc?export=view&id=XXX", ...)
IMAGE_PATTERN = r'=IMAGE\s*\(\s*"https://drive\.google\.com/uc\?export=view&id=([a-zA-Z0-9_-]+)"'

# Drive folder link patterns
FOLDER_PATTERNS = [
    r'drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)',
    r'drive\.google\.com/drive/u/\d+/folders/([a-zA-Z0-9_-]+)',
]

# Direct image link pattern (uc?id=XXX or uc?export=view&id=XXX)
DIRECT_IMAGE_PATTERNS = [
    r'drive\.google\.com/uc\?(?:export=view&)?id=([a-zA-Z0-9_-]+)',
    r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)',
]


def extract_image_id(cell_value: str) -> Optional[str]:
    """
    Extract image ID from =IMAGE(...) formula or direct link.

    Args:
        cell_value: Cell content (formula or URL)

    Returns:
        Drive file ID or None if not found
    """
    if not cell_value:
        return None

    # Try IMAGE formula pattern
    match = re.search(IMAGE_PATTERN, cell_value)
    if match:
        return match.group(1)

    # Try direct image link patterns
    for pattern in DIRECT_IMAGE_PATTERNS:
        match = re.search(pattern, cell_value)
        if match:
            return match.group(1)

    return None


def extract_folder_id(cell_value: str) -> Optional[str]:
    """
    Extract folder ID from Drive folder link.

    Args:
        cell_value: Cell content (URL)

    Returns:
        Drive folder ID or None if not found
    """
    if not cell_value:
        return None

    for pattern in FOLDER_PATTERNS:
        match = re.search(pattern, cell_value)
        if match:
            return match.group(1)

    return None


def column_letter_to_index(letter: str) -> int:
    """
    Convert column letter to 0-based index.

    Args:
        letter: Column letter (e.g., "A", "B", "AA")

    Returns:
        0-based column index
    """
    result = 0
    for char in letter.upper():
        result = result * 26 + (ord(char) - ord('A') + 1)
    return result - 1


@dataclass
class ImageReference:
    """Reference to an image in Google Drive."""
    row_id: str
    column: str
    drive_id: str
    source_type: str  # "image" or "folder"


@dataclass
class SheetData:
    """Parsed data from the Google Sheet."""
    image_references: List[ImageReference]
    row_ids: List[str]
    raw_data: List[List[str]]


class SheetReader:
    """Reads Google Sheet and extracts Drive IDs for images."""

    def __init__(
        self,
        sheets_client: GoogleSheetsClient,
        spreadsheet_id: str,
        worksheet_name: str
    ):
        """
        Initialize the sheet reader.

        Args:
            sheets_client: GoogleSheetsClient instance
            spreadsheet_id: Google Sheets spreadsheet ID
            worksheet_name: Name of the worksheet tab
        """
        self.sheets_client = sheets_client
        self.spreadsheet_id = spreadsheet_id
        self.worksheet_name = worksheet_name

    def read_image_references(
        self,
        row_id_column: str,
        image_columns: List[str],
        column_types: Dict[str, str],
        skip_header: bool = True
    ) -> SheetData:
        """
        Read the sheet and extract all image references.

        Args:
            row_id_column: Column containing row identifiers (e.g., "A")
            image_columns: List of columns containing images/folders (e.g., ["D", "E", "F"])
            column_types: Dict mapping column to type ("image" or "folder")
            skip_header: Whether to skip the first row (header)

        Returns:
            SheetData with all extracted image references
        """
        # Get all values with formulas
        all_values = self.sheets_client.get_all_values(
            self.spreadsheet_id,
            self.worksheet_name,
            value_render_option="FORMULA"
        )

        if not all_values:
            return SheetData(image_references=[], row_ids=[], raw_data=[])

        # Skip header if needed
        start_idx = 1 if skip_header else 0
        data_rows = all_values[start_idx:]

        # Get column indices
        row_id_idx = column_letter_to_index(row_id_column)
        image_col_indices = {col: column_letter_to_index(col) for col in image_columns}

        image_references = []
        row_ids = []

        for row in data_rows:
            # Get row ID
            if row_id_idx >= len(row):
                continue
            row_id = row[row_id_idx]
            if not row_id:
                continue

            row_ids.append(row_id)

            # Process each image column
            for col, col_idx in image_col_indices.items():
                if col_idx >= len(row):
                    continue

                cell_value = row[col_idx]
                if not cell_value:
                    continue

                col_type = column_types.get(col, "image")

                if col_type == "folder":
                    # Extract folder ID
                    folder_id = extract_folder_id(cell_value)
                    if folder_id:
                        image_references.append(ImageReference(
                            row_id=row_id,
                            column=col,
                            drive_id=folder_id,
                            source_type="folder"
                        ))
                else:
                    # Extract image ID
                    image_id = extract_image_id(cell_value)
                    if image_id:
                        image_references.append(ImageReference(
                            row_id=row_id,
                            column=col,
                            drive_id=image_id,
                            source_type="image"
                        ))

        return SheetData(
            image_references=image_references,
            row_ids=row_ids,
            raw_data=all_values
        )

    def get_row_count(self, skip_header: bool = True) -> int:
        """
        Get the number of data rows in the sheet.

        Args:
            skip_header: Whether to exclude the header row

        Returns:
            Number of data rows
        """
        all_values = self.sheets_client.get_all_values(
            self.spreadsheet_id,
            self.worksheet_name,
            value_render_option="FORMATTED_VALUE"
        )

        count = len(all_values)
        if skip_header and count > 0:
            count -= 1

        return count
