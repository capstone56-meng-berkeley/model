"""Feature loader scaffold for retrieving features/labels from Google Sheet."""


import pandas as pd

from .sheet_reader import column_letter_to_index
from .sheets_client import GoogleSheetsClient


class FeatureLoader:
    """
    Loads feature and label data from Google Sheet.

    This is a scaffold - configure feature_columns and label_columns
    in config.json once the sheet structure is finalized.
    """

    def __init__(
        self,
        sheets_client: GoogleSheetsClient,
        spreadsheet_id: str,
        worksheet_name: str
    ):
        """
        Initialize the feature loader.

        Args:
            sheets_client: GoogleSheetsClient instance
            spreadsheet_id: Google Sheets spreadsheet ID
            worksheet_name: Name of the worksheet tab
        """
        self.sheets_client = sheets_client
        self.spreadsheet_id = spreadsheet_id
        self.worksheet_name = worksheet_name

    def load_features(
        self,
        row_id_column: str,
        feature_columns: list[str],
        label_columns: list[str],
        skip_header: bool = True
    ) -> pd.DataFrame:
        """
        Load features and labels from the sheet.

        Args:
            row_id_column: Column containing row identifiers (e.g., "A")
            feature_columns: List of columns containing features (e.g., ["B", "C"])
            label_columns: List of columns containing labels (e.g., ["G", "H", "I"])
            skip_header: Whether to skip the first row (header)

        Returns:
            DataFrame with row_id, features, and labels
        """
        # Get all values (formatted, not formulas)
        all_values = self.sheets_client.get_all_values(
            self.spreadsheet_id,
            self.worksheet_name,
            value_render_option="FORMATTED_VALUE"
        )

        if not all_values:
            return pd.DataFrame()

        # Get headers
        headers = all_values[0] if all_values else []

        # Skip header if needed
        start_idx = 1 if skip_header else 0
        data_rows = all_values[start_idx:]

        # Get column indices
        row_id_idx = column_letter_to_index(row_id_column)
        feature_indices = {col: column_letter_to_index(col) for col in feature_columns}
        label_indices = {col: column_letter_to_index(col) for col in label_columns}

        # Build data records
        records = []

        for row in data_rows:
            # Get row ID
            if row_id_idx >= len(row):
                continue
            row_id = row[row_id_idx]
            if not row_id:
                continue

            record = {"row_id": row_id}

            # Extract features
            for col, col_idx in feature_indices.items():
                col_name = headers[col_idx] if col_idx < len(headers) else f"feature_{col}"
                value = row[col_idx] if col_idx < len(row) else None
                record[col_name] = self._parse_value(value)

            # Extract labels
            for col, col_idx in label_indices.items():
                col_name = headers[col_idx] if col_idx < len(headers) else f"label_{col}"
                value = row[col_idx] if col_idx < len(row) else None
                record[col_name] = self._parse_value(value)

            records.append(record)

        return pd.DataFrame(records)

    def _parse_value(self, value: str | None) -> float | None:
        """
        Parse a cell value to numeric.

        Args:
            value: Cell string value

        Returns:
            Parsed float or None
        """
        if value is None or value == "":
            return None

        try:
            # Handle percentage values
            if isinstance(value, str) and value.endswith('%'):
                return float(value[:-1]) / 100

            return float(value)
        except (ValueError, TypeError):
            # Return as string if not numeric
            return value

    def get_available_columns(self) -> list[str]:
        """
        Get list of available column headers.

        Returns:
            List of column header names
        """
        headers = self.sheets_client.get_headers(
            self.spreadsheet_id,
            self.worksheet_name
        )
        return headers

    def load_raw_data(self, skip_header: bool = True) -> pd.DataFrame:
        """
        Load all data from the sheet as a DataFrame.

        Args:
            skip_header: Whether to use first row as headers

        Returns:
            DataFrame with all sheet data
        """
        all_values = self.sheets_client.get_all_values(
            self.spreadsheet_id,
            self.worksheet_name,
            value_render_option="FORMATTED_VALUE"
        )

        if not all_values:
            return pd.DataFrame()

        if skip_header and len(all_values) > 1:
            headers = all_values[0]
            data = all_values[1:]
            return pd.DataFrame(data, columns=headers)

        return pd.DataFrame(all_values)
