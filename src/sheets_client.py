"""Google Sheets client for reading spreadsheet data."""

from typing import List, Optional

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build


class GoogleSheetsClient:
    """Handles Google Sheets read operations."""

    def __init__(self, creds: Credentials):
        """
        Initialize the Google Sheets client.

        Args:
            creds: Google OAuth credentials (shared with Drive client)
        """
        self.service = build('sheets', 'v4', credentials=creds)
        print("✓ Initialized Google Sheets client")

    def get_all_values(
        self,
        spreadsheet_id: str,
        worksheet_name: str,
        value_render_option: str = "FORMULA"
    ) -> List[List[str]]:
        """
        Get all values from a worksheet.

        Args:
            spreadsheet_id: Google Sheets spreadsheet ID
            worksheet_name: Name of the worksheet tab
            value_render_option: How to render values:
                - "FORMATTED_VALUE": Display values
                - "UNFORMATTED_VALUE": Raw values
                - "FORMULA": Formulas (e.g., =IMAGE(...))

        Returns:
            2D list of cell values
        """
        result = self.service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=worksheet_name,
            valueRenderOption=value_render_option
        ).execute()

        return result.get('values', [])

    def get_range(
        self,
        spreadsheet_id: str,
        range_notation: str,
        value_render_option: str = "FORMULA"
    ) -> List[List[str]]:
        """
        Get values from a specific range.

        Args:
            spreadsheet_id: Google Sheets spreadsheet ID
            range_notation: A1 notation (e.g., "Sheet1!A1:F100")
            value_render_option: How to render values

        Returns:
            2D list of cell values
        """
        result = self.service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=range_notation,
            valueRenderOption=value_render_option
        ).execute()

        return result.get('values', [])

    def get_column_values(
        self,
        spreadsheet_id: str,
        worksheet_name: str,
        column: str,
        start_row: int = 1,
        value_render_option: str = "FORMULA"
    ) -> List[str]:
        """
        Get all values from a specific column.

        Args:
            spreadsheet_id: Google Sheets spreadsheet ID
            worksheet_name: Name of the worksheet tab
            column: Column letter (e.g., "A", "D")
            start_row: Row to start from (1-indexed)
            value_render_option: How to render values

        Returns:
            List of cell values from the column
        """
        range_notation = f"{worksheet_name}!{column}{start_row}:{column}"

        result = self.service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=range_notation,
            valueRenderOption=value_render_option
        ).execute()

        values = result.get('values', [])
        # Flatten the 2D list to 1D
        return [row[0] if row else "" for row in values]

    def get_headers(
        self,
        spreadsheet_id: str,
        worksheet_name: str,
        header_row: int = 1
    ) -> List[str]:
        """
        Get header row values.

        Args:
            spreadsheet_id: Google Sheets spreadsheet ID
            worksheet_name: Name of the worksheet tab
            header_row: Row number containing headers (1-indexed)

        Returns:
            List of header values
        """
        range_notation = f"{worksheet_name}!{header_row}:{header_row}"

        result = self.service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=range_notation,
            valueRenderOption="FORMATTED_VALUE"
        ).execute()

        values = result.get('values', [])
        return values[0] if values else []
