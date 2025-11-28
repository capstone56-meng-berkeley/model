"""Google Drive client for file operations."""

import io
import os
from typing import List, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = [
    'https://www.googleapis.com/auth/drive.readonly',
    'https://www.googleapis.com/auth/spreadsheets.readonly'
]


class GoogleDriveClient:
    """Handles Google Drive authentication and file operations."""

    def __init__(
        self,
        credentials_path: str = 'credentials.json',
        token_path: str = 'token.json'
    ):
        """
        Initialize the Google Drive client.

        Args:
            credentials_path: Path to OAuth credentials JSON file
            token_path: Path to store/load cached token
        """
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service = None
        self.creds = None
        self._authenticate()

    def _authenticate(self) -> None:
        """Authenticate with Google Drive API."""
        creds = None

        # Check if token.json exists (previously authenticated)
        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)

        # If no valid credentials, authenticate
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_path):
                    raise FileNotFoundError(
                        f"Credentials file not found: {self.credentials_path}\n"
                        f"Please download OAuth 2.0 credentials from Google Cloud Console."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Save credentials for next run
            with open(self.token_path, 'w') as token:
                token.write(creds.to_json())

        self.creds = creds
        self.service = build('drive', 'v3', credentials=creds)
        print("✓ Authenticated with Google Drive")

    def download_file(self, file_id: str, destination: str) -> None:
        """
        Download a file from Google Drive by file ID.

        Args:
            file_id: Google Drive file ID
            destination: Local path to save the file
        """
        request = self.service.files().get_media(fileId=file_id)

        with io.FileIO(destination, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()

    def get_file_metadata(self, file_id: str) -> dict:
        """
        Get metadata for a file.

        Args:
            file_id: Google Drive file ID

        Returns:
            File metadata dictionary
        """
        return self.service.files().get(
            fileId=file_id,
            fields='id, name, mimeType'
        ).execute()

    def list_files_in_folder(
        self,
        folder_id: str,
        file_extensions: Optional[List[str]] = None
    ) -> List[dict]:
        """
        List all files in a Google Drive folder.

        Args:
            folder_id: Google Drive folder ID
            file_extensions: Optional list of file extensions to filter (e.g., ['.png', '.jpg'])

        Returns:
            List of file dictionaries with id, name, mimeType
        """
        query = f"'{folder_id}' in parents and trashed=false"

        # Exclude Google Workspace native formats
        google_mime_types = [
            'application/vnd.google-apps.document',
            'application/vnd.google-apps.spreadsheet',
            'application/vnd.google-apps.presentation',
            'application/vnd.google-apps.form',
            'application/vnd.google-apps.drawing',
            'application/vnd.google-apps.folder',
        ]
        for gmt in google_mime_types:
            query += f" and mimeType!='{gmt}'"

        results = self.service.files().list(
            q=query,
            pageSize=1000,
            fields="files(id, name, mimeType)"
        ).execute()

        files = results.get('files', [])

        # Filter by file extensions if specified
        if file_extensions:
            normalized_exts = [
                ext if ext.startswith('.') else f'.{ext}'
                for ext in file_extensions
            ]
            normalized_exts = [ext.lower() for ext in normalized_exts]

            files = [
                f for f in files
                if any(f['name'].lower().endswith(ext) for ext in normalized_exts)
            ]

        return files
