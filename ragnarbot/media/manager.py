"""Central media manager for file storage and downloads."""

import time
from pathlib import Path
from typing import Awaitable, Callable

from loguru import logger

DownloadCallback = Callable[[str], Awaitable[tuple[bytes, str]]]
# Takes file_id, returns (file_bytes, suggested_filename)


class MediaManager:
    """Manages media storage with per-session directories.

    Directory layout::

        base_dir/
        └── {session_key}/
            ├── photos/
            │   └── photo_1738934400.jpg
            └── files/
                ├── report.pdf
                └── data.csv
    """

    def __init__(self, base_dir: Path):
        self._base_dir = base_dir
        self._download_callbacks: dict[str, DownloadCallback] = {}

    def register_download_callback(self, channel: str, cb: DownloadCallback) -> None:
        """Register a download callback for a channel."""
        self._download_callbacks[channel] = cb

    async def save_photo(self, session_key: str, data: bytes, ext: str) -> str:
        """Save photo bytes to the session's photos directory.

        Args:
            session_key: The session identifier.
            data: Raw photo bytes.
            ext: File extension (e.g. "jpg", "png").

        Returns:
            The filename (not full path) of the saved photo.
        """
        photos_dir = self._base_dir / session_key / "photos"
        photos_dir.mkdir(parents=True, exist_ok=True)

        filename = f"photo_{int(time.time())}.{ext}"
        path = self._unique_path(photos_dir / filename)
        path.write_bytes(data)
        logger.debug(f"Saved photo: {path}")
        return path.name

    async def download_file(
        self, file_id: str, channel: str, session_key: str, filename: str = ""
    ) -> str:
        """Download a file via the channel's callback and save it.

        Args:
            file_id: Platform-specific file identifier.
            channel: Channel name (e.g. "telegram").
            session_key: The session identifier.
            filename: Optional cosmetic filename.

        Returns:
            Full path to the saved file.
        """
        cb = self._download_callbacks.get(channel)
        if not cb:
            raise RuntimeError(f"No download callback registered for channel '{channel}'")

        data, suggested_name = await cb(file_id)

        name = filename or suggested_name or f"file_{int(time.time())}"
        files_dir = self._base_dir / session_key / "files"
        files_dir.mkdir(parents=True, exist_ok=True)

        path = self._unique_path(files_dir / name)
        path.write_bytes(data)
        logger.debug(f"Downloaded file: {path}")
        return str(path)

    def get_photo_path(self, session_key: str, filename: str) -> Path:
        """Get full path for a photo in a session."""
        return self._base_dir / session_key / "photos" / filename

    @staticmethod
    def _unique_path(path: Path) -> Path:
        """Append _1, _2, etc. if the path already exists."""
        if not path.exists():
            return path

        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        counter = 1
        while True:
            candidate = parent / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1
