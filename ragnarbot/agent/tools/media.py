"""Media tools for downloading user-shared files."""

from typing import Any

from ragnarbot.agent.tools.base import Tool
from ragnarbot.media.manager import MediaManager


class DownloadFileTool(Tool):
    """Download a file shared by the user in chat."""

    def __init__(self, media_manager: MediaManager):
        self._media = media_manager
        self._channel: str = ""
        self._session_key: str = ""

    def set_context(self, channel: str, session_key: str) -> None:
        """Set the current message context (called per-message by AgentLoop)."""
        self._channel = channel
        self._session_key = session_key

    @property
    def name(self) -> str:
        return "download_file"

    @property
    def description(self) -> str:
        return (
            "Download a file shared by the user. Use when you need to access "
            "a file the user sent in the conversation. Pass the file_id from "
            "the [file available: ...] marker."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "The file_id from the [file available] marker",
                },
                "filename": {
                    "type": "string",
                    "description": "Optional filename for saving the file",
                },
            },
            "required": ["file_id"],
        }

    async def execute(self, file_id: str, filename: str = "", **kwargs: Any) -> str:
        if not self._channel or not self._session_key:
            return "Error: No message context set (channel/session unknown)"

        try:
            path = await self._media.download_file(
                file_id, self._channel, self._session_key, filename
            )
            return f"File saved: {path}"
        except Exception as e:
            return f"Error downloading file: {e}"
