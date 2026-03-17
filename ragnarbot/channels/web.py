"""Web UI channel implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from ragnarbot.bus.events import OutboundMessage
from ragnarbot.bus.queue import MessageBus
from ragnarbot.channels.base import BaseChannel
from ragnarbot.config.schema import WebConfig

if TYPE_CHECKING:
    from ragnarbot.agent.loop import AgentLoop
    from ragnarbot.cron.service import CronService
    from ragnarbot.heartbeat.service import HeartbeatService
    from ragnarbot.media.manager import MediaManager
    from ragnarbot.providers.transcription import TranscriptionProvider
    from ragnarbot.session.manager import SessionManager
    from ragnarbot.web.server import WebServer


class WebChannel(BaseChannel):
    """Web UI channel backed by FastAPI + WebSocket."""

    name = "web"

    def __init__(
        self,
        config: WebConfig,
        bus: MessageBus,
        password_hash: str = "",
        transcription_provider: "TranscriptionProvider | None" = None,
        media_manager: "MediaManager | None" = None,
        cron_service: "CronService | None" = None,
        heartbeat_service: "HeartbeatService | None" = None,
        agent_loop: "AgentLoop | None" = None,
        session_manager: "SessionManager | None" = None,
        workspace: Any = None,
    ):
        super().__init__(config, bus)
        self._password_hash = password_hash
        self._transcriber = transcription_provider
        self._media_manager = media_manager
        self._cron_service = cron_service
        self._heartbeat_service = heartbeat_service
        self._agent_loop = agent_loop
        self._session_manager = session_manager
        self._workspace = workspace
        self._server: WebServer | None = None

    async def start(self) -> None:
        """Create and start the FastAPI/uvicorn server."""
        from ragnarbot.web.server import WebServer

        self._server = WebServer(
            config=self.config,
            password_hash=self._password_hash,
            bus=self.bus,
            agent_loop=self._agent_loop,
            cron_service=self._cron_service,
            heartbeat_service=self._heartbeat_service,
            media_manager=self._media_manager,
            transcription_provider=self._transcriber,
            session_manager=self._session_manager,
            workspace=self._workspace,
        )
        self._running = True
        logger.info("Web UI channel starting")
        await self._server.start()

    async def stop(self) -> None:
        """Stop the web server."""
        if self._server:
            await self._server.stop()
        self._running = False
        logger.info("Web UI channel stopped")

    async def send(self, msg: OutboundMessage) -> None:
        """Translate OutboundMessage → WebSocket JSON and push to clients."""
        if not self._server:
            return
        try:
            await self._server.send_outbound(msg)
        except Exception as e:
            logger.error(f"Error sending web message: {e}")

    def is_allowed(self, sender_id: str) -> bool:
        """Web channel: password-based auth. Always True after login (IP check is in middleware)."""
        return True
