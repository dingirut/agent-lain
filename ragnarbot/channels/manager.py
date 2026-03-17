"""Channel manager for coordinating chat channels."""

import asyncio
from typing import Any

from loguru import logger

from ragnarbot.auth.credentials import Credentials
from ragnarbot.bus.events import OutboundMessage
from ragnarbot.bus.queue import MessageBus
from ragnarbot.channels.base import BaseChannel
from ragnarbot.config.schema import Config
from ragnarbot.media.manager import MediaManager


class ChannelManager:
    """
    Manages chat channels and coordinates message routing.

    Responsibilities:
    - Initialize enabled channels (Telegram, Web)
    - Start/stop channels
    - Route outbound messages
    """

    def __init__(
        self,
        config: Config,
        bus: MessageBus,
        credentials: Credentials | None = None,
        media_manager: MediaManager | None = None,
        cron_service: Any = None,
        heartbeat_service: Any = None,
        agent_loop: Any = None,
    ):
        self.config = config
        self.bus = bus
        self.credentials = credentials or Credentials()
        self.media_manager = media_manager
        self.cron_service = cron_service
        self.heartbeat_service = heartbeat_service
        self.agent_loop = agent_loop
        self.channels: dict[str, BaseChannel] = {}
        self._dispatch_task: asyncio.Task | None = None

        self._init_channels()

    def _init_channels(self) -> None:
        """Initialize channels based on config."""

        # Telegram channel
        if self.config.channels.telegram.enabled:
            try:
                from ragnarbot.channels.telegram import TelegramChannel
                from ragnarbot.providers.transcription import create_transcription_provider

                transcriber = create_transcription_provider(
                    self.config.transcription.provider,
                    self.credentials.services,
                )
                self.channels["telegram"] = TelegramChannel(
                    self.config.channels.telegram,
                    self.bus,
                    bot_token=self.credentials.channels.telegram.bot_token,
                    transcription_provider=transcriber,
                    media_manager=self.media_manager,
                )
                logger.info("Telegram channel enabled")
            except ImportError as e:
                logger.warning(f"Telegram channel not available: {e}")

        # Web UI channel
        if self.config.channels.web.enabled:
            try:
                from ragnarbot.channels.web import WebChannel
                from ragnarbot.providers.transcription import create_transcription_provider

                transcriber = create_transcription_provider(
                    self.config.transcription.provider,
                    self.credentials.services,
                )

                # Get session manager from agent loop if available
                session_manager = None
                workspace = self.config.workspace_path
                if self.agent_loop and hasattr(self.agent_loop, "sessions"):
                    session_manager = self.agent_loop.sessions

                self.channels["web"] = WebChannel(
                    config=self.config.channels.web,
                    bus=self.bus,
                    password_hash=self.credentials.channels.web.password_hash,
                    transcription_provider=transcriber,
                    media_manager=self.media_manager,
                    cron_service=self.cron_service,
                    heartbeat_service=self.heartbeat_service,
                    agent_loop=self.agent_loop,
                    session_manager=session_manager,
                    workspace=workspace,
                )
                logger.info("Web UI channel enabled")
            except ImportError as e:
                logger.warning(f"Web UI channel not available: {e}")

    
    async def start_all(self) -> None:
        """Start all channels and the outbound dispatcher."""
        if not self.channels:
            logger.warning("No channels enabled")
            return
        
        # Start outbound dispatcher
        self._dispatch_task = asyncio.create_task(self._dispatch_outbound())
        
        # Start channels
        tasks = []
        for name, channel in self.channels.items():
            logger.info(f"Starting {name} channel...")
            tasks.append(asyncio.create_task(channel.start()))
        
        # Wait for all to complete (they should run forever)
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop_all(self) -> None:
        """Stop all channels and the dispatcher."""
        logger.info("Stopping all channels...")
        
        # Stop dispatcher
        if self._dispatch_task:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass
        
        # Stop all channels
        for name, channel in self.channels.items():
            try:
                await channel.stop()
                logger.info(f"Stopped {name} channel")
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
    
    async def _dispatch_outbound(self) -> None:
        """Dispatch outbound messages to the appropriate channel."""
        logger.info("Outbound dispatcher started")
        
        while True:
            try:
                msg = await asyncio.wait_for(
                    self.bus.consume_outbound(),
                    timeout=1.0
                )
                
                channel = self.channels.get(msg.channel)
                if channel:
                    try:
                        await channel.send(msg)
                    except Exception as e:
                        logger.error(f"Error sending to {msg.channel}: {e}")
                else:
                    logger.warning(f"Unknown channel: {msg.channel}")
                    
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
    
    def get_channel(self, name: str) -> BaseChannel | None:
        """Get a channel by name."""
        return self.channels.get(name)
    
    def get_status(self) -> dict[str, Any]:
        """Get status of all channels."""
        return {
            name: {
                "enabled": True,
                "running": channel.is_running
            }
            for name, channel in self.channels.items()
        }
    
    @property
    def enabled_channels(self) -> list[str]:
        """Get list of enabled channel names."""
        return list(self.channels.keys())
