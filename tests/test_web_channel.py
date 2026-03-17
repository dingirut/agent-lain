"""Tests for web channel outbound message translation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ragnarbot.bus.events import OutboundMessage
from ragnarbot.bus.queue import MessageBus
from ragnarbot.config.schema import WebConfig


class TestWebChannelSend:
    """Test that WebChannel.send() correctly translates OutboundMessage metadata."""

    @pytest.fixture
    def channel(self):
        from ragnarbot.channels.web import WebChannel
        config = WebConfig(enabled=True, port=8080)
        bus = MessageBus()
        ch = WebChannel(config=config, bus=bus, password_hash="testhash")

        # Mock the server's send_outbound
        ch._server = MagicMock()
        ch._server.send_outbound = AsyncMock()
        return ch

    @pytest.mark.asyncio
    async def test_send_typing(self, channel):
        msg = OutboundMessage(
            channel="web", chat_id="user1", content="",
            metadata={"chat_action": "typing"},
        )
        await channel.send(msg)
        channel._server.send_outbound.assert_called_once_with(msg)

    @pytest.mark.asyncio
    async def test_send_stop_typing(self, channel):
        msg = OutboundMessage(
            channel="web", chat_id="user1", content="",
            metadata={"stop_typing": True},
        )
        await channel.send(msg)
        channel._server.send_outbound.assert_called_once_with(msg)

    @pytest.mark.asyncio
    async def test_send_text_message(self, channel):
        msg = OutboundMessage(
            channel="web", chat_id="user1", content="Hello!",
            metadata={},
        )
        await channel.send(msg)
        channel._server.send_outbound.assert_called_once_with(msg)

    @pytest.mark.asyncio
    async def test_send_with_no_server(self, channel):
        channel._server = None
        msg = OutboundMessage(
            channel="web", chat_id="user1", content="Hello!",
        )
        # Should not raise
        await channel.send(msg)


class TestWebChannelIsAllowed:
    def test_always_allowed(self):
        from ragnarbot.channels.web import WebChannel
        config = WebConfig(enabled=True)
        bus = MessageBus()
        ch = WebChannel(config=config, bus=bus)
        assert ch.is_allowed("anyone") is True
        assert ch.is_allowed("") is True


class TestWebConfig:
    def test_default_config(self):
        config = WebConfig()
        assert config.enabled is False
        assert config.host == "0.0.0.0"
        assert config.port == 80
        assert config.allow_from == []

    def test_custom_config(self):
        config = WebConfig(
            enabled=True, host="127.0.0.1", port=8080,
            allow_from=["192.168.1.0/24"],
        )
        assert config.enabled is True
        assert config.port == 8080
        assert config.allow_from == ["192.168.1.0/24"]
