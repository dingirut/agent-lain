"""Tests for the config tool."""

import json
from unittest.mock import MagicMock, patch

import pytest

from ragnarbot.agent.tools.config_tool import ConfigTool
from ragnarbot.config.schema import Config

LOAD_CONFIG = "ragnarbot.config.loader.load_config"
SAVE_CONFIG = "ragnarbot.config.loader.save_config"


@pytest.fixture
def mock_agent():
    """Create a mock agent for ConfigTool."""
    agent = MagicMock()
    agent.provider = MagicMock()
    agent.provider.set_temperature = MagicMock()
    agent.provider.set_max_tokens = MagicMock()
    agent.stream_steps = True
    agent.debounce_seconds = 0.5
    agent.context_mode = "normal"
    agent.max_context_tokens = 200_000
    agent.cache_manager = MagicMock()
    agent.compactor = MagicMock()
    agent.brave_api_key = None
    agent.tools = MagicMock()
    agent.tools.get = MagicMock(return_value=None)
    agent.tools.unregister = MagicMock()
    agent.tools.register = MagicMock()
    return agent


@pytest.fixture
def config_tool(mock_agent):
    return ConfigTool(agent=mock_agent)


@pytest.mark.asyncio
async def test_schema_action_returns_all_fields(config_tool):
    with patch(LOAD_CONFIG, return_value=Config()):
        result = await config_tool.execute(action="schema")
    assert "agents.defaults.temperature" in result
    assert "float" in result
    assert "[hot]" in result


@pytest.mark.asyncio
async def test_schema_action_filter_by_path(config_tool):
    with patch(LOAD_CONFIG, return_value=Config()):
        result = await config_tool.execute(action="schema", path="tools.web")
    assert "tools.web.search.engine" in result
    assert "agents.defaults" not in result


@pytest.mark.asyncio
async def test_get_action_returns_value(config_tool):
    with patch(LOAD_CONFIG, return_value=Config()):
        result = await config_tool.execute(action="get", path="agents.defaults.temperature")
    data = json.loads(result)
    assert data["value"] == 0.7
    assert data["reload"] == "hot"


@pytest.mark.asyncio
async def test_get_action_missing_path(config_tool):
    result = await config_tool.execute(action="get")
    assert "Error" in result


@pytest.mark.asyncio
async def test_set_action_saves_and_hot_reloads(config_tool, mock_agent):
    with (
        patch(LOAD_CONFIG, return_value=Config()),
        patch(SAVE_CONFIG),
    ):
        result = await config_tool.execute(
            action="set", path="agents.defaults.temperature", value="0.5"
        )

    data = json.loads(result)
    assert data["new_value"] == 0.5
    assert data["status"] == "applied"
    mock_agent.provider.set_temperature.assert_called_once_with(0.5)


@pytest.mark.asyncio
async def test_set_action_warm_field(config_tool):
    with (
        patch(LOAD_CONFIG, return_value=Config()),
        patch(SAVE_CONFIG),
    ):
        result = await config_tool.execute(
            action="set", path="agents.defaults.model", value="openai/gpt-4"
        )

    data = json.loads(result)
    assert data["status"] == "saved"
    assert "restart" in data["detail"].lower()


@pytest.mark.asyncio
async def test_set_action_rejects_invalid(config_tool):
    with patch(LOAD_CONFIG, return_value=Config()):
        result = await config_tool.execute(
            action="set", path="agents.defaults.context_mode", value="invalid"
        )
    assert "Error" in result


@pytest.mark.asyncio
async def test_set_action_missing_value(config_tool):
    result = await config_tool.execute(action="set", path="agents.defaults.temperature")
    assert "Error" in result


@pytest.mark.asyncio
async def test_list_action_returns_full_config(config_tool):
    with patch(LOAD_CONFIG, return_value=Config()):
        result = await config_tool.execute(action="list")
    assert "agents.defaults.temperature = 0.7" in result
    assert "gateway.port = 18790" in result


@pytest.mark.asyncio
async def test_diff_action_default_config(config_tool):
    with patch(LOAD_CONFIG, return_value=Config()):
        result = await config_tool.execute(action="diff")
    assert "defaults" in result.lower()


@pytest.mark.asyncio
async def test_diff_action_shows_differences(config_tool):
    config = Config()
    config.agents.defaults.temperature = 0.9

    with patch(LOAD_CONFIG, return_value=config):
        result = await config_tool.execute(action="diff")
    assert "temperature" in result
    assert "0.9" in result
