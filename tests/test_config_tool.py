"""Tests for the config tool."""

import json
from unittest.mock import MagicMock, patch

import pytest

from ragnarbot.agent.tools.config_tool import ConfigTool
from ragnarbot.auth.credentials import (
    Credentials,
    ProviderCredentials,
    ProvidersCredentials,
)
from ragnarbot.config.schema import Config

LOAD_CONFIG = "ragnarbot.config.loader.load_config"
SAVE_CONFIG = "ragnarbot.config.loader.save_config"
LOAD_CREDS = "ragnarbot.auth.credentials.load_credentials"
LOAD_CREDS_HELPERS = "ragnarbot.agent.tools.secrets_helpers.load_credentials"
SAVE_CREDS = "ragnarbot.auth.credentials.save_credentials"


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
    creds = Credentials(
        providers=ProvidersCredentials(
            openai=ProviderCredentials(api_key="sk-openai-test"),
        ),
    )
    with (
        patch(LOAD_CONFIG, return_value=Config()),
        patch(SAVE_CONFIG),
        patch(LOAD_CREDS_HELPERS, return_value=creds),
    ):
        result = await config_tool.execute(
            action="set", path="agents.defaults.model", value="openai/gpt-5.2"
        )

    data = json.loads(result)
    assert data["status"] == "saved"
    assert "restart" in data["detail"].lower()


@pytest.mark.asyncio
async def test_set_model_rejects_unknown(config_tool):
    with patch(LOAD_CONFIG, return_value=Config()):
        result = await config_tool.execute(
            action="set", path="agents.defaults.model", value="openai/gpt-4"
        )
    assert "Error" in result
    assert "not available" in result


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


# --- Secrets integration tests ---


@pytest.mark.asyncio
async def test_get_secret_returns_value(config_tool):
    creds = Credentials(
        providers=ProvidersCredentials(
            anthropic=ProviderCredentials(api_key="sk-ant-test"),
        ),
    )
    with patch(LOAD_CREDS, return_value=creds):
        result = await config_tool.execute(
            action="get", path="secrets.providers.anthropic.api_key"
        )
    data = json.loads(result)
    assert data["value"] == "sk-ant-test"
    assert data["reload"] == "warm"


@pytest.mark.asyncio
async def test_set_secret_saves_credentials(config_tool):
    creds = Credentials()
    with (
        patch(LOAD_CREDS, return_value=creds),
        patch(SAVE_CREDS) as mock_save,
    ):
        result = await config_tool.execute(
            action="set", path="secrets.extra.github_token", value="ghp_xxx"
        )
    data = json.loads(result)
    assert data["status"] == "saved"
    mock_save.assert_called_once()
    saved_creds = mock_save.call_args[0][0]
    assert saved_creds.extra["github_token"] == "ghp_xxx"


@pytest.mark.asyncio
async def test_list_includes_secrets(config_tool):
    creds = Credentials(
        providers=ProvidersCredentials(
            anthropic=ProviderCredentials(api_key="sk-ant-test"),
        ),
    )
    with (
        patch(LOAD_CONFIG, return_value=Config()),
        patch(LOAD_CREDS, return_value=creds),
    ):
        result = await config_tool.execute(action="list")
    assert "secrets.providers.anthropic.api_key = ****" in result
    assert "secrets.providers.anthropic.oauth_key = [not set]" in result


@pytest.mark.asyncio
async def test_schema_includes_secrets(config_tool):
    creds = Credentials(
        providers=ProvidersCredentials(
            anthropic=ProviderCredentials(api_key="sk-ant-test"),
        ),
    )
    with (
        patch(LOAD_CONFIG, return_value=Config()),
        patch(LOAD_CREDS, return_value=creds),
    ):
        result = await config_tool.execute(action="schema")
    assert "secrets.providers.anthropic.api_key [set \u2713]" in result


@pytest.mark.asyncio
async def test_schema_secrets_only(config_tool):
    creds = Credentials()
    with patch(LOAD_CREDS, return_value=creds):
        result = await config_tool.execute(action="schema", path="secrets")
    assert "secrets.providers" in result
    assert "agents.defaults" not in result


@pytest.mark.asyncio
async def test_diff_excludes_secrets(config_tool):
    with patch(LOAD_CONFIG, return_value=Config()):
        result = await config_tool.execute(action="diff")
    assert "Secrets excluded from diff" in result


@pytest.mark.asyncio
async def test_set_config_blocked_by_missing_credential(config_tool):
    creds = Credentials()
    with (
        patch(LOAD_CONFIG, return_value=Config()),
        patch(LOAD_CREDS_HELPERS, return_value=creds),
    ):
        result = await config_tool.execute(
            action="set", path="agents.defaults.model", value="gemini/gemini-3-pro-preview"
        )
    assert "Error" in result
    assert "Blocked" in result
    assert "gemini" in result
