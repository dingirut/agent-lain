"""Tests for the restart tool."""

import json
from unittest.mock import MagicMock

import pytest

from ragnarbot.agent.tools.restart import RestartTool


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent._restart_requested = False

    def set_restart():
        agent._restart_requested = True

    agent.request_restart = MagicMock(side_effect=set_restart)
    return agent


@pytest.fixture
def restart_tool(mock_agent):
    return RestartTool(agent=mock_agent)


@pytest.mark.asyncio
async def test_execute_sets_restart_flag(restart_tool, mock_agent):
    result = await restart_tool.execute()
    data = json.loads(result)
    assert data["status"] == "restart_scheduled"
    mock_agent.request_restart.assert_called_once()
    assert mock_agent._restart_requested is True


@pytest.mark.asyncio
async def test_execute_returns_json(restart_tool):
    result = await restart_tool.execute()
    data = json.loads(result)
    assert "status" in data
    assert "note" in data


def test_tool_metadata(restart_tool):
    assert restart_tool.name == "restart"
    assert "restart" in restart_tool.description.lower()
    assert restart_tool.parameters == {"type": "object", "properties": {}}
