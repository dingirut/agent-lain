"""Restart tool for scheduling graceful gateway restarts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ragnarbot.agent.tools.base import Tool

if TYPE_CHECKING:
    from ragnarbot.agent.loop import AgentLoop


class RestartTool(Tool):
    """Tool to schedule a graceful gateway restart."""

    name = "restart"
    description = (
        "Schedule a graceful gateway restart. "
        "The restart happens after the current response is fully sent. "
        "Use after changing 'warm' config values that require a restart to apply."
    )

    parameters = {
        "type": "object",
        "properties": {},
    }

    def __init__(self, agent: AgentLoop):
        self._agent = agent

    async def execute(self, **kwargs: Any) -> str:
        self._agent.request_restart()
        return json.dumps({
            "status": "restart_scheduled",
            "note": "Gateway will restart after this response completes.",
        })
