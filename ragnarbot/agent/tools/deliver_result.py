"""Deliver result tool for isolated cron jobs."""

from typing import Any

from ragnarbot.agent.tools.base import Tool


class DeliverResultTool(Tool):
    """Tool that captures output from isolated cron jobs."""

    def __init__(self):
        self._result: str | None = None

    @property
    def name(self) -> str:
        return "deliver_result"

    @property
    def description(self) -> str:
        return (
            "Deliver the final result of a cron job to the user. "
            "This is the ONLY way the user sees your output in isolated mode."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The result content to deliver to the user",
                },
            },
            "required": ["content"],
        }

    async def execute(self, content: str = "", **kwargs: Any) -> str:
        self._result = content
        return "Result captured."

    @property
    def result(self) -> str | None:
        return self._result

    def reset(self) -> None:
        self._result = None
