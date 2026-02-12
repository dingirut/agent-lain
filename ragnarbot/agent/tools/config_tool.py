"""Config tool for viewing and modifying bot configuration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ragnarbot.agent.tools.base import Tool

if TYPE_CHECKING:
    from ragnarbot.agent.loop import AgentLoop


class ConfigTool(Tool):
    """Tool to view and modify bot configuration at runtime."""

    name = "config"
    description = (
        "View and modify bot configuration. "
        "Actions: schema (discover fields), get (read value), set (change value), "
        "list (all current values), diff (non-default values)."
    )

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["schema", "get", "set", "list", "diff"],
                "description": "Action to perform",
            },
            "path": {
                "type": "string",
                "description": "Dot-notation path (e.g. 'agents.defaults.model'). For schema/get/set.",
            },
            "value": {
                "type": "string",
                "description": "New value to set (for 'set' action). Strings are auto-coerced to the target type.",
            },
        },
        "required": ["action"],
    }

    def __init__(self, agent: AgentLoop):
        self._agent = agent

    async def execute(
        self,
        action: str,
        path: str | None = None,
        value: str | None = None,
        **kwargs: Any,
    ) -> str:
        if action == "schema":
            return self._action_schema(path)
        elif action == "get":
            return self._action_get(path)
        elif action == "set":
            return self._action_set(path, value)
        elif action == "list":
            return self._action_list()
        elif action == "diff":
            return self._action_diff()
        return f"Unknown action: {action}"

    def _action_schema(self, path: str | None) -> str:
        from ragnarbot.config.loader import load_config
        from ragnarbot.config.path_utils import get_all_paths, get_field_meta
        from ragnarbot.config.schema import Config

        config = load_config()
        all_paths = get_all_paths(config)
        lines = []
        for p in sorted(all_paths.keys()):
            if path and not p.startswith(path):
                continue
            try:
                meta = get_field_meta(Config, p)
                reload_tag = f" [{meta['reload']}]" if meta.get("reload") else ""
                pattern = f" pattern={meta['pattern']}" if meta.get("pattern") else ""
                label = f" — {meta['label']}" if meta.get("label") else ""
                lines.append(
                    f"{p}: {meta['type']} = {meta['default']!r}{reload_tag}{pattern}{label}"
                )
            except ValueError:
                lines.append(f"{p}: (metadata unavailable)")

        if not lines:
            return f"No fields matching '{path}'" if path else "No fields found"
        return "\n".join(lines)

    def _action_get(self, path: str | None) -> str:
        if not path:
            return "Error: 'path' is required for get action"

        from ragnarbot.config.loader import load_config
        from ragnarbot.config.path_utils import get_by_path, get_field_meta
        from ragnarbot.config.schema import Config

        try:
            config = load_config()
            value = get_by_path(config, path)
            meta = get_field_meta(Config, path)
            return json.dumps({
                "path": path,
                "value": value,
                "type": meta.get("type", "unknown"),
                "reload": meta.get("reload", "unknown"),
                "label": meta.get("label", ""),
            }, default=str)
        except ValueError as e:
            return f"Error: {e}"

    def _action_set(self, path: str | None, value: str | None) -> str:
        if not path:
            return "Error: 'path' is required for set action"
        if value is None:
            return "Error: 'value' is required for set action"

        from ragnarbot.config.loader import load_config, save_config
        from ragnarbot.config.path_utils import get_by_path, get_field_meta, set_by_path
        from ragnarbot.config.schema import Config

        try:
            config = load_config()
            old_value = get_by_path(config, path)
            set_by_path(config, path, value)
            save_config(config)
            new_value = get_by_path(config, path)
            meta = get_field_meta(Config, path)
            reload_level = meta.get("reload", "unknown")

            result = {
                "path": path,
                "old_value": old_value,
                "new_value": new_value,
            }

            if reload_level == "hot":
                apply_msg = self._apply_hot_reload(path, new_value)
                result["status"] = "applied"
                result["detail"] = apply_msg or "Applied immediately."
            elif reload_level == "warm":
                result["status"] = "saved"
                result["detail"] = "Saved. Use the `restart` tool to apply."
            else:
                result["status"] = "saved"
                result["detail"] = "Saved. Requires full re-onboard to apply."

            return json.dumps(result, default=str)
        except ValueError as e:
            return f"Error: {e}"

    def _action_list(self) -> str:
        from ragnarbot.config.loader import load_config
        from ragnarbot.config.path_utils import get_all_paths

        config = load_config()
        all_paths = get_all_paths(config)
        lines = [f"{p} = {v!r}" for p, v in sorted(all_paths.items())]
        return "\n".join(lines)

    def _action_diff(self) -> str:
        from ragnarbot.config.loader import load_config
        from ragnarbot.config.path_utils import get_all_paths
        from ragnarbot.config.schema import Config

        current = load_config()
        defaults = Config()
        current_paths = get_all_paths(current)
        default_paths = get_all_paths(defaults)

        diffs = []
        for p in sorted(current_paths.keys()):
            cur = current_paths[p]
            dflt = default_paths.get(p)
            if cur != dflt:
                diffs.append(f"{p}: {dflt!r} → {cur!r}")

        if not diffs:
            return "All values are at defaults."
        return "\n".join(diffs)

    def _apply_hot_reload(self, path: str, value: Any) -> str | None:
        """Apply a hot-reloadable config change to live agent/tool state."""
        agent = self._agent

        if path == "agents.defaults.temperature":
            agent.provider.set_temperature(float(value))
            return "Temperature updated on provider."

        if path == "agents.defaults.max_tokens":
            agent.provider.set_max_tokens(int(value))
            return "Max tokens updated on provider."

        if path == "agents.defaults.stream_steps":
            agent.stream_steps = bool(value) if isinstance(value, bool) else str(value).lower() in ("true", "1")
            return "Stream steps updated."

        if path == "agents.defaults.debounce_seconds":
            agent.debounce_seconds = float(value)
            return "Debounce seconds updated."

        if path == "agents.defaults.context_mode":
            agent.context_mode = str(value)
            return "Context mode updated."

        if path == "agents.defaults.max_context_tokens":
            tokens = int(value)
            agent.max_context_tokens = tokens
            agent.cache_manager.max_context_tokens = tokens
            agent.compactor.max_context_tokens = tokens
            return "Max context tokens updated."

        if path in ("tools.web.search.engine", "tools.web.search.max_results"):
            return self._reload_web_search()

        if path == "tools.exec.timeout":
            exec_tool = agent.tools.get("exec")
            if exec_tool and hasattr(exec_tool, "timeout"):
                exec_tool.timeout = int(value)
            return "Exec timeout updated."

        if path == "tools.exec.restrict_to_workspace":
            exec_tool = agent.tools.get("exec")
            if exec_tool and hasattr(exec_tool, "restrict_to_workspace"):
                val = bool(value) if isinstance(value, bool) else str(value).lower() in ("true", "1")
                exec_tool.restrict_to_workspace = val
            return "Exec restrict_to_workspace updated."

        return None

    def _reload_web_search(self) -> str:
        """Re-register WebSearchTool with current config values."""
        from ragnarbot.agent.tools.web import WebSearchTool
        from ragnarbot.config.loader import load_config

        config = load_config()
        agent = self._agent

        agent.tools.unregister("web_search")
        agent.tools.register(WebSearchTool(
            engine=config.tools.web.search.engine,
            api_key=agent.brave_api_key,
            max_results=config.tools.web.search.max_results,
        ))
        return "Web search tool re-registered."
