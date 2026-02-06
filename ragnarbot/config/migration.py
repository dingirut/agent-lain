"""Smart config schema migration for ragnarbot."""

import json
from pathlib import Path
from typing import Any

from rich.console import Console

from ragnarbot.config.loader import convert_keys, convert_to_camel
from ragnarbot.config.schema import Config
from ragnarbot.auth.credentials import Credentials


# Paths that contain sensitive data (warn before removing)
SENSITIVE_PATHS = {"providers", "api_key", "oauth_key", "bot_token", "api_key_url"}


def _deep_diff(
    existing: dict, default: dict, path: str = ""
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Walk two dicts, return (added, removed) as flat path->value maps."""
    added: dict[str, Any] = {}
    removed: dict[str, Any] = {}

    all_keys = set(existing.keys()) | set(default.keys())
    for key in all_keys:
        current_path = f"{path}.{key}" if path else key
        in_existing = key in existing
        in_default = key in default

        if in_default and not in_existing:
            added[current_path] = default[key]
        elif in_existing and not in_default:
            removed[current_path] = existing[key]
        elif in_existing and in_default:
            if isinstance(existing[key], dict) and isinstance(default[key], dict):
                sub_added, sub_removed = _deep_diff(existing[key], default[key], current_path)
                added.update(sub_added)
                removed.update(sub_removed)

    return added, removed


def _is_sensitive(path: str) -> bool:
    """Check if a config path contains sensitive data."""
    parts = path.split(".")
    return any(p in SENSITIVE_PATHS for p in parts)


def _has_meaningful_data(value: Any) -> bool:
    """Check if a value contains non-default data."""
    if value is None:
        return False
    if isinstance(value, str) and value == "":
        return False
    if isinstance(value, bool) and value is False:
        return False
    if isinstance(value, dict):
        return any(_has_meaningful_data(v) for v in value.values())
    if isinstance(value, list) and len(value) == 0:
        return False
    return True


def _set_nested(d: dict, path: str, value: Any) -> None:
    """Set a value at a dotted path in a nested dict."""
    keys = path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _delete_nested(d: dict, path: str) -> None:
    """Delete a key at a dotted path in a nested dict."""
    keys = path.split(".")
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            return
        d = d[key]
    d.pop(keys[-1], None)


def migrate_config(
    config_path: Path,
    console: Console | None = None,
    auto_confirm: bool = False,
) -> dict:
    """Migrate an existing config file to the current schema.

    Returns the migrated config as a snake_case dict.
    """
    with open(config_path) as f:
        raw = json.load(f)
    existing = convert_keys(raw)

    default = Config().model_dump()
    added, removed = _deep_diff(existing, default)

    # Apply additions (safe — just new defaults)
    for path, value in added.items():
        _set_nested(existing, path, value)

    # Handle removals
    sensitive_removals = {}
    for path, value in removed.items():
        if _has_meaningful_data(value) and _is_sensitive(path):
            sensitive_removals[path] = value
        else:
            _delete_nested(existing, path)

    if sensitive_removals and console and not auto_confirm:
        console.print("\n[yellow]Schema migration found removed fields with data:[/yellow]")
        for path, value in sensitive_removals.items():
            display = value
            if isinstance(value, str) and len(value) > 20:
                display = value[:4] + "****" + value[-4:]
            console.print(f"  {path}: {display}")
        console.print("[dim]These fields are no longer in the schema and will be removed.[/dim]")
        # In interactive mode, we keep them to let the user decide
        # For onboarding, auto_confirm=True skips this
    elif sensitive_removals and auto_confirm:
        for path in sensitive_removals:
            _delete_nested(existing, path)

    return existing


def migrate_credentials(
    creds_path: Path,
    console: Console | None = None,
    auto_confirm: bool = False,
) -> dict:
    """Migrate an existing credentials file to the current schema.

    Returns the migrated credentials as a snake_case dict.
    """
    with open(creds_path) as f:
        raw = json.load(f)
    existing = convert_keys(raw)

    default = Credentials().model_dump()
    added, removed = _deep_diff(existing, default)

    for path, value in added.items():
        _set_nested(existing, path, value)

    for path, value in removed.items():
        if not (_has_meaningful_data(value) and _is_sensitive(path)):
            _delete_nested(existing, path)

    return existing
