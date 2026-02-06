"""Tests for config schema migration."""

import json
from pathlib import Path

import pytest

from ragnarbot.config.migration import (
    _deep_diff,
    _has_meaningful_data,
    _is_sensitive,
    _set_nested,
    _delete_nested,
    migrate_config,
    migrate_credentials,
)


class TestDeepDiff:
    def test_no_diff(self):
        a = {"x": 1, "y": 2}
        b = {"x": 1, "y": 2}
        added, removed = _deep_diff(a, b)
        assert added == {}
        assert removed == {}

    def test_added_key(self):
        existing = {"x": 1}
        default = {"x": 1, "y": 2}
        added, removed = _deep_diff(existing, default)
        assert added == {"y": 2}
        assert removed == {}

    def test_removed_key(self):
        existing = {"x": 1, "old": "val"}
        default = {"x": 1}
        added, removed = _deep_diff(existing, default)
        assert added == {}
        assert removed == {"old": "val"}

    def test_nested_diff(self):
        existing = {"a": {"b": 1}}
        default = {"a": {"b": 1, "c": 2}}
        added, removed = _deep_diff(existing, default)
        assert added == {"a.c": 2}
        assert removed == {}

    def test_nested_removal(self):
        existing = {"a": {"b": 1, "old": "x"}}
        default = {"a": {"b": 1}}
        added, removed = _deep_diff(existing, default)
        assert removed == {"a.old": "x"}


class TestHelpers:
    def test_meaningful_data_string(self):
        assert _has_meaningful_data("hello") is True
        assert _has_meaningful_data("") is False

    def test_meaningful_data_dict(self):
        assert _has_meaningful_data({"key": "val"}) is True
        assert _has_meaningful_data({"key": ""}) is False

    def test_meaningful_data_none(self):
        assert _has_meaningful_data(None) is False

    def test_meaningful_data_list(self):
        assert _has_meaningful_data([1, 2]) is True
        assert _has_meaningful_data([]) is False

    def test_is_sensitive(self):
        assert _is_sensitive("providers.anthropic.api_key") is True
        assert _is_sensitive("providers.openai.oauth_key") is True
        assert _is_sensitive("channels.telegram.bot_token") is True
        assert _is_sensitive("agents.defaults.model") is False

    def test_set_nested(self):
        d = {"a": {"b": 1}}
        _set_nested(d, "a.c", 2)
        assert d == {"a": {"b": 1, "c": 2}}

    def test_set_nested_creates_parents(self):
        d = {}
        _set_nested(d, "a.b.c", 1)
        assert d == {"a": {"b": {"c": 1}}}

    def test_delete_nested(self):
        d = {"a": {"b": 1, "c": 2}}
        _delete_nested(d, "a.c")
        assert d == {"a": {"b": 1}}

    def test_delete_nested_missing(self):
        d = {"a": 1}
        _delete_nested(d, "b.c")  # should not raise
        assert d == {"a": 1}


class TestMigrateConfig:
    def test_adds_missing_fields(self, tmp_path):
        config = {"agents": {"defaults": {"model": "anthropic/claude-opus-4-6"}}}
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        result = migrate_config(path, auto_confirm=True)
        # Should have added gateway, channels, tools, etc.
        assert "gateway" in result
        assert "channels" in result
        assert "tools" in result

    def test_preserves_existing_values(self, tmp_path):
        config = {
            "agents": {
                "defaults": {
                    "model": "openai/gpt-5.2",
                    "maxTokens": 4096,
                }
            }
        }
        # Write as camelCase (like real config)
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        result = migrate_config(path, auto_confirm=True)
        assert result["agents"]["defaults"]["model"] == "openai/gpt-5.2"
        assert result["agents"]["defaults"]["max_tokens"] == 4096

    def test_removes_unknown_non_sensitive(self, tmp_path):
        config = {
            "agents": {"defaults": {"model": "anthropic/claude-opus-4-6", "oldField": "test"}},
        }
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        result = migrate_config(path, auto_confirm=True)
        assert "old_field" not in result.get("agents", {}).get("defaults", {})


class TestMigrateCredentials:
    def test_adds_missing_sections(self, tmp_path):
        creds = {"providers": {"anthropic": {"apiKey": "sk-test"}}}
        path = tmp_path / "creds.json"
        path.write_text(json.dumps(creds))

        result = migrate_credentials(path, auto_confirm=True)
        assert "services" in result
        assert "channels" in result

    def test_preserves_existing_keys(self, tmp_path):
        creds = {
            "providers": {
                "anthropic": {"apiKey": "sk-test-123"},
                "openai": {"apiKey": ""},
                "gemini": {"apiKey": ""},
            },
            "services": {"transcription": {"apiKey": ""}, "webSearch": {"apiKey": ""}},
            "channels": {"telegram": {"botToken": ""}},
        }
        path = tmp_path / "creds.json"
        path.write_text(json.dumps(creds))

        result = migrate_credentials(path, auto_confirm=True)
        assert result["providers"]["anthropic"]["api_key"] == "sk-test-123"
