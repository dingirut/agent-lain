"""Tests for web UI authentication module."""

import pytest

from ragnarbot.web.auth import (
    check_ip_allowed,
    create_session_token,
    hash_password,
    verify_password,
    verify_session_token,
)


class TestPasswordHashing:
    def test_hash_password(self):
        h = hash_password("mypassword")
        assert len(h) == 32  # MD5 hex digest
        assert h == hash_password("mypassword")  # deterministic

    def test_verify_password_correct(self):
        h = hash_password("secret123")
        assert verify_password("secret123", h) is True

    def test_verify_password_wrong(self):
        h = hash_password("secret123")
        assert verify_password("wrong", h) is False

    def test_verify_password_empty(self):
        h = hash_password("")
        assert verify_password("", h) is True
        assert verify_password("x", h) is False


class TestSessionTokens:
    def test_create_and_verify(self):
        pw_hash = hash_password("test")
        token = create_session_token(pw_hash, ttl=3600)
        assert "." in token

        payload = verify_session_token(token, pw_hash)
        assert payload is not None
        assert "exp" in payload

    def test_verify_invalid_token(self):
        pw_hash = hash_password("test")
        assert verify_session_token("invalid", pw_hash) is None
        assert verify_session_token("abc.def", pw_hash) is None
        assert verify_session_token("", pw_hash) is None

    def test_verify_wrong_secret(self):
        pw_hash1 = hash_password("test1")
        pw_hash2 = hash_password("test2")
        token = create_session_token(pw_hash1)
        assert verify_session_token(token, pw_hash2) is None

    def test_expired_token(self):
        pw_hash = hash_password("test")
        token = create_session_token(pw_hash, ttl=-1)
        assert verify_session_token(token, pw_hash) is None


class TestIPAllowlist:
    def test_empty_list_allows_all(self):
        assert check_ip_allowed("192.168.1.1", []) is True
        assert check_ip_allowed("10.0.0.1", []) is True

    def test_exact_ip_match(self):
        assert check_ip_allowed("192.168.1.1", ["192.168.1.1"]) is True
        assert check_ip_allowed("192.168.1.2", ["192.168.1.1"]) is False

    def test_cidr_match(self):
        allow = ["192.168.1.0/24"]
        assert check_ip_allowed("192.168.1.1", allow) is True
        assert check_ip_allowed("192.168.1.254", allow) is True
        assert check_ip_allowed("192.168.2.1", allow) is False

    def test_multiple_entries(self):
        allow = ["192.168.1.0/24", "10.0.0.0/8"]
        assert check_ip_allowed("192.168.1.5", allow) is True
        assert check_ip_allowed("10.5.5.5", allow) is True
        assert check_ip_allowed("172.16.0.1", allow) is False

    def test_ipv6(self):
        assert check_ip_allowed("::1", ["::1"]) is True
        assert check_ip_allowed("::1", ["::2"]) is False

    def test_invalid_client_ip(self):
        assert check_ip_allowed("not-an-ip", ["192.168.1.0/24"]) is False

    def test_invalid_allowlist_entry(self):
        # Invalid entries are skipped, but valid ones still work
        assert check_ip_allowed("192.168.1.1", ["bad-entry", "192.168.1.0/24"]) is True
        assert check_ip_allowed("192.168.1.1", ["bad-entry"]) is False
