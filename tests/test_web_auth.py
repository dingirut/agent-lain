"""Web console password protection: hashing, tokens, middleware, endpoints."""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from aiohttp import web

from ragnarbot.auth.credentials import load_credentials
from ragnarbot.web.auth import (
    COOKIE_NAME,
    WebAuth,
    auth_middleware,
    hash_password,
    mint_token,
    verify_password,
    verify_token,
)


class JsonRequest(SimpleNamespace):
    async def json(self):
        return self.body


def make_server():
    """A WebServer shell with just the auth handlers wired."""
    from ragnarbot.web.server import WebServer

    server = object.__new__(WebServer)
    server.auth = WebAuth()
    return server


# ── primitives ───────────────────────────────────────────────────

def test_password_hash_roundtrip():
    stored = hash_password("hunter42")
    assert verify_password("hunter42", stored)
    assert not verify_password("hunter43", stored)
    assert not verify_password("hunter42", "garbage")
    assert not verify_password("hunter42", "")


def test_token_mint_verify_and_expiry():
    assert verify_token("s3cret", mint_token("s3cret"))
    assert not verify_token("other", mint_token("s3cret"))
    expired = f"{int(time.time()) - 10}.deadbeef"
    assert not verify_token("s3cret", expired)
    assert not verify_token("s3cret", "not-a-token")


# ── state persistence ────────────────────────────────────────────

def test_set_change_disable_password_persists_and_rotates():
    auth = WebAuth()
    assert not auth.enabled

    auth.set_password("first-pass")
    assert auth.enabled
    first_secret = auth.session_secret
    creds = load_credentials()
    assert creds.web.password_hash.startswith("pbkdf2$")
    assert verify_password("first-pass", creds.web.password_hash)

    auth.set_password("second-pass")
    assert auth.session_secret != first_secret  # old sessions invalidated

    auth.disable()
    assert not auth.enabled
    creds = load_credentials()
    assert creds.web.password_hash == "" and creds.web.session_secret == ""


def test_login_rate_limit():
    auth = WebAuth()
    auth.set_password("правильный")
    for _ in range(5):
        assert auth.check_login("wrong", "10.0.0.9") is False
    with pytest.raises(web.HTTPTooManyRequests):
        auth.check_login("правильный", "10.0.0.9")
    # a different address is unaffected
    assert auth.check_login("правильный", "10.0.0.10") is True


# ── middleware ───────────────────────────────────────────────────

def _request(path, cookies=None, auth=None):
    return SimpleNamespace(
        path=path,
        cookies=cookies or {},
        app={"rb_auth": auth},
    )


def test_middleware_gates_api_but_serves_spa():
    async def go():
        auth = WebAuth()
        auth.set_password("secret-pass")

        async def handler(_request):
            return web.json_response({"ok": True})

        # SPA shell + login endpoints stay open
        for path in ("/", "/assets/app.js", "/api/auth/status", "/api/auth/login"):
            response = await auth_middleware(_request(path, auth=auth), handler)
            assert response.status == 200, path

        # API and ws are gated
        for path in ("/api/models", "/api/sessions", "/ws"):
            with pytest.raises(web.HTTPUnauthorized):
                await auth_middleware(_request(path, auth=auth), handler)

        # a valid cookie unlocks
        cookie = {COOKIE_NAME: mint_token(auth.session_secret)}
        response = await auth_middleware(_request("/api/models", cookie, auth), handler)
        assert response.status == 200

        # disabled protection → everything open
        auth.disable()
        response = await auth_middleware(_request("/api/models", auth=auth), handler)
        assert response.status == 200

    asyncio.run(go())


# ── endpoints ────────────────────────────────────────────────────

def test_password_endpoint_set_change_disable():
    async def go():
        server = make_server()

        # set (no current needed while disabled)
        resp = await server._handle_auth_password(JsonRequest(body={"new_password": "secret-1"}))
        assert json.loads(resp.text)["protected"] is True
        assert COOKIE_NAME in resp.cookies

        # too short
        resp = await server._handle_auth_password(
            JsonRequest(body={"current_password": "secret-1", "new_password": "abc"})
        )
        assert resp.status == 400

        # change requires the correct current password
        resp = await server._handle_auth_password(
            JsonRequest(body={"current_password": "wrong", "new_password": "secret-2"})
        )
        assert resp.status == 403
        resp = await server._handle_auth_password(
            JsonRequest(body={"current_password": "secret-1", "new_password": "secret-2"})
        )
        assert json.loads(resp.text)["protected"] is True

        # disable
        resp = await server._handle_auth_password(
            JsonRequest(body={"current_password": "secret-2", "new_password": ""})
        )
        assert json.loads(resp.text)["protected"] is False
        assert not server.auth.enabled

    asyncio.run(go())


def test_login_endpoint():
    async def go():
        server = make_server()
        server.auth.set_password("secret-1")

        resp = await server._handle_auth_login(
            JsonRequest(body={"password": "nope"}, headers={}, remote="10.1.1.1")
        )
        assert resp.status == 401

        resp = await server._handle_auth_login(
            JsonRequest(body={"password": "secret-1"}, headers={}, remote="10.1.1.1")
        )
        assert resp.status == 200
        assert COOKIE_NAME in resp.cookies

    asyncio.run(go())
