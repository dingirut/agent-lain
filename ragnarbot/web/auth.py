"""Optional password protection for the web console.

Single-user model: one password guards the whole console. The PBKDF2 hash
and the HMAC session secret live in the credentials file (0600). Session
cookies are stateless (`exp.signature`); changing or disabling the password
rotates the secret, which invalidates every existing session.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from typing import Any

from aiohttp import web

COOKIE_NAME = "rb_session"
SESSION_TTL_S = 30 * 24 * 3600
MIN_PASSWORD_LEN = 6

_PBKDF2_ITERS = 200_000
_LOGIN_WINDOW_S = 60
_LOGIN_MAX_FAILURES = 5

# Always reachable: the login flow itself. The SPA shell and its assets are
# also served unauthenticated (the UI code is public anyway) so the login
# screen can render; everything under /api and /ws requires a session.
PUBLIC_API_PATHS = frozenset({"/api/auth/status", "/api/auth/login"})


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt), _PBKDF2_ITERS
    ).hex()
    return f"pbkdf2${_PBKDF2_ITERS}${salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt, digest = stored.split("$")
        if algo != "pbkdf2":
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt), int(iters)
        ).hex()
        return hmac.compare_digest(candidate, digest)
    except (ValueError, TypeError):
        return False


def mint_token(secret: str, ttl_s: int = SESSION_TTL_S) -> str:
    expires = str(int(time.time()) + ttl_s)
    sig = hmac.new(secret.encode(), expires.encode(), hashlib.sha256).hexdigest()
    return f"{expires}.{sig}"


def verify_token(secret: str, token: str) -> bool:
    try:
        expires, sig = token.split(".", 1)
        if int(expires) < time.time():
            return False
        expected = hmac.new(secret.encode(), expires.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, sig)
    except (ValueError, TypeError):
        return False


class WebAuth:
    """Holds the live password state and answers per-request auth checks."""

    def __init__(self) -> None:
        self.password_hash = ""
        self.session_secret = ""
        self.auto_lock_minutes = 0
        self._failures: dict[str, list[float]] = {}
        self.reload()

    # ── persistence ──────────────────────────────────────────────

    def reload(self) -> None:
        from ragnarbot.auth.credentials import load_credentials

        creds = load_credentials()
        web_creds = getattr(creds, "web", None)
        self.password_hash = getattr(web_creds, "password_hash", "") if web_creds else ""
        self.session_secret = getattr(web_creds, "session_secret", "") if web_creds else ""
        self.auto_lock_minutes = getattr(web_creds, "auto_lock_minutes", 0) if web_creds else 0

    def _persist(self) -> None:
        from ragnarbot.auth.credentials import load_credentials, save_credentials

        creds = load_credentials()
        creds.web.password_hash = self.password_hash
        creds.web.session_secret = self.session_secret
        creds.web.auto_lock_minutes = self.auto_lock_minutes
        save_credentials(creds)

    # ── state changes ────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return bool(self.password_hash)

    def set_password(self, password: str) -> None:
        """Set or change the password; rotates the secret (logs out everyone)."""
        self.password_hash = hash_password(password)
        self.session_secret = secrets.token_hex(32)
        self._persist()

    def disable(self) -> None:
        self.password_hash = ""
        self.session_secret = ""
        self.auto_lock_minutes = 0
        self._persist()

    def set_auto_lock(self, minutes: int) -> None:
        self.auto_lock_minutes = minutes
        self._persist()

    # ── request checks ───────────────────────────────────────────

    def check_login(self, password: str, remote: str) -> bool:
        """Verify a login attempt with a small per-IP failure budget."""
        now = time.time()
        attempts = [t for t in self._failures.get(remote, []) if now - t < _LOGIN_WINDOW_S]
        if len(attempts) >= _LOGIN_MAX_FAILURES:
            self._failures[remote] = attempts
            raise web.HTTPTooManyRequests(reason="too many attempts — wait a minute")
        if self.enabled and verify_password(password, self.password_hash):
            self._failures.pop(remote, None)
            return True
        attempts.append(now)
        self._failures[remote] = attempts
        return False

    def authenticated(self, request: web.Request) -> bool:
        token = request.cookies.get(COOKIE_NAME, "")
        return bool(self.session_secret) and verify_token(self.session_secret, token)

    def issue_cookie(self, response: web.StreamResponse) -> None:
        response.set_cookie(
            COOKIE_NAME,
            mint_token(self.session_secret),
            max_age=SESSION_TTL_S,
            httponly=True,
            samesite="Lax",
            path="/",
        )

    @staticmethod
    def clear_cookie(response: web.StreamResponse) -> None:
        response.del_cookie(COOKIE_NAME, path="/")


@web.middleware
async def auth_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    auth: WebAuth | None = request.app.get("rb_auth")
    if auth is None or not auth.enabled:
        return await handler(request)
    path = request.path
    guarded = path.startswith("/api/") or path == "/ws"
    if not guarded or path in PUBLIC_API_PATHS or auth.authenticated(request):
        return await handler(request)
    raise web.HTTPUnauthorized(reason="login required")
