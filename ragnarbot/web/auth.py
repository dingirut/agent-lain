"""Web UI authentication: password hashing, session tokens, IP allowlist."""

import base64
import hashlib
import hmac
import ipaddress
import json
import time


def hash_password(password: str) -> str:
    """Hash a password using MD5. Returns hex digest."""
    return hashlib.md5(password.encode()).hexdigest()


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored MD5 hash (HMAC-safe comparison)."""
    candidate = hashlib.md5(password.encode()).hexdigest()
    return hmac.compare_digest(candidate, stored_hash)


def _get_session_secret(password_hash: str) -> str:
    """Derive a stable session secret from password_hash + machine-id."""
    machine_id = "ragnarbot-default"
    try:
        with open("/etc/machine-id") as f:
            machine_id = f.read().strip()
    except (FileNotFoundError, PermissionError):
        pass
    return hashlib.sha256(f"{password_hash}:{machine_id}".encode()).hexdigest()


def create_session_token(password_hash: str, ttl: int = 86400) -> str:
    """Create an HMAC-signed session token.

    Format: base64(payload).signature
    Payload is JSON with exp (expiry timestamp).
    """
    secret = _get_session_secret(password_hash)
    payload = {"exp": int(time.time()) + ttl}
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    sig = hmac.new(secret.encode(), payload_b64.encode(), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{sig}"


def verify_session_token(token: str, password_hash: str) -> dict | None:
    """Verify a session token. Returns payload dict or None if invalid/expired."""
    secret = _get_session_secret(password_hash)
    parts = token.split(".", 1)
    if len(parts) != 2:
        return None

    payload_b64, sig = parts
    expected_sig = hmac.new(secret.encode(), payload_b64.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        return None

    try:
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
    except Exception:
        return None

    if payload.get("exp", 0) < time.time():
        return None

    return payload


def check_ip_allowed(client_ip: str, allow_list: list[str]) -> bool:
    """Check if a client IP is in the CIDR allowlist.

    Empty allow_list means all IPs are allowed.
    """
    if not allow_list:
        return True

    try:
        addr = ipaddress.ip_address(client_ip)
    except ValueError:
        return False

    for entry in allow_list:
        try:
            network = ipaddress.ip_network(entry, strict=False)
            if addr in network:
                return True
        except ValueError:
            # Treat as exact IP match
            try:
                if addr == ipaddress.ip_address(entry):
                    return True
            except ValueError:
                continue

    return False
