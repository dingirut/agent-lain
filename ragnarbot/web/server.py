"""FastAPI server for ragnarbot Web UI."""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from ragnarbot.bus.events import InboundMessage, MediaAttachment, OutboundMessage
from ragnarbot.config.schema import WebConfig
from ragnarbot.web.auth import (
    check_ip_allowed,
    create_session_token,
    verify_password,
    verify_session_token,
)

if TYPE_CHECKING:
    from ragnarbot.agent.loop import AgentLoop
    from ragnarbot.bus.queue import MessageBus
    from ragnarbot.cron.service import CronService
    from ragnarbot.heartbeat.service import HeartbeatService
    from ragnarbot.media.manager import MediaManager
    from ragnarbot.providers.transcription import TranscriptionProvider
    from ragnarbot.session.manager import SessionManager

STATIC_DIR = Path(__file__).parent / "static"
SESSION_COOKIE = "ragnarbot_session"


# ── Middleware ────────────────────────────────────────────────


class IPAllowlistMiddleware(BaseHTTPMiddleware):
    """Block requests from IPs not in the allowlist."""

    async def dispatch(self, request: Request, call_next):
        allow_list = request.app.state.allow_from
        client_ip = request.client.host if request.client else "0.0.0.0"
        if not check_ip_allowed(client_ip, allow_list):
            return JSONResponse({"error": "Forbidden"}, status_code=403)
        return await call_next(request)


class AuthMiddleware(BaseHTTPMiddleware):
    """Require valid session token for /api/* and /ws (except /api/auth/login)."""

    OPEN_PATHS = {"/api/auth/login", "/", "/favicon.ico"}
    OPEN_PREFIXES = ("/static/",)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Allow open paths
        if path in self.OPEN_PATHS or any(path.startswith(p) for p in self.OPEN_PREFIXES):
            request.state.session_user = None
            return await call_next(request)

        # Check for auth on API and WS paths
        if path.startswith("/api/") or path == "/ws":
            password_hash = request.app.state.password_hash
            token = request.cookies.get(SESSION_COOKIE, "")
            payload = verify_session_token(token, password_hash)
            if not payload:
                return JSONResponse({"error": "Unauthorized"}, status_code=401)
            request.state.session_user = payload.get("user", "web_user")
            return await call_next(request)

        request.state.session_user = None
        return await call_next(request)


# ── WebSocket Manager ────────────────────────────────────────


class ConnectionManager:
    """Manage WebSocket connections per chat_id."""

    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = {}

    def register(self, chat_id: str, ws: WebSocket):
        self._connections.setdefault(chat_id, []).append(ws)

    def unregister(self, chat_id: str, ws: WebSocket):
        conns = self._connections.get(chat_id, [])
        if ws in conns:
            conns.remove(ws)
            if not conns:
                del self._connections[chat_id]

    async def send_to_chat(self, chat_id: str, data: dict):
        conns = self._connections.get(chat_id, [])
        dead = []
        for ws in conns:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.unregister(chat_id, ws)

    def get_chat_ids(self) -> list[str]:
        return list(self._connections.keys())


# ── Server ────────────────────────────────────────────────────


class WebServer:
    """FastAPI + uvicorn server for the ragnarbot Web UI."""

    def __init__(
        self,
        config: WebConfig,
        password_hash: str,
        bus: "MessageBus",
        agent_loop: "AgentLoop | None" = None,
        cron_service: "CronService | None" = None,
        heartbeat_service: "HeartbeatService | None" = None,
        media_manager: "MediaManager | None" = None,
        transcription_provider: "TranscriptionProvider | None" = None,
        session_manager: "SessionManager | None" = None,
        workspace: Path | None = None,
    ):
        self.config = config
        self.password_hash = password_hash
        self.bus = bus
        self.agent_loop = agent_loop
        self.media_manager = media_manager
        self.transcriber = transcription_provider
        self.workspace = workspace
        self.connections = ConnectionManager()

        # Register web download callback so download_file tool works
        if media_manager:
            async def _web_download(file_id: str) -> tuple[bytes, str]:
                """Read file from disk by path (file_id = saved path)."""
                p = Path(file_id)
                if p.exists():
                    return p.read_bytes(), p.name
                raise FileNotFoundError(f"File not found: {file_id}")

            media_manager.register_download_callback("web", _web_download)
        self._server: uvicorn.Server | None = None
        self._outbound_task: asyncio.Task | None = None

        self.app = FastAPI(title="ragnarbot", docs_url=None, redoc_url=None)

        # Global exception handler — log unhandled errors via loguru
        @self.app.exception_handler(Exception)
        async def _unhandled_exception(request: Request, exc: Exception):
            logger.opt(exception=exc).error(
                f"Unhandled error in {request.method} {request.url.path}"
            )
            return JSONResponse(
                {"error": "Internal server error"},
                status_code=500,
            )

        # Store references on app.state for route handlers
        self.app.state.password_hash = password_hash
        self.app.state.allow_from = config.allow_from
        self.app.state.agent_loop = agent_loop
        self.app.state.cron_service = cron_service
        self.app.state.heartbeat_service = heartbeat_service
        self.app.state.session_manager = session_manager
        self.app.state.workspace = workspace or Path.home() / ".ragnarbot" / "workspace"
        self.app.state.connections = self.connections

        # Middleware (order matters: IP check first, then auth)
        self.app.add_middleware(AuthMiddleware)
        self.app.add_middleware(IPAllowlistMiddleware)

        self._register_routes()

    def _register_routes(self):
        """Register all HTTP and WebSocket routes."""
        app = self.app

        # ── Static / SPA ──────────────────────────────────────

        @app.get("/")
        async def index():
            html_path = STATIC_DIR / "index.html"
            if html_path.exists():
                return FileResponse(html_path, media_type="text/html")
            return Response("Web UI not found", status_code=404)

        @app.get("/static/{path:path}")
        async def static_file(path: str):
            file_path = (STATIC_DIR / path).resolve()
            if not str(file_path).startswith(str(STATIC_DIR.resolve())):
                return Response("Forbidden", status_code=403)
            if file_path.exists():
                return FileResponse(file_path)
            return Response("Not found", status_code=404)

        # ── Auth ──────────────────────────────────────────────

        @app.post("/api/auth/login")
        async def login(request: Request):
            body = await request.json()
            password = body.get("password", "")
            if not verify_password(password, self.password_hash):
                return JSONResponse({"error": "Invalid password"}, status_code=401)

            token = create_session_token(self.password_hash)
            response = JSONResponse({"ok": True})
            response.set_cookie(
                SESSION_COOKIE, token,
                httponly=True, samesite="strict", max_age=86400,
            )
            return response

        @app.post("/api/auth/logout")
        async def logout():
            response = JSONResponse({"ok": True})
            response.delete_cookie(SESSION_COOKIE)
            return response

        @app.get("/api/auth/check")
        async def check_auth(request: Request):
            return {"ok": True, "user": request.state.session_user}

        # ── Media serving ─────────────────────────────────────

        @app.get("/api/media/{path:path}")
        async def serve_media(path: str, request: Request):
            if not self.media_manager:
                return Response("Media not available", status_code=503)
            media_base = self.media_manager._base_dir
            file_path = (media_base / path).resolve()
            if not str(file_path).startswith(str(media_base.resolve())):
                return Response("Forbidden", status_code=403)
            if file_path.exists():
                return FileResponse(file_path)
            return Response("Not found", status_code=404)

        # ── Artifacts (workspace HTML files) ──────────────────

        @app.get("/api/artifacts/{path:path}")
        async def serve_artifact(path: str, request: Request):
            """Serve HTML/CSS/JS files from the workspace for preview."""
            if not self.workspace:
                return Response("Workspace not available", status_code=503)
            file_path = (self.workspace / path).resolve()
            if not str(file_path).startswith(str(self.workspace.resolve())):
                return Response("Forbidden", status_code=403)
            if not file_path.exists():
                return Response("Not found", status_code=404)
            return FileResponse(file_path)

        # ── File upload ───────────────────────────────────────

        @app.post("/api/upload")
        async def upload_file(request: Request):
            from starlette.datastructures import UploadFile as StarletteUploadFile
            form = await request.form()
            file = form.get("file")
            if not file or not hasattr(file, "read"):
                return JSONResponse({"error": "No file"}, status_code=400)
            data = await file.read()
            filename = getattr(file, "filename", "upload")
            # Return base64 for WebSocket injection
            b64 = base64.b64encode(data).decode()
            return {"filename": filename, "size": len(data), "data": b64}

        # ── Dashboard API routers ─────────────────────────────

        from ragnarbot.web.api import (
            extensions_router,
            jobs_router,
            logs_router,
            memory_router,
            routines_router,
            sessions_router,
            skills_router,
        )
        app.include_router(sessions_router)
        app.include_router(jobs_router)
        app.include_router(routines_router)
        app.include_router(extensions_router)
        app.include_router(memory_router)
        app.include_router(skills_router)
        app.include_router(logs_router)

        # ── WebSocket ─────────────────────────────────────────

        @app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            # Auth check (cookie)
            token = ws.cookies.get(SESSION_COOKIE, "")
            payload = verify_session_token(token, self.password_hash)
            if not payload:
                await ws.close(code=4001, reason="Unauthorized")
                return

            await ws.accept()
            chat_id = payload.get("user", "web_user")
            self.connections.register(chat_id, ws)
            logger.info(f"WebSocket connected: {chat_id}")

            # Send history on connect
            await self._send_history(ws, chat_id)

            try:
                while True:
                    raw = await ws.receive_text()
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        await ws.send_json({"type": "error", "message": "Invalid JSON"})
                        continue

                    logger.debug(f"WS msg type={msg.get('type')}, keys={list(msg.keys())}")
                    await self._handle_ws_message(ws, chat_id, msg)
            except WebSocketDisconnect:
                pass
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self.connections.unregister(chat_id, ws)
                logger.info(f"WebSocket disconnected: {chat_id}")

    async def _send_history(self, ws: WebSocket, chat_id: str):
        """Send chat history to a newly connected WebSocket."""
        if not self.app.state.session_manager:
            return

        sm = self.app.state.session_manager
        user_key = f"web:{chat_id}"
        session = sm.get_or_create(user_key)
        messages = []
        for m in session.messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({
                    "id": m.get("metadata", {}).get("msg_id", str(uuid.uuid4())[:8]),
                    "role": role,
                    "content": content,
                    "metadata": m.get("metadata", {}),
                })
        await ws.send_json({"type": "history", "messages": messages})

    async def _handle_ws_message(self, ws: WebSocket, chat_id: str, msg: dict):
        """Handle a single WebSocket message from the client."""
        msg_type = msg.get("type", "")

        if msg_type == "message":
            content = msg.get("content", "").strip()
            if not content:
                return

            metadata: dict[str, Any] = {
                "msg_id": str(uuid.uuid4())[:8],
            }

            reply_to = msg.get("reply_to_id")
            if reply_to:
                metadata["reply_to"] = {"content": reply_to}

            await self._publish_inbound(chat_id, content, metadata=metadata)

        elif msg_type == "command":
            name = msg.get("name", "")
            value = msg.get("value", "")

            # Session switch — reload history without going through AgentLoop
            if name == "switch_session":
                await self._send_history(ws, chat_id)
                return

            cmd_map = {
                "new": ("new_chat", "/new"),
                "context": ("context_info", "/context"),
                "context_mode": ("context_mode", f"/context_mode {value}".strip()),
                "trace": ("trace", f"/trace {value}".strip()),
                "compact": ("compact", "/compact"),
                "stop": ("stop", "/stop"),
            }
            if name in cmd_map:
                cmd_key, content = cmd_map[name]
                metadata = {"command": cmd_key}
                if name == "context_mode" and value:
                    metadata["context_mode"] = value
                if name == "trace" and value:
                    metadata["trace_mode"] = value
                await self._publish_inbound(chat_id, content, metadata=metadata)

        elif msg_type == "voice":
            await self._handle_voice(chat_id, msg)

        elif msg_type == "file":
            await self._handle_file(chat_id, msg)

        elif msg_type == "reaction":
            emoji = msg.get("emoji", "")
            message_id = msg.get("message_id", "")
            if emoji:
                metadata = {
                    "reaction": emoji,
                    "target_message_id": message_id,
                }
                await self._publish_inbound(
                    chat_id, f"[reaction: {emoji}]", metadata=metadata,
                )

        elif msg_type == "keyboard_callback":
            data = msg.get("data", "")
            if data.startswith("ctx_mode:"):
                mode = data.split(":", 1)[1]
                await self._publish_inbound(
                    chat_id, f"/context_mode {mode}",
                    metadata={"command": "set_context_mode", "context_mode": mode},
                )
            elif data.startswith("trace_mode:"):
                value = data.split(":", 1)[1]
                await self._publish_inbound(
                    chat_id, f"/trace {value}",
                    metadata={"command": "set_trace_mode", "trace_mode": value},
                )

    async def _handle_voice(self, chat_id: str, msg: dict):
        """Decode and transcribe a voice message."""
        b64_data = msg.get("data", "")
        if not b64_data:
            return

        try:
            audio_bytes = base64.b64decode(b64_data)
        except Exception:
            await self.connections.send_to_chat(
                chat_id, {"type": "error", "message": "Invalid voice data"},
            )
            return

        # Save to temp file
        import tempfile
        mime = msg.get("mime", "audio/webm")
        ext = ".webm" if "webm" in mime else ".ogg"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        content = f"[voice: {tmp_path}]"

        if self.transcriber:
            try:
                text = await self.transcriber.transcribe(Path(tmp_path))
                content = f"[Voice message transcription: {text}]"
            except Exception as e:
                logger.error(f"Voice transcription failed: {e}")
                content = f"[voice: transcription failed — {tmp_path}]"

        await self._publish_inbound(chat_id, content, media=[tmp_path])

    async def _handle_file(self, chat_id: str, msg: dict):
        """Handle a file attachment from WebSocket."""
        b64_data = msg.get("data", "")
        filename = msg.get("filename", "upload")
        mime = msg.get("mime", "")
        caption = msg.get("caption", "").strip()

        logger.debug(f"_handle_file: filename={filename}, mime={mime}, data_len={len(b64_data)}, caption={caption!r}")

        if not b64_data:
            logger.warning("_handle_file: empty b64_data, skipping")
            return

        try:
            file_bytes = base64.b64decode(b64_data)
        except Exception as e:
            logger.error(f"_handle_file: base64 decode failed: {e}")
            return

        # Save file to disk immediately so download_file tool works
        user_key = f"web:{chat_id}"
        saved_path = ""
        if self.media_manager:
            files_dir = self.media_manager._base_dir / user_key / "files"
            files_dir.mkdir(parents=True, exist_ok=True)
            save_path = files_dir / filename
            # Ensure unique name
            if save_path.exists():
                stem, suffix = save_path.stem, save_path.suffix
                counter = 1
                while save_path.exists():
                    save_path = files_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
            save_path.write_bytes(file_bytes)
            saved_path = str(save_path)

        file_id = saved_path or f"web_{uuid.uuid4().hex[:8]}"
        att_type = "photo" if mime.startswith("image/") else "file"

        attachments = [MediaAttachment(
            type=att_type,
            file_id=file_id,
            data=file_bytes,
            filename=filename,
            mime_type=mime,
        )]

        content = f"[file available: {file_id}]"
        if caption:
            content = caption + "\n" + content
        await self._publish_inbound(
            chat_id, content, attachments=attachments,
        )

    async def _publish_inbound(
        self,
        chat_id: str,
        content: str,
        media: list[str] | None = None,
        attachments: list[MediaAttachment] | None = None,
        metadata: dict | None = None,
    ):
        """Create and publish an InboundMessage to the bus."""
        msg = InboundMessage(
            channel="web",
            sender_id=chat_id,
            chat_id=chat_id,
            content=content,
            media=media or [],
            attachments=attachments or [],
            metadata=metadata or {},
        )
        await self.bus.publish_inbound(msg)

    async def send_outbound(self, msg: OutboundMessage):
        """Translate OutboundMessage to WebSocket JSON and push to clients."""
        chat_id = msg.chat_id
        meta = msg.metadata

        # Typing indicator
        if meta.get("chat_action") == "typing":
            await self.connections.send_to_chat(
                chat_id, {"type": "typing", "active": True},
            )
            return

        if meta.get("stop_typing"):
            await self.connections.send_to_chat(
                chat_id, {"type": "typing", "active": False},
            )
            return

        # Reaction
        if meta.get("reaction"):
            await self.connections.send_to_chat(chat_id, {
                "type": "reaction",
                "emoji": meta["reaction"],
                "message_id": str(meta.get("target_message_id", "")),
            })
            return

        # Media
        media_type = meta.get("media_type")
        if media_type and meta.get("media_path"):
            media_path = meta["media_path"]
            # Build a relative URL for the media
            if self.media_manager:
                media_base = self.media_manager._base_dir
                src = Path(media_path).resolve()
                try:
                    rel = src.relative_to(media_base.resolve())
                except ValueError:
                    # File is outside media dir — copy it in
                    import shutil
                    served_dir = media_base / "_served"
                    served_dir.mkdir(parents=True, exist_ok=True)
                    dest = served_dir / src.name
                    # Avoid name collisions
                    if dest.exists() and dest.stat().st_size != src.stat().st_size:
                        dest = served_dir / f"{src.stem}_{uuid.uuid4().hex[:6]}{src.suffix}"
                    if not dest.exists():
                        shutil.copy2(str(src), str(dest))
                    rel = dest.relative_to(media_base)
                url = f"/api/media/{rel}"
            else:
                url = f"/api/media/{Path(media_path).name}"

            await self.connections.send_to_chat(chat_id, {
                "type": "media",
                "media_type": media_type,
                "url": url,
                "caption": msg.content or "",
            })
            return

        # Inline keyboard
        if meta.get("inline_keyboard"):
            msg_id = str(meta.get("edit_message_id", uuid.uuid4().hex[:8]))
            await self.connections.send_to_chat(chat_id, {
                "type": "inline_keyboard",
                "buttons": meta["inline_keyboard"],
                "message_id": msg_id,
                "content": msg.content or "",
            })
            return

        # Edit existing message
        if meta.get("edit_message_id"):
            await self.connections.send_to_chat(chat_id, {
                "type": "edit",
                "message_id": str(meta["edit_message_id"]),
                "content": msg.content,
            })
            return

        # Regular message
        is_intermediate = meta.get("intermediate", False)
        msg_id = str(meta.get("msg_id", uuid.uuid4().hex[:8]))

        await self.connections.send_to_chat(chat_id, {
            "type": "message",
            "id": msg_id,
            "role": "assistant",
            "content": msg.content,
            "metadata": {k: v for k, v in meta.items() if k not in (
                "chat_action", "stop_typing", "reaction", "target_message_id",
                "media_type", "media_path", "inline_keyboard", "edit_message_id",
            )},
        })

        # Stop typing after final message
        if not is_intermediate:
            await self.connections.send_to_chat(
                chat_id, {"type": "typing", "active": False},
            )

    async def start(self) -> None:
        """Start uvicorn server as asyncio task."""
        config = uvicorn.Config(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)

        logger.info(f"Web UI starting on http://{self.config.host}:{self.config.port}")
        await self._server.serve()

    async def stop(self) -> None:
        """Signal uvicorn to exit."""
        if self._server:
            self._server.should_exit = True
