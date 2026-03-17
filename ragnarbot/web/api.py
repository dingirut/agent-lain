"""REST API routers for Web UI dashboard tabs."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.responses import StreamingResponse

if TYPE_CHECKING:
    from ragnarbot.agent.loop import AgentLoop
    from ragnarbot.cron.service import CronService
    from ragnarbot.heartbeat.service import HeartbeatService
    from ragnarbot.session.manager import SessionManager


# ── Sessions ──────────────────────────────────────────────────

sessions_router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@sessions_router.get("")
async def list_sessions(request: Request) -> dict:
    sm: SessionManager = request.app.state.session_manager
    user_key = f"web:{request.state.session_user or 'default'}"
    sessions = sm.list_sessions(user_key=user_key)
    active_id = sm.get_active_id(user_key)
    return {"sessions": sessions, "active_session_id": active_id}


@sessions_router.post("/new")
async def new_session(request: Request) -> dict:
    sm: SessionManager = request.app.state.session_manager
    user_key = f"web:{request.state.session_user or 'default'}"
    session = sm.create_new(user_key)
    return {"session_id": session.key}


@sessions_router.post("/switch")
async def switch_session(request: Request) -> dict:
    sm: SessionManager = request.app.state.session_manager
    user_key = f"web:{request.state.session_user or 'default'}"
    body = await request.json()
    session_id = body.get("session_id", "")
    if not session_id:
        return JSONResponse({"error": "session_id required"}, status_code=400)
    # Verify session exists and belongs to user
    sessions = sm.list_sessions(user_key=user_key)
    valid_ids = {s["session_id"] for s in sessions}
    if session_id not in valid_ids:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    sm.set_active(user_key, session_id)
    # Clear session cache so AgentLoop picks up the new active session
    sm._cache.pop(session_id, None)
    return {"ok": True, "session_id": session_id}


@sessions_router.delete("/{session_id}")
async def delete_session(session_id: str, request: Request) -> dict:
    sm: SessionManager = request.app.state.session_manager
    user_key = f"web:{request.state.session_user or 'default'}"
    # Verify session belongs to user
    sessions = sm.list_sessions(user_key=user_key)
    valid_ids = {s["session_id"] for s in sessions}
    if session_id not in valid_ids:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    # Don't allow deleting the active session
    active_id = sm.get_active_id(user_key)
    if session_id == active_id:
        return JSONResponse({"error": "Cannot delete the active session. Switch to another session first."}, status_code=400)
    if sm.delete(session_id):
        return {"ok": True}
    return JSONResponse({"error": "Failed to delete"}, status_code=500)


@sessions_router.post("/rename")
async def rename_session(request: Request) -> dict:
    sm: SessionManager = request.app.state.session_manager
    user_key = f"web:{request.state.session_user or 'default'}"
    body = await request.json()
    session_id = body.get("session_id", "")
    title = body.get("title", "").strip()
    if not session_id or not title:
        return JSONResponse({"error": "session_id and title required"}, status_code=400)
    # Verify session belongs to user
    sessions = sm.list_sessions(user_key=user_key)
    valid_ids = {s["session_id"] for s in sessions}
    if session_id not in valid_ids:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    if sm.rename(session_id, title):
        return {"ok": True}
    return JSONResponse({"error": "Failed to rename"}, status_code=500)


# ── Jobs ──────────────────────────────────────────────────────

jobs_router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@jobs_router.get("")
async def list_jobs(request: Request) -> list[dict]:
    agent: AgentLoop | None = request.app.state.agent_loop
    if not agent:
        return []

    jobs: list[dict] = []

    # Sub-agent tasks
    if hasattr(agent, "subagents"):
        for task in agent.subagents.list_tasks():
            jobs.append({
                "id": task.get("id", ""),
                "label": task.get("label", ""),
                "type": "subagent",
                "status": task.get("status", "unknown"),
                "elapsed": task.get("elapsed"),
            })

    # Background shell jobs
    if hasattr(agent, "bg_processes"):
        for job in agent.bg_processes.list_jobs():
            jobs.append({
                "id": job.get("id", ""),
                "label": job.get("label", job.get("command", "")),
                "type": "background",
                "status": job.get("status", "unknown"),
                "elapsed": job.get("elapsed"),
            })

    return jobs


# ── Routines ──────────────────────────────────────────────────

routines_router = APIRouter(prefix="/api/routines", tags=["routines"])


@routines_router.get("")
async def list_routines(request: Request) -> dict:
    cron: CronService | None = request.app.state.cron_service
    heartbeat: HeartbeatService | None = request.app.state.heartbeat_service

    result: dict[str, Any] = {"cron_jobs": [], "heartbeat": None}

    if cron:
        result["cron_jobs"] = [j.model_dump() for j in cron.list_jobs()]

    if heartbeat:
        result["heartbeat"] = {
            "enabled": heartbeat.enabled,
            "interval_m": getattr(heartbeat, "interval_m", None),
            "running": getattr(heartbeat, "_running", False),
        }

    return result


@routines_router.post("")
async def create_routine(request: Request) -> dict:
    cron: CronService | None = request.app.state.cron_service
    if not cron:
        return JSONResponse({"error": "Cron service not available"}, status_code=503)

    body = await request.json()
    try:
        job = cron.add_job(body)
        return {"id": job.id, "name": job.name}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@routines_router.put("/{job_id}")
async def update_routine(job_id: str, request: Request) -> dict:
    cron: CronService | None = request.app.state.cron_service
    if not cron:
        return JSONResponse({"error": "Cron service not available"}, status_code=503)

    body = await request.json()
    try:
        job = cron.update_job(job_id, body)
        return {"id": job.id, "name": job.name}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@routines_router.delete("/{job_id}")
async def delete_routine(job_id: str, request: Request) -> dict:
    cron: CronService | None = request.app.state.cron_service
    if not cron:
        return JSONResponse({"error": "Cron service not available"}, status_code=503)

    try:
        cron.remove_job(job_id)
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@routines_router.post("/{job_id}/run")
async def run_routine(job_id: str, request: Request) -> dict:
    cron: CronService | None = request.app.state.cron_service
    if not cron:
        return JSONResponse({"error": "Cron service not available"}, status_code=503)

    try:
        await cron.run_job(job_id)
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# ── Extensions ────────────────────────────────────────────────

extensions_router = APIRouter(prefix="/api/extensions", tags=["extensions"])


@extensions_router.get("")
async def list_extensions(request: Request) -> list[dict]:
    agent: AgentLoop | None = request.app.state.agent_loop
    if not agent:
        return []

    defs = agent.tools.get_definitions()
    result = []
    for d in defs:
        fn = d.get("function", {})
        result.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
        })
    return result


# ── Memory ────────────────────────────────────────────────────

memory_router = APIRouter(prefix="/api/memory", tags=["memory"])


_WALK_TREE_MAX_DEPTH = 6
_WALK_TREE_MAX_ENTRIES = 2000
_WALK_TREE_SKIP_DIRS = {
    "__pycache__", "node_modules", ".git", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
}


def _walk_tree(
    base: Path,
    rel: Path | None = None,
    *,
    _depth: int = 0,
    _counter: list[int] | None = None,
) -> list[dict]:
    """Recursively build a file tree with depth/entry limits."""
    if _counter is None:
        _counter = [0]

    if _depth > _WALK_TREE_MAX_DEPTH or _counter[0] >= _WALK_TREE_MAX_ENTRIES:
        return []

    target = base / rel if rel else base
    if not target.exists() or target.is_symlink():
        return []

    items = []
    try:
        for entry in sorted(target.iterdir()):
            if _counter[0] >= _WALK_TREE_MAX_ENTRIES:
                break
            if entry.name.startswith("."):
                continue
            if entry.is_symlink():
                continue
            entry_rel = entry.relative_to(base)
            _counter[0] += 1
            if entry.is_dir():
                if entry.name in _WALK_TREE_SKIP_DIRS:
                    continue
                items.append({
                    "name": entry.name,
                    "path": str(entry_rel),
                    "type": "directory",
                    "children": _walk_tree(
                        base, entry_rel,
                        _depth=_depth + 1, _counter=_counter,
                    ),
                })
            else:
                items.append({
                    "name": entry.name,
                    "path": str(entry_rel),
                    "type": "file",
                    "size": entry.stat().st_size,
                })
    except PermissionError:
        pass
    return items


@memory_router.get("/tree")
async def memory_tree(request: Request) -> list[dict]:
    workspace: Path = request.app.state.workspace
    return await asyncio.to_thread(_walk_tree, workspace)


@memory_router.get("/read")
async def memory_read(request: Request, path: str = Query(...)) -> dict:
    workspace: Path = request.app.state.workspace
    target = (workspace / path).resolve()

    # Safety: prevent path traversal
    if not str(target).startswith(str(workspace.resolve())):
        return JSONResponse({"error": "Access denied"}, status_code=403)

    if not target.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)

    try:
        content = target.read_text(encoding="utf-8")
        return {"path": path, "content": content}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@memory_router.put("/write")
async def memory_write(request: Request) -> dict:
    workspace: Path = request.app.state.workspace
    body = await request.json()
    path = body.get("path", "")
    content = body.get("content", "")

    target = (workspace / path).resolve()
    if not str(target).startswith(str(workspace.resolve())):
        return JSONResponse({"error": "Access denied"}, status_code=403)

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"ok": True, "path": path}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@memory_router.delete("/delete")
async def memory_delete(request: Request) -> dict:
    workspace: Path = request.app.state.workspace
    body = await request.json()
    path = body.get("path", "")

    target = (workspace / path).resolve()
    if not str(target).startswith(str(workspace.resolve())):
        return JSONResponse({"error": "Access denied"}, status_code=403)

    if not target.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)

    # Safety: don't allow deleting directories or critical files
    if target.is_dir():
        return JSONResponse({"error": "Cannot delete directories"}, status_code=400)

    try:
        target.unlink()
        return {"ok": True, "path": path}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@memory_router.get("/graph")
async def memory_graph(request: Request) -> dict:
    """Return knowledge graph nodes and edges for visualization."""
    agent_loop = getattr(request.app.state, "agent_loop", None)
    memory_store = getattr(agent_loop, "memory_store", None) if agent_loop else None

    if not memory_store or not getattr(memory_store, "has_vector", False):
        return {"nodes": [], "edges": [], "status": "unavailable"}

    graph = getattr(memory_store, "_graph", None)
    if not graph or not graph._driver:
        return {"nodes": [], "edges": [], "status": "no_graph"}

    try:
        async with graph._driver.session() as session:
            # Get all entities
            node_result = await session.run(
                """
                MATCH (e:Entity)
                RETURN e.name AS name, e.entity_type AS type,
                       coalesce(e.aliases, []) AS aliases
                ORDER BY e.name
                LIMIT 500
                """
            )
            nodes = []
            node_set = set()
            async for record in node_result:
                name = record["name"]
                nodes.append({
                    "id": name,
                    "label": name,
                    "type": record["type"] or "other",
                    "aliases": list(record["aliases"] or []),
                })
                node_set.add(name)

            # Get all active relations
            edge_result = await session.run(
                """
                MATCH (s:Entity)-[r:RELATION]->(o:Entity)
                WHERE r.valid_until IS NULL
                RETURN s.name AS source, o.name AS target,
                       r.type AS label, r.confidence AS confidence,
                       r.fact_id AS fact_id
                ORDER BY r.confidence DESC
                LIMIT 1000
                """
            )
            edges = []
            async for record in edge_result:
                src, tgt = record["source"], record["target"]
                # Auto-add nodes that appear only in edges
                for n in (src, tgt):
                    if n not in node_set:
                        nodes.append({"id": n, "label": n, "type": "other", "aliases": []})
                        node_set.add(n)
                edges.append({
                    "source": src,
                    "target": tgt,
                    "label": (record["label"] or "").replace("_", " "),
                    "confidence": record["confidence"] or 1.0,
                })

        # Also get fact counts from pgvector
        backend = getattr(memory_store, "_backend", None)
        fact_count = 0
        if backend:
            try:
                async with backend._pool.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT count(*) AS cnt FROM facts WHERE valid_until IS NULL"
                    )
                    fact_count = row["cnt"] if row else 0
            except Exception:
                pass

        return {
            "nodes": nodes,
            "edges": edges,
            "fact_count": fact_count,
            "status": "ok",
        }
    except Exception as e:
        logger.warning(f"Memory graph API error: {e}")
        return {"nodes": [], "edges": [], "status": f"error: {e}"}


@memory_router.get("/facts")
async def memory_facts(
    request: Request,
    query: str = Query("", description="Search query"),
    limit: int = Query(20, ge=1, le=100),
) -> dict:
    """Search or list facts from semantic memory."""
    agent_loop = getattr(request.app.state, "agent_loop", None)
    memory_store = getattr(agent_loop, "memory_store", None) if agent_loop else None

    if not memory_store or not getattr(memory_store, "has_vector", False):
        return {"facts": [], "status": "unavailable"}

    try:
        if query:
            results = await memory_store.search(query=query, limit=limit)
        else:
            # List recent facts
            backend = getattr(memory_store, "_backend", None)
            results = []
            if backend:
                async with backend._pool.acquire() as conn:
                    rows = await conn.fetch(
                        """
                        SELECT id, subject, predicate, object, confidence,
                               category, extraction_type, created_at
                        FROM facts
                        WHERE valid_until IS NULL
                        ORDER BY created_at DESC
                        LIMIT $1
                        """,
                        limit,
                    )
                    for r in rows:
                        results.append({
                            "fact": f"{r['subject']} {r['predicate']} {r['object']}",
                            "subject": r["subject"],
                            "predicate": r["predicate"],
                            "object": r["object"],
                            "confidence": float(r["confidence"]),
                            "category": r["category"],
                            "score": 1.0,
                            "sources": [],
                            "issues": [],
                        })
        return {"facts": results, "status": "ok"}
    except Exception as e:
        logger.warning(f"Memory facts API error: {e}")
        return {"facts": [], "status": f"error: {e}"}


# ── Memory Debug ──────────────────────────────────────────────


@memory_router.get("/debug/config")
async def memory_debug_config(request: Request) -> dict:
    """Return current pipeline configuration (prompts, filter sets, models)."""
    from ragnarbot.agent.memory.extraction import EXTRACTION_PROMPT
    from ragnarbot.agent.memory.enrichment import HQ_PROMPT
    from ragnarbot.agent.memory.validation import (
        ALWAYS_IMPORTANT,
        CONDITIONAL,
        GROUNDING_PROMPT,
        NOISE,
        NOISE_SUBJECTS,
    )

    agent_loop = getattr(request.app.state, "agent_loop", None)
    memory_store = getattr(agent_loop, "memory_store", None) if agent_loop else None

    models = {}
    if memory_store:
        pipeline = getattr(memory_store, "_pipeline", None)
        if pipeline:
            models = {
                "extraction": getattr(pipeline.extractor, "model", ""),
                "grounding": getattr(pipeline.grounding, "model", ""),
                "enrichment": getattr(pipeline.enricher, "model", ""),
            }

    return {
        "extraction_prompt": EXTRACTION_PROMPT,
        "enrichment_prompt": HQ_PROMPT,
        "grounding_prompt": GROUNDING_PROMPT,
        "noise_predicates": sorted(NOISE),
        "noise_subjects": sorted(NOISE_SUBJECTS),
        "always_important": sorted(ALWAYS_IMPORTANT),
        "conditional": sorted(CONDITIONAL),
        "models": models,
    }


@memory_router.post("/debug/extract")
async def memory_debug_extract(request: Request) -> dict:
    """Dry-run pipeline with full per-stage trace."""
    agent_loop = getattr(request.app.state, "agent_loop", None)
    memory_store = getattr(agent_loop, "memory_store", None) if agent_loop else None
    if not memory_store or not getattr(memory_store, "has_vector", False):
        return {"error": "Semantic memory not available"}

    pipeline = getattr(memory_store, "_pipeline", None)
    if not pipeline:
        return {"error": "Pipeline not initialized"}

    body = await request.json()
    text = body.get("text", "")
    if not text or len(text.strip()) < 10:
        return {"error": "Text too short (minimum 10 chars)"}

    overrides = body.get("overrides", {})
    trace = await pipeline.process_debug(text=text, overrides=overrides, dry_run=True)
    return trace


@memory_router.post("/debug/store")
async def memory_debug_store(request: Request) -> dict:
    """Run pipeline and persist results to DB."""
    agent_loop = getattr(request.app.state, "agent_loop", None)
    memory_store = getattr(agent_loop, "memory_store", None) if agent_loop else None
    if not memory_store or not getattr(memory_store, "has_vector", False):
        return {"error": "Semantic memory not available"}

    pipeline = getattr(memory_store, "_pipeline", None)
    if not pipeline:
        return {"error": "Pipeline not initialized"}

    body = await request.json()
    text = body.get("text", "")
    if not text or len(text.strip()) < 10:
        return {"error": "Text too short (minimum 10 chars)"}

    overrides = body.get("overrides", {})
    trace = await pipeline.process_debug(text=text, overrides=overrides, dry_run=False)
    return trace


@memory_router.post("/debug/purge")
async def memory_debug_purge(request: Request) -> dict:
    """Purge ALL facts from pgvector and ALL nodes from Neo4j."""
    agent_loop = getattr(request.app.state, "agent_loop", None)
    memory_store = getattr(agent_loop, "memory_store", None) if agent_loop else None
    if not memory_store or not getattr(memory_store, "has_vector", False):
        return {"error": "Semantic memory not available"}

    body = await request.json()
    if body.get("confirm") != "PURGE ALL":
        return {"error": "Confirmation required: send {\"confirm\": \"PURGE ALL\"}"}

    result: dict[str, Any] = {}

    backend = getattr(memory_store, "_backend", None)
    if backend:
        try:
            deleted = await backend.purge_all()
            result["pgvector"] = {"deleted_facts": deleted}
        except Exception as e:
            result["pgvector"] = {"error": str(e)}

    graph = getattr(memory_store, "_graph", None)
    if graph and getattr(graph, "_driver", None):
        try:
            neo_result = await graph.purge_all()
            result["neo4j"] = neo_result
        except Exception as e:
            result["neo4j"] = {"error": str(e)}

    return result


@memory_router.post("/debug/upload")
async def memory_debug_upload(request: Request) -> dict:
    """Upload a file and return its text content for extraction."""
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        from starlette.datastructures import UploadFile as _UF
        form = await request.form()
        file_item = form.get("file")
        if not file_item or not hasattr(file_item, "read"):
            return {"error": "No file in request"}
        raw = await file_item.read()
        filename = getattr(file_item, "filename", "upload.txt") or "upload.txt"
    else:
        body = await request.json()
        import base64
        raw = base64.b64decode(body.get("data", ""))
        filename = body.get("filename", "upload.txt")

    ext = Path(filename).suffix.lower()

    # Parse based on extension
    if ext in (".md", ".txt", ".log", ".rst"):
        text = raw.decode("utf-8", errors="replace")
    elif ext == ".csv":
        text = raw.decode("utf-8", errors="replace")
    elif ext == ".json":
        import json
        try:
            data = json.loads(raw)
            text = json.dumps(data, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            text = raw.decode("utf-8", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")

    # Truncate very large files
    max_chars = 50_000
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]

    return {
        "filename": filename,
        "text": text,
        "chars": len(text),
        "truncated": truncated,
    }


# ── Skills ────────────────────────────────────────────────────

skills_router = APIRouter(prefix="/api/skills", tags=["skills"])


@skills_router.get("")
async def list_skills(request: Request) -> list[dict]:
    agent: AgentLoop | None = request.app.state.agent_loop
    if not agent:
        return []

    try:
        skills = agent.context.skills.list_skills(filter_unavailable=False)
        return [
            {
                "name": s.get("name", ""),
                "description": s.get("description", ""),
                "source": s.get("source", ""),
                "available": s.get("available", True),
            }
            for s in skills
        ]
    except Exception:
        return []


# ── Logs ──────────────────────────────────────────────────────

logs_router = APIRouter(prefix="/api/logs", tags=["logs"])

# Shared log buffer for SSE broadcast
_log_subscribers: list[asyncio.Queue] = []
_log_sink_installed = False


def _install_log_sink():
    """Install a loguru sink that broadcasts to SSE subscribers."""
    global _log_sink_installed
    if _log_sink_installed:
        return
    _log_sink_installed = True

    def _sink(message):
        record = message.record
        entry = {
            "time": str(record["time"]),
            "level": record["level"].name,
            "message": str(record["message"]),
            "module": record.get("module", ""),
        }
        import json
        line = json.dumps(entry)
        for q in list(_log_subscribers):
            try:
                q.put_nowait(line)
            except asyncio.QueueFull:
                pass

    logger.add(_sink, level="DEBUG", format="{message}")


@logs_router.get("/stream")
async def logs_stream(request: Request):
    """SSE endpoint for real-time log streaming."""
    _install_log_sink()
    queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    _log_subscribers.append(queue)

    async def event_generator():
        try:
            while True:
                try:
                    line = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {line}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            _log_subscribers.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
