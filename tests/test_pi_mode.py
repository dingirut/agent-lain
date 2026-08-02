"""Pi mode: minimal system prompt for small/slow local models."""

from types import SimpleNamespace

from ragnarbot.agent.context import ContextBuilder
from ragnarbot.agent.tools.base import Tool
from ragnarbot.agent.tools.registry import ToolRegistry


class _Tool(Tool):
    def __init__(self, name: str, description: str):
        self._name = name
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        return "ok"


# ── one-line tool snippets ───────────────────────────────────────

def test_snippet_takes_the_first_sentence():
    tool = _Tool("read", "Read a file. Supports offset and limit for big files.")
    assert tool.prompt_snippet == "Read a file"


def test_snippet_does_not_split_on_abbreviations():
    tool = _Tool("glob", "Find files by pattern (e.g. '**/*.md'). Sorted by mtime.")
    assert tool.prompt_snippet == "Find files by pattern (e.g. '**/*.md')"


def test_snippet_is_truncated_with_ellipsis():
    tool = _Tool("recall", "Hybrid semantic and keyword search over " + "memory " * 30)
    snippet = tool.prompt_snippet
    assert len(snippet) <= 91 and snippet.endswith("…")


def test_snippet_handles_single_sentence_and_empty():
    assert _Tool("web_search", "Search the web").prompt_snippet == "Search the web"
    assert _Tool("noop", "").prompt_snippet == ""


def test_registry_exposes_snippets():
    registry = ToolRegistry()
    registry.register(_Tool("read", "Read a file. More detail here."))
    registry.register(_Tool("bash", "Run a shell command."))
    assert registry.prompt_snippets() == [
        ("read", "Read a file"),
        ("bash", "Run a shell command"),
    ]


# ── the minimal prompt ───────────────────────────────────────────

def _builder(tmp_path, *, identity: str | None = None) -> ContextBuilder:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    builder = ContextBuilder(workspace)
    if identity is not None:
        (workspace / "IDENTITY.md").write_text(identity, encoding="utf-8")
    builder.pi_mode = True
    builder.tool_snippets_provider = lambda: [
        ("file_read", "Read a file"),
        ("exec", "Run a shell command"),
    ]
    # Keep catalogues out of the way unless a test asks for them
    builder.skills = SimpleNamespace(
        build_skills_summary=lambda **_: "", get_always_skills=lambda: [],
        load_skills_for_context=lambda names: "",
    )
    builder.agents = SimpleNamespace(build_agents_summary=lambda: "")
    return builder


def test_pi_prompt_keeps_identity_tools_and_pointers(tmp_path):
    builder = _builder(tmp_path, identity="# Identity\n\nI am Lain.")
    prompt = builder.build_system_prompt(channel="web")

    assert "I am Lain." in prompt
    assert "- file_read: Read a file" in prompt
    assert "- exec: Run a shell command" in prompt
    assert "AGENTS.md" in prompt  # pointer to the full manual, not its content
    assert "recall" in prompt
    assert "Workspace:" in prompt and "Timezone:" in prompt


def test_pi_prompt_drops_the_builtin_manuals(tmp_path):
    builder = _builder(tmp_path, identity="I am Lain.")
    pi_prompt = builder.build_system_prompt(channel="web")
    builder.pi_mode = False
    full_prompt = builder.build_system_prompt(channel="web")

    # The full prompt ships SOUL/AGENTS/BUILTIN_TOOLS inline; pi mode does not.
    assert len(pi_prompt) < len(full_prompt) / 5
    for marker in ("## Core Philosophy", "# Built-in Tools", "## Browser Protocol"):
        assert marker in full_prompt
        assert marker not in pi_prompt


def test_pi_prompt_lists_skills_and_agents_when_present(tmp_path):
    builder = _builder(tmp_path)
    builder.skills = SimpleNamespace(
        build_skills_summary=lambda **_: "<skills><skill>weather</skill></skills>",
        get_always_skills=lambda: [],
        load_skills_for_context=lambda names: "",
    )
    builder.agents = SimpleNamespace(
        build_agents_summary=lambda: "<agents><agent>researcher</agent></agents>",
    )
    prompt = builder.build_system_prompt(channel="web")

    assert "weather" in prompt and "file_read" in prompt
    assert "researcher" in prompt
    # skills are referenced, not inlined
    assert "read its `<location>` with file_read" in prompt


def test_pi_prompt_survives_a_missing_snippet_provider(tmp_path):
    builder = _builder(tmp_path)
    builder.tool_snippets_provider = None
    prompt = builder.build_system_prompt(channel="web")
    assert "## Tools" not in prompt
    assert "Workspace:" in prompt

    def broken():
        raise RuntimeError("registry unavailable")

    builder.tool_snippets_provider = broken
    assert "Workspace:" in builder.build_system_prompt(channel="web")


def test_pi_prompt_keeps_isolated_run_rules(tmp_path):
    """Cron/heartbeat/hook runs carry their own execution rules — always."""
    builder = _builder(tmp_path)
    builder._load_builtin_cron_isolated = lambda ctx: "CRON RULES for " + ctx["job_name"]
    prompt = builder.build_system_prompt(
        session_metadata={"cron_isolated": {"job_name": "digest"}}, channel="cli",
    )
    assert "CRON RULES for digest" in prompt
