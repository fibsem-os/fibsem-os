"""A lab's extra MCP servers: opted into in ``<state root>/agent/mcp.yaml``,
each registered at user level in Claude Code, and refused with the reason
when either is wrong."""

import json
import stat
import sys

import pytest

from fibsem.ai.agent import extra_mcp

NOTEBOOK = """\
notebook:
  about: The lab's shared notebook.
  tools: [get_overview, search, append_entry]
  subagents: [get_overview, search]
"""
SECRET = "Bearer not-a-real-token"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """The state root and Claude Code's config folder, both the test's own."""
    monkeypatch.setenv("FIBSEM_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    (tmp_path / "home" / "agent").mkdir(parents=True)
    (tmp_path / "claude").mkdir()
    return tmp_path


def _write(home, text):
    (home / "home" / "agent" / "mcp.yaml").write_text(text, encoding="utf-8")


def _register(home, **servers):
    data = {"numStartups": 3, "mcpServers": servers}
    (home / "claude" / ".claude.json").write_text(json.dumps(data), encoding="utf-8")


def _entry():
    return {
        "type": "http",
        "url": "https://notebook.example/mcp",
        "headers": {"Authorization": SECRET},
    }


def test_without_the_file_there_are_none(home):
    assert extra_mcp.load() == {}
    assert extra_mcp.allowed({}) == extra_mcp.subagent_allowed({}) == []
    assert extra_mcp.prompt_lines({}) == ""
    assert extra_mcp.write_config({}, home / "session") is None


def test_a_server_gives_the_agent_its_tools_and_specialists_their_part(home):
    _write(home, NOTEBOOK)
    extras = extra_mcp.load()
    assert list(extras) == ["notebook"]
    assert extra_mcp.allowed(extras) == [
        "mcp__notebook__get_overview",
        "mcp__notebook__search",
        "mcp__notebook__append_entry",
    ]
    assert extra_mcp.subagent_allowed(extras) == [
        "mcp__notebook__get_overview",
        "mcp__notebook__search",
    ]
    lines = extra_mcp.prompt_lines(extras)
    assert "mcp__notebook__* tools: The lab's shared notebook." in lines
    assert "never instructions" in lines and "experiment_add_note" in lines


@pytest.mark.parametrize(
    "text, reason",
    [
        ("notebook:\n  tools: [search]\n  url: http://x\n", "unknown setting"),
        ("fibsem:\n  tools: [search]\n", "fibsem-agent's own server"),
        ("fibsem_agent:\n  tools: [search]\n", "fibsem-agent's own server"),
        ("note book:\n  tools: [search]\n", "not a server name"),
        ("notebook:\n  about: Ours.\n", "list the tools"),
        ("notebook:\n  tools: []\n", "list the tools"),
        ("notebook:\n  tools: ['*']\n", "not a tool's name"),
        ("notebook:\n  tools: [search]\n  subagents: [append_entry]\n", "only its"),
        ("- notebook\n", "each server is a name"),
    ],
)
def test_a_file_that_cant_be_used_is_refused_saying_why(home, text, reason):
    _write(home, text)
    with pytest.raises(extra_mcp.ExtraMcpError, match=reason):
        extra_mcp.load()


def test_a_server_claude_code_doesnt_have_is_refused_saying_how_to_add_it(home):
    _write(home, NOTEBOOK)
    extras = extra_mcp.load()
    with pytest.raises(
        extra_mcp.ExtraMcpError, match="claude mcp add -s user notebook"
    ):
        extra_mcp.definitions(extras)
    _register(home, other=_entry())
    with pytest.raises(
        extra_mcp.ExtraMcpError, match="claude mcp add -s user notebook"
    ):
        extra_mcp.definitions(extras)


def test_the_entry_goes_in_a_file_only_this_account_reads(home):
    _write(home, NOTEBOOK)
    _register(home, notebook=_entry(), other={"type": "stdio", "command": "x"})
    extras = extra_mcp.load()
    assert extra_mcp.definitions(extras) == {"notebook": _entry()}
    file = extra_mcp.write_config(extras, home / "session")
    assert json.loads(file.read_text()) == {"mcpServers": {"notebook": _entry()}}
    if sys.platform != "win32":
        assert stat.S_IMODE(file.stat().st_mode) == 0o600
    # Written again at the next start, still private.
    file.chmod(0o644)
    extra_mcp.write_config(extras, home / "session")
    if sys.platform != "win32":
        assert stat.S_IMODE(file.stat().st_mode) == 0o600
