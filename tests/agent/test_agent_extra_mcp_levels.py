"""A lab server's tools and the agent level: each needs Drive unless mcp.yaml
marks it ``watch``, the harness refuses it below that, and the model never sees
the server's tools mcp.yaml doesn't list."""

import asyncio
import sys

import pytest

from fibsem.ai.agent import extra_mcp, harness

NOTEBOOK = """\
notebook:
  command: [notebook-mcp]
  tools:
    - search: watch
    - read_page: watch
    - append_entry
    - tidy: drive
  subagents: [search]
"""


@pytest.fixture
def extras(tmp_path):
    file = tmp_path / "mcp.yaml"
    file.write_text(NOTEBOOK, encoding="utf-8")
    return extra_mcp.load(file)


def test_a_tool_needs_drive_unless_marked_watch(extras):
    notebook = extras["notebook"]
    assert notebook.tools == ["search", "read_page", "append_entry", "tidy"]
    assert notebook.watch == ["search", "read_page"]
    assert extra_mcp.allowed(extras) == [
        f"mcp__notebook__{tool}" for tool in notebook.tools
    ]
    assert "These need the agent at Drive: append_entry, tidy." in (
        extra_mcp.prompt_lines(extras)
    )


def test_a_tools_level_is_watch_or_drive(tmp_path):
    file = tmp_path / "mcp.yaml"
    file.write_text(
        "notebook:\n  command: [x]\n  tools:\n    - search: assist\n", encoding="utf-8"
    )
    with pytest.raises(extra_mcp.ExtraMcpError, match="a tool is watch or drive"):
        extra_mcp.load(file)


def test_a_server_marked_watch_marks_its_tools(tmp_path):
    file = tmp_path / "mcp.yaml"
    file.write_text(
        "notebook:\n  command: [x]\n  level: watch\n"
        "  tools:\n    - search\n    - read_page\n    - tidy: drive\n",
        encoding="utf-8",
    )
    notebook = extra_mcp.load(file)["notebook"]
    assert notebook.watch == ["search", "read_page"]
    assert (
        extra_mcp.refusal(
            {"notebook": notebook}, "mcp__notebook__search", lambda: "watch"
        )
        is None
    )
    assert "Drive" in extra_mcp.refusal(
        {"notebook": notebook}, "mcp__notebook__tidy", lambda: "watch"
    )

    file.write_text(
        "notebook:\n  command: [x]\n  level: assist\n  tools: [search]\n",
        encoding="utf-8",
    )
    with pytest.raises(extra_mcp.ExtraMcpError, match="a server is watch or drive"):
        extra_mcp.load(file)


@pytest.mark.parametrize(
    "tool, level, refused",
    [
        ("mcp__notebook__search", "watch", None),
        ("mcp__notebook__search", "off", None),  # the server refuses Off itself
        ("mcp__notebook__append_entry", "drive", None),
        ("mcp__notebook__append_entry", "assist", "the operator has it at Assist"),
        ("mcp__notebook__append_entry", "watch", "the operator has it at Watch"),
        ("mcp__notebook__append_entry", None, "can't be read"),
        ("mcp__notebook__delete_everything", "drive", "not one of the notebook tools"),
        ("mcp__fibsem__stage_move", "watch", None),  # the App Server gates its own
        ("Skill", "watch", None),
    ],
)
def test_the_level_decides_a_lab_tool(extras, tool, level, refused):
    why = extra_mcp.refusal(extras, tool, lambda: level)
    if refused is None:
        assert why is None
    else:
        assert refused in why


def _decide(hook, tool):
    return asyncio.run(hook({"tool_name": tool}, "id", None))


def test_the_harness_hook_refuses_below_drive(extras, monkeypatch):
    level = ["watch"]
    monkeypatch.setattr(extra_mcp, "read_level", lambda url, token: level[0])
    hook = harness.level_gate(extras, "http://app", "t")

    refused = _decide(hook, "mcp__notebook__append_entry")
    assert refused["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "Drive" in refused["hookSpecificOutput"]["permissionDecisionReason"]
    assert _decide(hook, "mcp__notebook__search") == {}

    level[0] = "drive"  # read at each call: the operator's change holds at once
    assert _decide(hook, "mcp__notebook__append_entry") == {}


def test_the_gate_joins_the_playbooks_hooks(extras):
    pytest.importorskip("claude_agent_sdk")
    from claude_agent_sdk import HookMatcher

    reminder = HookMatcher(matcher="mcp__fibsem__(x)", hooks=[])
    hooks = harness._with_level_gate({"PreToolUse": [reminder]}, extras, None, None)
    first, second = hooks["PreToolUse"]
    assert first.matcher == "mcp__(notebook)__.*" and second is reminder
    assert harness._with_level_gate(None, {}, None, None) is None


SERVER = """\
import sys

from mcp.server.mcpserver import MCPServer

app = MCPServer("notebook")

@app.tool()
def search(text: str) -> str:
    return text

@app.tool()
def append_entry(text: str) -> str:
    return text

@app.tool()
def delete_everything() -> str:
    return "gone"

app.run(*sys.argv[1:])
"""


def test_the_tools_mcp_yaml_doesnt_list_are_hidden(tmp_path):
    pytest.importorskip("mcp")
    script = tmp_path / "notebook_server.py"
    script.write_text(SERVER, encoding="utf-8")
    file = tmp_path / "mcp.yaml"
    file.write_text(
        f"notebook:\n  command: [{sys.executable}, {script}]\n"
        "  tools:\n    - search: watch\n    - append_entry\n",
        encoding="utf-8",
    )
    extras = extra_mcp.load(file)
    assert extra_mcp.hidden(extras, {}) == ["mcp__notebook__delete_everything"]


def test_a_server_that_cant_be_listed_is_warned_about_not_fatal(extras, caplog):
    pytest.importorskip("mcp")
    assert harness._hidden(extras, {}) == []
    assert "may see tools mcp.yaml doesn't list" in caplog.text


def test_a_server_at_an_address_is_listed_with_its_token(tmp_path):
    """Over HTTP, with the token in the Authorization header."""
    pytest.importorskip("mcp")
    import socket
    import subprocess
    import time

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    script = tmp_path / "notebook_server.py"
    script.write_text(
        SERVER.replace(
            "app.run(*sys.argv[1:])", f'app.run("streamable-http", port={port})'
        ),
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 20
        while True:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
                break
            except OSError:
                assert time.monotonic() < deadline, "the server never came up"
                time.sleep(0.1)
        file = tmp_path / "mcp.yaml"
        file.write_text(
            f"notebook:\n  url: http://127.0.0.1:{port}/mcp\n"
            "  token: NOTEBOOK_TOKEN\n  tools: [search]\n",
            encoding="utf-8",
        )
        extras = extra_mcp.load(file)
        hidden = extra_mcp.hidden(extras, {"NOTEBOOK_TOKEN": "t"})
        assert sorted(hidden) == [
            "mcp__notebook__append_entry",
            "mcp__notebook__delete_everything",
        ]
    finally:
        process.terminate()
        process.wait(timeout=10)


def _all(tmp_path, level):
    file = tmp_path / "mcp.yaml"
    file.write_text(
        f"notebook:\n  command: [x]\n  level: {level}\n  tools: all\n"
        "  subagents: [search]\n",
        encoding="utf-8",
    )
    return extra_mcp.load(file)


def test_tools_all_allows_every_tool_the_server_offers(tmp_path):
    extras = _all(tmp_path, "drive")
    notebook = extras["notebook"]
    assert notebook.every and notebook.tools == []
    # The whole server, by its prefix; nothing to list or hide.
    assert extra_mcp.allowed(extras) == ["mcp__notebook"]
    assert extra_mcp.hidden(extras, {}) == []
    assert extra_mcp.subagent_allowed(extras) == ["mcp__notebook__search"]
    assert "They all need the agent at Drive." in extra_mcp.prompt_lines(extras)
    # Its level still decides.
    assert extra_mcp.refusal(extras, "mcp__notebook__anything", lambda: "watch")
    assert extra_mcp.refusal(extras, "mcp__notebook__anything", lambda: "drive") is None


def test_tools_all_at_watch_works_at_watch(tmp_path):
    extras = _all(tmp_path, "watch")
    assert extra_mcp.refusal(extras, "mcp__notebook__anything", lambda: "watch") is None
    assert "Drive" not in extra_mcp.prompt_lines(extras)
