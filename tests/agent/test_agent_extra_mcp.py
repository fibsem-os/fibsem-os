"""A lab's extra MCP servers: set up in one place, ``<state root>/agent/mcp.yaml``,
with each token in the key store, and refused with the reason when wrong."""

import pytest

from fibsem.ai.agent import extra_mcp

NOTEBOOK = """\
notebook:
  url: https://notebook.example/mcp
  token: NOTEBOOK_TOKEN
  about: The lab's shared notebook.
  tools: [get_overview, search, append_entry]
  subagents: [get_overview, search]
"""
SECRET = "not-a-real-token"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """The test's own state root, and no token stored."""
    monkeypatch.setenv("FIBSEM_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("NOTEBOOK_TOKEN", raising=False)
    monkeypatch.setattr(extra_mcp.keystore, "stored_secret", lambda name: None)
    (tmp_path / "home" / "agent").mkdir(parents=True)
    return tmp_path


def _write(home, text):
    (home / "home" / "agent" / "mcp.yaml").write_text(text, encoding="utf-8")


def test_without_the_file_there_are_none(home):
    assert extra_mcp.load() == {}
    assert extra_mcp.allowed({}) == extra_mcp.subagent_allowed({}) == []
    assert extra_mcp.prompt_lines({}) == ""
    assert extra_mcp.definitions({}) == extra_mcp.tokens({}) == {}


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


def test_the_entry_names_the_token_and_never_holds_it(home, monkeypatch):
    _write(home, NOTEBOOK)
    extras = extra_mcp.load()
    assert extra_mcp.definitions(extras) == {
        "notebook": {
            "type": "http",
            "url": "https://notebook.example/mcp",
            "headers": {"Authorization": "Bearer ${NOTEBOOK_TOKEN}"},
        }
    }
    monkeypatch.setenv("NOTEBOOK_TOKEN", SECRET)
    assert extra_mcp.tokens(extras) == {"NOTEBOOK_TOKEN": SECRET}


def test_the_token_comes_from_the_key_store(home, monkeypatch):
    _write(home, NOTEBOOK)
    stored = {"NOTEBOOK_TOKEN": SECRET}
    monkeypatch.setattr(extra_mcp.keystore, "stored_secret", stored.get)
    assert extra_mcp.tokens(extra_mcp.load()) == {"NOTEBOOK_TOKEN": SECRET}


def test_a_token_not_stored_is_refused_saying_how_to_store_it(home):
    _write(home, NOTEBOOK)
    with pytest.raises(
        extra_mcp.ExtraMcpError, match="fibsem-cli key set NOTEBOOK_TOKEN"
    ):
        extra_mcp.tokens(extra_mcp.load())


def test_a_server_started_by_a_command_gets_its_token_in_its_environment(home):
    _write(
        home,
        "papers:\n  command: [papers-mcp, --stdio]\n  token: PAPERS_KEY\n"
        "  tools: [search]\n"
        "open:\n  url: http://127.0.0.1:9000/mcp\n  tools: [read]\n",
    )
    assert extra_mcp.definitions(extra_mcp.load()) == {
        "papers": {
            "type": "stdio",
            "command": "papers-mcp",
            "args": ["--stdio"],
            "env": {"PAPERS_KEY": "${PAPERS_KEY}"},
        },
        "open": {"type": "http", "url": "http://127.0.0.1:9000/mcp"},
    }


URL = "  url: https://n.example/mcp\n"


@pytest.mark.parametrize(
    "text, reason",
    [
        ("notebook:\n  tools: [search]\n" + URL + "  headers: {}\n", "unknown setting"),
        ("fibsem:\n  tools: [search]\n" + URL, "fibsem-agent's own server"),
        ("fibsem_agent:\n  tools: [search]\n" + URL, "fibsem-agent's own server"),
        ("note book:\n  tools: [search]\n" + URL, "not a server name"),
        ("notebook:\n  about: Ours.\n" + URL, "list the tools"),
        ("notebook:\n  tools: []\n" + URL, "list the tools"),
        ("notebook:\n  tools: ['*']\n" + URL, "not a tool's name"),
        (
            "notebook:\n  tools: [search]\n  subagents: [append_entry]\n" + URL,
            "only its",
        ),
        ("notebook:\n  tools: [search]\n", "one of the two"),
        (
            "notebook:\n  tools: [search]\n" + URL + "  command: [x]\n",
            "one of the two",
        ),
        ("notebook:\n  tools: [search]\n  url: n.example\n", "http:// or https://"),
        ("notebook:\n  tools: [search]\n  command: []\n", "command is a list"),
        ("notebook:\n  tools: [search]\n  command: x\n", "command is a list"),
        ("notebook:\n  tools: [search]\n" + URL + "  token: abc\n", "NOTEBOOK_TOKEN"),
        ("notebook:\n  tools: [search]\n" + URL + "  token: NOTEBOOK\n", "KEY, TOKEN"),
        ("- notebook\n", "each server is a name"),
    ],
)
def test_a_file_that_cant_be_used_is_refused_saying_why(home, text, reason):
    _write(home, text)
    with pytest.raises(extra_mcp.ExtraMcpError, match=reason):
        extra_mcp.load()
