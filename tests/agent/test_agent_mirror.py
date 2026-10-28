"""The Claude Code transcript, copied into the session folder as it is written."""

import asyncio
import json

from fibsem.ai.agent import mirror
from fibsem.ai.agent.mirror import TranscriptMirror


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_the_main_session_and_a_specialists_go_in_their_own_files(tmp_path):
    store = TranscriptMirror(tmp_path, "main-id")
    main = {"project_key": "p", "session_id": "main-id"}
    specialist = {**main, "subpath": "subagents/agent-1"}
    asyncio.run(store.append(main, [{"type": "user", "uuid": "1"}]))
    asyncio.run(store.append(main, [{"type": "assistant", "uuid": "2"}]))
    asyncio.run(store.append(specialist, [{"type": "assistant", "uuid": "3"}]))

    assert [e["uuid"] for e in _lines(tmp_path / mirror.FILE)] == ["1", "2"]
    copied = tmp_path / mirror.FOLDER / "main-id" / "subagents" / "agent-1.jsonl"
    assert _lines(copied) == [{"type": "assistant", "uuid": "3"}]


def test_resuming_reads_the_clis_own_copy():
    store = TranscriptMirror(None, "main-id")
    assert (
        asyncio.run(store.load({"project_key": "p", "session_id": "main-id"})) is None
    )


def test_a_folder_that_cant_be_written_costs_the_run_nothing(tmp_path):
    (tmp_path / "file").write_text("")
    store = TranscriptMirror(tmp_path / "file", "s")
    asyncio.run(store.append({"project_key": "p", "session_id": "s"}, [{"type": "x"}]))


def test_fibsem_agent_mirrors_into_the_session_folder(tmp_path):
    from fibsem.ai.agent import harness

    options = harness.build_options(
        harness.find_checkout(), session_id="abc", transcript=tmp_path
    )
    assert options.session_store.folder == tmp_path
    assert options.session_store.session_id == "abc"
    assert harness.build_options(harness.find_checkout()).session_store is None
