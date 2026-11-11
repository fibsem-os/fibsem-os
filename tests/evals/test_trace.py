import json

from fibsem.ai.evals import trace
from fibsem.ai.evals.__main__ import main

LOG = """\
13:50:03 Model:    Claude Code's default
13:50:23 Read the fibsem-agent skill
13:50:23 Read task.md
13:50:35 Read the autolamella-polishing skill
13:50:41 Called plan_set_section {"section": "Brief"}
13:50:41 failed: Error executing tool plan_set_section: refused (409)
13:50:41 Called experiment_read_file {"path": "P2b_after_20um_eb.tif"}
13:50:44 agent: The band stops thinning after P2b.
It stays near 340 nm.
13:51:13 Wrote answer.json
"""


def run_folder(tmp_path, repeat=1, result="hit"):
    folder = tmp_path / "out" / "polish" / f"r{repeat}"
    (folder / "work").mkdir(parents=True)
    (folder / "agent.log").write_text(LOG)
    (folder / "work" / "answer.json").write_text(json.dumps({"stop_after": "P2b"}))
    scores = [{"part": None, "result": result, "detail": "stopped after P2b"}]
    record = {"case": "polish", "repeat": repeat, "scores": scores, "spent_usd": 0.4}
    (folder / "result.json").write_text(json.dumps(record))
    return folder


def test_the_log_says_skills_images_tools_and_words(tmp_path):
    got = trace.read_run(run_folder(tmp_path))
    assert got.skills == ["fibsem-agent", "autolamella-polishing"]
    assert got.looked_at == ["task.md", "P2b_after_20um_eb.tif"]
    assert got.tools == {"plan_set_section": 1, "experiment_read_file": 1}
    assert got.said == ["The band stops thinning after P2b.\nIt stays near 340 nm."]
    assert got.failed and "refused" in got.failed[0]


def test_the_transcript_gives_the_agents_words_in_full(tmp_path):
    folder = run_folder(tmp_path)
    (folder / "session").mkdir()
    lines = [
        {"type": "user", "message": {"content": "go"}},
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "All of it."}]},
        },
    ]
    (folder / "session" / "transcript.jsonl").write_text(
        "\n".join(json.dumps(line) for line in lines)
    )
    assert trace.read_run(folder).said == ["All of it."]


def test_trace_puts_repeats_side_by_side(tmp_path, capsys):
    run_folder(tmp_path, 1)
    run_folder(tmp_path, 2, result="late")
    assert main(["trace", str(tmp_path / "out")]) == 0
    text = capsys.readouterr().out
    assert "| r1 | hit | fibsem-agent, autolamella-polishing | 2 | 2 | $0.40 |" in text
    assert "### r2: late" in text and "> It stays near 340 nm." in text
    assert 'Answer: `{"stop_after": "P2b"}`' in text
