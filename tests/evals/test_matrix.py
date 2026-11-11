import json

import pytest

from fibsem.ai.evals import matrix
from fibsem.ai.evals.runner import CASE_PLAYBOOK

from .test_runner import STOP_AT_P3, case, config, home  # noqa: F401 (fixtures)


@pytest.fixture
def polish_case(case):
    case.data["playbook"] = "autolamella"
    return case


def session(out, cell, repeat):
    folder = out / cell / "polish" / f"r{repeat}" / "session" / "session.json"
    return json.loads(folder.read_text())


def test_every_cell_runs_repeat_by_repeat(home, polish_case, tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_FAKE_ANSWER", json.dumps(STOP_AT_P3))
    out = tmp_path / "matrix"
    grid = matrix.cells(["opus", "sonnet"], ["files", "code"], ["base", "case"])
    assert len(grid) == 8 and grid[0].name == "opus-files-base"
    matrix.start(out, grid, {"commit": "abc", "skills": "s", "knowledge": "k"})
    done = matrix.run_matrix(
        [polish_case], out, grid, config(tmp_path, timeout_s=20, repeats=2)
    )
    # Repeat 1 of every cell before any repeat 2.
    assert [r["repeat"] for r in done] == [1] * 8 + [2] * 8
    first = session(out, "sonnet-code-case", 1)
    assert (first["model"], first["code"], first["playbook"]) == (
        "sonnet",
        True,
        "autolamella",
    )
    assert session(out, "opus-files-base", 2)["playbook"] is None
    text = matrix.matrix_markdown(out)
    assert "Commit abc" in text and "| opus-code-case |" in text
    assert "| polish | " + " | ".join(["2/2 ($0.25)"] * 8) + " |" in text
    # The fake agent reads no skill.
    assert "| polish | " + " | ".join(["0/2"] * 8) + " |" in text


def test_a_matrix_continues_only_on_the_same_code(tmp_path):
    grid = matrix.cells(["opus"], ["files"], ["base"])
    version = {"commit": "abc", "skills": "s", "knowledge": "k"}
    matrix.start(tmp_path, grid, version)
    matrix.start(tmp_path, grid, version)  # the same: carries on
    with pytest.raises(matrix.MatrixError, match="skills"):
        matrix.start(tmp_path, grid, {**version, "skills": "t"})


def test_the_case_playbook_cell_runs_each_cases_own(polish_case, tmp_path):
    cell = matrix.Cell("opus", "files", CASE_PLAYBOOK)
    assert cell.config(config(tmp_path), tmp_path).playbook == CASE_PLAYBOOK
    assert matrix.Cell("opus", "code", "base").config(config(tmp_path), tmp_path).code
