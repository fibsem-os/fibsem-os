"""Report section flags and column renames, both of which failed silently.

A section key that matches nothing is a no-op wearing the costume of a working
control: the checkbox moves, the default wins, the report is unchanged. A column
rename that matches nothing leaves the raw column name in the PDF. Neither raises,
so neither surfaced until someone compared the output against what they asked for.

Both sides are read as source text rather than imported. The reporting module pulls in
reportlab (the `reporting` extra) and the dialog pulls in PyQt5 (the `ui` extra), and
CI installs neither. Importing them would make these skip in CI, and a regression test
that never runs is not much of a guard against a regression. What is being checked is
the literal spelling on each side, which the text carries.

(This used to name matplotlib_scalebar as a third reason and put all three in the `ui`
extra. Neither was right: scalebar is a core dependency now (FIB-562), and reportlab
was always its own extra. reportlab alone still makes the import unavailable here.)
"""

import re
from pathlib import Path

import pandas as pd
import pytest

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaWorkflowConfig,
    Experiment,
)

_FIBSEM = Path(__file__).resolve().parents[2] / "fibsem"
_REPORTING = _FIBSEM / "applications" / "autolamella" / "tools" / "reporting.py"
_DATA = _FIBSEM / "applications" / "autolamella" / "tools" / "data.py"
_WIDGET = _FIBSEM / "applications" / "autolamella" / "ui" / "autolamella_generate_report_widget.py"


def _report_sections() -> set:
    """The section names generate_report2 builds its defaults from."""
    block = _REPORTING.read_text(encoding="utf-8").split("REPORT_SECTIONS = (", 1)[1].split(")", 1)[0]
    return set(re.findall(r'"(\w+)"', block))


def _widget_section_keys() -> set:
    """The section keys the report dialog emits."""
    src = _WIDGET.read_text(encoding="utf-8")
    block = src.split("def _get_sections_config", 1)[1].split("def ", 1)[0]
    return set(re.findall(r'"(\w+)":\s*self\.checkbox', block))


def _workflow_rename_keys() -> set:
    """The columns format_pretty_dataframes renames on the workflow frame."""
    block = _DATA.read_text(encoding="utf-8").split("# WORKFLOW", 1)[1].split("# TASK HISTORY", 1)[0]
    return set(re.findall(r'"(\w+)":\s*"', block))


def test_the_widget_emits_exactly_the_sections_the_report_reads() -> None:
    """The defect: the widget said "lamella_images", the report read
    "lamella_workflow_images". The default True always won, so unchecking
    "Per-Lamella Workflow Images" did nothing."""
    assert _widget_section_keys() == _report_sections()


def test_the_report_sections_are_not_empty() -> None:
    """Guard on the parsing above -- two empty sets would compare equal."""
    assert len(_report_sections()) >= 6
    assert "lamella_workflow_images" in _report_sections()


def test_the_workflow_rename_targets_columns_that_exist() -> None:
    """The rename asked for "task"; workflow_dataframe() produces "task_name".
    pandas ignores unknown labels by default, so the PDF showed the raw name.

    Checked against the real dataframe rather than a hand-written column list, so
    this fails if either side moves.
    """
    exp = Experiment(path="/tmp", name="x")
    exp.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[
                AutoLamellaTaskDescription(
                    name="MillTrench", supervise=False, required=True
                )
            ]
        )
    )

    rename_keys = _workflow_rename_keys()

    assert rename_keys, "parsed no rename keys -- the source layout moved"
    assert rename_keys <= set(exp.workflow_dataframe().columns)


def test_a_stale_rename_key_raises_rather_than_no_opping() -> None:
    """errors="raise" is what turns the silent case loud, so pin the behaviour."""
    df = pd.DataFrame([{"order": 1, "task_name": "MillTrench"}])

    with pytest.raises(KeyError):
        df.rename(columns={"task": "Task Name"}, errors="raise")
