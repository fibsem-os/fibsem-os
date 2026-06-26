"""Tests for the Qt-free run preflight (block reasons + confirmation note)."""

import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fibsem.applications.autolamella.structures import (  # noqa: E402
    Experiment,
    GridRecord,
    Lamella,
)
from fibsem.applications.autolamella.workflows.run_preflight import (  # noqa: E402
    _exchange_note,
    _skip_note,
    build_run_preflight,
)
from fibsem.applications.autolamella.workflows.tasks.scheduling import Plan  # noqa: E402


def _exp(spec):
    exp = Experiment.create(path=tempfile.mkdtemp(), name="pf")
    recs = {}
    for g, n in spec.items():
        rec = GridRecord(name=g)
        exp.add_grid(rec)
        recs[g] = rec
        for i in range(n):
            exp.positions.append(
                Lamella(petname=f"{g}{i + 1}", path=f"{exp.path}/{g}{i + 1}",
                        number=i + 1, grid_id=rec._id))
    return exp, recs


# --- note builders (pure) ---

def test_exchange_note_none_without_exchanges():
    assert _exchange_note(Plan()) is None


def test_exchange_note_lists_grids_and_count():
    note = _exchange_note(Plan(n_exchanges=2, items_per_grid={"A": 9, "B": 6, "C": 9}))
    assert "2 grid exchange" in note and "A, B, C" in note


def test_skip_note_explains_reason():
    note = _skip_note(Plan(skipped=[("L1", "polish", "missing_prereqs")]))
    assert "1 item" in note and "prerequisite not in this run" in note


def test_skip_note_caps_long_lists():
    sk = [(f"L{i}", "polish", "failure") for i in range(10)]
    assert "+4 more" in _skip_note(Plan(skipped=sk))


# --- preflight against a real (static, no-loader) Demo holder ---

def _static_holder(microscope):
    """Force a clean static (no-loader) holder — robust to shared-session state."""
    microscope._stage.loader = None
    for slot in microscope._stage.holder.slots.values():
        slot.loaded_grid = None


def test_preflight_blocks_when_grid_not_placed():
    from fibsem import utils
    microscope, _ = utils.setup_session(manufacturer="Demo")
    _static_holder(microscope)  # no loader, nothing placed → A is unreachable
    exp, _ = _exp({"A": 1})
    pf = build_run_preflight(exp, microscope, ["mill_trench"], ["A1"])
    assert pf.blocked is not None and "A" in pf.blocked


def test_preflight_ok_when_grid_loaded():
    from fibsem import utils
    from fibsem.microscopes._stage import SampleGrid
    microscope, _ = utils.setup_session(manufacturer="Demo")
    _static_holder(microscope)
    next(iter(microscope._stage.holder.slots.values())).loaded_grid = SampleGrid(name="A")
    exp, _ = _exp({"A": 1})
    pf = build_run_preflight(exp, microscope, ["mill_trench"], ["A1"])
    assert pf.blocked is None and pf.note is None
