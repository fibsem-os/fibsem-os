"""ScriptRunner: what may run, how it runs, and what the user is told (FIB-338/340).

Driven with no menu and no dialog around it -- the runner is the one path both
surfaces go through, so this is where the running rules are pinned down.
"""

import time
from pathlib import Path

import pytest

pytest.importorskip("PyQt5")

from fibsem.cancellation import OperationCancelledError  # noqa: E402
from fibsem.ui.widgets.script_runner import ScriptRunner  # noqa: E402


@pytest.fixture
def qapp():
    from PyQt5.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


class FakeContext:
    """Stands in for whatever an application hands its scripts.

    Same shape as the real ScriptContext, including the cancellation surface the
    runner injects into for threaded scripts.
    """

    def __init__(self):
        self.saved = False
        self.value = "from-context"
        self.stop_event = None

    def save(self):
        self.saved = True

    @property
    def cancelled(self):
        return self.stop_event is not None and self.stop_event.is_set()

    def raise_if_cancelled(self):
        from fibsem.cancellation import raise_if_cancelled

        raise_if_cancelled(self.stop_event)


def _drain(qapp, predicate, timeout=5.0):
    """Pump the Qt event loop until predicate() is true, or give up.

    Threaded scripts report back through a queued signal, so nothing is delivered
    unless the loop runs.
    """
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    return predicate()


def _write(directory: Path, name: str, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(body)
    return path


def _runner(tmp_path, context=None, reason="", notes=None, confirm=True):
    return ScriptRunner(
        scripts_directory=lambda: tmp_path,
        context_factory=lambda: (context, reason),
        notify=lambda msg, level: (notes if notes is not None else []).append(
            (level, msg)
        ),
        # the real one is a modal box; a stub keeps the tests from blocking on it
        confirm=lambda question, detail: confirm,
    )


def test_running_passes_the_context_through(qapp, tmp_path):
    _write(tmp_path, "read.py", "def run(ctx):\n    return ctx.value\n")
    context = FakeContext()
    runner = _runner(tmp_path, context=context)

    result = runner.run(runner.discover()[0])

    assert result.ok and result.value == "from-context"


def test_writes_flag_triggers_the_context_save_hook(qapp, tmp_path):
    _write(tmp_path, "w.py", "writes = True\ndef run(ctx):\n    pass\n")
    context = FakeContext()
    runner = _runner(tmp_path, context=context)

    runner.run(runner.discover()[0])

    assert context.saved


def test_a_writes_script_is_confirmed_before_it_runs(qapp, tmp_path):
    """It saves the moment it finishes and there is no undo."""
    _write(tmp_path, "w.py", "writes = True\ndef run(ctx):\n    pass\n")
    asked = []
    runner = _runner(tmp_path, context=FakeContext())
    runner.confirm = lambda q, detail: asked.append((q, detail)) or True

    runner.run(runner.discover()[0])

    assert asked, "a writes script ran without asking"
    question, detail = asked[0]
    assert "w" in question and "no undo" in detail


def test_declining_the_confirmation_does_not_run_or_save(qapp, tmp_path):
    _write(tmp_path, "w.py", "writes = True\ndef run(ctx):\n    ctx.ran = True\n")
    context = FakeContext()
    runner = _runner(tmp_path, context=context, confirm=False)

    assert runner.run(runner.discover()[0]) is None
    assert not context.saved
    assert not getattr(context, "ran", False)


def test_the_real_confirmation_box_builds(qapp, tmp_path, monkeypatch):
    """Every other test injects a stub for it, so the box itself is otherwise
    never constructed and a mistake in it would only show up in the app."""
    from PyQt5.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "exec_", lambda self: 0)
    runner = _runner(tmp_path, context=FakeContext())

    # no button clicked -> not confirmed, which is the safe direction
    assert runner._default_confirm("Run 'x'?", "It writes.") is False


def test_a_read_only_script_is_not_confirmed(qapp, tmp_path):
    """The common case must stay one click -- a prompt on every run gets clicked
    through without reading, which is worse than not having one."""
    _write(tmp_path, "r.py", "def run(ctx):\n    pass\n")
    runner = _runner(tmp_path, context=FakeContext())
    runner.confirm = lambda q, detail: pytest.fail("asked to confirm a read")

    assert runner.run(runner.discover()[0]).ok


def test_without_the_flag_the_save_hook_is_not_called(qapp, tmp_path):
    _write(tmp_path, "r.py", "def run(ctx):\n    pass\n")
    context = FakeContext()
    runner = _runner(tmp_path, context=context)

    runner.run(runner.discover()[0])

    assert not context.saved


def test_a_failing_script_notifies_instead_of_raising(qapp, tmp_path):
    """PyQt5 aborts the process on an exception escaping a slot (FIB-329)."""
    _write(tmp_path, "bad.py", "def run(ctx):\n    raise RuntimeError('nope')\n")
    notes = []
    runner = _runner(tmp_path, context=FakeContext(), notes=notes)

    result = runner.run(runner.discover()[0])

    assert not result.ok
    assert notes and notes[-1][0] == "error"


def test_a_microscope_script_runs_off_the_gui_thread(qapp, tmp_path):
    """Hardware operations take seconds to minutes. Inline they would freeze the
    window for the whole run, which is indistinguishable from a crash."""
    _write(
        tmp_path,
        "hw.py",
        "import threading\n"
        "uses_microscope = True\n"
        "def run(ctx):\n"
        "    return threading.current_thread() is threading.main_thread()\n",
    )
    done = []
    runner = _runner(tmp_path, context=FakeContext())

    # returns None because it has not finished yet -- the result arrives by callback
    assert runner.run(runner.discover()[0], on_finished=done.append) is None
    assert _drain(qapp, lambda: bool(done)), "the script never reported back"
    assert done[0].value is False, "ran on the main thread"


def test_the_microscope_is_withheld_without_the_flag(qapp, tmp_path):
    """Otherwise a script could drive the stage while the dialog still labels it
    'Data / Read-only' -- an active falsehood, worse than saying nothing."""
    _write(tmp_path, "sneaky.py", "def run(ctx):\n    return ctx.microscope\n")
    context = FakeContext()
    context.microscope = object()
    runner = _runner(tmp_path, context=context)

    result = runner.run(runner.discover()[0])

    assert result.ok and result.value is None
    # put back, so a host that reuses one context object is not left without it
    assert context.microscope is not None


def test_forgetting_the_flag_says_which_flag_to_add(qapp, tmp_path):
    """The traceback names the line, but the toast is all most users see, and
    "'NoneType' has no attribute acquire_image" does not say what to do."""
    _write(
        tmp_path, "forgot.py", "def run(ctx):\n    return ctx.microscope.acquire()\n"
    )
    context = FakeContext()
    context.microscope = object()
    notes = []
    runner = _runner(tmp_path, context=context, notes=notes)

    runner.run(runner.discover()[0])

    level, message = notes[-1]
    assert level == "error"
    assert "uses_microscope = True" in message


def test_a_declared_script_gets_the_real_microscope(qapp, tmp_path):
    _write(
        tmp_path,
        "hw.py",
        "uses_microscope = True\ndef run(ctx):\n    return ctx.microscope\n",
    )
    context = FakeContext()
    context.microscope = microscope = object()
    done = []
    runner = _runner(tmp_path, context=context)

    runner.run(runner.discover()[0], on_finished=done.append)

    assert _drain(qapp, lambda: bool(done))
    assert done[0].value is microscope


def test_a_microscope_script_is_confirmed_before_it_runs(qapp, tmp_path):
    """It has already moved the hardware by the time it finishes."""
    _write(tmp_path, "hw.py", "uses_microscope = True\ndef run(ctx):\n    pass\n")
    asked = []
    runner = _runner(tmp_path, context=FakeContext())
    runner.confirm = lambda q, detail: asked.append(detail) or True

    runner.run(runner.discover()[0])
    _drain(qapp, lambda: not runner.is_running)

    assert asked and "drives the microscope" in asked[0]


def test_declining_a_microscope_confirmation_does_not_start_a_thread(qapp, tmp_path):
    _write(
        tmp_path,
        "hw.py",
        "uses_microscope = True\ndef run(ctx):\n    raise AssertionError\n",
    )
    runner = _runner(tmp_path, context=FakeContext(), confirm=False)

    assert runner.run(runner.discover()[0]) is None
    assert not runner.is_running


def test_a_script_that_both_writes_and_drives_says_both(qapp, tmp_path):
    _write(
        tmp_path,
        "hw.py",
        "uses_microscope = True\nwrites = True\ndef run(ctx):\n    pass\n",
    )
    asked = []
    runner = _runner(tmp_path, context=FakeContext())
    runner.confirm = lambda q, detail: asked.append(detail) or True

    runner.run(runner.discover()[0])
    _drain(qapp, lambda: not runner.is_running)

    assert "drives the microscope" in asked[0] and "modifies the experiment" in asked[0]


def test_two_consequences_are_two_sentences_not_a_chain_of_ands(qapp, tmp_path):
    """Joining the clauses with "and" produced "nothing checks what it does and
    modifies the experiment", where the second and attaches to "nothing checks" --
    so the warning stated the opposite of what it meant, on the worst script type."""
    _write(
        tmp_path,
        "hw.py",
        "uses_microscope = True\nwrites = True\ndef run(ctx):\n    pass\n",
    )
    runner = _runner(tmp_path, context=FakeContext())

    detail = runner._consequence(runner.discover()[0])

    assert detail == (
        "It drives the microscope, and nothing checks what it does. "
        "It modifies the experiment and saves it when it finishes. "
        "There is no undo."
    )
    assert "does and modifies" not in detail


def test_only_one_script_runs_at_a_time(qapp, tmp_path):
    """Two threads commanding one microscope is the failure this prevents."""
    _write(
        tmp_path,
        "slow.py",
        "uses_microscope = True\n"
        "def run(ctx):\n"
        "    ctx.stop_event.wait(3)\n"
        "    return 'done'\n",
    )
    notes = []
    runner = _runner(tmp_path, context=FakeContext(), notes=notes)
    script = runner.discover()[0]

    runner.run(script)
    assert _drain(qapp, lambda: runner.is_running)

    assert runner.run(script) is None
    assert notes[-1] == ("warning", "A script is already running.")

    runner.stop()
    assert _drain(qapp, lambda: not runner.is_running)


def test_stop_is_cooperative_and_reports_as_cancelled(qapp, tmp_path):
    """A Python thread cannot be killed, so Stop only sets a flag -- the script
    has to look at it."""
    _write(
        tmp_path,
        "slow.py",
        "uses_microscope = True\n"
        "def run(ctx):\n"
        "    ctx.stop_event.wait(3)\n"
        "    ctx.raise_if_cancelled()\n"
        "    return 'ran to completion'\n",
    )
    done, notes = [], []
    runner = _runner(tmp_path, context=FakeContext(), notes=notes)

    runner.run(runner.discover()[0], on_finished=done.append)
    assert _drain(qapp, lambda: runner.is_running)
    runner.stop()

    assert _drain(qapp, lambda: bool(done)), "the script never reported back"
    assert isinstance(done[0].error, OperationCancelledError)
    # a cancel is not a failure, and must not be reported as one
    assert notes[-1] == ("warning", "slow cancelled.")


def test_a_script_that_ignores_stop_still_finishes(qapp, tmp_path):
    """Documented behaviour, not a bug: nothing can force it to stop."""
    _write(
        tmp_path,
        "stubborn.py",
        "uses_microscope = True\n"
        "def run(ctx):\n"
        "    ctx.stop_event.wait(0.05)\n"
        "    return 'finished anyway'\n",
    )
    done = []
    runner = _runner(tmp_path, context=FakeContext())

    runner.run(runner.discover()[0], on_finished=done.append)
    runner.stop()

    assert _drain(qapp, lambda: bool(done))
    assert done[0].ok and done[0].value == "finished anyway"


def test_running_is_refused_when_the_host_has_no_context(qapp, tmp_path):
    _write(tmp_path, "export.py", "def run(ctx):\n    raise AssertionError\n")
    notes = []
    runner = _runner(tmp_path, context=None, reason="Load an experiment", notes=notes)

    assert runner.run(runner.discover()[0]) is None
    assert notes and notes[-1] == ("warning", "Load an experiment")
