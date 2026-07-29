"""The Scripts menu, exercised without any application around it (FIB-338).

The controller takes a folder, a context factory and a notifier, so it can be
driven standalone -- which is the point of it not living in AutoLamellaMainUI.
"""

from pathlib import Path

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QMenu  # noqa: E402

from fibsem.ui.widgets.script_menu import MANAGE_LABEL, ScriptMenuController  # noqa: E402


@pytest.fixture
def qapp():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


class FakeContext:
    """Stands in for whatever an application hands its scripts."""

    def __init__(self):
        self.saved = False
        self.value = "from-context"

    def save(self):
        self.saved = True


def _write(directory: Path, name: str, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(body)
    return path


def _controller(tmp_path, context=None, reason="", notes=None, confirm=True):
    menu = QMenu()
    controller = ScriptMenuController(
        menu=menu,
        scripts_directory=lambda: tmp_path,
        context_factory=lambda: (context, reason),
        notify=lambda msg, level: (notes if notes is not None else []).append((level, msg)),
        # the real one is a modal box; a stub keeps the tests from blocking on it
        confirm=lambda question, detail: confirm,
    )
    return controller, menu


def test_menu_lists_scripts_by_name(qapp, tmp_path):
    _write(tmp_path, "export.py", '"""Export a summary."""\ndef run(ctx):\n    pass\n')
    controller, menu = _controller(tmp_path, context=FakeContext())

    controller.rebuild()

    labels = [a.text() for a in menu.actions() if a.text()]
    assert "export" in labels


def test_a_script_that_writes_is_labelled(qapp, tmp_path):
    """A script that rewrites state must not look identical to one that reads."""
    _write(tmp_path, "mark.py", "writes = True\ndef run(ctx):\n    pass\n")
    controller, menu = _controller(tmp_path, context=FakeContext())

    controller.rebuild()

    assert any("writes" in a.text() for a in menu.actions())


def test_a_broken_script_is_summarised_and_points_at_the_manager(qapp, tmp_path):
    """A menu has nowhere to put a failure reason, so it reports the count and
    sends the user to the dialog. Omitting it entirely is what we are avoiding."""
    _write(tmp_path, "broken.py", "def main(ctx):\n    pass\n")
    controller, menu = _controller(tmp_path, context=FakeContext())

    controller.rebuild()

    labels = [a.text() for a in menu.actions()]
    assert not any(t.startswith("broken") for t in labels)
    assert any("failed to load" in t for t in labels)
    assert any(t == MANAGE_LABEL for t in labels)


def test_the_manager_entry_is_always_offered(qapp, tmp_path):
    controller, menu = _controller(tmp_path, context=FakeContext())
    controller.rebuild()
    assert any(a.text() == MANAGE_LABEL for a in menu.actions())


def test_entries_are_disabled_and_explained_when_the_host_is_not_ready(qapp, tmp_path):
    _write(tmp_path, "export.py", "def run(ctx):\n    pass\n")
    controller, menu = _controller(tmp_path, context=None, reason="Load an experiment")

    controller.rebuild()

    export = next(a for a in menu.actions() if a.text().startswith("export"))
    assert not export.isEnabled()
    assert any(a.text() == "Load an experiment" for a in menu.actions())


def test_empty_folder_says_so_rather_than_showing_nothing(qapp, tmp_path):
    controller, menu = _controller(tmp_path, context=FakeContext())

    controller.rebuild()

    assert any(a.text() == "No scripts found" for a in menu.actions())


def test_open_folder_action_is_always_offered(qapp, tmp_path):
    controller, menu = _controller(tmp_path, context=FakeContext())
    controller.rebuild()
    assert any(a.text() == "Open scripts folder" for a in menu.actions())


def test_running_passes_the_context_through(qapp, tmp_path):
    _write(tmp_path, "read.py", "def run(ctx):\n    return ctx.value\n")
    context = FakeContext()
    controller, _ = _controller(tmp_path, context=context)

    result = controller.runner.run(controller.runner.discover()[0])

    assert result.ok and result.value == "from-context"


def test_writes_flag_triggers_the_context_save_hook(qapp, tmp_path):
    _write(tmp_path, "w.py", "writes = True\ndef run(ctx):\n    pass\n")
    context = FakeContext()
    controller, _ = _controller(tmp_path, context=context)

    controller.runner.run(controller.runner.discover()[0])

    assert context.saved


def test_a_writes_script_is_confirmed_before_it_runs(qapp, tmp_path):
    """It saves the moment it finishes and there is no undo."""
    _write(tmp_path, "w.py", "writes = True\ndef run(ctx):\n    pass\n")
    asked = []
    controller, _ = _controller(tmp_path, context=FakeContext())
    controller.runner.confirm = lambda q, detail: asked.append((q, detail)) or True

    controller.runner.run(controller.runner.discover()[0])

    assert asked, "a writes script ran without asking"
    question, detail = asked[0]
    assert "w" in question and "no undo" in detail


def test_declining_the_confirmation_does_not_run_or_save(qapp, tmp_path):
    _write(tmp_path, "w.py", "writes = True\ndef run(ctx):\n    ctx.ran = True\n")
    context = FakeContext()
    controller, _ = _controller(tmp_path, context=context, confirm=False)

    assert controller.runner.run(controller.runner.discover()[0]) is None
    assert not context.saved
    assert not getattr(context, "ran", False)


def test_the_real_confirmation_box_builds(qapp, tmp_path, monkeypatch):
    """Every other test injects a stub for it, so the box itself is otherwise
    never constructed and a mistake in it would only show up in the app."""
    from PyQt5.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "exec_", lambda self: 0)
    controller, _ = _controller(tmp_path, context=FakeContext())

    # no button clicked -> not confirmed, which is the safe direction
    assert controller.runner._default_confirm("Run 'x'?", "It writes.") is False


def test_a_read_only_script_is_not_confirmed(qapp, tmp_path):
    """The common case must stay one click -- a prompt on every run gets clicked
    through without reading, which is worse than not having one."""
    _write(tmp_path, "r.py", "def run(ctx):\n    pass\n")
    controller, _ = _controller(tmp_path, context=FakeContext())
    controller.runner.confirm = lambda q, detail: pytest.fail("asked to confirm a read")

    assert controller.runner.run(controller.runner.discover()[0]).ok


def test_without_the_flag_the_save_hook_is_not_called(qapp, tmp_path):
    _write(tmp_path, "r.py", "def run(ctx):\n    pass\n")
    context = FakeContext()
    controller, _ = _controller(tmp_path, context=context)

    controller.runner.run(controller.runner.discover()[0])

    assert not context.saved


def test_a_failing_script_notifies_instead_of_raising(qapp, tmp_path):
    """PyQt5 aborts the process on an exception escaping a slot (FIB-329)."""
    _write(tmp_path, "bad.py", "def run(ctx):\n    raise RuntimeError('nope')\n")
    notes = []
    controller, _ = _controller(tmp_path, context=FakeContext(), notes=notes)

    result = controller.runner.run(controller.runner.discover()[0])

    assert not result.ok
    assert notes and notes[-1][0] == "error"


def test_a_microscope_script_is_refused_until_the_strict_runner_exists(qapp, tmp_path):
    """FIB-340 owns the worker thread, hardware lock and state restoration."""
    _write(tmp_path, "hw.py", "uses_microscope = True\ndef run(ctx):\n    raise AssertionError\n")
    notes = []
    controller, _ = _controller(tmp_path, context=FakeContext(), notes=notes)

    assert controller.runner.run(controller.runner.discover()[0]) is None
    assert notes and notes[-1][0] == "warning"


def test_running_is_refused_when_the_host_has_no_context(qapp, tmp_path):
    _write(tmp_path, "export.py", "def run(ctx):\n    raise AssertionError\n")
    notes = []
    controller, _ = _controller(tmp_path, context=None, reason="Load an experiment", notes=notes)

    assert controller.runner.run(controller.runner.discover()[0]) is None
    assert notes and notes[-1] == ("warning", "Load an experiment")
