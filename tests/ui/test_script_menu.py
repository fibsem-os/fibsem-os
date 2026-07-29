"""The Scripts menu, exercised without any application around it (FIB-338).

The controller takes a folder, a context factory and a notifier, so it can be
driven standalone -- which is the point of it not living in AutoLamellaMainUI.
"""

from pathlib import Path

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QMenu  # noqa: E402

from fibsem.ui.widgets.script_menu import ScriptMenuController  # noqa: E402


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


def _controller(tmp_path, context=None, reason="", notes=None):
    menu = QMenu()
    controller = ScriptMenuController(
        menu=menu,
        scripts_directory=lambda: tmp_path,
        context_factory=lambda: (context, reason),
        notify=lambda msg, level: (notes if notes is not None else []).append((level, msg)),
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


def test_a_broken_script_is_shown_disabled_with_the_reason(qapp, tmp_path):
    """Omitting it would leave the author with nothing to diagnose."""
    _write(tmp_path, "broken.py", "def main(ctx):\n    pass\n")
    controller, menu = _controller(tmp_path, context=FakeContext())

    controller.rebuild()

    action = next(a for a in menu.actions() if a.text().startswith("broken"))
    assert not action.isEnabled()
    assert "run()" in action.toolTip()


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

    result = controller.run(controller.discover()[0])

    assert result.ok and result.value == "from-context"


def test_writes_flag_triggers_the_context_save_hook(qapp, tmp_path):
    _write(tmp_path, "w.py", "writes = True\ndef run(ctx):\n    pass\n")
    context = FakeContext()
    controller, _ = _controller(tmp_path, context=context)

    controller.run(controller.discover()[0])

    assert context.saved


def test_without_the_flag_the_save_hook_is_not_called(qapp, tmp_path):
    _write(tmp_path, "r.py", "def run(ctx):\n    pass\n")
    context = FakeContext()
    controller, _ = _controller(tmp_path, context=context)

    controller.run(controller.discover()[0])

    assert not context.saved


def test_a_failing_script_notifies_instead_of_raising(qapp, tmp_path):
    """PyQt5 aborts the process on an exception escaping a slot (FIB-329)."""
    _write(tmp_path, "bad.py", "def run(ctx):\n    raise RuntimeError('nope')\n")
    notes = []
    controller, _ = _controller(tmp_path, context=FakeContext(), notes=notes)

    result = controller.run(controller.discover()[0])

    assert not result.ok
    assert notes and notes[-1][0] == "error"


def test_a_microscope_script_is_refused_until_the_strict_runner_exists(qapp, tmp_path):
    """FIB-340 owns the worker thread, hardware lock and state restoration."""
    _write(tmp_path, "hw.py", "uses_microscope = True\ndef run(ctx):\n    raise AssertionError\n")
    notes = []
    controller, _ = _controller(tmp_path, context=FakeContext(), notes=notes)

    assert controller.run(controller.discover()[0]) is None
    assert notes and notes[-1][0] == "warning"


def test_running_is_refused_when_the_host_has_no_context(qapp, tmp_path):
    _write(tmp_path, "export.py", "def run(ctx):\n    raise AssertionError\n")
    notes = []
    controller, _ = _controller(tmp_path, context=None, reason="Load an experiment", notes=notes)

    assert controller.run(controller.discover()[0]) is None
    assert notes and notes[-1] == ("warning", "Load an experiment")
