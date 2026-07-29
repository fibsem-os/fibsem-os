"""The Scripts menu, exercised without any application around it (FIB-338).

The menu is a way *in*, not a way to run: it offers the manager and the folder
and nothing else. Running lives in the dialog, which is the only surface with a
Stop button. What the runner does once something is started is pinned down in
test_script_runner.py.
"""

from pathlib import Path

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QMenu  # noqa: E402

from fibsem.ui.widgets.script_menu import (  # noqa: E402
    MANAGE_LABEL,
    OPEN_FOLDER_LABEL,
    ScriptMenuController,
)


@pytest.fixture
def qapp():
    from PyQt5.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


class FakeContext:
    def save(self):
        pass


def _write(directory: Path, name: str, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(body)
    return path


def _controller(tmp_path, context=None, reason=""):
    menu = QMenu()
    controller = ScriptMenuController(
        menu=menu,
        scripts_directory=lambda: tmp_path,
        context_factory=lambda: (context, reason),
        notify=lambda msg, level: None,
    )
    return controller, menu


def test_the_menu_offers_the_manager_and_the_folder(qapp, tmp_path):
    _, menu = _controller(tmp_path, context=FakeContext())
    assert [a.text() for a in menu.actions()] == [MANAGE_LABEL, OPEN_FOLDER_LABEL]


def test_the_menu_never_offers_to_run_a_script(qapp, tmp_path):
    """A menu item cannot be stopped. A microscope script started from one would
    run to completion with no way to interrupt it, so nothing here starts one."""
    _write(tmp_path, "export.py", '"""Export a summary."""\ndef run(ctx):\n    pass\n')
    _write(tmp_path, "hw.py", "uses_microscope = True\ndef run(ctx):\n    pass\n")

    _, menu = _controller(tmp_path, context=FakeContext())

    assert not any("export" in a.text() or "hw" in a.text() for a in menu.actions())


def test_the_menu_is_the_same_whatever_is_in_the_folder(qapp, tmp_path):
    """Nothing here depends on the folder, so it is built once rather than
    re-importing every script each time the Tools menu opens."""
    _, empty = _controller(tmp_path, context=FakeContext())
    before = [a.text() for a in empty.actions()]

    _write(tmp_path, "broken.py", "def main(ctx):\n    pass\n")
    _write(tmp_path, "good.py", "def run(ctx):\n    pass\n")
    _, populated = _controller(tmp_path, context=FakeContext())

    assert [a.text() for a in populated.actions()] == before


def test_the_menu_is_usable_with_no_experiment_loaded(qapp, tmp_path):
    """Both entries still work -- the dialog is where "why can I not run this"
    gets explained, and you have to be able to reach it to read that."""
    _, menu = _controller(tmp_path, context=None, reason="Load an experiment")
    assert all(a.isEnabled() for a in menu.actions())


def test_the_controller_exposes_the_runner_it_built(qapp, tmp_path):
    """AutoLamellaUI reaches through it to refuse a workflow while a microscope
    script is running (FIB-340)."""
    controller, _ = _controller(tmp_path, context=FakeContext())
    assert controller.runner.is_running is False
