"""The coincidence viewer records its own task output.

It writes its FIB image directly rather than going through a task, so it appends a
history entry by hand and has to record the output itself. Lives here rather than
with the other discovery tests because importing the viewer pulls in the UI stack.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.structures import Lamella
from fibsem.applications.autolamella.task_outputs import final_reference_images


def _lamella(tmp_path: Path) -> Lamella:
    lamella = Lamella(path=tmp_path / "lam", number=0, petname="test")
    os.makedirs(lamella.path, exist_ok=True)
    return lamella


def test_the_coincidence_viewer_records_what_the_panel_will_find(tmp_path, monkeypatch):
    """The coincidence viewer writes its FIB image itself rather than going through a
    task, so it has to record the output on the history entry it appends. What it
    records and what the panel discovers must line up.
    """
    from types import SimpleNamespace

    import numpy as np

    from fibsem.applications.autolamella.ui import (
        fluorescence_coincidence_viewer_widget as viewer,
    )
    from fibsem.structures import FibsemImage

    monkeypatch.setattr(viewer.notification_service, "show_toast", lambda *a, **k: None)

    lamella = _lamella(tmp_path)
    widget = viewer.FluorescenceCoincidenceViewerWidget.__new__(
        viewer.FluorescenceCoincidenceViewerWidget
    )
    widget._selected_lamella = lamella
    widget.experiment = SimpleNamespace(save=lambda: None)
    widget._latest_fib_image = FibsemImage(data=np.zeros((8, 8), dtype=np.uint8))

    widget._record_coincidence_result()

    assert lamella.task_history, "no history entry was appended"
    task = lamella.task_history[-1]
    assert task.name == viewer.COINCIDENCE_REVIEW_TASK_NAME

    recorded = task.outputs["final_fib"]
    assert recorded == [os.path.basename(recorded[0])], "path must be relative"

    found = final_reference_images(lamella, task)
    assert found == [str(Path(lamella.path, recorded[0]))]
    assert os.path.isfile(found[0])
