"""The overview plot dialog opens the experiment's most recent overview.

It used to open ``glob("*overview*.tif")[-1]`` -- whichever file the filesystem
listed last. With a re-acquired overview whose name sorted first, that was the older
one, with nothing on screen to say a newer one existed.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_overview_image_dialog_default.py
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

# CI installs `.[test]`, not `.[ui]`, so PyQt5 is absent there.
pytest.importorskip("PyQt5")

import numpy as np  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem.applications.autolamella.structures import Experiment  # noqa: E402
from fibsem.applications.autolamella.ui.autolamella_overview_image_widget import (  # noqa: E402
    OverviewImageWidget,
)
from fibsem.structures import FibsemImage  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv)


def _save_overview(folder, name: str, age_s: float) -> str:
    image = FibsemImage.generate_blank_image(resolution=(96, 32), hfw=100e-6)
    image.data = np.zeros((32, 96), dtype=np.uint8)
    path = os.path.join(str(folder), name)
    image.save(path)
    stamp = time.time() - age_s
    os.utime(path, (stamp, stamp))
    return path


def test_the_dialog_opens_the_most_recent_overview(tmp_path):
    experiment = Experiment(path=tmp_path, name="overview-default-test")
    os.makedirs(str(experiment.path), exist_ok=True)
    _save_overview(experiment.path, "overview-image-2026-07-29_18-03-35.tif", 3600)
    newer = _save_overview(
        experiment.path, "overview-image-2026-07-29_09-00-00-reacquired.tif", 0
    )

    widget = OverviewImageWidget()
    widget.set_experiment(experiment)

    assert widget.overview_image is not None, "no overview was opened"
    assert os.path.samefile(widget.overview_image.filepath, newer)
