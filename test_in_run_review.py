"""A hand harness for in-run reviews (FIB-1025): ask a detection through the
record and answer it in the Review tab.

Run it:

    python test_in_run_review.py

The real app opens, connected to the Demo microscope with a throwaway
experiment. A worker thread plays the part of a milling task: it takes an ion
image, invents a detection on it, records it as a question and parks until a
decision lands on it. So:

* a row appears in the **Review** tab, on the lamella, holding the workflow;
* the panel shows the image with a magenta marker per feature, draggable;
* **Confirm** hands the points back and the worker prints what it got, with
  how far each one moved;
* **Reject** raises on the worker instead, the way a failed task unwinds;
* **Attention Required** lights, and brings you back to it if you look
  elsewhere; so does **Go to Review** on the Microscope tab's prompt bar;
* closing the window while it is parked takes the question back.

The question goes through the window's own ``QtResponder``, exactly as a task's
does. Interactive review is switched on for this window only -- the preference
on disk is not touched -- because without the Review tab the same question is
the old Detection tab prompt.

The prediction is fabricated and randomly placed -- no model, no ``ml`` extra
-- so each round is a different correction to make. A real task's detection
takes the same path whenever interactive review is on: this is a way to see it
without a model, a sample or a mill.

Not a pytest module despite the name: ``test_*.py`` outside ``tests/`` is this
repo's convention for a GUI harness.
"""

import logging
import os
import random
import sys
import threading
import time
from copy import deepcopy

os.environ.setdefault("QT_QPA_PLATFORM", "")  # a real window, not offscreen

import numpy as np
from psygnal.containers import EventedDict
from PyQt5.QtWidgets import QApplication

from fibsem import acquire, utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    Experiment,
)
from fibsem.applications.autolamella.ui import AutoLamellaMainUI as main_ui_module
from fibsem.applications.autolamella.workflows.interaction import ReviewDetection, ask
from fibsem.detection.detection import DetectedFeatures, ImageCentre, LamellaCentre
from fibsem.structures import BeamType, ImageSettings, MicroscopeState, Point

TASK = "Mill Rough"
RUN = "harness-run-1"


def _fake_detection(image) -> DetectedFeatures:
    """A prediction with no model behind it: two features dropped at random,
    well inside the frame so both are reachable with the mouse."""
    rows, cols = image.data.shape[:2]

    def somewhere(lo: float, hi: float) -> Point:
        return Point(
            x=float(random.uniform(cols * lo, cols * hi)),
            y=float(random.uniform(rows * 0.2, rows * 0.8)),
        )

    # One in each half of the frame. Random over the whole image puts them on
    # top of each other often enough to make the markers fiddly to grab, which
    # is the one thing this harness exists to let you do.
    features = []
    for cls, half in ((LamellaCentre, (0.15, 0.4)), (ImageCentre, (0.6, 0.85))):
        feature = cls()
        feature.px = somewhere(*half)
        features.append(feature)
    mask = np.zeros((rows, cols), dtype=np.uint8)
    return DetectedFeatures(
        features=features,
        image=image.data,
        mask=mask,
        rgb=np.zeros((rows, cols, 3), dtype=np.uint8),
        pixelsize=image.metadata.pixel_size.x,
        # A copy, as ``detect_features_v2`` takes one; it keeps the filepath.
        fibsem_image=deepcopy(image),
    )


def _pretend_to_be_a_task(window, experiment, lamella) -> None:
    """The workflow thread's part: acquire, predict, ask, use the answer.

    Exactly the shape ``update_detection_ui`` has today -- the only difference
    is which responder it asks.
    """
    microscope = window.autolamella_ui.microscope
    # Acquired the way ``take_image_and_detect_features`` acquires: through
    # ``acquire.new_image``, always saved, into the lamella's folder, named
    # ``ml-<timestamp>``. The save is what gives the image its ``filepath`` --
    # beam suffix and extension included -- and that is the file the question
    # records, so the panel shows the task's own picture and not a copy.
    image = acquire.new_image(
        microscope,
        ImageSettings(
            beam_type=BeamType.ION,
            hfw=80e-6,
            resolution=[768, 512],
            save=True,
            path=str(lamella.path),
            filename=f"ml-{utils.current_timestamp_v2()}",
        ),
    )
    detection = _fake_detection(image)
    proposed = {f.name: Point(f.px.x, f.px.y) for f in detection.features}
    print("\n--- the task is asking ---")
    for name, px in proposed.items():
        print(f"    {name}: proposed at ({px.x:.0f}, {px.y:.0f}) px")
    print("    Answer it in the Review tab.\n")

    # The window's own responder, as a task's parent_ui hands it over. It
    # records the question because the request says who is asking and the
    # Review tab is showing; otherwise this would be the Detection tab prompt.
    responder = window.autolamella_ui.ui_responder
    try:
        answer = ask(
            responder,
            ReviewDetection(detection=detection, item_id=lamella.id, task_name=TASK),
        )
    except Exception as exc:  # noqa: BLE001 - the harness reports it
        print(f"\n--- the task unwound: {type(exc).__name__}: {exc} ---\n")
        return
    print("\n--- the task resumed ---")
    for feature in answer.features:
        was = proposed[feature.name]
        moved = ((feature.px.x - was.x) ** 2 + (feature.px.y - was.y) ** 2) ** 0.5
        print(
            f"    {feature.name}: ({feature.px.x:.0f}, {feature.px.y:.0f}) px "
            f"· moved {moved:.0f} px "
            f"· {feature.feature_m.x * 1e6:+.2f}, {feature.feature_m.y * 1e6:+.2f} µm"
        )
    print("    Milling would carry on from here.\n")


def _keep_the_fabrication_out_of_the_training_data() -> None:
    """A confirmed detection writes its image, mask and corrections to the ML
    data directory, for training. This one is invented, so it must not."""
    from fibsem.detection import utils as det_utils

    def not_written(det, initial_features=None) -> None:
        print("    (training data not written: this detection is fabricated)")

    det_utils.save_ml_feature_data = not_written


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    _keep_the_fabrication_out_of_the_training_data()
    app = QApplication(sys.argv)
    window = main_ui_module.AutoLamellaSingleWindowUI()
    window.autolamella_ui.system_widget.connect_to_microscope()

    path = os.path.join(os.path.expanduser("~"), "Documents", "in-run-review-harness")
    os.makedirs(path, exist_ok=True)
    experiment = Experiment(path=path, name=f"in-run-{int(time.time())}")
    os.makedirs(experiment.path, exist_ok=True)
    experiment.task_protocol = AutoLamellaTaskProtocol()
    experiment.add_new_lamella(MicroscopeState(), EventedDict())
    lamella = experiment.positions[0]
    lamella.path.mkdir(parents=True, exist_ok=True)
    # The task is mid-flight: what makes this an in-run question rather than a
    # result left for later, and what the decision has to name.
    lamella.task_history.append(
        AutoLamellaTaskState(name=TASK, status=AutoLamellaTaskStatus.InProgress)
    )
    lamella.task_state.name = TASK
    lamella.task_state.task_id = RUN
    lamella.task_state.status = AutoLamellaTaskStatus.InProgress

    window.autolamella_ui.experiment = experiment
    window._on_experiment_update()
    window._preferences.features.proposer_reviewer_workflow_enabled = True
    window._apply_review_visibility()
    window.tab_widget.setCurrentWidget(window.review_tab)
    window.show()

    thread = threading.Thread(
        target=_pretend_to_be_a_task,
        args=(window, experiment, lamella),
        daemon=True,
    )
    # After the window is up, so the question lands on a Review tab that exists.
    threading.Timer(1.5, thread.start).start()

    print(f"Experiment: {experiment.path}")
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
