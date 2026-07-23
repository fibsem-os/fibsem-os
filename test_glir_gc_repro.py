"""Manual harness: reproduce / verify the Windows vispy-gloo GC crash.

Recreates the conditions of the 2026-07-22 Tescan session crashes
(``OSError: access violation reading 0x1C`` in ``glDrawArrays``): a napari
viewer whose Image/Shapes layers are churned from the main thread (as
``update_lamella_ui`` / ``_update_lamella_display`` do) while background
threads allocate heavily (as workflow/movement/acquisition threads do).

The crash mechanism is Python's GC running in a worker thread and finalizing
vispy ``GLObject``s there — each finalizer appends a GLIR ``DELETE`` command
to an unlocked list raced by the GUI thread's ``paintGL``. Rather than waiting
for a (driver-dependent) hard crash, this harness instruments
``_GlirQueueShare.command`` and counts every GLIR command queued from a
non-main thread — the crash *precondition* — making the result deterministic
and platform-independent.

Usage (manual, opens a real window)::

    python test_glir_gc_repro.py                 # stock GC: expect detections
    python test_glir_gc_repro.py --fixed         # main-thread GC: expect zero
    python test_glir_gc_repro.py --duration 60   # run longer (default 20 s)

Exit code 0 = no cross-thread GLIR commands observed, 1 = at least one.
On a Windows machine without ``--fixed`` this may also genuinely hard-crash —
that is the bug reproducing.
"""
from __future__ import annotations

import argparse
import threading
import time
import traceback

import napari
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

import vispy.gloo.glir as glir

# ---------------------------------------------------------------- detector

_DETECTIONS: list = []  # (command-name, thread-name, stack-summary)
_MAX_STACKS = 5


def install_glir_thread_detector() -> None:
    """Record every GLIR command queued from a non-main thread."""
    original = glir._GlirQueueShare.command

    def command(self, *args):
        if threading.current_thread() is not threading.main_thread():
            stack = (
                "".join(traceback.format_stack(limit=8))
                if len(_DETECTIONS) < _MAX_STACKS
                else ""
            )
            _DETECTIONS.append((args[0], threading.current_thread().name, stack))
        return original(self, *args)

    glir._GlirQueueShare.command = command


# ---------------------------------------------------------------- churn

RESOLUTIONS = [(1024, 1536), (512, 768)]  # alternate → texture realloc + garbage


class LayerChurner:
    """Main-thread layer churn mimicking update_lamella_ui / acquisitions."""

    def __init__(self, viewer: napari.Viewer):
        self.viewer = viewer
        self.tick = 0
        self.image_layer = viewer.add_image(
            np.zeros(RESOLUTIONS[0], dtype=np.uint8), name="live-image"
        )

    def churn(self) -> None:
        self.tick += 1
        shape = RESOLUTIONS[self.tick % len(RESOLUTIONS)]
        self.image_layer.data = np.random.randint(
            0, 255, shape, dtype=np.uint8
        )
        # remove/re-add an overlay layer, as _update_lamella_display does
        if "overlay" in self.viewer.layers:
            self.viewer.layers.remove("overlay")
        lines = np.random.rand(6, 2, 2) * shape[0]
        self.viewer.add_shapes(
            lines, shape_type="line", edge_color="lime", name="overlay"
        )


def allocation_storm(stop: threading.Event) -> None:
    """Background allocation pressure, as workflow threads produce.

    With stock automatic GC, generation thresholds are crossed *in this
    thread*, so cyclic garbage from the layer churn (old vispy visuals,
    textures, buffers) gets finalized here — off the GUI thread. The sleep
    yields the GIL so the GUI thread keeps churning (real workflow threads
    block on hardware I/O the same way); without it the storms starve the
    main loop and nothing renders.
    """
    while not stop.is_set():
        sink = []
        for i in range(20_000):
            sink.append([i])
        del sink
        time.sleep(0.002)


# ---------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fixed",
        action="store_true",
        help="install the main-thread GC fix before running",
    )
    parser.add_argument(
        "--duration", type=float, default=20.0, help="run time in seconds"
    )
    parser.add_argument(
        "--workers", type=int, default=3, help="background allocation threads"
    )
    args = parser.parse_args()

    install_glir_thread_detector()

    viewer = napari.Viewer(title="GLIR GC repro")
    app = QApplication.instance()

    if args.fixed:
        from fibsem.ui.qt.gc import install_main_thread_gc

        collector = install_main_thread_gc(parent=app)  # noqa: F841

    churner = LayerChurner(viewer)
    churn_timer = QTimer()
    churn_timer.setInterval(50)
    churn_timer.timeout.connect(churner.churn)
    churn_timer.start()

    stop = threading.Event()
    workers = [
        threading.Thread(target=allocation_storm, args=(stop,), daemon=True)
        for _ in range(args.workers)
    ]
    for w in workers:
        w.start()

    QTimer.singleShot(int(args.duration * 1000), app.quit)
    app.exec_()
    stop.set()

    print(
        f"\n{'=' * 70}\n"
        f"mode: {'FIXED (main-thread GC)' if args.fixed else 'STOCK (automatic GC)'} | "
        f"duration: {args.duration:.0f}s | churn ticks: {churner.tick}\n"
        f"GLIR commands queued from non-main threads: {len(_DETECTIONS)}"
    )
    if _DETECTIONS:
        names = sorted({f"{cmd} ({thread})" for cmd, thread, _ in _DETECTIONS})
        print("detected:", ", ".join(names))
        for cmd, thread, stack in _DETECTIONS[:_MAX_STACKS]:
            if stack:
                print(f"\n--- first stacks: {cmd} on {thread} ---\n{stack}")
        print(
            "\nRESULT: FAIL — off-main-thread GLIR traffic observed "
            "(crash precondition present)"
        )
        return 1
    print("RESULT: PASS — all GLIR traffic stayed on the main thread")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
