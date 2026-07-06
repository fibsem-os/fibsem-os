# Unifying GUI background threading

The PyQt5 GUI runs microscope operations off the GUI thread with **two competing patterns**,
and this change unifies them onto one small wrapper — `fibsem/ui/qt/threading.py`
`FunctionWorker`. A third pattern (a hand-rolled `QThread` subclass) already does the right
thing and is left alone.

1. **napari `@thread_worker`** — `napari.qt.threading.thread_worker` is one of the last hard
   napari dependencies in the quad-view control widgets.
2. **manual `threading.Thread`** — raw threads, some fire-and-forget, some stored on the widget
   for lifecycle (`self._acquisition_thread`, polled via `.is_alive()` / `.join()`).

Both reduce to the same need: *run a function on a background thread, then deliver its outcome
back on the GUI thread* — plus, for the stored ones, *let the widget observe and join it*. That
is a small thing to reproduce without napari.

## The wrapper — `fibsem/ui/qt/threading.py`

A `FunctionWorker(QObject)` plus a drop-in `@thread_worker` decorator.

```python
class FunctionWorker(QObject):
    started  = pyqtSignal()
    returned = pyqtSignal(object)   # result, on success only
    errored  = pyqtSignal(object)   # exception, on failure only (also logged)
    finished = pyqtSignal()         # always, after returned/errored

    def start(self): ...            # daemon thread; self-registers in _ACTIVE_WORKERS
    def is_alive(self) -> bool: ... # threading.Thread parity, for stored-thread sites
    def join(self, timeout=None): ...
```

Because the worker is a `QObject` built on the calling (GUI) thread, emitting its completion
signals *from the worker thread* makes Qt deliver them through the GUI event loop — exactly how
napari's `WorkerBase` behaves, so **signal-delivery threading is unchanged at every call site**.

### Why it's a faithful drop-in for both patterns

- **napari sites** — `.started/.returned/.errored/.finished` mirror the napari names, delivered
  on the GUI thread. `finished` always fires after `returned`/`errored`. Migration is a one-line
  import swap; call sites are untouched:
  ```python
  # from napari.qt.threading import thread_worker
  from fibsem.ui.qt.threading import thread_worker
  ```
- **stored-thread sites** — `FunctionWorker` also exposes the slice of the `threading.Thread`
  API those widgets use (`start` / `is_alive` / `join`), so `self._acquisition_thread` can hold a
  `FunctionWorker` with **no change** to the `is_acquiring` property, the `cancel_*` join, or the
  `closeEvent` join. Cancellation stays **widget-owned**: the widget keeps its own
  `threading.Event`, sets it, then `join(timeout=)`s the body out. The worker only needs to be
  *observable*, not to own the stop signal.
- **Liveness** — call sites keep only a local `worker = self.x_worker(...)`, GC'd the moment the
  method returns. `_ACTIVE_WORKERS` holds a strong reference until `finished` (napari keeps its
  own registry for the same reason).
- **Errors never swallowed** — the body's exception is `logging.exception`-logged before
  `.errored`, so sites that don't connect `.errored` still surface failures.

Intentionally *not* implemented: generators / `.yielded` / pause / abort /
`@thread_worker(connect=...)`. None are used; a generator body or `connect=` kwarg fails loudly.

## The map — what changes, what doesn't

### A1 — napari `thread_worker`, napari-free win (import swap)

| File | Workers | Signals |
|---|---|---|
| `FibsemImageSettingsWidget.py` | autocontrast, autofocus, acquisition | `.finished` |
| `FibsemMovementWidget.py` | absolute move, double-click, orientation | `.finished` |
| `FibsemSpotBurnWidget.py` | spot burn | `.returned` / `.errored` |

All plain (non-generator) bodies; progress reported through each widget's own `pyqtSignal`s.
`FibsemSpotBurnWidget` already imports **both** `FunctionWorker` and `thread_worker` from napari
and is already in returned/errored shape — so it too is a plain import swap (the wrapper exports
both names).

### B1 — manual `threading.Thread`, fire-and-forget

`fluorescence_coincidence_viewer_widget.py` (×7): move-to-lamella, FIB/FM acquire, FIB
autocontrast/autofocus, two double-click stage moves. Each becomes `FunctionWorker(body)` with
its slots connected to `returned`/`finished`. **Fix in passing:** the autocontrast and autofocus
bodies mutate a button (`btn.setText(...)`) from the worker thread in a `finally` block — that
moves into a slot on `finished`, where it belongs.

### B2 — manual `threading.Thread`, stored + cancel/lifecycle

These store the thread and poll `.is_alive()` / `.join(timeout=5)` on cancel and close. Swapping
the stored `threading.Thread` for a `FunctionWorker` leaves those call sites unchanged.

| File | Sites | Lifecycle surface |
|---|---|---|
| `FMAcquisitionWidget.py` | stage move, image, overview, autofocus (shared `_acquisition_thread`) | `is_acquiring` (`is_alive`), `cancel_acquisition` (`join`), `closeEvent` (`join`); `_acquisition_stop_event` propagated into the blocking call |
| `fluorescence_control_widget.py` | image, autofocus | `is_acquiring`, `cancel_acquisition` |
| `FibsemMinimapWidget.py` | tile collection | returns a result dict → natural `.returned`; `tile_collection_finished(result)` |
| `milling_widget.py` | milling | `is_milling`, `_milling_stop_event`; `finally` → `finished` slot |
| `AutoLamellaUI.py` | task worker | `is_workflow_running`, `_workflow_stop_event`, `_workflow_finished_signal(bool)` |

### Out of scope

- **Deferred napari trio** — `FibsemSegmentationModelWidget`, `FibsemModelTrainingWidget`,
  `correlation/ui/fm_import_wizard`. They still import `napari.qt.threading`, but their bodies use
  `self.viewer` / `napari.utils.notifications`, so removing `thread_worker` there buys no
  napari-free win. Same one-line swap when those widgets are otherwise de-napari'd.
- **Backend / non-GUI threads** — `microscope.py`, `fm/microscope.py` (emit `psygnal.Signal`, not
  `pyqtSignal`; driver-level), `tools/_streamlit.py` (subprocess launcher), `workflows/tasks/
  hooks.py` (fire-and-forget webhook). These belong to the non-GUI orchestration layer, not the
  GUI-responsiveness concern this wrapper addresses.
- **`lamella_task_image_widget._ImageLoaderWorker(QThread)`** — a third pattern, already correct
  (`pyqtSignal` + `cancel()`/`quit()`/`wait()`). Could fold onto `FunctionWorker` later; no need.

## Slices

- **T0 — wrapper + unit test.** `fibsem/ui/qt/threading.py` + `test_qt_threading.py` (success
  delivers on the main thread while the body runs off it; error emits `errored` + `finished` but
  not `returned`; a worker with no surviving local ref is not GC'd mid-run; `is_alive`/`join`
  mirror `threading.Thread` for the stored-thread cancel path).
- **T1 — A1.** Import swap in `FibsemImageSettingsWidget`, `FibsemMovementWidget`,
  `FibsemSpotBurnWidget` + a migration test driving a real worker against a Demo microscope.
- **T2 — B1.** `fluorescence_coincidence_viewer_widget` ×7, incl. the `setText`-in-`finally` fix.
- **T3 — B2.** The stored-thread sites, one file per commit.
- **T5 — verify.** Full `fibsem/ui/widgets/tests` suite green.

## Risks / watch-items

- **Lambda-slot threading** — relies on Qt delivering even lambda `.finished` slots on the GUI
  thread when the sender lives there; the same assumption napari already makes at those sites.
- **Toasts from worker threads** (a couple of pre-existing spots) — unchanged; the body already
  ran off-thread under napari / the raw thread.
- **Double-log on error** for sites that also log in their `.errored` handler — acceptable,
  strictly more information.
