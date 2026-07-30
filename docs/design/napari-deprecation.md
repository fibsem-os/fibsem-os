# Napari Deprecation — the matplotlib canvas subsystem

Consolidates ten design documents written 2026-06-26 → 2026-07-29 (quad-view display, canvas
overlay state model, overview/minimap cutover, lamella editor cutover, quad-view UX features,
FM z-slider, spot-burn consolidation + workflow unify, thread-worker removal, minimap
correlation rework).

This is the **durable architecture record**. Open work is tracked in Linear under
[Napari Deprecation](https://linear.app/fibsemos/project/napari-deprecation-18be0d19e5d9),
not here — a doc that doubles as a task list goes stale the moment it merges.

---

## Why

napari brought a heavy dependency, a GL canvas implicated in a hard crash on Windows/Tescan
(garbage-collected vispy `GLObject`s finalised on worker threads racing the GLIR queue), and an
overlay model where every widget reached into the viewer directly. Six widgets each attached,
drove and tore down their own overlays against a shared viewer — the scatter behind a connection
leak, an acquisition-time UI freeze, and two alignment overlays that could both be visible at once.

The replacement is a plain matplotlib canvas plus a reducer: overlay state is data, rendering is
one coalesced pass.

---

## The pieces

### `FibsemImageCanvas` — `fibsem/ui/widgets/canvas/image_canvas.py`

Zoom/pan, scalebar, crosshair, contrast/gamma popover, overlay toolbar, hint / info-bar / flash
text, and pixel-coordinate click + scroll signals.

Two entry points: `set_image(FibsemImage)` pulls `metadata.pixel_size.x` + `filtered_data` and
delegates to `set_array(arr, pixel_size, cmap)`, which also serves composites and RGB that have no
backing `FibsemImage`. The canvas **never stores the image or its metadata** — only `_pixel_size`
for the scalebar. It is a display surface and a pixel-coordinate emitter, nothing more; consumers
source images from their own widget state.

Signals carry keyboard modifiers as a trailing `object` (a napari-style tuple such as `("Alt",)`),
read from `event.guiEvent.modifiers()` — Qt is the source of truth, matplotlib's `MouseEvent.key`
is focus-flaky. Click modifiers are captured at *press* and emitted on release. PyQt5 truncates
extra arguments, so 2-arg slots kept working across the widening.

Modified scroll emits and returns **without zooming**: the gesture belongs to whoever claimed it
(Shift+scroll drives FM objective focus and FIB-SEM working distance). Plain scroll still zooms.

Zoom/pan is preserved across same-resolution image updates — auto-fit happens only on the first
image or a resolution change, so live acquisition doesn't re-frame on every frame.

### Overlays — `fibsem/ui/widgets/canvas/overlays/`

`CanvasOverlay` base plus point, rect, ruler, pattern, alignment-area, segmentation-mask,
milling-pattern and minimap-shape overlays.

**Design rule, carried from the Windows GC/GLIR crash:** overlay and crosshair refreshes must
update existing artists in place (`set_data` etc.), never destroy and recreate per refresh. The
napari implementation's remove/re-add churn was the garbage factory behind that crash. The rule
outlived the code that caused it.

### `FMCanvasWidget` — `fm_canvas.py`, `fm_composite.py`

Per-channel normalize-by-clim → gamma → tint → **additive** sum → clip → uint8 RGB. Channel tints
are the napari colormap endpoints (`_CANONICAL_TINTS`), so colours match what users had, while
`fm_composite` itself stays napari-free and Qt-free.

Auto-contrast downsamples before `np.percentile` and caches per-layer clim keyed on data-array
identity — the naive version cost ~485 ms per 3-channel frame on the GUI thread, roughly 10× the
20 fps budget.

Holds per-channel z-stacks with a z-slider and max-projection toggle; controls appear only for
multi-plane stacks, so live single-frame FM shows neither.

### `MicroscopeViewController` — `quad_view.py`

The reducer, and the seam that replaced the napari `Viewer`. Producers mutate a declarative
`SceneModel` through `set_image` / `set_overlay` / `remove_overlay`; each mutation marks the scene
dirty and a queued signal drives one coalesced `_reconcile` pass per canvas on the GUI thread.
Producers never touch canvases or overlay objects.

Because the render is queued onto the GUI thread, **the reducer API is safe to call from worker
threads**.

#### Settled decisions

- **Overlay objects stay the renderer; specs are data; the reducer bridges.** The objects already
  blit and emit committed values — rewriting them as pure-data render functions would throw that
  away.
- **`armed_overlay` in the model is the single input arbiter**, replacing convention-coordinated
  mode entry/exit.
- **The reducer injects the image; specs carry none.** A producer updating patterns shouldn't have
  to pass the image just to do so. Bonus: a bare image swap re-renders patterns against the *new*
  image, where the old code redrew against the overlay's stale cached copy.
- **The reducer is beam-generic** (SEM / FIB / FM handled uniformly) even though overlays only
  landed on FIB initially.

#### Active-overlay input model

At most one overlay per canvas is *armed* and owns input. While armed, the canvas suppresses its
three semantic click signals, so stage-move and the milling menu stand down; pan/zoom/scroll stay
live. Default `armed is None` reproduces the pre-migration behaviour, which is what let the
coincidence viewer and correlation widgets carry on untouched.

`modal=True` overlays respond *only* while armed — that is what lets spot-burn use right-click-add
on the same FIB canvas the milling widget uses for its right-click menu.

### The minimap is deliberately **not** on the reducer

`FibsemMinimapWidget` composites its overview through `composite_fm_layers` and hosts three
entity-grouped overlays (`LamellaMarkers`, `CurrentPosition`, `ReferenceFrame`) on one generic
`MinimapShapesOverlay`. It keeps its imperative `draw_current_stage_position()` / `update_viewer()`
structure — those methods just call canvas APIs instead of napari layer APIs.

A single standalone image with bespoke overlays is a poor fit for the quad-view reducer; the
imperative model already there is the right altitude. Two rendering models, chosen on purpose.

---

## Background threading

`fibsem/ui/qt/threading.py` provides `FunctionWorker` and a `thread_worker` decorator covering the
non-generator subset of napari's. It re-emits outcomes as Qt signals *and* mirrors the slice of the
`threading.Thread` API the manual sites use (`start` / `is_alive` / `join`), so it is a drop-in for
both patterns the GUI had.

Because the worker is a `QObject` constructed on the GUI thread, emitting from the worker thread
makes Qt deliver through the GUI event loop — the same mechanism napari used, so signal-delivery
threading is unchanged at every call site.

**The Qt-free layers keep plain threads on purpose**: `fibsem/microscope.py`, `fibsem/fm/microscope.py`
(driver level, `psygnal` not `pyqtSignal`) and `workflows/tasks/hooks.py` (fire-and-forget webhook).
`FunctionWorker` is a `QObject`; using it there would drag Qt into non-GUI code.
`tests/ui/test_gui_thread_migration.py` asserts both halves of that boundary.

---

## Hard-won lessons

**Never trigger canvas redraws from inside an `@ensure_main_thread` update method.** Wiring an
info-bar refresh into `FibsemMovementWidget.update_ui` produced infinite recursion
(`handle_acquisition_update → update_ui → _update_canvas_info → update_ui → …`) and froze the app
during acquisition. Drive such refreshes from a decoupled, non-reentrant path, and test with live
acquisition running.

**Theme bleed is invisible in standalone demos.** Any objectName- or type-scoped themed widget
needs explicit `background: transparent` on its labels and sliders, or the app-global napari
stylesheet bleeds through. Test under a global stylesheet, not a bare demo.

**Native slider repaint flicker.** A panel parented as a child overlay over the matplotlib canvas
repaints on every canvas redraw, which *looks* like sliders moving on their own. Making
`FMLayersPanel` a top-level `Qt.Tool | FramelessWindowHint` window decoupled it.

**Verify cross-branch review claims.** A review agent reported two HIGH "signal arity mismatch →
feature dead" findings that were false — it had read `main`, where the modifier widening didn't
exist yet.

---

## Shipped

| PR | |
| --- | --- |
| #148 | Milling cooperative cancellation |
| #149 | User-`Cancelled` task status |
| #150 | Ctrl+C killable, icon constants, notification theme |
| #151 | Napari-free `FunctionWorker` |
| #199 | Canvas subsystem as a standalone reusable package |
| #200 | Manipulator widget napari-free |
| #201 | Finished the GUI threading migration onto `FunctionWorker` |

**#111** carries the remainder: main-tab cutover, minimap, lamella editor, FM canvas, main-tab UX
features, spot-burn settings refactor.

---

## Known constraint

`fibsem/ui/__init__.py` eagerly imports all eight widgets, one of which imports napari — so
importing *any* symbol under `fibsem.ui` pulls napari into the graph. That is why UI tests skip on
CI (which installs neither PyQt5 nor napari), and why "is this widget napari-free?" can't currently
be answered by import. Tracked as FIB-352.
