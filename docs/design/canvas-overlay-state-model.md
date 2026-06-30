# Canvas State & Overlay Model — Consolidating the Quad-View Overlays

> Follow-on to [quad-view-microscope-display.md](quad-view-microscope-display.md).
> That doc migrated the overlays *onto* the mpl canvases. This one consolidates
> how they are **owned, updated, and gated**.

## Summary

The quad-view overlays (milling patterns, spot burn, alignment area, detection
mask/features, POI, FM points) are already migrated onto the `FibsemImageCanvas`
widgets behind `FEATURE_QUAD_VIEW_ENABLED` and working, *including* the
active-overlay input model. This is **not** a napari migration and **not** a
rewrite of the overlay objects.

The problem is the *management*: every producing widget reaches
`controller.fib_canvas` and calls `add_overlay` / `set_stages` /
`enter_overlay_mode` **directly**, bypassing the controller (which has no overlay
surface). Each widget hand-rolls the overlay's full lifecycle, and the
active-overlay mode is coordinated by convention across independent widgets with
no central arbiter.

We replace that with a single per-canvas **state model** (`CanvasState`) owned by
`MicroscopeViewController`, mutated through a small **reducer** API, rendered by
**one debounced pass** on the GUI thread. The existing overlay objects stay as the
renderer. The migration is **behaviour-preserving, flag-gated, one overlay at a
time**, starting with milling patterns.

## What's "pretty bad" today (grounded)

Six direct-attach sites bypass the controller:

| Site | Overlay | Lifecycle owner |
|---|---|---|
| `FibsemSpotBurnWidget.py:77,132` | spot `PointOverlay` (+ mode) | the widget |
| `FibsemImageSettingsWidget.py:770,811` | editable `AlignmentAreaOverlay` (+ mode) | the widget |
| `FibsemEmbeddedDetectionWidget.py:112,119,127` | `MaskOverlay` + features `PointOverlay` (+ mode) | the widget |
| `AutoLamellaUI.py:2016,2021` | POI `PointOverlay` (+ mode) | the window |
| `milling_task_viewer_widget.py:176` | `MillingPatternOverlay` | the widget |
| `milling_task_viewer_widget.py:178` | read-only `AlignmentAreaOverlay` | the widget |

Consequences:

1. **Scattered ownership.** `MicroscopeViewController` mediates images only; for
   overlays it's a hollow `fib_canvas` pass-through. There is no place that knows
   "what is on the FIB canvas."
2. **Per-widget teardown, repeated 6×.** Each widget must `remove_overlay` +
   disconnect in `closeEvent` against the *app-lifetime* canvas. Miss it and you
   leak a handler that fires into a dead widget — the change-A class of bug.
3. **No central input arbiter.** The active-overlay mode is correct per-overlay
   but coordinated by convention. Two `AlignmentAreaOverlay`s (read-only from the
   milling viewer + editable from image settings) can sit on the FIB canvas at
   once — the "two alignment overlays visible" follow-up.
4. **No home for canvas-level state.** The info/text overlay had nowhere to live,
   so it was pushed through `FibsemMovementWidget.update_ui` (`@ensure_main_thread`)
   and re-entered it → the acquisition freeze.

## The model

```python
# ---- model: pure data, the source of truth (one per canvas) ----
@dataclass
class CanvasState:
    image: Optional["FibsemImage"] = None   # richer ImageLayer (clim/cmap, FM composite) deferred
    overlays: Dict[str, OverlaySpec] = field(default_factory=dict)   # keyed by id
    info: List[Tuple[str, str]] = field(default_factory=list)        # bottom info bar
    armed_overlay: Optional[str] = None    # id that owns edit input (None = view/move)

@dataclass
class SceneModel:
    sem: CanvasState; fib: CanvasState; fm: CanvasState

# ---- overlay specs: data records the reducer maps onto overlay objects ----
@dataclass
class MillingSpec(OverlaySpec):
    id: str = "milling"
    stages: Sequence = ()                   # NO image — reducer injects CanvasState.image
    background_stages: Sequence = ()
    selected_index: Optional[int] = None
# AlignmentSpec, PointsSpec, MaskSpec, … added per slice
```

### Reducer (the only mutation path)

```python
class MicroscopeViewController(QObject):
    def set_image(beam, layer): ...
    def set_overlay(beam, spec): ...          # upsert by spec.id → mark dirty
    def remove_overlay(beam, overlay_id): ...
    def set_info(beam, key, value): ...
    def arm_overlay(beam, overlay_id | None): ...
```

**Render & threading contract** — what earns the model its keep:

- The reducer owns a `Dict[id, CanvasOverlay]` of overlay **objects** per canvas;
  their lifetime matches the canvas (the controller), so **producers never attach
  or tear down** — killing the per-widget leak surface.
- Each `set_*` marks the canvas dirty and schedules a **single** `singleShot(0)`
  render that coalesces every mutation in this event-loop turn into one
  `_reconcile()` + `draw_idle()`. Rendering is always deferred and runs exactly
  once → the info-bar recursion is **impossible by construction**, not merely
  avoided.
- `set_*` is **thread-safe**: a worker-thread call hops to the GUI thread via an
  internal `Qt.QueuedConnection` signal, so producers stop sprinkling
  `@ensure_main_thread`.
- `_reconcile(beam)` diffs `state.overlays` (specs) against the owned overlay
  objects by id: create+`add_overlay` on first sight, drive via the overlay's
  existing method (`MillingSpec → MillingPatternOverlay.set_stages(...)`),
  `remove_overlay` when a spec disappears.
- `set_image(beam, img)` stashes the `FibsemImage` as `CanvasState.image` and marks
  the canvas dirty, so a bare image swap re-renders image-dependent overlays
  (milling) against the new image — the reconciler injects that stashed image when
  driving them, so producers never pass it.
- **Interactive overlays** keep their blit-on-drag + commit-on-release signals
  (`RectOverlay.rect_changed`, `PointOverlay.point_*`); the reducer subscribes and
  folds the committed value back into the model (skipping the echo to the
  originating overlay to avoid a feedback loop). This is the existing de-facto
  pattern, just routed through one place.
- `arm_overlay(beam, id)` replaces the convention-coordinated
  `enter_overlay_mode`/`exit_overlay_mode` dance with a single arbiter: the
  reducer sets the canvas's active overlay (still via `set_active_overlay` under
  the hood) so only one overlay per canvas is armed.

**Coexistence during migration:** a reducer-owned overlay and a still-directly-
attached overlay share the canvas's `_overlays` list with no conflict
(`add_overlay` just appends). So we migrate one overlay at a time; the rest keep
working untouched.

## Decisions (settled)

- **Keep overlay objects as the renderer; specs are data; the reducer bridges.**
  The objects already blit + emit committed values; rewriting them to pure-data
  render functions would throw that away.
- **`armed_overlay` in the model is the single input arbiter**, replacing
  convention-coordinated mode entry/exit.
- **Behaviour-preserving vertical slices, flag-gated.** Milling first (display-
  only, no input, no napari) to prove the spine at near-zero risk.
- **The reducer injects the image; specs carry none.** `set_stages` needs px-size
  + shape, but the producer shouldn't have to pass the image just to update
  patterns. The controller already receives the `FibsemImage` via `set_image`,
  stashes it per beam as `CanvasState.image`, and supplies it when driving
  image-dependent overlays. Bonus: a bare image swap then re-renders patterns
  against the *new* image (today `on_image_changed` redraws against the overlay's
  stale cached image).
- **The reducer is beam-generic** (SEM / FIB / FM handled uniformly) even though
  overlays only land on FIB today — cheap now, exercised as later slices add SEM/FM.

## Out of scope

- **Qt screen-space tools** (measure / loupe / pins / radial menu) — `QGraphicsView`-
  based, not on this canvas, ephemeral modes. Stay imperative.
- **Fluorescence coincidence viewer** — self-contained; a reference for the
  overlay pattern, not folded in.
- **The napari path** — the `else`-branch of each producer; deleted in quad-view
  Phase 7, not here.

## Slice 1 — milling patterns (behaviour-preserving)

**Today** (`milling_task_viewer_widget.py`):
- `_init_canvas_overlay()` (159) grabs `controller.fib_canvas`, creates +
  `add_overlay`s a `MillingPatternOverlay` (and a read-only `AlignmentAreaOverlay`).
- `_update_canvas_patterns()` (460) drives `set_stages(stages, self._fib_image,
  background_stages=…, selected_index=…)` or `.clear()`.
- `closeEvent()` (191) disconnects + `remove_overlay`s both.

**After:**
- Add `CanvasState`/`SceneModel`/`OverlaySpec`/`MillingSpec` + the reducer
  (`set_overlay`/`remove_overlay`), the owned-overlay reconciler (the one branch:
  `MillingSpec → MillingPatternOverlay.set_stages`), and the debounced render loop
  to `MicroscopeViewController`.
- `milling_task_viewer`: drop the create/attach/teardown of `_canvas_overlay`.
  `_update_canvas_patterns()` becomes
  `controller.set_overlay(BeamType.ION, MillingSpec(stages=…,
  background_stages=…, selected_index=…))`, and the empty case
  `controller.remove_overlay(BeamType.ION, "milling")`.
- **Unchanged this slice:** the read-only `_canvas_alignment` display (stays direct;
  folded into the alignment slice with its editable twin), the right-click
  reposition wiring (`canvas_right_clicked` → menu — that's input, not an overlay),
  and the entire napari `else`-branch.

**Verify:** patterns appear / update / clear / selected-highlight on the FIB canvas
identically with the flag on; closing the milling widget no longer needs to remove
the overlay (controller owns it); napari path byte-for-byte unchanged with the flag
off. Cover the reducer + reconcile in the standalone `test_image_canvas`-style
harness (no hardware).

## Migration order (after slice 1)

| Slice | Overlay(s) | Proves |
|---|---|---|
| 1 | milling patterns | reducer + debounced render (display-only) |
| 2 | alignment area (read-only + editable, unified) | view→model round-trip + `arm_overlay`; retires the two-overlay bug |
| 3 | spot burn | modal `PointOverlay` via `arm_overlay` |
| 4 | detection (mask + features) | multi-overlay per canvas + beam re-host |
| 5 | POI | window-owned producer onto the reducer |
| 6 | FM points + **info bar** | FM canvas parity; `set_info` on the same loop |

The **info bar** rides the render loop as `set_info` → one text artist; the path
the recursion exploited can't recur because the render is deferred and single.

## Open questions

- **Toolbar mode UX (slice 2).** `enter_overlay_mode` shows a checkable "Move"
  toggle button — the user-facing escape hatch back to stage movement. Open: does
  `arm_overlay` fully own that button, or only decide *which* overlay is armed and
  leave the button on the canvas? Default: reducer owns the full mode incl. the
  button. Does **not** affect slice 1 (milling is display-only, never armed).
- **Input routing (later).** Overlay *rendering* moves behind the controller;
  *input* (right-click→menu reposition, double-click-to-move) stays widget-owned for
  now. Open: do we eventually pull input routing behind the controller too, alongside
  `armed_overlay`? Default: leave input as-is, revisit as its own thread. Does **not**
  affect slice 1.

**Resolved:** image is injected by the reducer (specs carry none); reducer built
beam-generic.
