# Quad-View Microscope Display — Napari Deprecation (Main UI)

## Summary

Replace the single napari `Viewer` that backs the **main microscope tab** of the
AutoLamella UI with a re-usable 2×2 grid of matplotlib `FibsemImageCanvas`
widgets — one per beam view (SEM, FIB, FM, + an empty placeholder). The existing
napari workflow must keep working at every phase; the new path ships behind a
`FEATURE_QUAD_VIEW_ENABLED` feature flag and the napari code is only deleted once
the quad path reaches parity.

This is **not** a from-scratch build. `fibsem/ui/widgets/image_canvas.py` already
provides the canvas, the overlay system, and the click/scroll signals, and it is
already in production in the fluorescence coincidence viewer. The 2×2 layout is
already prototyped in `fibsem/ui/widgets/tests/test_splitter_layouts.py`.

## Scope (decided)

| Decision | Choice |
|---|---|
| Surface | **Only** the main microscope tab in `AutoLamellaMainUI`. |
| Minimap / protocol editor / labelling UIs | **Deferred.** They use separate napari viewers; the napari *dependency* cannot be removed until they migrate too. |
| 4th panel | Empty, with a **"No Data"** placeholder. No API in v1. |
| Contrast controls | **Contrast limits + gamma only** for now. Multi-layer / opacity / blending / colormap controls deferred. |
| First code task | **Close the modifier-key ("Alt") gap** in `image_canvas.py` before any other code. See [Focus](#focus-closing-the-modifier-alt-gap). |

## What exists today

### The napari coupling (main tab)

- `AutoLamellaMainUI.AutoLamellaSingleWindowUI` owns `self.main_viewer =
  napari.Viewer(...)` and passes it into `AutoLamellaUI(viewer=...)`, which shares
  it with the control widgets.
- Widgets reach the viewer as `self.viewer` and use napari-native APIs:
  `viewer.add_image/add_points/add_shapes/add_labels`, `layer.world_to_data`,
  and `viewer.mouse_*_callbacks` / `layer.mouse_*_callbacks`.
- Beams are disambiguated by **which layer was clicked** (`image_widget.eb_layer`
  vs `ib_layer`).

### The matplotlib foundation (already built)

`fibsem/ui/widgets/image_canvas.py`:

| Component | Capability |
|---|---|
| `FibsemImageCanvas` | scroll-zoom, drag-pan, auto-scalebar, crosshair, corner toolbar buttons |
| signals | `canvas_clicked`, `canvas_double_clicked`, `canvas_right_clicked`, `canvas_scrolled` — **already in pixel coords** |
| `PointOverlay` | interactive points: add / select / drag / delete + signals |
| `PointsOverlay` | static scatter markers |
| `RectOverlay` | drag + optional resize rectangle |
| `PatternOverlay` | milling shapes: rectangle / circle / line / polygon (pixel space) |
| `ScanDirectionArrowOverlay` | scan-direction arrow |

`fibsem/ui/widgets/tests/test_splitter_layouts.py` → `splitter_2x2()` is the target
layout (nested `QSplitter`s, each cell a labelled `FibsemImageCanvas`).

## Target architecture

```
QuadViewWidget (QSplitter 2×2)          ← reusable; from test_splitter_layouts.py
  ├─ FibsemImageCanvas[ELECTRON]  (SEM)
  ├─ FibsemImageCanvas[ION]       (FIB)
  ├─ FibsemImageCanvas[FM]        (fluorescence)
  └─ FibsemImageCanvas[placeholder]  → "No Data"

MicroscopeViewController                 ← the seam that replaces the napari.Viewer
  • get_canvas(beam) / set_image(beam, image)
  • per-beam overlay helpers (patterns, points, mask, rect)
  • exposes each canvas's Qt signals to control widgets
```

Two structural wins fall out of the quad layout:

1. **Each canvas *is* a beam**, so the "which layer was clicked" branching goes away.
2. Signals already carry pixel coords, so the napari `layer.world_to_data(event.position)`
   step is removed — handlers feed `Point(x, y)` straight into
   `conversions.image_to_microscope_image_coordinates(...)`.

### Compatibility strategy

Add `cfg.FEATURE_QUAD_VIEW_ENABLED` (default `False`), mirroring the existing
`FEATURE_VIEWER_MOVEMENT_EVENTS` flag pattern. In `AutoLamellaMainUI`, branch the
central widget: napari viewer (today) vs `QuadViewWidget`. Each interactive widget
grows one `if self.use_quad_view:` branch that wires canvas signals; the napari
path is untouched. The napari branches are deleted wholesale in Phase 7.

We considered a full `IViewer` adapter interface (napari + mpl implementations,
widgets unaware). It is cleaner long-term but the layer-model-vs-overlay-model
impedance makes it a leaky, heavy abstraction, and most of the canvas already
exists. The flagged parallel path is the pragmatic choice.

## Interaction map: napari → matplotlib

| Interaction | Current (napari) | Target (`FibsemImageCanvas`) | Status |
|---|---|---|---|
| single / double / right click, scroll | `mouse_*_callbacks` + `world_to_data` | `canvas_clicked` / `canvas_double_clicked` / `canvas_right_clicked` / `canvas_scrolled` | exists |
| stage move (dbl-click) | `FibsemMovementWidget._double_click` | connect `canvas_double_clicked` per beam | wire-up |
| **Alt+dbl-click vertical move** | `event.modifiers` | signals carry **no modifiers** | **gap (do first)** |
| milling display | `draw_milling_patterns_in_napari` | `PatternOverlay.set_patterns` (µm→px) | exists |
| milling reposition | right-click + context menu | `canvas_right_clicked` + `QMenu` | wire-up |
| FOV / alignment area | shapes layers, `mode="select"` | `RectOverlay` | exists |
| spots / points | `add_points` + drag callbacks | `PointOverlay` | exists |
| mask / labels | `viewer.add_labels` | alpha-blended `imshow` overlay | **gap** |
| crosshair / scalebar | napari draw utils | built into canvas | exists |
| contrast / gamma | napari layer controls | per-canvas popup (limits + gamma) | **gap** |
| ruler / measure | points + line shapes + drag | `RulerOverlay` | **gap** |
| stage-position text | `update_text_overlay(viewer)` | `ax.text` / corner `QLabel` | **gap (trivial)** |
| FM add/update position | Alt / Shift + click | `canvas_clicked` + modifiers | **gap (Phase 6)** |

## Capability gaps to add to `image_canvas.py`

All additive, all unit-testable in the standalone `test_*` harness (no hardware):

1. **Modifier keys on mouse signals** — load-bearing; see Focus below.
2. **`MaskOverlay`** — alpha-blended `imshow` of a label array + colormap.
3. **Per-canvas contrast control** — limits + gamma popup behind an overlay button
   (model on the coincidence viewer's existing histogram button).
4. **`RulerOverlay`** — two draggable endpoints + line + distance label (port the
   math from `FibsemImageSettingsWidget.update_ruler_points`).
5. **Stage-position text overlay** + **per-overlay visibility toggles** — minor.

## Phases (each independently shippable, flag-gated)

| Phase | Content | Ships |
|---|---|---|
| 0 — Foundation & seam | `QuadViewWidget` + `MicroscopeViewController`; flag wired in main window | nothing user-visible (flag off) |
| 1 — Canvas parity | close the 5 gaps in `image_canvas.py`, unit-tested | library only |
| 2 — Image display (read-only) | populate SEM/FIB/FM canvases; native crosshair/scalebar/contrast | live quad view, no interaction |
| 3 — Stage movement | `canvas_double_clicked`(+modifiers) → existing `_double_click_worker`; drop `world_to_data` | double-click-to-move |
| 4 — Milling patterns | `PatternOverlay` (µm→px from `napari/patterns.py`), FOV/alignment `RectOverlay`, right-click `QMenu` | milling display + interaction |
| 5 — Points & masks | spot burn → `PointOverlay`; detection → `MaskOverlay` | spot burn + detection |
| 6 — Fluorescence | FM add/update (Alt/Shift), relative move, camera transforms, FM crosshairs | full FM workflow |
| 7 — Cutover & removal | flip default to `True`, soak, delete napari branches for the main tab **and** migrate standalone `FibsemUI` onto the canvases | quad view default; napari-free main tab + `fibsem_ui` |

> Note: Phase 7 removes napari from the **main microscope tab** and the **standalone
> `FibsemUI`** app only. The napari dependency stays for every other surface (minimap,
> protocol/lamella editor, FM acquisition, correlation, labelling/training, coincidence
> viewer). **Detailed plan: see "Phase 7 — Cutover & removal (detailed plan)" below.**

---

## Focus: closing the modifier ("Alt") gap

This is the first code task. It unblocks Phase 3 (stage movement) and Phase 6 (FM),
and is a small, isolated, fully-testable change to `image_canvas.py`.

### The gap

The canvas mouse signals emit only `(x, y)`:

```python
canvas_clicked        = pyqtSignal(float, float)
canvas_double_clicked = pyqtSignal(float, float)
canvas_right_clicked  = pyqtSignal(float, float)
canvas_scrolled       = pyqtSignal(float, float, int)
```

But the napari handlers branch on `event.modifiers`. In the **main tab** scope,
the consumer is `FibsemMovementWidget._double_click_worker`:

```python
if event.button != 1 or "Shift" in event.modifiers:   # line 408 — Shift aborts the move
    return
...
vertical_move = True if "Alt" in event.modifiers else False   # line 440 — Alt = vertical / eucentric
```

#### Modifier consumers across the UI

| Location | Modifier | Meaning | Scope |
|---|---|---|---|
| `FibsemMovementWidget.py:408` | Shift | abort double-click move | **main tab (now)** |
| `FibsemMovementWidget.py:440` | Alt | vertical / eucentric move | **main tab (now)** |
| `FMAcquisitionWidget.py:758/788/809` | Alt / Shift | add / update FM position | Phase 6 |
| `fluorescence_control_widget.py:340` | Alt | relative-move modifier | Phase 6 |
| `fm/widgets/objective_control_widget.py:410` | Shift | objective scroll modifier | Phase 6 |
| `correlation/app.py` | Control/Shift/Alt | separate app | out of scope |
| `ui/widgets/drag_distance.py` | Shift | square constraint | **no change** — already native Qt `event.modifiers()` |

### Mechanism

matplotlib exposes `MouseEvent.key`, but it is populated from matplotlib's own
key-tracking and depends on canvas keyboard focus — unreliable in a Qt embedding.
The robust source of truth is the underlying Qt event, available on every mpl
event as `event.guiEvent`:

```python
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

# napari-compatible strings, so handler bodies port verbatim
_QT_MODS = (
    (Qt.AltModifier, "Alt"),
    (Qt.ShiftModifier, "Shift"),
    (Qt.ControlModifier, "Control"),
    (Qt.MetaModifier, "Meta"),
)

def _modifiers_from_event(event) -> tuple:
    gui = getattr(event, "guiEvent", None)
    mods = gui.modifiers() if gui is not None else QApplication.keyboardModifiers()
    return tuple(name for flag, name in _QT_MODS if mods & flag)
```

Emitting a **tuple of napari-style strings** (`("Alt",)`, `("Shift",)`) is
deliberate: the consuming code ports with a one-line change —
`"Alt" in event.modifiers` → `"Alt" in modifiers`.

### Proposed signal change

Widen all four mouse signals with a 3rd `modifiers` argument:

```python
canvas_clicked        = pyqtSignal(float, float, object)  # x, y, modifiers
canvas_double_clicked = pyqtSignal(float, float, object)
canvas_right_clicked  = pyqtSignal(float, float, object)
canvas_scrolled       = pyqtSignal(float, float, int, object)
```

**Subtlety — click modifiers come from press, not release.** `canvas_clicked` is
emitted in `_on_release` (after the `dist < 3` test). The modifier state must be
captured at **press** time and stashed in `self._pan_start`, then emitted on
release. `canvas_double_clicked`, `canvas_right_clicked`, and `canvas_scrolled`
read modifiers directly from their triggering event.

### Backward compatibility

Widening is low-risk: **PyQt5 truncates extra arguments** when a signal connects
to a Python slot that accepts fewer parameters. Existing 2-arg consumers (the
coincidence viewer) keep working unchanged; only consumers that *want* modifiers
opt into the 3rd argument.

- Alternative considered: add parallel `*_with_mods` signals. Rejected — doubles
  the signal surface and every consumer eventually wants modifiers anyway.

### Test plan (standalone, no hardware)

Extend `fibsem/ui/widgets/tests/test_image_canvas.py`:

1. `_modifiers_from_event` maps each `Qt.*Modifier` (and combos) to the right
   string tuple; empty event → `()`.
2. Synthesize press/release `QMouseEvent`s with `Qt.AltModifier` → assert
   `canvas_clicked` emits `("Alt",)`; modifiers captured at press survive a
   press-with-Alt / release-without-Alt sequence.
3. `canvas_double_clicked` / `canvas_right_clicked` emit current modifiers.
4. A legacy 2-arg slot still fires (truncation) after the widen.

### Estimated cost

| Change | LoC | Complexity |
|---|---|---|
| `_modifiers_from_event` helper + signal widen + press-time capture | ~25 | Low |
| Tests | ~40 | Low |

---

## Open questions

- Gamma: apply as a display-only LUT on the canvas, or bake into
  `FibsemImage.filtered_data`? (Leaning display-only LUT so the underlying image
  data is untouched.)
- Does the placeholder panel need to participate in pan/zoom sync with the other
  canvases later, or stay fully inert? (v1: inert.)
- Per-beam vs shared toolbar for the contrast/gamma popup.

---

## Volume-milling branch (`feat-proj-volume-milling`) — reuse & reconciliation

That branch forked `image_canvas.py` from an **older base** (2-arg signals, no
modifiers / contrast) and added volume-milling overlays. Decisions:

- **Phase 4 milling overlays live in a separate module**
  (`fibsem/ui/widgets/milling_overlay.py`), **not** `image_canvas.py`, to minimise
  the merge-conflict surface with that branch.
- That branch has **no multi-stage pattern renderer** — `MillingPatternOverlay`
  (port of `draw_milling_patterns_in_napari`) is still ours to build.

**To reuse / port later (not now):**
- Editable alignment area: `RectOverlay(resizable=True)` + normalized
  `FibsemRectangle` + an `alignment_area_changed` signal (their
  `VolumeMillingImageCanvas`).
- `PointOverlay.point_dragging` (continuous drag signal) — port into our `PointOverlay`.
- Overlay toggle-button UX (show/hide pattern / scan arrow / alignment).
- `ScanDirectionArrowOverlay` (already common to both files).
- `_ResizableMillingPatternOverlay` for single-rect interactive needs.
- Canvas-subclass architecture (`VolumeMillingImageCanvas` /
  `VolumeSEMAcquisitionCanvas`): a use-case canvas that owns its overlays +
  metre↔pixel conversion + high-level signals (metres / normalized rects).

**Reconciliation:** our modifier + contrast base is the superset (3-arg signals
are backward-compatible via PyQt arg-truncation). Direction: land our base, rebase
their 4 additive classes + `point_dragging` on top. Coordinate with the
volume-branch owner before both files diverge further.

---

## Phase 4 — MillingPatternOverlay scope (locked, parity-first)

`MillingPatternOverlay` — a **display-only** `CanvasOverlay` in
`fibsem/ui/widgets/milling_overlay.py`. Captures no mouse events (coexists with
Phase 3 double-click-to-move + right-click menu on the FIB canvas).

In scope now:
- Render **patterns only**: rectangle (rotated), circle, line, polygon —
  **per-stage colours** (reuse `COLOURS`) + per-stage crosshair at `stage.pattern.point`.
- Input = milling stages + FIB image; convert internally by reusing
  `convert_pattern_to_napari_*` (already emit rotated, y-flipped pixel geometry).
- Cache inputs, redraw on `on_image_changed`; refresh via `draw_idle()` (no debounce
  unless slider-drag feels laggy).
- Movement via the widget: `fib_canvas.canvas_right_clicked` → `ContextMenu` →
  existing `_move_patterns` (unchanged).

Deferred (note for later):
- Direct drag-to-move patterns (interactive overlay) — wanted eventually.
- FOV rect, alignment area (editable; reuse volume branch `RectOverlay` approach).
- Selected-stage highlight, background milling stages.
- Annulus / bitmap shapes (disabled even in napari).

---

## Active-overlay input model (input gating)

Replaces napari's *active-layer* gating, which the quad-view canvas dropped.

**Problem.** In napari the active layer (e.g. the POI / spot-burn points layer,
set via `viewer.layers.selection.active`) is the only thing that handles clicks,
so the milling layer and stage-movement don't also react. Without that, on the
**shared FIB canvas** a single right-click both adds a point *and* opens the
milling reposition menu, and a double-click moves the stage mid-selection — both
`canvas_right_clicked` and the overlay's own handler fire, because matplotlib
callbacks have no propagation-stop and the canvas (connected first, in
`__init__`) emits *before* any overlay handler runs. So the canvas must **consult
its overlays before emitting**, not the other way round.

**Model.** At most one overlay per canvas is **active**. Invariant:
- the active overlay owns mouse/key input on that canvas;
- every *other* interactive overlay stands down;
- the canvas suppresses its **semantic click signals** (`canvas_clicked` /
  `canvas_double_clicked` / `canvas_right_clicked` — i.e. select / stage-move /
  milling menu);
- **navigation (pan / zoom / scroll) always stays live**, even with an active overlay.

Default `_active_overlay = None` ⇒ the canvas behaves exactly as today. Opt-in,
per canvas — every other surface (coincidence viewer, correlation, lamella
editor) never calls it and is untouched.

**Event availability** (answers "can we still move?" — yes, when nothing is active):

| Gesture | No active overlay (default / "Move") | An overlay is active |
|---|---|---|
| left-drag empty | pan | pan |
| scroll | zoom | zoom |
| double-click | **stage move** (`canvas_double_clicked`) | suppressed |
| right-click | **milling menu** (`canvas_right_clicked`) | suppressed (active overlay handles, e.g. add spot) |
| left-click | `canvas_clicked` (e.g. coincidence select) | suppressed |
| overlay gesture | all attached interactive overlays respond | only the active overlay |

**Canvas API** (`FibsemImageCanvas`):
- `set_active_overlay(overlay_or_None)` — primitive gating, no UI; `active_overlay` getter.
- `_overlay_input_allowed(overlay)` = `_active_overlay is None or _active_overlay is overlay`.
- Gate the 3 semantic emits behind `_active_overlay is None` (`_on_press` dbl/right, `_on_release` click).
- `remove_overlay` auto-clears `_active_overlay` if it removes the active one (no dangling ref).

**Interactive overlays** (`PointOverlay`, `RectOverlay`): one guard at the top of
each input handler — `if self._canvas and not self._canvas._overlay_input_allowed(self): return`.
Display overlays (milling, scalebar) need no change.

**Toolbar mode toggle (discoverability + escape hatch).** Layered on top of the
primitive:
- `enter_overlay_mode(overlay, label, icon="mdi:cursor-default-click")` — sets active **and**
  shows a checkable toolbar button (checked).
- The button: **checked = active (overlay owns input); uncheck = Move (active→None, stage
  movement back); re-check = re-activate** — toggles without destroying the button.
- `exit_overlay_mode()` — active→None and removes the button (called when the step ends).

**Callers:**
- POI selection — `enter_overlay_mode(poi_overlay, "POI")` on show; `exit_overlay_mode()` on compute/clear.
- Alignment-area editing — entered while `editable=True` (`set_alignment_layer`), exited in
  `clear_alignment_area`; also retires the "editable + read-only alignment both visible" follow-up.
- Spot burn (Phase 5a, DONE) — `enter_overlay_mode(spot_overlay, "Spot burn")`; the spot
  overlay is `PointOverlay(add_on_right_click=True, removable=True, modal=True)`, so
  right-click drops a spot with the milling menu suppressed, and toggling to **Move** makes
  it fully **inert** (see the modal gate below). `set_active`/`set_inactive` enter/exit the
  mode + `set_visible`; coords convert normalized(0–1)↔full-res pixel (no napari translate).
  - **Modal gate (implemented).** `_overlay_input_allowed` lets an overlay respond whenever
    *nothing* is active (required for backward-compat: always-on overlays). A `PointOverlay`
    constructed with `modal=True` instead responds **only while it is the active overlay**
    (`_input_allowed()` → `active is self`), so in Move mode it is display-only. POI stays
    non-modal (move-only, harmless in Move). `PointOverlay.set_visible()` hides/shows markers
    without discarding points (survives image rebuilds) for active/inactive parity.
- Detection (Phase 5b, DONE) — `FibsemEmbeddedDetectionWidget` hosts on the canvas for
  `det.fibsem_image.metadata.beam_type` (fallback FIB): a display-only `MaskOverlay`
  (`set_mask(det.mask)`) + a modal, move-only features `PointOverlay` with per-feature
  colours + name labels; `enter_overlay_mode(features, "Detection")` so drag-to-correct can't
  move the stage. `point_moved` → `feature.px`; Continue reads it back unchanged. Mask is
  **display-only** (no painting): `confirm` keeps `det.mask` as the model output and still
  saves the corrected features. Overlays re-host if the detection beam changes. Crosshairs
  dropped (markers + labels suffice).

**Scope boundary (deliberate).** Per-canvas, not global: during FIB POI selection
the SEM canvas has no active overlay, so SEM double-click-move stays live. Strictly
≥ today (today nothing is gated). A controller-level movement freeze across all
canvases is an easy follow-on if it's ever wanted.

---

## Phase 7 — Cutover & removal (detailed plan)

The overlay state-model migration (`canvas-overlay-state-model.md`) is **complete**: every
producer mutates the reducer behind `FEATURE_QUAD_VIEW_ENABLED`. But the main tab is still
**napari-first for image display** — each widget updates napari layers and *also* mirrors
into the quad-view (`if FEATURE_QUAD_VIEW_ENABLED and controller: controller.set_image(...)`).
The napari `main_viewer` is created on every launch and merely kept out of the splitter when
the flag is on. Phase 7 makes the quad-view the **sole** main-tab display, stops creating
`main_viewer`, **and** migrates the standalone `FibsemUI` app onto the canvases so the shared
image/movement widgets lose their last napari host.

### End state

- AutoLamella **main microscope tab**: quad-view only; no `main_viewer`.
- Standalone **`fibsem_ui`** console app: a `MicroscopeViewController` left pane + the existing
  control tabs on the right; no napari viewer.
- `FibsemImageSettingsWidget` / `FibsemMovementWidget`: **napari path deleted** (no host left).
- `FEATURE_QUAD_VIEW_ENABLED` retired; the main tab is unconditional.

### Stays on napari (do not touch)

`minimap_viewer` (Minimap tab), `lamella_viewer` (Protocol / Lamella Editor),
`FMAcquisitionWidget`, `fibsem/correlation/*`, labelling/training UIs, the coincidence viewer.
`fibsem/ui/napari/*` draw utilities stay — the deferred surfaces above still use them; only the
main-tab call-sites go.

### The `main_viewer` entanglement surface (what Phase 7 must sever)

`AutoLamellaMainUI.main_viewer` ([:955]) currently backs, via the control widgets:

| Widget | napari coupling to remove from the main-tab path |
|---|---|
| `FibsemImageSettingsWidget` | SEM/FIB image layers, scalebar, crosshair, **ruler**, alignment shapes, text overlay, `_update_layer_positions`, `reset_view` (~29 `self.viewer` refs) |
| `fluorescence_control_widget` | FM image layers (`update_image` → `viewer.add_image`) |
| `FibsemMovementWidget` | text overlay, napari `mouse_double_click_callbacks` |
| `FibsemSpotBurnWidget`, `FibsemEmbeddedDetectionWidget` | `self.viewer = parent.viewer` ref (overlays already migrated) |
| `milling_task_viewer_widget` | FIB image layer + the napari right-click drag callback |

Parity already exists on the canvas for: images, scalebar, crosshair (canvas draws its own from
`FibsemImage` metadata), all overlays, the info bar, and **click-to-move** (`canvas_double_clicked`
→ `_on_canvas_double_click`). The one true gap is the **ruler** (napari-only;
`FibsemImageSettingsWidget.update_ruler`).

### Shared-widget constraint → why `FibsemUI` is in scope

`FibsemImageSettingsWidget` and `FibsemMovementWidget` are constructed in **two** places:
AutoLamella's main tab and standalone `FibsemUI` ([FibsemUI.py:97,102]). `milling_task_viewer_widget`
is shared with the **Lamella Editor** (napari). So:

- The milling viewer **keeps both paths** (Lamella Editor still napari) — `_controller is None`
  already selects the napari path there; no change needed.
- The image/movement widgets would otherwise have to keep a guarded napari path forever — *unless*
  `FibsemUI` also moves onto the canvases. **It does** (this phase), which lets us delete their
  napari paths outright instead of guarding them.

`FibsemUI` is a napari-docked `QMainWindow` (viewer = canvas, tabs docked right: Connection / Image
/ Movement / Milling / Manipulator). Migrating it = give it its own `MicroscopeViewController` left
pane (same seam as `AutoLamellaMainUI`), exposed as `view_controller` so the widgets' `_view_controller()`
parent-walk resolves it. Caveat: `FibsemUI` also hosts **`FibsemManipulatorWidget`** (`viewer=self.viewer`),
another napari consumer — see slice 7.4.

### Slices (each flag-gated where it can be, reviewable, committed on approval)

- **7.0 — Make quad-view the real default (the felt cutover).** Replace the two `# DEV`
  force-`True` lines with a clean default: `FeatureFlags.quad_view_enabled = True` and the module
  constant `True`, *as the intended default* (not a dev override). `main_viewer` stays
  created-but-hidden as the rollback path (toggle the user pref → napari returns). Smallest diff,
  instantly reversible. **Soak here** before going further.
- **7.1 — Port the ruler to the canvas.** A matplotlib ruler on `FibsemImageCanvas` (line artist +
  length label in image/µm units), reachable from the Image widget. Closes the last parity gap.
  Until this lands, removing `main_viewer` would drop the measure tool.
- **7.2 — `FibsemUI` onto the canvases.** Give `FibsemUI` a `MicroscopeViewController` left pane;
  feed SEM/FIB through the controller; expose `view_controller`. Image + Movement + Milling tabs
  now drive the canvas. Keep the napari path alive *in parallel* this slice (still flag-gated) so
  `fibsem_ui` is verifiable side-by-side. **`FibsemUI` is the napari-free regression harness for 7.3.**
- **7.3 — Delete the napari path from the shared widgets.** With no napari host left for them, drop
  the `parent.viewer` requirement and every napari-layer call from `FibsemImageSettingsWidget` /
  `FibsemMovementWidget` (and the FM `update_image` napari branch). The mirror becomes the only path.
- **7.4 — Stop creating `main_viewer`; handle `FibsemUI` leftovers.** Remove `main_viewer` creation
  for the main tab; decide the **Manipulator** widget (migrate its overlay to the canvas, or keep a
  small dedicated napari viewer for `FibsemUI`'s manipulator only). Re-smoke minimap + lamella editor.
- **7.5 — Retire the flag.** Make the main tab unconditional, delete the dead napari main-tab
  branches and the now-unused `_view_controller() is None` fallbacks, drop the `quad_view_enabled`
  pref field (`UserPreferences.from_dict` already ignores unknown keys, so old configs still load).

### Verification harness

- **`fibsem_ui` standalone** is the cheapest end-to-end check for the shared widgets after 7.2 —
  run it napari-free and exercise Image / Movement / Milling without hardware where possible.
- After **7.4**, smoke the surfaces that *keep* napari (Minimap, Lamella Editor) to prove the
  removal didn't touch them.
- Headless: extend `tests/test_canvas_overlay_reducer.py` as the image-feed path becomes
  controller-only; add a ruler test in 7.1.

### Risks / watch-items

- **Manipulator widget** (`FibsemUI` only) is the one napari consumer with no canvas equivalent yet
  — it can gate how clean 7.4 is (migrate vs keep a scoped napari viewer).
- **`milling_task_viewer_widget` stays dual-path** (Lamella Editor) — do **not** delete its napari
  branch; only the main-tab/`FibsemUI` call-sites take the controller path.
- **Ordering:** 7.3 (delete napari path) must land *after* 7.2 (`FibsemUI` migrated), or `fibsem_ui`
  breaks. 7.4 (remove `main_viewer`) must land *after* 7.1 (ruler) or the main tab loses the ruler.
