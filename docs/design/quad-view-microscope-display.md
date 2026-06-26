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
| 7 — Cutover & removal | flip default to `True`, soak, delete napari branches + `fibsem/ui/napari/*` draw utils for the main tab | quad view default |

> Note: Phase 7 removes napari from the **main tab only**. The napari dependency
> stays until the deferred surfaces (minimap, protocol editor, labelling) migrate.

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
