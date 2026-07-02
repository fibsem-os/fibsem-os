# Overview (Minimap) Tab — Napari Deprecation

Follow-on to the main-tab quad-view cutover (`quad-view-microscope-display.md`,
`canvas-overlay-state-model.md`) and the lamella-editor cutover
(`lamella-editor-cutover.md`). Migrate the **Overview tab** — `FibsemMinimapWidget`
(`FibsemMinimapWidget.py`, ~1508 lines) — off its dedicated `napari.Viewer` onto the
matplotlib `FibsemImageCanvas`, for the same reasons: napari's interaction model is awkward
to extend, and each viewer pays for a full `napari.Viewer` (a `QMainWindow` + vispy/OpenGL
context + dock machinery).

This is the **last napari viewer in the AutoLamella main UI** — the main tab, the standalone
`FibsemUI`, and the lamella/protocol editor are already migrated. Removing it clears the way to
drop the napari dependency from the app's main window entirely.

## Scope

- **In:** `FibsemMinimapWidget` — the widget itself owns the migration.
- **Two call sites** construct it with a napari viewer today; both get updated:
  - `AutoLamellaMainUI.add_minimap_tab` (the Overview tab) — creates `minimap_viewer` and docks
    `minimap_viewer.window._qt_window` beside the controls in a splitter.
  - `FibsemUI.open_minimap_widget` — creates `viewer_minimap` on demand for the standalone UI.
- **Out (deferred, see below):** the correlation *drag-to-align* interaction and the *milling-pattern
  overlay* — both deliberate scope cuts, not oversights.

## What the minimap actually does

A large, tiled **overview image** of the grid with stage-linked overlays and click-to-navigate.
For the current experiment it:

1. **Displays an overview image** — acquired via tiled collection (`run_tile_collection`), loaded
   from disk (`load_image`), or a blank placeholder (`draw_blank_image`). Grayscale FIB/SEM tile.
2. **Composites correlation images** — N loaded FM/other images (`add_correlation_image`) blended
   over the overview with per-layer colormap + opacity, plus a synthetic **gridbar** overlay
   (`generate_gridbar_image`, toggled + regenerated on spacing/width change).
3. **Draws stage-linked shape overlays** (`_collect_all_overlays` / `_draw_overlay_shapes`):
   current-overview FoV (magenta rect), stage limits (yellow rect), grid-boundary circle (red
   ellipse), TEM limits (orange rect), and one rect per **saved lamella position** (selection-tinted),
   each with a text label.
4. **Draws crosshairs** (`_collect_crosshair_overlays` / `_draw_position_crosshairs`): origin, grid
   slots, current stage position, and saved positions — lines + labels.
5. **Overlays milling patterns** for all saved positions (`_draw_milling_pattern_overlay` →
   `draw_milling_patterns_in_napari`), reprojected onto the overview, toggled + task-selected via a
   checkbox/combobox.
6. **Shows a microscope-state info bar** (`update_text_overlay`): stage, milling angle, grid, objective.
7. **Click-to-navigate** via three viewer mouse callbacks:
   - **single-click** → select the nearest saved position within 50 µm (`on_single_click`).
   - **double-click** → move the stage to the clicked point (`on_double_click`, stage-limit checked).
   - **right-click** → context menu: *Add New Position Here* / *Move Selected Position Here*
     (`_on_right_click`).

Everything is anchored to stage coordinates via pure, napari-free helpers:
`tiled.reproject_stage_positions_onto_image2`, `conversions.image_to_microscope_image_coordinates`,
`microscope.project_stable_move`. **These port unchanged.**

## What's reuse vs. genuinely new

The headline: **no new canvas class and no new compositor.** Almost every capability already exists
from the prior cutovers; this is mostly *rewiring* napari layer ops onto canvas equivalents. The
net-new code is a small set of **entity-grouped overlays** (see below) on one generic shape primitive.

Reuse (no new capability):

- **Image display + multi-image compositing.** `FibsemImageCanvas.set_array()` already accepts an
  `HxWx3` RGB composite (built for the FM canvas), and `FMLayer` + `composite_fm_layers()`
  (`fm_composite.py`) already blend a gray base + colored, opacity-weighted layers additively. The
  minimap's overview (gray `FMLayer`) + correlation images (colored `FMLayer`s) + gridbar (colored
  `FMLayer`) map onto this **directly**. Composite → `set_array` on shape change / `update_display`
  in steady state (exactly what `FMCanvasWidget._recomposite` does).
- **Clicks.** The canvas emits `canvas_clicked` / `canvas_double_clicked` / `canvas_right_clicked`
  with **full-resolution image-pixel `(x, y)` + modifiers** (confirmed:
  `FibsemMovementWidget._canvas_double_click_worker` — "the canvas emits data coords, so no
  `world_to_data` needed"). So `get_coordinate_in_microscope_coordinates` collapses to a bounds check
  + the existing `image_to_microscope_image_coordinates` call. The three napari callbacks become three
  signal handlers with identical downstream logic.
- **Info bar.** `FibsemImageCanvas.set_info_text()` + the info-bar artist already exist; the content
  logic is already reimplemented once in `quad_view.update_info` (mirrors `update_text_overlay`). Port
  the same fields.
- **Scalebar, pan/zoom, reset-view, crosshair toggle, contrast** — all free from `FibsemImageCanvas`.
- **Context menu** — `ContextMenu`/`ContextMenuConfig` are Qt, already napari-free.

### Overlay grouping — by entity, not by shape type

napari grouped overlays by *shape type* (all crosshairs on one Shapes layer, all boxes on another)
because a napari Shapes layer holds one kind of geometry — so a single lamella was smeared across two
layers. The matplotlib overlays have no such constraint, so we group the way the domain thinks: **by
entity + update cadence.** This fixes a real problem the napari code even flags — a `# Performance
Note` that *every* shape is rebuilt whenever a position is added or the stage moves — because
type-grouping forces `draw_current_stage_position` to redraw everything. Entity groups redraw
independently.

One generic primitive underneath: a **`MinimapShapesOverlay`** (`CanvasOverlay` subclass) rendering a
list of shape specs — `Rectangle` (FoV / limit / TEM / position boxes), `Ellipse` (grid boundary),
line-pair (crosshair), each with an edge colour + optional text label. Geometry comes straight from the
reprojected centre point + width/height (no napari vertex arrays; `create_rectangle_shape` /
`create_circle_shape` / `create_crosshair_shape` are dropped).

Three overlay instances own the specs, grouped by entity + refresh trigger:

| Overlay | Owns | Refreshes when |
|---|---|---|
| **LamellaMarkers** | per lamella: FOV box + crosshair + name label, coloured by defect/selection | add / remove / select / defect change |
| **CurrentPosition** | current overview-acquisition FOV box + current crosshair | every stage move |
| **ReferenceFrame** | origin + grid-slot crosshairs, stage-limit rect, grid-boundary circle, TEM rect | image load / holder change (≈static) |

This maps 1:1 onto the existing display-option toggles (which are *already* functional, not
type-based): "Show Saved Positions FOV" → LamellaMarkers box visibility, "Show Stage Limits / Circle /
TEM" → ReferenceFrame sub-visibility, "Show Overview FOV" → CurrentPosition box. It also localises the
one cross-cutting quirk — a lamella's name label shows on its box when the FOV is on, else on its
crosshair — which was awkward split across napari layers and is trivial inside one marker.

## Disabled: correlation + gridbar (rework pending)

**Update (2026-07-02): correlation *and* gridbar are now disabled entirely** — controls hidden, handlers
no-op — pending a proper rework. See **`minimap-correlation-gridbar-rework.md`** for the why and the
requirements.

Short version: the first cut composited correlation images as colour-tinted `FMLayer`s, stretched to
the overview shape when the resolution didn't match. But correlation is fundamentally about *aligning*
two coordinate frames — a stretch-to-fit overlay is geometrically wrong except in the same-FOV case,
and napari's old inline "transform mode" never persisted its affine either. Doing it right (a real,
saved alignment transform, ideally reusing `fibsem/correlation/`) is a sizeable piece of work, so it's
been reverted rather than shipped half-working. The gridbar composited fine but its model
(image-in-composite vs. a vector overlay tied to real grid geometry) is reconsidered in the same rework.

## Deferred: milling-pattern overlay

napari drew the selected task's milling pattern reprojected onto every saved position
(`draw_milling_patterns_in_napari`), toggled by a "Display Pattern" checkbox + task combobox. This is
*portable* — `MillingPatternOverlay` (live in the main tab + lamella editor) already renders
`FibsemMillingStage`s on the canvas, so it's just "reproject the stage list → hand to the overlay."

Decision (2026-07-02): **defer it** — a display-only convenience (the code even carries a perf caveat
about redrawing every position on each change), not needed to get the minimap off napari.
`_draw_milling_pattern_overlay` stays a no-op stub; the "Display Pattern" checkbox + task combobox are
disabled with a tooltip. Re-add later by feeding the reprojected stage list to a `MillingPatternOverlay`.

## Architecture

```
overview (gray FMLayer) ─┐
correlation imgs (FMLayers)├─ composite_fm_layers() ─→ RGB ─→ canvas.set_array()/update_display()
gridbar (FMLayer) ───────┘

canvas overlays:  LamellaMarkers   (per-lamella box + crosshair + label + colour)
                  CurrentPosition  (current FOV box + crosshair)
                  ReferenceFrame   (origin/grid crosshairs, stage limits, grid boundary, TEM)
                  — all on one generic MinimapShapesOverlay primitive
                  (milling-pattern overlay deferred)

canvas signals:   canvas_clicked      → select nearest saved position (≤50 µm)
                  canvas_double_clicked→ project_stable_move → move stage (limit-checked)
                  canvas_right_clicked → ContextMenu (add / move position)

info bar:         canvas.set_info_text(...)  (stage / angle / grid / objective)
```

The widget keeps its imperative `draw_current_stage_position()` / `update_viewer()` structure — those
methods just call canvas + overlay APIs instead of napari layer APIs. No reducer/`SceneModel` is
introduced: the minimap is a single standalone image with bespoke overlays, so the quad-view reducer
would be a poor fit; the imperative model that's already here is the right altitude.

## Staged plan

Each stage keeps the widget **runnable** (imports clean, no crash) so the cutover can be reviewed and
smoke-tested incrementally. Napari overlay methods early-return while unported, then get their bodies
swapped stage by stage; the final stage deletes the dead napari branches + imports.

- **M0 — canvas seam + image display.** This doc. Create `FibsemImageCanvas` inside the widget; drop
  the `viewer` constructor arg; update both call sites (`AutoLamellaMainUI.add_minimap_tab`,
  `FibsemUI.open_minimap_widget`) to stop building a `napari.Viewer`. Route `update_viewer` /
  `draw_blank_image` / `load_image` through the `FMLayer` composite → `set_array`. Info bar ported.
  Overlay-drawing methods stubbed to no-ops (guarded). **Result:** overview image + pan/zoom + info
  bar on matplotlib; no overlays or clicks yet.
- **M1 — click parity.** Wire `canvas_clicked` / `canvas_double_clicked` / `canvas_right_clicked` to
  the ported single/double/right-click logic (bounds check + `image_to_microscope_image_coordinates`
  + `project_stable_move` + the existing context menu).
- **M2 — generic primitive + ReferenceFrame + CurrentPosition.** Build `MinimapShapesOverlay`; render
  the static/current geometry: origin + grid-slot crosshairs, stage-limit rect, grid-boundary circle,
  TEM rect (ReferenceFrame), plus the current overview-FOV box + current crosshair (CurrentPosition).
  Wire the display-option toggles to group visibility. Geometry ports from `_collect_all_overlays` /
  `_collect_crosshair_overlays`.
- **M3 — LamellaMarkers.** Per-lamella FOV box + crosshair + name label, coloured by defect/selection;
  refreshed on add/remove/select/defect. The box↔crosshair label switch lives here. Selection sync via
  `update_current_selected_position`.
- **M4 — correlation + gridbars.** ~~Load correlation images + gridbar as `FMLayer`s in the
  composite.~~ **Reverted + disabled** — the composite/stretch-to-fit model can't do aligned
  correlation; controls hidden, handlers stubbed. See `minimap-correlation-gridbar-rework.md`.
- **Extras (post-review, kept):** black canvas background + ~2x view margin (opt-in
  `FibsemImageCanvas.set_background_color` / `set_view_margin`; the margin also stops the
  stage-limit / grid-boundary overlays clipping at the image edge); the display-option toggles moved
  into a canvas-toolbar popover (napari-style layer controls).
- **M5 — remove napari + finish.** Delete remaining `napari` imports + dead layer branches from the
  widget; hide the disabled correlation/gridbar + milling-pattern controls; headless verify (offscreen)
  + smoke both call sites; update memory + this doc's status.

## Risks / watch-list

- **Composite performance on large overviews.** Tiled overviews can be large (e.g. 4096²+).
  `composite_fm_layers` runs in numpy per recomposite; `FibsemImageCanvas` already downsamples for
  display (`_downsample(_MAX_DISPLAY_PX)`). Watch recomposite cost when many correlation layers are
  present; composite lazily (only on layer change), not on every overlay redraw.
- **Coordinate parity.** The single/double/right-click all reproject through the same helpers; verify
  a click at a known pixel maps to the same stage position as napari did (headless assert against
  `image_to_microscope_image_coordinates`).
- **Overlay redraw on pan/zoom.** Shape/crosshair overlays are in data coords, so they pan/zoom with
  the image for free (matplotlib patches on the axes) — no per-frame reprojection needed, unlike the
  napari text-scaling. Confirm labels don't rescale awkwardly.
- **Selection sync.** `lamella_list` selection tints a lamella's box + crosshair (both owned by
  LamellaMarkers now); keep the redraw hook (`update_current_selected_position` → LamellaMarkers refresh).
