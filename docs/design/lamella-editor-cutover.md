# Lamella (Protocol) Editor — Napari Deprecation

Follow-on to the main-tab quad-view cutover (`quad-view-microscope-display.md`,
`canvas-overlay-state-model.md`). Migrate the **Lamella tab → Protocol** sub-tab off its
napari viewer (`lamella_viewer`) onto the matplotlib canvas + reducer, for the same reasons:
napari's interaction model is awkward to extend, and each tab pays for a full
`napari.Viewer` (a `QMainWindow` + vispy/OpenGL context + dock machinery).

## Scope

- **In:** the Protocol editor only — `AutoLamellaProtocolEditorWidget`
  (`autolamella_lamella_protocol_editor.py`, ~1072 lines) + the `lamella_viewer` it's docked into
  (`AutoLamellaMainUI._add_lamella_editor_tab`, ~line 1196).
- **Out (for now):** the sibling **Review** tab (`LamellaTaskImageWidget`) — separate widget, not
  on `lamella_viewer`. The Minimap and all other napari surfaces stay.

`lamella_viewer` is used *only* by the protocol editor, so removing it = migrating that one widget.

## What the protocol editor actually does

A per-lamella, per-task protocol editor working on **saved reference images** (not live
acquisition). For the selected lamella + task it:

1. **Loads reference images from disk** — FIB (primary editing surface), SEM (optional), FM
   (optional) — via `FibsemImage.load(path)` / `FluorescenceImage.load(path)`, swapped on
   lamella/task/image-combobox changes.
2. Displays them in `lamella_viewer`: **SEM is placed to the *left* of FIB** (`translate=(0,
   -sem.shape[1])`) so you pan one continuous SEM|FIB view; FM overlays.
3. **Embeds `MillingTaskViewerWidget`** (`self.milling_task_editor`, built with
   `viewer=lamella_viewer`) — displays + edits the milling patterns for the selected task,
   including **background stages from other tasks** (dimmed, for context).
4. Draws a **Point of Interest** marker, moved via a right-click **"Move Point of Interest Here"**
   context-menu action (registered on the milling editor via `set_right_click_menu_actions`).
5. Draws an **editable alignment-area** rectangle with an **editable on/off toggle**
   (`setup_editable_alignment_layer`, `_on_alignment_area_updated → save`).
6. Swaps **per-task custom widgets** by selected task type: a **fluorescence acquisition** config
   form (form only — no viewer coupling) that drives FM-channel display, and a **spot-burn
   coordinates** editor (`AutoLamellaSpotBurnCoordinatesWidget`, its *own* napari points layer) for
   placing editable burn points on the FIB image.

### How it differs from the main-tab widgets (important)

| Aspect | Main tab | Protocol editor |
|---|---|---|
| Image source | live microscope acquisition | **static** `FibsemImage.load()`, per lamella/task |
| Layout | one canvas per beam (quad) | **SEM beside FIB** in one view + FM overlay |
| Switching | n/a (live) | per-lamella + per-task + per-image-file |
| POI | drag-only marker | **right-click "Move POI Here"** menu action |
| Alignment | workflow-driven | **editable toggle**, per-lamella, saves to the lamella |
| Patterns | display + reposition | display + reposition + **background stages** (already supported) |

## What's reuse vs. genuinely new

Nearly all of the editor is **wiring** onto machinery that already exists — no new overlay capability
is needed once right-click-to-add is accepted for spot burn (below). The one bit of *new structure*
is extracting a **shared spot-burn coordinate-editor** (see gap 6 / L3).

Reuse (no new capability):
- **Milling pattern editing is parameter-driven, not freeform.** Edits come from the milling-config
  form (`milling_task_editor.settings_changed → save`) + right-click reposition — *not* drawing on
  the image. `MillingTaskViewerWidget`'s canvas path (live in the main tab) already does display +
  reposition + background stages. No new shape engine.
- **POI + alignment overlays already exist:** `PointsSpec`/`PointOverlay` (POI) with the canvas's
  `canvas_right_clicked` + the milling editor's right-click hook ("move here");
  `AlignmentAreaOverlay`/`AlignmentSpec` (editable alignment).
- **FM display is reuse:** the controller's `set_fm_image` / `FMCanvasWidget` already renders the
  multi-channel composite; the fluorescence config widget is a **form only** (its only napari is a
  `__main__` demo), so it just emits `settings_changed`.
- **Static image feed is trivial:** `controller.set_image(beam, FibsemImage.load(...))` — the
  reducer doesn't care whether the image is live or loaded.

Spot-burn — with **right-click-to-add accepted (decided)** — is also reuse, not new capability:
- `PointOverlay` already does right-click-add (`add_on_right_click`), drag-move, Delete-remove,
  select-highlight, and `modal=True` (docstring: "handles input only while it's the active overlay —
  e.g. spot burn — inert in Move mode"). It emits `point_added` / `point_moved` / `point_removed`,
  which the controller already forwards to `overlay_edited`; `set_points()` drives the reverse
  (table → overlay) and does *not* echo (only user gestures emit).
- **Working precedent in-repo:** the live `FibsemSpotBurnWidget` already runs
  `add_on_right_click=True, removable=True, modal=True` on the canvas — the editor's coordinates
  widget copies that pattern. Two-way *data* sync (table ↔ overlay) is fully supported; the old
  Add/Select **mode buttons disappear** (right-click adds, left-click selects/drags, Delete removes).
  Only a table-row → highlight-point *selection* nicety would need a small overlay helper.

## The display surface: task-driven single view (revised)

**Key realization: the editor is *already* a single, task-driven view.** Today it uses one napari
viewer and swaps *what's shown* by the selected task type (`_on_selected_task_changed`,
`layer.visible = is_fluorescence_task`):

| Selected task | Shows |
|---|---|
| milling | FIB (+ optional SEM) + milling patterns + POI + alignment |
| fluorescence | FM channels (FIB / patterns / POI hidden) |
| spot burn | FIB + spot-burn points |

So "one visible canvas that follows the selected task" isn't a redesign — it's a faithful port of the
current UX, and a **better fit than a static 2×2 quad** (which would show three always-on panes the
editor doesn't want).

**Decision (revised): reuse the controller, not the quad layout.** Keep `MicroscopeViewController` as
the overlay reducer — every widget (milling / POI / alignment / detection) already talks to it — but
give the editor a **task-driven stacked view** instead of `QuadViewWidget`:

- a **FIB `FibsemImageCanvas`** (patterns + POI + alignment + spot-burn points), and
- the **`FMCanvasWidget`** (multi-channel composite),
- in a `QStackedWidget` (one visible at a time); the selected task type picks which page is on top —
  mirroring today's visibility toggle exactly.

The controller only reaches into its view via `.fib_canvas` / `.fm_canvas` / `.fm_widget` /
`.sem_canvas`, so the seam is small: let the controller **accept a pre-built view widget** exposing
those four attributes, and hand it a `LamellaEditorView` instead of the quad. SEM stays a **toggle**
(a second canvas shown on demand, or deferred) — *not* the old SEM-beside-FIB translate, so the
offset math stays dead.

> **Decided: reuse the controller (overlay reducer) with a task-driven stacked view** (FIB canvas ⟷
> FM widget), not the 2×2 quad. Supersedes the earlier "reuse the quad as-is" note.

## Capability / wiring gaps to close

1. **Controller accepts an alternate view.** Let `MicroscopeViewController(view=…)` take a pre-built
   `LamellaEditorView` (FIB canvas + FM widget in a stack) instead of hardcoding `QuadViewWidget`;
   the beam→canvas maps stay identical. Task type drives which stacked page is visible.
2. **`MillingTaskViewerWidget` controller injection.** It currently resolves the controller via
   `self._image_widget._view_controller()`; the editor has no `image_widget`. Add a direct
   `set_controller(controller)` (or `controller=` ctor arg). Keep its existing napari path additive
   for any other caller.
3. **Static reference-image feed** per selection (FIB/SEM/FM) → `controller.set_image` /
   `set_fm_image`, replacing `_update_fib/sem/fm_image_layer`.
4. **POI overlay + right-click "move here"** — a `PointsSpec` on `fib_canvas`; the existing
   right-click hook repositions. (Reuse the main-tab POI pattern.)
5. **Editable alignment** — `AlignmentSpec` on `fib_canvas` driven by the editable toggle
   (`set_alignment_edit`), `overlay_edited → save to lamella.alignment_area`.
6. **Spot-burn interactive points → a *shared* coordinate-editor.** Rather than port
   `AutoLamellaSpotBurnCoordinatesWidget` one-off, extract a reusable spot-burn coordinate-editor: it
   drives a `PointsSpec`/`PointOverlay` on the controller (`add_on_right_click=True, removable=True,
   modal=True`, per the `FibsemSpotBurnWidget` precedent) + the 0–1↔pixel conversion + table, driven
   by `SpotBurnFiducialTaskConfig.coordinates`. Wiring: `overlay_edited` → px→relative(0–1) → table +
   save; table edits → `controller.set_points`. Drop the Add/Select mode buttons; coordinates simplify
   to single-image relative(0–1)↔pixel (no layer `translate`). The **protocol editor adopts it in the
   cutover**; retrofitting the live `FibsemSpotBurnWidget` is a **follow-up** (below).
7. **Per-image contrast/gamma persistence** across swaps (the napari path kept layer params on image
   switch) — minor; the canvas has `ContrastGammaControl`, decide whether to persist it.

## Slices (each independently shippable, reviewable)

- **L0 — controller seam + stacked view.** `LamellaEditorView` (FIB canvas + FM widget in a
  `QStackedWidget`); `MicroscopeViewController(view=…)`; `MillingTaskViewerWidget.set_controller()`.
  Build behind the existing layout — nothing user-visible yet.
- **L1 — reference images + patterns.** Feed FIB/FM (SEM via toggle) to the controller on selection;
  swap the editor's left pane from `lamella_viewer._qt_window` to `controller.widget`; task type
  drives the visible page. Milling patterns render via the milling editor's canvas path.
- **L2 — POI + alignment.** Right-click "Move POI Here" marker + editable alignment overlay/toggle
  saving to the lamella. Both reuse existing overlays.
- **L3 — spot-burn coordinate-editor (shared, extracted).** Build a reusable spot-burn coordinate-
  editor (overlay + 0–1↔pixel + table, driven by `SpotBurnFiducialTaskConfig.coordinates`) and adopt
  it in the protocol editor behind the spot-burn task type. Cutover scope stops here — the live-widget
  retrofit is a follow-up.
- **L4 — fluorescence.** Render FM channels on the FM page by task type (mostly `set_fm_image`); the
  config widget is form-only, just wire `settings_changed`.
- **L5 — remove `lamella_viewer`.** Drop the napari viewer + placeholder layer, the
  `_update_*_image_layer` / `_draw_*` / `setup_editable_alignment_layer` napari paths, the spot-burn
  widget's napari layer, and the napari imports.

## Risks / watch-items

- **Spot-burn right-click-add is gated by `modal=True`** — the overlay is inert unless the spot-burn
  task is the active overlay, so it won't collide with the milling/POI right-click menu. Only the
  table-row → highlight-point selection nicety may need a small overlay helper.
- **`MillingTaskViewerWidget` is shared** (main tab + this editor + its own napari path). Add the
  controller seam *additively* — don't disturb the existing napari path other callers may use.
- **Stacked view is a visible change** — one canvas at a time instead of napari's layer stack, and
  SEM moves from beside-FIB to an on-demand toggle. Faithful to today's per-task visibility, but
  confirm the SEM handling.
- **Coordinate space:** single-image pixel space per canvas — do *not* reintroduce the side-by-side
  translate offsets.
- **Reference-image edge cases:** missing FIB image (blank fallback), SEM/FM optional, contrast/gamma
  persistence across swaps.
- Verification: build the editor viewer-less headless against a saved experiment (like the
  `test_viewer_less_widgets` harness); live-test milling edit + reposition, POI, alignment,
  **spot-burn add/move/remove**, and fluorescence per lamella/task.

## Follow-ups (deferred — not in this cutover)

- **Retrofit the live `FibsemSpotBurnWidget` onto the shared coordinate-editor.** The cutover extracts
  the editor and uses it in the *offline* protocol editor; adopting it in the live widget too yields
  one point-placement spine and a clean **define-config vs run** split:
  - *authoring shell* (protocol editor) = shared editor + table + generic params → writes
    `SpotBurnFiducialTaskConfig`, no microscope;
  - *execution shell* (live/workflow) = shared editor + current/exposure + a runner (worker around
    `run_spot_burn`, `fibsem/imaging/spot.py`).

  Deferred because `FibsemSpotBurnWidget` is on the **live running-workflow path** (higher blast
  radius) and still uses the `thread_worker` we've separately deferred — best not entangled with a
  napari-removal change. Keep `SpotBurnFiducialTaskConfig` as the single source of truth throughout.
- **Unify the supervised workflow confirm.** Once the live widget uses the shared editor, the
  supervised spot-burn confirm step (`SpotBurnFiducialTask.update_spot_burn_parameters_ui` →
  `update_spot_burn_parameters` / `ask_user(spot_burn=True)`) shows the *same* editor as protocol
  authoring instead of a second, divergent point UI. (Unsupervised already calls `run_spot_burn`
  headless — unchanged.)
- **FM z-scrubbing + max projection.** The FM page currently shows a fixed composite. Add a **z-slider**
  to scrub through the fluorescence z-stack and a **"max projection" checkbox** in the FM layer
  controls. Touches `FMCanvasWidget` (per-channel z-index + projection mode), not the reducer.
