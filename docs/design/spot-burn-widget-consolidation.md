# Spot Burn Widget Consolidation — live widget onto the shared coordinate editor

Follow-on to the lamella-editor cutover (`lamella-editor-cutover.md`). The protocol editor now
authors spot-burn coordinates through a shared, controller-based
`AutoLamellaSpotBurnCoordinatesWidget` (app-styled list + canvas points overlay). The **live**
`FibsemSpotBurnWidget` (the supervised-workflow confirm step) still carries its *own* inline copy of
the same overlay logic. Consolidate onto the shared editor and make the roles explicit:
**define the coordinates (shared editor) vs run the burn (live shell).**

## What each does today

- **Shared editor** (`AutoLamellaSpotBurnCoordinatesWidget`) — controller-based. `set_task_config` /
  `get_task_config` / `set_image_shape` / `settings_changed(SpotBurnFiducialTaskConfig)`; a
  `PointsSpec` `"spot_burn"` overlay (right-click add / drag / Delete / numbered markers / two-way
  row↔point selection); a titled list with add + per-row trash.
- **Live widget** (`FibsemSpotBurnWidget`, ~305 lines) — a Qt-Designer `.ui` form
  (`comboBox_beam_current`, `doubleSpinBox_exposure_time`, `pushButton_run_spot_burn`, `progressBar`,
  `label_information`) **plus an inline `"spot"` `PointsSpec` overlay** (`_ensure_spot`,
  `_set_coordinates`, `get_coordinates`, `_on_data_changed`, arming/visibility) — a near-duplicate of
  the shared editor's overlay code, but with only a *count* label (no list) and the extra bits the
  editor doesn't have: **beam current / exposure controls + run / cancel / progress** via
  `@thread_worker` → `run_spot_burn` (`fibsem/imaging/spot.py`).
- **Workflow integration** (`SpotBurnFiducialTask`, supervised): `update_spot_burn_parameters(dict)`
  → `workflow_update_signal` → UI handler → `spot_burn_widget.update_parameters(dict)`; the user
  places points and burns; `get_coordinates()` is read back into the config; `clear_points_layer()`
  on exit. Activation via `set_spot_burn_widget_active` → `set_active()` / `set_inactive()`. The
  widget is a lazily-created "Spot Burn" tab. The **unsupervised** path calls `run_spot_burn`
  headless and is untouched.

## The duplication we delete

Every overlay/coordinate method on the live widget already exists (better) on the shared editor:
`_ensure_spot` · `_on_spot_edited` · `_set_coordinates` (0-1→px) · `get_coordinates` (px→0-1) ·
`_on_data_changed` (count) · arming/visibility in `set_active` / `set_inactive` ·
`clear_points_layer` overlay teardown. All of it goes; the live widget delegates to an embedded
editor instance.

## Data model: `SpotBurnSettings` (the shared currency)

Today the shared editor speaks `SpotBurnFiducialTaskConfig` — an *autolamella* structure — even though
it lives under `fibsem/ui`. Embedding it in the **fibsem-level** `FibsemSpotBurnWidget` would drag an
autolamella dependency down into fibsem (a layering smell). Fix both at once with a small, fibsem-level
payload next to `run_spot_burn`:

    # fibsem/imaging/spot.py
    @dataclass
    class SpotBurnSettings:
        coordinates: list[Point] = field(default_factory=list)
        milling_current: float = 60e-12   # A
        exposure_time: float = 10.0       # s
        # to_dict / from_dict; optional .run(microscope, beam_type=ION, parent_ui, stop_event)

It maps 1:1 to `run_spot_burn`'s real arguments. `orientation` is workflow stage-movement, *not* part
of the burn, so it stays on the task config. Everyone shares this object:

- **Shared editor** switches to `set_settings` / `get_settings` / `settings_changed(SpotBurnSettings)`,
  editing `.coordinates` and passing current/exposure through untouched. This drops its autolamella
  import → it becomes genuinely fibsem-level (worth renaming to `SpotBurnCoordinatesWidget`, optional).
- **`SpotBurnFiducialTaskConfig`** gains `to_settings()` / `apply_settings()` converters — no change to
  its on-disk `to_dict` / `from_dict`, so saved experiments are safe. Orientation + task machinery stay.

## Design: define vs run

- **Execution shell = `FibsemSpotBurnWidget`** = embedded editor (coordinates) + beam-current/exposure
  form + run/cancel/progress, all over one `SpotBurnSettings`: the editor writes `.coordinates`, the
  form writes current/exposure, and `run_spot_burn` consumes the whole thing.
- **Authoring = the shared editor** (same class + `SpotBurnSettings` in both hosts).
- The **protocol editor** feeds `config.to_settings()` and reads `settings.coordinates` back into the
  lamella's task config (current/exposure there stay owned by the generic parameters widget).

## Wiring / gaps to close

**Shared (data model) — S0:**
1. Add `SpotBurnSettings` (+ `to_dict` / `from_dict`, optional `.run(...)`) to `fibsem/imaging/spot.py`.
2. Move the shared editor onto `set_settings` / `get_settings` / `settings_changed(SpotBurnSettings)`;
   drop the `SpotBurnFiducialTaskConfig` import.
3. Add `SpotBurnFiducialTaskConfig.to_settings()` / `apply_settings()`; adapt the protocol editor —
   `set_settings(config.to_settings())`, and on `settings_changed` set `config.coordinates =
   settings.coordinates` and save (current/exposure stay owned by the generic parameters widget).

**Live widget (`FibsemSpotBurnWidget`) — S1:**
4. **Rebuild the widget in code (drop the `.ui`).** Replace the Qt-Designer `Ui_Form` base with a
   code-built layout — a `QVBoxLayout` of: the embedded editor, a small form (beam-current combo +
   exposure spin), the info label, the progress bar, and the run/cancel button — matching the other
   migrated widgets. Delete `qtdesigner_files/FibsemSpotBurnWidget.ui` and
   `qtdesigner_files/FibsemSpotBurnWidget.py` and the `FibsemSpotBurnWidgetUI` import (the generated UI
   is imported *only* by this widget, so it's self-contained). Delete the inline `"spot"` overlay code
   at the same time.
5. Live image-shape feed (the one live-specific difference): subscribe
   `image_widget.viewer_update_signal` + feed on activate → `editor.set_image_shape(
   ib_image.data.shape[:2])` (re-syncs the overlay from the relative coords across acquisitions).
6. Hold a `SpotBurnSettings`: the form writes current/exposure, the editor writes coordinates.
   `settings_changed` → `label_information` (N points, N·exposure s) + run-button state.
   `run_spot_burn_worker` runs the settings; run/cancel/progress unchanged.
7. Public API stays thin adapters (workflow unchanged): `update_parameters(dict)` → set the form +
   `editor.set_settings(settings-from-dict)`; `get_coordinates()` → `editor.get_settings().coordinates`
   (bounds-filtered); `set_active` / `set_inactive` (feed shape + refresh button; arming via show/hide);
   `clear_points_layer()` → `editor.set_settings(SpotBurnSettings())`.

## Out of scope (separate follow-ups)

- **`thread_worker` stays.** The run still uses `@napari.qt.threading.thread_worker`. After this
  retrofit that is the **only** remaining napari import in the widget; removing it belongs to the
  separate `thread_worker` effort.
- **Workflow signal stays dict-adapted.** `update_spot_burn_parameters` keeps emitting the parameter
  dict; the live widget builds a `SpotBurnSettings` from it at the boundary. Threading `SpotBurnSettings`
  through `workflow_update_signal` end-to-end (and having `SpotBurnFiducialTask` hand off its settings
  directly) is a later step — keeps this retrofit off the running-workflow signal path.
- Optional: have the shared editor set the FIB-canvas hint on activate (nice for both hosts).

## Slices

- **S0 — `SpotBurnSettings` + editor swap.** Add the dataclass; move the shared editor onto it; add the
  config converters; adapt the protocol editor. Re-verify the protocol editor (the lamella headless
  smoke + the widget suite) — this re-touches code shipped in the lamella cutover.
- **S1 — retrofit the live widget (pure code).** Rebuild `FibsemSpotBurnWidget` in code (drop the
  `.ui`, delete both qtdesigner files); embed the editor; delete the inline overlay code; adapt the
  public API to settings; feed image shape; wire the info label + run button. **Zero workflow changes.**
- **S2 — verify.** Headless build of the live widget (Demo microscope + controller, like
  `test_viewer_less_widgets`); confirm adapters, count, overlay, clear. Live-test the supervised step:
  place → burn → continue → clear.

## Risks / watch-items

- **S0 re-touches shipped code** — the shared editor + protocol editor just merged; the config→settings
  swap changes their integration. Re-run the lamella headless smoke + the widget suite after S0.
- **Overlay id `"spot"` → `"spot_burn"`** — confirm nothing else references `"spot"` (it's internal
  to the widget).
- **Controller availability** — the widget is created lazily inside `AutoLamellaUI` (which has
  `view_controller`); still guard for `None`.
- **Image-shape timing** — coords are relative(0-1); feeding on activate + on image-update covers
  re-acquisition, and `set_image_shape` repositions the overlay.
- **Arming lifecycle in a tab** — verify show/hideEvent fire on tab switch (they do); if the editor
  doesn't arm reliably there, expose an explicit `set_active(bool)` on the editor (idempotent with
  the show/hide path).
- **Not fully napari-free** — `thread_worker` remains until its own effort.
