# Correlation widget: FM surface point + pre-correlation refractive-index correction

**Status:** implemented (pending live GUI test on real data)
**Branch:** `claude/correlation-widget-surface-point-2904d2`

## Decisions (resolved 2026-07-15)

1. **LUT tilt in pre mode:** locked to 0° (FM optical axis ⊥ sample surface);
   the tilt spinbox is disabled with an explanatory tooltip and restored on
   returning to post mode.
2. **POI scope in pre mode:** all POIs are corrected (each has its own depth);
   post mode remains POI 1 only.
3. **Apply UX:** Apply in pre mode stores the factor and re-runs the correlation;
   a "Re-run on apply" checkbox (default on) can defer application to the next
   manual run.
4. **Surface exclusivity:** only one surface point may exist at a time — adding a
   FIB surface removes the FM surface and vice versa (`set_data` prefers the FM
   surface if a hand-edited file contains both). Consequently the RI-tab mode
   needs no selector: it is implied by which surface exists.
5. **Enum value:** `PointType.SURFACE_FM = "FM-SURFACE"` (JSON token, menu label
   "Add FM-SURFACE", row names "FM-SURFACE 1").

## Feature request

The correlation widget currently lets the user pick a **surface point on the FIB image**
and apply a refractive-index (RI) depth correction to the **correlated POI result**
(post-correlation, in FIB image space). Users want to additionally be able to pick a
**surface point in the FM image** and apply the correction to the **pre-correlation POI**
(in FM volume space, before the 3D→2D transform is fit/applied).

## Current implementation

### Data model — `fibsem/correlation/structures.py`

- `PointType`: `FIB | FM | POI | SURFACE`. `SURFACE` is implicitly FIB-image space.
- `CorrelationInputData.surface_coordinate: Optional[Coordinate]` — single FIB-space
  surface point (x, y in FIB pixels).
- `CorrelationResult.apply_refractive_index_correction(factor)` — post-correlation,
  in-place on `poi[0]` only, FIB image space:
  `corrected_y = surface_y + (poi.image_px.y - surface_y) * factor`,
  then recomputes `px` / `px_m` from `image_px`. Stores
  `refractive_index_correction_factor` on the result.

### Engine — `fibsem/correlation/correlation_v2.py`

- `run_correlation_from_data(data)` → builds arrays from coordinates,
  fits `Rigid3D.find_32` from **fiducials only** (FM 3D ↔ FIB 2D), then reprojects the
  POI array through the fitted transform. POI does not influence the transform.
- The fit uses random restarts (`ninit=10`) → run-to-run results vary slightly.
- FM coordinates are passed in **voxel units** and assumed isotropic
  (`io.py`: "assume isotropic").

### Widget — `fibsem/correlation/ui/widgets/correlation_tab_widget.py`

- FIB canvas: `allowed_point_types=[FIB, SURFACE]` → right-click "Add SURFACE".
- FM display: `allowed_point_types=[FM, POI]`; added points get `z = current_z` slider value.
- `_CoordinatesTab` has four `CoordinateListWidget` panels (FIB / FM / POI / Surface);
  surface is max-1 (replace on add).
- `_RITab` (tab index 3, disabled until first result):
  - embeds `RefractiveIndexWidget` (5 optical params → ζ via 5D LUT,
    `fibsem/correlation/refractive_index.py`; factor spinbox, default 1.47),
  - shows surface→POI y-distance (px), Apply button populates a table of
    original/corrected Y and calls `result.apply_refractive_index_correction(factor)`,
    re-emits `result_changed`, re-draws FIB overlay.
- Consumer: `fibsem/ui/widgets/autolamella_lamella_protocol_editor.py` opens
  `CorrelationTabDialog` and reads `result.poi[0].px_m` on accept.
- `correlation_point_picker_widget.py` is an orphaned parallel prototype (referenced
  only by its own test script) — out of scope, do not touch.

### Correction math (FM space, requested mode)

RI mismatch stretches apparent depth along the FM optical axis (z). Pre-correlation
correction is a pure z-scaling of the POI about the surface z:

```
corrected_z = surface_z + (poi_z - surface_z) * factor
```

- Signed arithmetic → works regardless of stack z direction.
- Valid in voxel units when the z-step is uniform (scaling is unit-free); the engine
  already assumes isotropic voxels for the transform fit itself.
- x, y are unchanged (distortion is axial only).

## Proposed design

**Principle: pre-correction is part of the run pipeline, recorded in the input data.**
The correction is applied transiently inside `run_correlation_from_data` (the stored
`poi_coordinates` keep the user-picked z), so re-running never double-applies, headless
use gets the same behaviour, and the auto-saved `correlation_data.json` carries full
provenance (surface point + factor).

### 1. `structures.py`

- Add `PointType.SURFACE_FM` (value string TBD, e.g. `"FM-SURFACE"` — the enum value is
  both the JSON token and the display label used in canvas menus/row names).
- `CorrelationInputData`:
  - `fm_surface_coordinate: Optional[Coordinate] = None`
  - `ri_pre_correction_factor: Optional[float] = None` (None = off)
  - serialize both; `from_dict` must use `.get(...)` for the new keys (old JSON loads).
  - `to_input_dataframe()` / CSV export: add the FM-surface row.
- Pure helper (unit-testable, no Qt), e.g. in `structures.py` or `util.py`:
  `apply_z_correction(poi_coords, surface_z, factor)` → corrected copy.
- `CorrelationResult`: add `refractive_index_correction_mode: Optional[str]`
  (`"post" | "pre"`), `.get(...)` in `from_dict`. Set `"post"` in
  `apply_refractive_index_correction()`.

### 2. `correlation_v2.py`

In `run_correlation_from_data`: if `data.fm_surface_coordinate` and
`data.ri_pre_correction_factor` are set, scale the z of the POI array (all POIs — each
has its own depth) before `correlate()`; log original→corrected z; set
`result.refractive_index_correction_factor = factor` and mode `"pre"`.

### 3. `correlation_tab_widget.py`

- FM display: `allowed_point_types=[FM, POI, SURFACE_FM]`; `_on_fm_add_requested`
  routes SURFACE_FM to a new max-1 list (replace semantics like the FIB surface),
  `z = current_z`.
- `_CoordinatesTab`: new "FM Surface" panel + `fm_surface_list`
  (`CoordinateListWidget(point_type=SURFACE_FM)`), count label in `update_headers`.
- Wire the 5 list signals; add the new list to every selection-clearing cascade
  (`_on_*_selected` handlers, canvas add/move/remove routing on the FM side).
- `data` property / `set_data` / `load_data` / `_refresh_fm_canvas` /
  `set_fm_image` (axis maxima incl. z) include the FM surface.
- Refit: route SURFACE_FM through the FM path (reflection/hole via the FM-fiducial
  method+channel combos — the surface is typically picked in the reflection channel).
- `_RITab`:
  - mode selector (e.g. radio: "FIB surface → correct result (post)" /
    "FM surface → correct input POI & re-run (pre)"), enabled per surface availability;
    default to FM/pre when an FM surface exists.
  - pre mode Apply: emit `pre_correction_requested(factor)`; main widget stores the
    factor (widget state → included in `data`) and calls `_run()` (background worker,
    same as Run button). Button label "Apply && Re-run" in pre mode.
  - post mode: unchanged behaviour.
  - guard double correction: if the result already has a factor applied in one mode,
    block the other mode with a clear message.
  - distance label in pre mode: slices and µm (`|Δz| * pixel_size_z`);
    table shows Z original / Z corrected (FM voxels) instead of Y columns;
    drop the "POI 1 only" caveat in pre mode (all POIs corrected).
  - optional enhancement: auto-fill the ζ-LUT `depth_um` spinbox from
    `|z_poi − z_surf| × pixel_size_z` (µm).
- Colors/markers: add SURFACE_FM entries to `_POINT_COLORS`/`_POINT_MARKERS`
  (image_point_canvas.py) and `_POINT_TYPE_COLORS` (coordinate_list_widget.py) —
  lookups already `.get(...)` with fallbacks, so missing entries degrade, not crash.

### Alternative considered (not recommended)

Deterministic re-projection through the *stored* transform (rebuild `Rigid3D` from
saved q/s/d and `transform()` the corrected POI, no re-fit). Instant and jitter-free,
but duplicates projection math outside `correlate()`, needs verification of the stored
`q` semantics, and complicates provenance. The re-run approach reuses the existing
worker path; run-to-run transform jitter already exists and only affects the POI via
the (unchanged) fiducial fit.

## Testing plan

### Unit (pytest, no Qt) — extend `tests/correlation/`

1. z-correction helper: basic scaling, surface above/below POI (signed), factor=1
   no-op, multiple POIs, z=surface (zero depth).
2. `run_correlation_from_data` with FM surface + factor (synthetic 4-marker dataset
   with a known transform): corrected z used for reprojection; `data.poi_coordinates`
   NOT mutated; re-run gives same corrected input (idempotent); result records
   factor + mode `"pre"`.
3. Serialization: round-trip `CorrelationInputData` with new fields; loading legacy
   dicts *without* the new keys; `CorrelationResult` mode field round-trip;
   `PointType.SURFACE_FM` round-trip; CSV/`to_input_dataframe` row.
4. **Fix pre-existing broken tests**: `fibsem/correlation/tests/test_structures.py`
   calls `apply_refractive_index_correction(factor, surface_y=..., fib_shape=...,
   pixel_size=...)` with graceful no-op semantics — the current implementation takes
   only `factor` and raises. These fail today (verified). Update them to the current
   contract (and consider consolidating into `tests/correlation/`).

### Widget (pytest-qt, pattern exists in `tests/fm/test_autofocus_widget.py`)

- Add FM surface via `_on_fm_add_requested` → appears in list (max-1 replace),
  included in `widget.data`, survives `set_data` round-trip.
- RI tab mode enable/disable vs. which surfaces exist; double-apply guard.
- Pre-mode apply triggers a run with factor set (mock `run_correlation_from_data`).

### Manual / on-instrument checklist

1. Load FIB + FM, pick ≥4 pairs + POI; right-click FM canvas on the surface slice
   (reflection channel) → Add FM surface; run; apply pre-correction; POI overlay on
   FIB moves deeper along the projected axis.
2. Compare pre vs post mode on the same dataset — directions must agree, magnitudes
   similar (science sanity check).
3. Save/Load coordinates JSON (new fields), Load old JSON (no fields → no crash),
   Load Correlation Result round-trip, CSV export, auto-save files.
4. Refit on the FM surface point; move/delete/reorder interactions; selection
   highlighting across all five lists/canvases.
5. Full protocol-editor flow: Continue → POI lands in lamella editor.

## Impact / risks

- **Downstream consumers**: protocol editor reads `result.poi[0].px_m` — no interface
  change; the value simply reflects the corrected input when pre mode is used.
- **File compat**: old JSON loads fine (`.get` defaults). New JSON in *old* app
  versions: the unknown `fm_surface_coordinate` key is ignored by the old `from_dict`
  (silent drop of the FM surface, no crash).
- **Double correction** is the main correctness risk → explicit mode + guard.
- **Stochastic re-run**: pre-mode Apply re-fits the transform (random restarts), so
  the POI also shifts by ordinary run-to-run jitter. Accepted; alternative noted.
- **Anisotropic stacks**: correction itself is valid per-axis, but the transform fit
  already assumes isotropic voxels — unchanged, pre-existing assumption.
- Stale test file noted above fails today regardless of this feature.

## Implementation summary

- `structures.py`: `PointType.SURFACE_FM`; `CorrelationInputData.fm_surface_coordinate`
  + `ri_pre_correction_factor` (serialized, `.get()`-compatible with legacy JSON);
  `apply_z_surface_correction()` helper; `CorrelationResult.refractive_index_correction_mode`
  ("pre"/"post"); FM-surface row in `to_input_dataframe()`.
- `correlation_v2.py`: `run_correlation_from_data` applies the z-correction
  transiently to the POI array (all POIs) before `correlate()` and records
  factor + mode on the result. Stored coordinates keep the user-picked z.
- `refractive_index_widget.py`: `set_tilt_locked()` (0° in pre mode, restores on unlock).
- `correlation_tab_widget.py`: SURFACE_FM on the FM canvas (z = current slice,
  max-1 replace), "Surface (FM)" list panel + full signal wiring, surface
  exclusivity, dual-mode `_RITab` (mode banner, tilt lock, re-run checkbox,
  Z-column table in pre mode, post double-apply guard), `_menu_load_result`
  reordered so the RI tab sees loaded surfaces, `_logger` NameError fixed.
- Canvas/list color+marker entries (yellow "+" / "yellow").
- Canvas legend (proxy `Line2D` handles, upper right, dark-themed): shows the
  point types currently on screen plus labeled overlay groups (FM reprojected,
  POI, POI uncorrected); View → Show Legend toggle (default on); replicated in
  `render_to_axes` exports. Unfilled markers ("+") keep their own colour as the
  legend edge so they don't render white.
- Marker styling unified in `_marker_style()`: filled circles show selection as
  a white rim; unfilled "+" crosshairs keep their type colour always (white
  edge would swallow it, "none" erased it) and show selection via size/stroke.
- FIB surface datum line: dashed orange `axhline` across the canvas at the
  surface y (the only coordinate the post correction uses), tracks drags/edits
  live, emphasized while selected, exported by `render_to_axes`. FM surfaces
  are z-planes — no in-plane line.
- Visibility of the applied correction:
  - status line reads "Done — RI pre-correction ×N applied." after a corrected run;
  - **ghost marker (both modes)**: `CorrelationResult.poi_uncorrected` holds the
    POI position(s) *without* the correction. Pre mode: all POIs, reprojected
    deterministically through the same fitted transform
    (`_reproject_poi_via_transform`; note the parsed `rotation_quaternion` field
    actually stores `Rigid3D.q`, the 3×3 rotation matrix). Post mode: POI 1
    snapshotted inside `apply_refractive_index_correction` before the in-place
    mutation. Drawn as a hollow magenta ring (larger than the solid marker, so
    it stays visible under partial overlap; `add_overlay_points` gained
    `alpha`/`show_labels`/`hollow`). The status line reports the shift in px
    for both modes — a 0.0 px shift is the tell that surface z == POI z
    (e.g. points placed in MIP mode without moving the z slider).

Tests: `tests/correlation/test_pre_correction.py` (math + engine seam via
monkeypatched `run_correlation` + serialization), offscreen widget tests in
`tests/correlation/test_correlation_tab_widget.py`, and the stale
`fibsem/correlation/tests/test_structures.py` rewritten to the current
raise-based contract. End-to-end verified against the real `Rigid3D` fit with
synthetic ground truth (corrected POI lands on the projection of the z-corrected
3D point to <0.01 px; rms ≈ 0).

## Code-review findings — FIXED (2026-07-15, uncommitted follow-up)

Findings 1–9 of the multi-agent review are fixed with regression tests:
1. `from_dict` now restores `stored_fib_image_shape`/`stored_fib_image_pixel_size`
   (the properties fall back to them), so post-correcting a JSON-loaded result
   updates px/px_m; the skip branch warns instead of staying silent.
2. `_clear_pre_correction_factor()` runs on every FM-surface removal path
   (canvas remove, list remove, FIB-surface replacement); `set_data` refuses to
   arm a factor when the file has no FM surface.
3. `_apply_post` guards `result.input_data is None` with a warning.
4. Factor spinbox mirrors only on stored-value *change* (`_last_mirrored_factor`),
   armed input factor outranks the older result factor, and the label shows
   "stored — applied on next run" when they differ.
5. `set_tilt_locked` changes tilt with signals blocked and recomputes zeta
   quietly (`_recompute(update_factor=False)`) — mode switches never overwrite
   a manually entered factor.
6. `apply_refractive_index_correction` raises on double-apply;
   `run_correlation_from_data` raises when both surfaces are set.
7. `render_to_axes` copies `markerfacecolor` + `alpha` (hollow ghost survives
   Save Plot).
8. `_menu_load_result` logic extracted to `_load_result`; redundant trailing
   `_update_run_button()` removed (status no longer clobbered on load).
9. `_RITab._set_warning(text, level)` pairs text+color at every site.

Finding 10 (PointType→list registry refactor) deferred to a follow-up PR.

## Code-review follow-ups (verified, below the report cutoff)

- `_RITab._populate_pre_table` inlines the correction formula instead of calling
  `apply_z_surface_correction`; variants of `surface + (v−surface)·factor` now
  exist in 5 places (util.py:770, structures.py ×2, tab widget ×2) — extract one
  scalar helper used everywhere so preview and engine can't drift.
- `_reproject_poi_via_transform` re-derives the image-px→centred-px→metres
  conversion (3rd copy in correlation_v2.py); its `R.shape != (3,3)` guard is
  unreachable at the only call site and silently degrades (returns no ghost).
- `_RITab.set_result` explodes `input_data` into 5 parallel cached fields —
  holding the object would remove the sync burden; it also rebuilds the pre
  table + copies the POI list on every `data_changed` (minor churn).
- Apply-and-re-run pays a full `copy.deepcopy(self.data)` (entire FM volume)
  plus a stochastic re-fit per factor tweak — pre-existing Run cost, but the
  deterministic reprojection path added for ghosts shows the cheap alternative.

## Follow-up: registry refactor (review finding 10) + PR #111 canvas convergence

**Status: implemented — PR #140 (rebased onto main after #139 merged; CI
green on py3.8–3.13).** 82 tests pass incl. a parametrized per-type behaviour
sweep and a loud-failure (KeyError) routing test. Formula dedup included:
`scale_about_surface()` in structures.py is now the only implementation of
the depth-scaling formula (engine, model, util tuple helper, and both RI-tab
previews call it).

A medium-effort multi-agent review of the PR found zero correctness
regressions (line-scan, removed-behaviour audit, cross-file trace, efficiency
all clean) and six confirmed polish items, fixed in a follow-up commit:
`_POINT_TYPE_SIDES` is now the single source for canvas allow-lists and
adapter binding (a spec-only point type appears in the right-click menu);
the adapter set, axis-maxima updates, and refit routing all derive from the
registry; `_PointTypeSpec.__post_init__` rejects inconsistent side/fit-role
combinations and the build asserts registry↔map completeness; `on_cleared`
now fires only when the spec's LAST point is removed; the `_CanvasAdapter`
docstring states honestly that the seam is outbound-only (inbound signal
translation is part of the future #111 migration). A seventh candidate
(remove the adapter as pass-through indirection) was refuted against the
recorded design intent above.

Agreed plan (2026-07-15): after PR #139 merges, a separate PR replaces the
per-point-type plumbing in `correlation_tab_widget.py` with a
`_PointTypeSpec` registry (point_type → list widget, canvas side, max_one,
exclusive_group, fm_fit_role, on_cleared) + generic handlers; surface
exclusivity and the pre-correction-factor lifecycle move into that single
chokepoint; unknown types fail loudly (KeyError) instead of misfiling into the
POI list; tests updated to the generic entry points + a parametrized per-type
behaviour sweep. Includes the formula dedup (`scale_about_surface()` helper
for the five copies of `surface + (v−surface)·factor`).

**PR #111 consideration:** #111 introduces a shared canvas stack
(`fibsem/ui/widgets/canvas/`: `FibsemImageCanvas`, `CanvasOverlay`,
interactive `PointOverlay` with index-based signals and per-overlay legend,
FM canvas/compositor, CanvasState reducer). It does not touch
`fibsem/correlation/` (no conflicts), but `ImagePointCanvas` duplicates its
concepts (and vice versa: legend/marker/datum-line features here vs
PointOverlay's own legend). Therefore:
- the registry's generic handlers must NOT talk to canvases directly — each
  spec carries a thin canvas adapter (refresh/set_selected/set_points), so the
  later swap from ImagePointCanvas (identity-based Coordinates) to per-type
  PointOverlay (index-based) is localized to the adapter;
- the canvas migration itself is a THIRD PR, after #111 merges and
  stabilizes: one PointOverlay per point type, right-click type menu moves to
  canvas level, aggregated legend / ghost / datum line ported (possibly
  upstreamed into the canvas package), FM z-stack display onto fm_canvas.
#111's minimap-correlation doc already defers to `CorrelationResult` as the
canonical alignment product, so the data-model boundary stays as-is.

## Remaining / follow-ups

- Full validation on real METEOR data (initial live test done 2026-07-15:
  "looks pretty good"), incl. pre vs post comparison on the same dataset.
- Optional: auto-fill the ζ-LUT `depth_um` spinbox from `|Δz| × pixel_size_z`.
- Pre-existing, untouched: `run_correlation` crashes when `image_props is None`
  (no images loaded — GUI always has them); `correlation_point_picker_widget.py`
  remains an orphaned prototype and was not updated.
