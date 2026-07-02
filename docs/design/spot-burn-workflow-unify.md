# Spot Burn Workflow Unify — `SpotBurnSettings` end-to-end

Follow-up from `spot-burn-widget-consolidation.md`. That work made `SpotBurnSettings` the currency
*inside* the live widget, but the supervised workflow still passes a parameter **dict** through
`workflow_update_signal` and reads coordinates back one at a time. Thread `SpotBurnSettings` end to
end so it's the single payload from the task config → the UI → and back.

## Current dict path (what we replace)

1. `SpotBurnFiducialTask.update_spot_burn_parameters_ui` (supervised) builds
   `parameters = {milling_current, exposure_time, coordinates}` and calls
   `update_spot_burn_parameters(parent_ui, parameters)`.
2. `update_spot_burn_parameters` (`workflows/ui.py`) emits `workflow_update_signal` with
   `INFO = {"spot_burn_parameters": parameters, "clear_spot_burn": …}` and blocks on
   `WAITING_FOR_UI_UPDATE`.
3. The `AutoLamellaUI` handler: `spot_burn_parameters is not None → spot_burn_widget.update_parameters(dict)`.
4. After `ask_user(spot_burn=True)`, the task reads back
   `self.config.coordinates = spot_burn_widget.get_coordinates()`, then `clear_spot_burn_ui`.

The **unsupervised** path calls `run_spot_burn(config.coordinates, …)` inline.

## Target: `SpotBurnSettings` as the payload

1. **Task (supervised):** `update_spot_burn_parameters(parent_ui, settings=self.config.to_settings())`;
   read back with `self.config.apply_settings(spot_burn_widget.get_settings())` (see the decision below).
2. **`workflows/ui.py`:** `update_spot_burn_parameters(parent_ui, settings: Optional[SpotBurnSettings] =
   None, clear_spots=False)`; INFO key `"spot_burn_settings"`; `clear_spot_burn_ui` passes
   `settings=None, clear_spots=True`. (Keep the `WAITING_FOR_UI_UPDATE` mechanism — only the payload
   changes.)
3. **`AutoLamellaUI` handler:** `spot_burn_settings is not None → spot_burn_widget.set_settings(settings)`.
4. **Live widget:** replace `update_parameters(dict)` with `set_settings(SpotBurnSettings)`, and expose
   `get_settings() -> SpotBurnSettings` (the current `_current_settings()` — coordinates from the editor,
   current/exposure from the form). Drop the public dict surface (`update_parameters` /
   `get_coordinates`; keep the latter internal).
5. **Unsupervised path (bonus unify):** `self.config.to_settings().run(microscope=self.microscope,
   beam_type=BeamType.ION, stop_event=self._stop_event)` — replaces the inline `run_spot_burn` call.

## The one behavior decision: readback

Today only coordinates round-trip (`config.coordinates = get_coordinates()`); user-adjusted
current/exposure are used for the burn but **not** persisted to the task config.

- **A — behavior-neutral:** `config.coordinates = widget.get_settings().coordinates` (coordinates only).
- **B — recommended:** `config.apply_settings(widget.get_settings())` — also persists the current /
  exposure the user actually set. It's the natural use of the round-trip and keeps the config honest;
  the only change is that per-lamella current/exposure tweaks now stick.

> **Decided: B** — read back with `config.apply_settings(widget.get_settings())`; per-lamella
> current/exposure tweaks persist to the task config.

## Touch points (all on the running-workflow path)

- `fibsem/applications/autolamella/workflows/tasks/spot_burn.py` — the task method (+ unsupervised run).
- `fibsem/applications/autolamella/workflows/ui.py` — `update_spot_burn_parameters` / `clear_spot_burn_ui`.
- `fibsem/applications/autolamella/ui/AutoLamellaUI.py` — the `workflow_update_signal` handler.
- `fibsem/ui/FibsemSpotBurnWidget.py` — `set_settings` / `get_settings`.
- `fibsem/ui/widgets/tests/test_spot_burn_live_widget.py` — update to the new API.
- `base.py` imports `update_spot_burn_parameters` / `clear_spot_burn_ui` but does not use them; the
  names are unchanged so it's unaffected (optional: drop the unused imports).

## Slices

- **U0 — widget API.** Add `set_settings` / `get_settings`; drop the public dict adapter. Update the
  regression test. (No workflow change yet — the old callers still pass dicts, so do U0+U1 together or
  keep a temporary shim.)
- **U1 — workflow payload.** Swap `update_spot_burn_parameters` + the handler + the task to
  `SpotBurnSettings`; unify the unsupervised run via `to_settings().run(...)`. Apply the readback
  decision.
- **U2 — verify.** Headless: the widget test (new API) + a task-level unit check that `to_settings()`
  round-trips through a fake widget. Live-test both paths: supervised (place → burn → continue, config
  updated) and unsupervised (auto-run with stored coordinates).

## Risks / watch-items

- **Running-workflow path** — this is the blast-radius reason it was deferred. Keep it a mechanical
  payload swap; the cross-thread `WAITING_FOR_UI_UPDATE` handshake is unchanged.
- **U0 and U1 are coupled** — the widget API and its callers change together. Either land them in one
  commit, or keep `update_parameters(dict)` as a thin shim during U0 and delete it in U1.
- **Readback behavior** — option B persists current/exposure (intended, but a change).
- **`thread_worker` still stays** — the last napari in the widget; separate effort.
