# FM & Coincidence Settings Persistence

Status: **draft / in progress**
Owner: Patrick
Feedback items addressed: **#11** (load FM settings + restore channels in the coincidence viewer), **#12** (persist camera transform between restarts), **#13b** (reload FM + milling params after closing the viewer).

## Problem

The coincidence-milling viewer (`fibsem/ui/widgets/fluorescence_coincidence_viewer_widget.py`)
persists **nothing** about its own configurable state. Every time it opens:

- the FM channel widget is seeded with a single hard-coded default `ChannelSettings()`,
- the camera settings (gain / binning / **transform**) come live from hardware,
- the milling task config is a hard-coded `DEFAULT_MILLING_TASK_CONFIG`.

Users work with **different samples** that need **different channel setups** (GFP vs mCherry
vs DAPI — different excitation/emission, exposure, power), and they have to re-enter these
every session. Separately, the **camera transform** is an instrument-mount property that
should survive app restarts but currently does not.

## What already exists (reuse, don't reinvent)

- `FluorescenceConfiguration` (`fibsem/fm/structures.py:1539`) already aggregates the full FM
  config — `channel_settings[]`, `camera_settings` (gain/binning/**transform**),
  `focus_position`, `limit_position`, `z_parameters`, `overview_parameters`,
  `autofocus_settings`, `default_orientation` — with `to_dict` / `from_dict` /
  `export(yaml)` / `load(yaml)`.
- `FibsemMillingTaskConfig` (`fibsem/milling/tasks.py:81`) round-trips the coincidence
  `CoincidenceMillingStrategy` config + pattern geometry.
- Config conventions: `utils.save_yaml` / `load_yaml`; file dialogs via
  `fibsem.ui.utils.open_save_file_dialog` / `open_existing_file_dialog`; app-level
  persistence via `UserPreferences` / `user-preferences.yaml` (`fibsem/config.py`).
- A **named-configuration registry** precedent already exists for *microscope* configs:
  `USER_CONFIGURATIONS_PATH` + `add_configuration` / `remove_configuration` /
  `set_default_configuration` (`fibsem/config.py:167-189`). We mirror this for FM.
- `MicroscopeSettings.from_dict` already loads an FM config from a `fm.config` path pointer
  (`fibsem/structures.py:1724`) and it is applied on startup at
  `fibsem/applications/autolamella/ui/AutoLamellaUI.py:677`.

## Design decisions (agreed)

1. **Bundle vs split:** FM presets are **per-sample**; milling params are a **separate
   single default** (not multiplied per preset).
2. **Camera transform home:** kept **inside each FM preset** (accepting minor duplication;
   no instrument/sample split of `FluorescenceConfiguration`).
3. **Auto-load:** the viewer/FM widget reopen with the **last-active** preset / working state.
4. **Auto-save trigger:** working state saved **on close** (+ explicit Save). Not on every change.

## Data model — three artifacts

1. **Named FM preset library** — `fibsem/config/fm-configurations/` (registry mirroring the
   microscope-config registry). Each entry is a full `FluorescenceConfiguration`, named per
   sample. Explicit **Save-as / Save / Delete / Load**.
2. **Current working state** — `fibsem/config/fm-configuration.yaml`. The live FM config;
   **auto-saved on close, auto-loaded + applied on startup**. Loading a preset copies it into
   here; "Save as preset" copies here → the library. This is what makes the camera transform
   survive restarts **without silently overwriting a named preset**.
3. **Coincidence milling default** — a single persisted `FibsemMillingTaskConfig` (strategy +
   pattern params), separate from FM presets. Loaded into the milling widget on viewer open.

On viewer open: apply working state (FM) + milling default. Selecting a preset copies it into
the working state and applies it.

### Caveat

`CONFIG_PATH` (`fibsem/config/`) is inside the package dir, so in a non-editable pip install
these files live in site-packages (writable, but lost on reinstall). Pre-existing issue shared
by `user-preferences.yaml`; follow the precedent now, treat "move mutable config to a user data
dir" as a separate future cleanup.

## Pre-requisite fixes (Phase 1)

Both are correctness bugs that block clean persistence and should land first.

1. **Make every FM save write the full config.** `FluorescenceControlWidget.save_fm_configuration`
   was omitting `limit_position`; `FMAcquisitionWidget` (which has no camera widget) omitted
   `limit_position` too. Both now include it inline. (A shared cross-widget helper was tried and
   dropped — the two widgets don't share a base class or sub-widget set, so a generic helper
   degenerated into optional-kwarg soup; per-widget inline construction is clearer. The camera
   transform for #12 flows through the camera-owning widgets — `FluorescenceControlWidget` and
   the coincidence viewer's `CameraWidget` — not `FMAcquisitionWidget`.)

2. **Fix the `CoincidenceMillingStrategyConfig` round-trip for `bbox`.** `bbox` is a valid,
   useful param — the FM ROI rectangle that defines where intensity drop is monitored
   (`coincidence.py:403/440/705`). The base `MillingStrategyConfig.to_dict = asdict` flattens
   it to a plain dict and `from_dict = cls(**d)` passes that dict straight back, so
   `config.bbox` comes back as a `dict`, not a `FibsemRectangle` (then `.to_pixel_coordinates`
   fails). Fix: override `to_dict` / `from_dict` on `CoincidenceMillingStrategyConfig` to use
   `FibsemRectangle.to_dict()` / `FibsemRectangle.from_dict()` (with a `None` guard). Keep the
   field; do not drop it. Base behavior untouched for scalar-only strategy configs.

## Phased build

1. **Fixes** — the two above (small, self-contained; first commit).
2. **FM working state + registry** — `FM_CONFIGURATION_PATH` + `fm-configurations/` registry
   helpers in `config.py` (mirroring add/remove/set_default_configuration); fallback-load in
   `MicroscopeSettings.from_dict`; auto-save-on-close; apply-on-startup (extend the existing
   `AutoLamellaUI:677` hook).
3. **Preset UI** — dropdown + Save-as / Save / Delete / Load in the FM widget and the
   coincidence viewer's FM tab; remember last-active in `UserPreferences`.
4. **Milling default** — persist / load the coincidence `FibsemMillingTaskConfig`; wire into
   the viewer's milling tab on open.

## Mapping to feedback

| Item | Satisfied by |
|---|---|
| #11 | Preset dropdown applies channels/camera/objective; viewer seeds FM from working state (not a dummy channel). |
| #12 | Transform lives in the working state (+ presets), auto-persisted on close, re-applied on startup. |
| #13b | Reopening the viewer restores the last FM working state + milling default. |
