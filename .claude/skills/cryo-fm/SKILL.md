---
name: cryo-fm
description: Using a standalone cryo-FM (a fluorescence microscope and its x, y, z sample stage, with no FIB-SEM) through fibsem-os at any level - its devices, imaging a channel or a z-stack, focus, moving the stage and stopping. Read fibsem-agent first, then this before the first action on a cryo-FM.
---

# Using a standalone cryo-FM

The instrument is only devices: no FIB-SEM, so no beams, milling or
FIB/SEM services. You reach each device through the device tools:
`device_list` (each device, its parameters and commands), `device_get_parameter`,
`device_set_parameter`, `device_call_command` and `device_call_query`.
`server_get_capabilities` lists what is served; a route that needs a FIB-SEM
answers `not_connected` and says what is connected instead.

The devices, by their names in `device_list`:

| Device | What it is | What you use |
|---|---|---|
| `fm` | the FM as a whole | `acquire_frame`, `acquire_channel`, `acquire_z_stack`, `start_live`, `stop_live`, `cancel` |
| `camera` | the FM camera | `exposure_time`, `gain`, `binning` |
| `light_source` | the excitation light | `power` (0 to 1) |
| `filter_set` | the channel's filters | `excitation_wavelength`, `emission_filter` |
| `objective` | the objective and its focus | `position`, `move_relative`, `insert`, `retract`, `save_focus_position` |
| `stage` | the sample stage, x, y and z | `position`, `move_relative` |

## Imaging

- Set the channel first: `filter_set` `excitation_wavelength`, then
  `light_source` `power` and `camera` `exposure_time`. Keep the power as low as
  gives a usable image: light bleaches the sample.
- One image: `device_call_command` device=fm command=acquire_frame. It answers
  the image's summary and a preview you can look at, saved with the experiment.
- A z-stack: `acquire_z_stack` with kwargs `channels` (each
  `{"name", "excitation_wavelength", "power", "exposure_time"}`, the wavelength
  one of `filter_set`'s choices), `positions` (objective positions in metres)
  and `restore_position` (where focus goes back to). It takes a while;
  `cancel`, with a reason, stops it.
- Look at every image before you act on it. Judge it only once its focus and
  brightness are good enough; fix them first.

## Focus

The objective's `position` is focus. Step it with `move_relative`
(kwargs `{"delta": <metres>}`, a few micrometres at a time), image, and keep
the best. `save_focus_position` keeps it for later; the objective's
`limit_position` is the operator's safety limit and you can't change it.

## Moving the stage

`stage_get_position` reads where the stage is. Move with `device_call_command`
device=stage command=move_relative and kwargs
`{"delta": {"x": ..., "y": ..., "z": ..., "r": null, "t": null, "coordinate_system": null}}`
in metres, null for an axis that stays. An absolute move is the operator's: to
reach a saved position, read the position and move by the difference. Move
in steps you can check with an image, and retract the objective first for a
long move if the operator's rules say so.

## Stopping

`device_call_command` device=fm command=cancel (or stop_live) stops imaging,
with a reason for the operator. On a hardware error, stop and tell the
operator the error word for word.
