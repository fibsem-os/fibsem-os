# FibsemClient — Remaining Gaps

`FibsemClient` (`fibsem/server/client.py`) is a plain class, not a `FibsemMicroscope`
subclass. It keeps `FibsemMicroscope`'s names for imaging, stage moves, beam and
detector settings, the milling angle and milling, and reaches everything else the App
Server serves by name: `get_parameter`/`set_parameter`, `get_values`,
`call_command`/`call_query` for devices, and `get_service_parameter`,
`set_service_parameter`, `call_service_command`, `call_service_query` and
`wait_for_run` for services. What it still lacks, against the microscope:

## Not on the client

| What | Notes |
|------|-------|
| `client.devices` for the stage, chamber, chamber camera and manipulator | `client.devices` builds only the beams and the FM parts. `RemoteStage`, `RemoteChamber`, `RemoteChamberCamera` and `RemoteManipulator` exist in `fibsem/drivers/remote/devices.py`; `_connect_devices` doesn't build them yet. |
| A `sample_handling` attribute | The service is served (`/microscope/services/sample_handling`: `move_to_grid`, `move_to_slot`, `current`, `holder_slots`, `get_inventory`, ...) and reachable through `call_service_command`/`call_service_query`; there is no wrapper, and no client-side holder object for code that reads `microscope.sample_handling.holder`. |
| `microscope.fm` | The FM's parts are in `client.devices` and its acquisitions are `imaging`'s `fm_*` commands; there is no `FluorescenceMicroscope` over them on the client. |
| Manipulator methods | The allowlist blocks every manipulator write and command for an agent (`fibsem/server/allowlist.py`), so wrappers would only be refused. |
| Individual `draw_*` methods | `draw_patterns` sends them to the milling service's `draw`, which takes any pattern list; single-pattern wrappers could delegate to it locally. |
| `is_acquiring`, `manufacturer`, `current_grid` | `is_acquiring` on the microscope is computed from the beams' `is_live`; `manufacturer` is in `GET /microscope`; `current_grid` is `sample_handling`'s `current` query. None is a client property yet. |
| `connect_to_microscope`, `disconnect` | The client connects in its constructor and ends with `close()`. |
