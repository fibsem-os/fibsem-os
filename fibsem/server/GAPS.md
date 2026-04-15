# FibsemClient — Remaining Gaps

What is still missing before `FibsemClient` can be used as a drop-in `FibsemMicroscope` subclass.

## Abstract methods — not yet on server or client

### Manipulator
All manipulator methods are low-priority (most systems don't have one).

| Method | Signature |
|--------|-----------|
| `insert_manipulator` | `(name: str) -> None` |
| `retract_manipulator` | `() -> None` |
| `move_manipulator_relative` | `(position: FibsemManipulatorPosition) -> None` |
| `move_manipulator_absolute` | `(position: FibsemManipulatorPosition) -> None` |
| `move_manipulator_corrected` | `(dx: float, dy: float, beam_type: BeamType) -> None` |
| `move_manipulator_to_position_offset` | `(offset: FibsemManipulatorPosition, name: str) -> None` |
| `_get_saved_manipulator_position` | `(name: str) -> FibsemManipulatorPosition` |

### GIS / Sputter / Cryo
| Method | Signature |
|--------|-----------|
| `cryo_deposition_v2` | `(gis_settings: FibsemGasInjectionSettings) -> None` |
| `setup_sputter` | `(*args, **kwargs)` |
| `draw_sputter_pattern` | `(*args, **kwargs) -> None` |
| `run_sputter` | `(*args, **kwargs)` |
| `finish_sputter` | `()` |

### Connection lifecycle
| Method | Signature |
|--------|-----------|
| `connect_to_microscope` | `(ip_address: str, port: int, reset_beam_shift: bool = True) -> None` |
| `disconnect` | `() -> None` |

### Individual pattern drawing
The server exposes `draw_patterns` (batch) but not the individual `@abstractmethod` variants.
These need either stubs that delegate to `draw_patterns`, or individual endpoints.

| Method | Signature |
|--------|-----------|
| `draw_rectangle` | `(pattern_settings: FibsemRectangleSettings)` |
| `draw_line` | `(pattern_settings: FibsemLineSettings)` |
| `draw_circle` | `(pattern_settings: FibsemCircleSettings)` |
| `draw_bitmap_pattern` | `(pattern_settings: FibsemBitmapSettings) -> None` |
| `draw_polygon` | `(pattern_settings: FibsemPolygonSettings) -> None` |

### Internal methods (need stubs only, nothing external calls them directly)
| Method | Notes |
|--------|-------|
| `_get(key, beam_type)` | Low-level hardware getter — stub with `NotImplementedError` |
| `_set(key, value, beam_type)` | Low-level hardware setter — stub with `NotImplementedError` |
| `check_available_values(key, values, beam_type)` | Validation helper — stub with `NotImplementedError` |

---

## Direct attribute access — not yet on client

These are read directly from `microscope.<attr>` in the existing codebase, not via method calls.

| Attribute | Type | Used where | Notes |
|-----------|------|-----------|-------|
| `microscope._stage` | `Stage` | 8 places — movement UI, autolamella grid workflows | Needs `_stage.limits`, `_stage.holder.grids`, `_stage.current_grid`. Expose via a `GET /stage` endpoint or dedicated wrappers. |
| `microscope.fm` | `FluorescenceMicroscope \| None` | 4 places — image settings widget, minimap, autolamella UI | Always checked with `if microscope.fm is not None` first. Can be stubbed as `None` on client. |
| `microscope.milling_channel` | `BeamType` | Milling setup code | Defaults to `BeamType.ION`. Can be a fixed client attribute. |
| `microscope._last_imaging_settings` | `ImageSettings` | `get_imaging_settings()` fallback path | Internal — low risk. |

---

## Properties — not yet on client

| Property | Returns | Notes |
|----------|---------|-------|
| `is_acquiring` | `bool` | Could poll `GET /is_acquiring` or always return `False` |
| `current_grid` | `str` | Reads `_stage.current_grid` — depends on `_stage` being exposed |
| `manufacturer` | `str` | Can be populated from `GET /health` at connect time |

---

## Notes

- `microscope.system` and `microscope.stage_is_compustage` are now fetched and cached at `FibsemClient.__init__` — these are covered.
- `microscope.fm = None` is safe to hardcode on the client; fluorescence microscope is never remote-controlled through this server.
- The individual `draw_*` methods are the easiest to close — they can delegate to `draw_patterns([pattern])` locally on the client without a new server endpoint.
- `_stage` is the hardest gap: autolamella grid workflows call `microscope._stage.holder.grids` which is a deeply hardware-specific object. This would require either a dedicated endpoint or is out of scope for remote scripting.
