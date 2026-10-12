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

---

## Direct attribute access — not yet on client

These are read directly from `microscope.<attr>` in the existing codebase, not via method calls.

| Attribute | Type | Used where | Notes |
|-----------|------|-----------|-------|
| `microscope.sample_handling` | `SampleHandling` | movement UI, autolamella grid workflows | A service (`fibsem.services.sample_handling`): its commands can be served once services are (FIB-1280). Needs `holder`, `current_grid`, the inventory. |
| `microscope.fm` | `FluorescenceMicroscope \| None` | 4 places — image settings widget, minimap, autolamella UI | Always checked with `if microscope.fm is not None` first. Can be stubbed as `None` on client. |
| `microscope.milling_channel` | `BeamType` | Milling setup code | Defaults to `BeamType.ION`. Can be a fixed client attribute. |
| `microscope._last_imaging_settings` | `ImageSettings` | `get_imaging_settings()` fallback path | Internal — low risk. |

---

## Properties — not yet on client

| Property | Returns | Notes |
|----------|---------|-------|
| `is_acquiring` | `bool` | Could poll `GET /is_acquiring` or always return `False` |
| `current_grid` | `str` | Reads `sample_handling.current_grid` — depends on `sample_handling` being served |
| `manufacturer` | `str` | Can be populated from `GET /health` at connect time |

---

## Notes

- `microscope.system` is fetched and cached at `FibsemClient.__init__` — this is covered.
- `microscope.fm = None` is safe to hardcode on the client; fluorescence microscope is never remote-controlled through this server.
- The individual `draw_*` methods are the easiest to close — they can delegate to `draw_patterns([pattern])` locally on the client without a new server endpoint.
- `sample_handling` is the hardest gap: autolamella grid workflows read `microscope.sample_handling.holder` directly. Its commands become reachable when services are served (FIB-1280); the holder itself would need its own endpoint.
