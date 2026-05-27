from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from fibsem.structures import (
        BeamSettings,
        BeamType,
        FibsemDetectorSettings,
        FibsemImage,
        FibsemStagePosition,
        MicroscopeState,
    )

try:
    from mcap.writer import Writer as _McapWriter
    MCAP_AVAILABLE = True
except ImportError:
    MCAP_AVAILABLE = False

# ---------------------------------------------------------------------------
# JSON schemas
# ---------------------------------------------------------------------------

_STAGE_POSITION_SCHEMA = {
    "type": "object",
    "title": "fibsem.StagePosition",
    "properties": {
        "x": {"type": ["number", "null"]},
        "y": {"type": ["number", "null"]},
        "z": {"type": ["number", "null"]},
        "r": {"type": ["number", "null"]},
        "t": {"type": ["number", "null"]},
        "coordinate_system": {"type": ["string", "null"]},
    },
}

_RAW_IMAGE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "foxglove.RawImage",
    "type": "object",
    "properties": {
        "timestamp": {
            "type": "object",
            "properties": {
                "sec":  {"type": "integer", "minimum": 0},
                "nsec": {"type": "integer", "minimum": 0, "maximum": 999999999},
            },
        },
        "frame_id": {"type": "string"},
        "width":    {"type": "integer"},
        "height":   {"type": "integer"},
        "encoding": {"type": "string"},
        "step":     {"type": "integer"},
        "data":     {"type": "string", "contentEncoding": "base64"},
    },
}

_BEAM_SETTINGS_SCHEMA = {
    "type": "object",
    "title": "fibsem.BeamSettings",
    "properties": {
        "working_distance": {"type": ["number", "null"]},
        "current": {"type": ["number", "null"]},
        "voltage": {"type": ["number", "null"]},
        "hfw": {"type": ["number", "null"]},
        "shift_x": {"type": ["number", "null"]},
        "shift_y": {"type": ["number", "null"]},
        "stigmation_x": {"type": ["number", "null"]},
        "stigmation_y": {"type": ["number", "null"]},
        "scan_rotation": {"type": ["number", "null"]},
    },
}

_DETECTOR_SETTINGS_SCHEMA = {
    "type": "object",
    "title": "fibsem.DetectorSettings",
    "properties": {
        "type": {"type": ["string", "null"]},
        "mode": {"type": ["string", "null"]},
        "brightness": {"type": ["number", "null"]},
        "contrast": {"type": ["number", "null"]},
    },
}

_SCALAR_VALUE_SCHEMA = {
    "type": "object",
    "title": "fibsem.ScalarValue",
    "properties": {
        "value": {"type": ["number", "null"]},
        "unit": {"type": "string"},
    },
}

_MICROSCOPE_STATE_SCHEMA = {
    "type": "object",
    "title": "fibsem.MicroscopeState",
    "properties": {
        "timestamp": {"type": "number"},
        "stage": {"type": ["object", "null"]},
        "electron_beam": {"type": ["object", "null"]},
        "ion_beam": {"type": ["object", "null"]},
        "electron_detector": {"type": ["object", "null"]},
        "ion_detector": {"type": ["object", "null"]},
    },
}

# topic -> (schema_name, schema_dict)
_TOPICS: dict[str, tuple[str, dict]] = {
    "microscope/stage/position":            ("fibsem.StagePosition",    _STAGE_POSITION_SCHEMA),
    "microscope/sem/image":                 ("foxglove.RawImage",       _RAW_IMAGE_SCHEMA),
    "microscope/fib/image":                 ("foxglove.RawImage",       _RAW_IMAGE_SCHEMA),
    "microscope/sem/beam":                  ("fibsem.BeamSettings",     _BEAM_SETTINGS_SCHEMA),
    "microscope/fib/beam":                  ("fibsem.BeamSettings",     _BEAM_SETTINGS_SCHEMA),
    "microscope/sem/beam/working_distance": ("fibsem.ScalarValue",      _SCALAR_VALUE_SCHEMA),
    "microscope/fib/beam/working_distance": ("fibsem.ScalarValue",      _SCALAR_VALUE_SCHEMA),
    "microscope/sem/beam/current":          ("fibsem.ScalarValue",      _SCALAR_VALUE_SCHEMA),
    "microscope/fib/beam/current":          ("fibsem.ScalarValue",      _SCALAR_VALUE_SCHEMA),
    "microscope/sem/beam/voltage":          ("fibsem.ScalarValue",      _SCALAR_VALUE_SCHEMA),
    "microscope/fib/beam/voltage":          ("fibsem.ScalarValue",      _SCALAR_VALUE_SCHEMA),
    "microscope/sem/beam/hfw":              ("fibsem.ScalarValue",      _SCALAR_VALUE_SCHEMA),
    "microscope/fib/beam/hfw":             ("fibsem.ScalarValue",      _SCALAR_VALUE_SCHEMA),
    "microscope/sem/detector":              ("fibsem.DetectorSettings", _DETECTOR_SETTINGS_SCHEMA),
    "microscope/fib/detector":              ("fibsem.DetectorSettings", _DETECTOR_SETTINGS_SCHEMA),
    "microscope/state":                     ("fibsem.MicroscopeState",  _MICROSCOPE_STATE_SCHEMA),
}


class FibsemMCAPLogger:
    """Record FibsemMicroscope telemetry to an MCAP file for Foxglove Studio.

    Usage::

        with FibsemMCAPLogger("session.mcap") as logger:
            microscope.attach_logger(logger)
            ...
            microscope.detach_logger()
    """

    def __init__(self, path: Path | str) -> None:
        if not MCAP_AVAILABLE:
            raise ImportError(
                "mcap is required for MCAP logging. "
                "Install it with: pip install 'fibsem[logging]'"
            )
        self._path = Path(path)
        self._file = open(self._path, "wb")
        self._writer = _McapWriter(self._file)
        self._writer.start(library="fibsemOS", profile="")
        self._channels: dict[str, int] = {}
        self._register_topics()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _register_topics(self) -> None:
        schema_ids: dict[str, int] = {}
        for topic, (schema_name, schema_dict) in _TOPICS.items():
            if schema_name not in schema_ids:
                schema_ids[schema_name] = self._writer.register_schema(
                    name=schema_name,
                    encoding="jsonschema",
                    data=json.dumps(schema_dict).encode(),
                )
            self._channels[topic] = self._writer.register_channel(
                schema_id=schema_ids[schema_name],
                topic=topic,
                message_encoding="json",
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _now_ns(self) -> int:
        return time.time_ns()

    def _write(self, topic: str, msg: dict, timestamp_ns: int) -> None:
        self._writer.add_message(
            channel_id=self._channels[topic],
            log_time=timestamp_ns,
            data=json.dumps(msg).encode(),
            publish_time=timestamp_ns,
        )

    def _beam_prefix(self, beam_type: "BeamType") -> str:
        from fibsem.structures import BeamType
        return "microscope/sem" if beam_type is BeamType.ELECTRON else "microscope/fib"

    # ------------------------------------------------------------------
    # Public logging methods
    # ------------------------------------------------------------------

    def log_stage_position(
        self,
        position: "FibsemStagePosition",
        timestamp_ns: Optional[int] = None,
    ) -> None:
        ts = timestamp_ns or self._now_ns()
        msg = {
            "x": position.x,
            "y": position.y,
            "z": position.z,
            "r": position.r,
            "t": position.t,
            "coordinate_system": position.coordinate_system,
        }
        self._write("microscope/stage/position", msg, ts)

    def log_image(
        self,
        image: "FibsemImage",
        timestamp_ns: Optional[int] = None,
    ) -> None:
        import numpy as np

        ts = timestamp_ns or self._now_ns()

        beam_type = None
        if image.metadata is not None:
            beam_type = image.metadata.beam_type

        prefix = self._beam_prefix(beam_type) if beam_type is not None else "microscope/sem"
        topic = f"{prefix}/image"

        bits = image.data.dtype.itemsize * 8
        if image.data.ndim == 3:
            encoding = "rgb8" if image.data.shape[2] == 3 else "rgba8"
        else:
            encoding = f"mono{bits}"

        arr = np.ascontiguousarray(image.data)
        height, width = arr.shape[:2]
        step = arr.strides[0]

        msg = {
            "timestamp": {"sec": ts // 1_000_000_000, "nsec": ts % 1_000_000_000},
            "frame_id": "",
            "width": width,
            "height": height,
            "encoding": encoding,
            "step": step,
            "data": base64.b64encode(arr.tobytes()).decode("ascii"),
        }
        self._write(topic, msg, ts)

    def log_beam_settings(
        self,
        settings: "BeamSettings",
        timestamp_ns: Optional[int] = None,
    ) -> None:
        ts = timestamp_ns or self._now_ns()
        prefix = self._beam_prefix(settings.beam_type)

        shift_x = settings.shift.x if settings.shift is not None else None
        shift_y = settings.shift.y if settings.shift is not None else None
        stig_x = settings.stigmation.x if settings.stigmation is not None else None
        stig_y = settings.stigmation.y if settings.stigmation is not None else None

        aggregate = {
            "working_distance": settings.working_distance,
            "current": settings.beam_current,
            "voltage": settings.voltage,
            "hfw": settings.hfw,
            "shift_x": shift_x,
            "shift_y": shift_y,
            "stigmation_x": stig_x,
            "stigmation_y": stig_y,
            "scan_rotation": settings.scan_rotation,
        }
        self._write(f"{prefix}/beam", aggregate, ts)

        for key, value, unit in [
            ("working_distance", settings.working_distance, "m"),
            ("current", settings.beam_current, "A"),
            ("voltage", settings.voltage, "V"),
            ("hfw", settings.hfw, "m"),
        ]:
            self._write(f"{prefix}/beam/{key}", {"value": value, "unit": unit}, ts)

    def log_detector_settings(
        self,
        settings: "FibsemDetectorSettings",
        beam_type: "BeamType",
        timestamp_ns: Optional[int] = None,
    ) -> None:
        ts = timestamp_ns or self._now_ns()
        prefix = self._beam_prefix(beam_type)
        msg = {
            "type": settings.type,
            "mode": settings.mode,
            "brightness": settings.brightness,
            "contrast": settings.contrast,
        }
        self._write(f"{prefix}/detector", msg, ts)

    def log_microscope_state(
        self,
        state: "MicroscopeState",
        timestamp_ns: Optional[int] = None,
    ) -> None:
        """Write the full aggregate microscope/state message.

        Sub-topics (stage/position, sem/beam, etc.) are written by the individual
        log_* methods when microscope getters fire them — this method only writes
        the top-level aggregate so it is not called recursively.
        """
        ts = timestamp_ns or self._now_ns()

        def _stage_dict(pos):
            if pos is None:
                return None
            return {
                "x": pos.x, "y": pos.y, "z": pos.z,
                "r": pos.r, "t": pos.t,
                "coordinate_system": pos.coordinate_system,
            }

        def _beam_dict(b):
            if b is None:
                return None
            shift_x = b.shift.x if b.shift is not None else None
            shift_y = b.shift.y if b.shift is not None else None
            stig_x = b.stigmation.x if b.stigmation is not None else None
            stig_y = b.stigmation.y if b.stigmation is not None else None
            return {
                "working_distance": b.working_distance,
                "current": b.beam_current,
                "voltage": b.voltage,
                "hfw": b.hfw,
                "shift_x": shift_x,
                "shift_y": shift_y,
                "stigmation_x": stig_x,
                "stigmation_y": stig_y,
                "scan_rotation": b.scan_rotation,
            }

        def _detector_dict(d):
            if d is None:
                return None
            return {
                "type": d.type,
                "mode": d.mode,
                "brightness": d.brightness,
                "contrast": d.contrast,
            }

        msg = {
            "timestamp": state.timestamp,
            "stage": _stage_dict(state.stage_position),
            "electron_beam": _beam_dict(state.electron_beam),
            "ion_beam": _beam_dict(state.ion_beam),
            "electron_detector": _detector_dict(state.electron_detector),
            "ion_detector": _detector_dict(state.ion_detector),
        }
        self._write("microscope/state", msg, ts)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        try:
            self._writer.finish()
        finally:
            self._file.close()

    def __enter__(self) -> "FibsemMCAPLogger":
        return self

    def __exit__(self, *args) -> None:
        self.close()
