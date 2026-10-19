"""Older correlation files, read as the current format.

Every earlier spelling of a correlation file lives here, and nowhere else: the
container ``correlation.json`` at version 1, and the split
``correlation_data.json`` (the points) and ``correlation_result.json`` (a result
with its snapshot) that it replaced (FIB-264). :func:`upgrade` turns the dict any
of them holds into the current container, which is all the classes in
:mod:`fibsem.correlation.structures` read or write.

The file's own ``version`` decides what changes. A split file has none; its keys
say which of the two it is. Version 1 and the split files spell their fields the
same way, so one set of steps covers both:

* the result's snapshot was written as ``computed_from`` (and, before that, as
  ``input_data`` in a standalone result); the result's ``rotation_matrix`` was
  once ``rotation_quaternion``; ``updated_at`` may be a POSIX number;
* the ``stored_*`` values were written without their prefix, beside a derived
  ``fm_image_shape`` that nothing read;
* a point's ``point_type`` was written by value (``FM-SURFACE``), not by name,
  and its ``status`` and ``provenance`` could be null or one of the strings the
  2026-09-09 .. 09-14 builds wrote.

A split file has no version, and ``CorrelationInputData.save`` and
``CorrelationResult.save`` still write one in the current spelling, so every step
leaves a dict already in the current spelling as it is.

Nothing here touches a file. A file read this way is written in the current
format when the session next saves it, never by opening it.
"""

from __future__ import annotations

import copy

from fibsem.util.timestamps import to_iso

# The container version this build writes and reads.
CORRELATION_FILE_VERSION = 2

# Point statuses the 2026-09-09 .. 09-14 builds wrote, and what each one is now.
OLD_STATUS = {
    "adjusted": "placed",
    "confirmed": "accepted",
    "suggested": "predicted",
    "fit_failed": "placed",
}

# Point types were written by value; this is the one value that is not its name.
_OLD_POINT_TYPE = {"FM-SURFACE": "SURFACE_FM"}

# Written without the prefix of the field that holds them.
_STORED = (
    "fib_image_shape",
    "fib_image_pixel_size",
    "fm_pixel_size_z",
    "fib_image_filename",
    "fm_image_filename",
)

_COORDINATE_LISTS = ("fib_coordinates", "fm_coordinates", "poi_coordinates")
_SINGLE_COORDINATES = ("surface_coordinate", "fm_surface_coordinate")


def upgrade(data: dict) -> dict:
    """Any correlation file's dict as the current container.

    ``data`` is left as it is. A legacy result becomes a container whose points
    are a copy of the result's snapshot, as opening one always did; a legacy
    coordinates file becomes a container with no result. Raises ValueError for
    a dict that is no correlation file, or one from a newer fibsem.
    """
    data = copy.deepcopy(data)
    if "version" not in data:
        return _split_file(data)
    version = data["version"]
    if version > CORRELATION_FILE_VERSION:
        raise ValueError(
            f"a correlation file from a newer fibsem (version {version}; this one"
            f" reads up to {CORRELATION_FILE_VERSION})"
        )
    result = data.get("result")
    return {
        "version": CORRELATION_FILE_VERSION,
        "input_data": _input_data(data["input_data"], version),
        "result": _result(result, version) if result else None,
    }


def upgrade_input_data(data: dict) -> dict:
    """A coordinates file's dict (``correlation_data.json``) in the current spelling."""
    return _input_data(copy.deepcopy(data), 0)


def upgrade_result(data: dict) -> dict:
    """A result file's dict (``correlation_result.json``) in the current spelling."""
    return _result(copy.deepcopy(data), 0)


def _split_file(data: dict) -> dict:
    if "poi" in data or "computed_from" in data or "input_data" in data:
        result = _result(data, 0)
        # A copy, not the same dict: the snapshot is what the transform was
        # fitted to, the container's points are the ones being edited (FIB-295).
        current = copy.deepcopy(result["input_data"]) or {"fib_coordinates": []}
        return {
            "version": CORRELATION_FILE_VERSION,
            "input_data": current,
            "result": result,
        }
    if "fib_coordinates" in data:
        return {
            "version": CORRELATION_FILE_VERSION,
            "input_data": _input_data(data, 0),
            "result": None,
        }
    raise ValueError(
        "not a correlation file: no 'version', 'poi', or 'fib_coordinates' key."
    )


def _input_data(data: dict, version: int) -> dict:
    if version < 2:
        data.pop("fm_image_shape", None)
        for name in _STORED:
            if name in data:
                data[f"stored_{name}"] = data.pop(name)
        for key in _COORDINATE_LISTS:
            if key in data:
                data[key] = [_coordinate(c) for c in data[key]]
        for key in _SINGLE_COORDINATES:
            if data.get(key):
                data[key] = _coordinate(data[key])
    return data


def _result(data: dict, version: int) -> dict:
    if version < 2:
        computed_from = data.pop("computed_from", None)
        snapshot = computed_from or data.pop("input_data", None)
        data["input_data"] = _input_data(snapshot, version) if snapshot else None
        if "rotation_quaternion" in data:
            data.setdefault("rotation_matrix", data.pop("rotation_quaternion"))
        if data.get("updated_at") is not None:
            data["updated_at"] = to_iso(data["updated_at"])
    return data


def _coordinate(data: dict) -> dict:
    point_type = data.get("point_type")
    if point_type in _OLD_POINT_TYPE:
        data["point_type"] = _OLD_POINT_TYPE[point_type]
    status = data.get("status") or ""
    data["status"] = OLD_STATUS.get(status, status)
    data["provenance"] = data.get("provenance") or ""
    return data
