"""Old correlation files read through the one upgrader, and the format written now.

``fixtures/correlation_file_pins.json`` holds every older shape of a correlation
file -- the version 1 container, legacy result files before and after FIB-264, and
legacy coordinates files -- with what fibsem loaded from each before the format was
consolidated (commit 08ea60c11). Each must load to the same structures today, by
the same loaders: ``load_correlation_file`` for any file, and the type-specific
``CorrelationInputData.load`` / ``CorrelationResult.load`` the quickstart uses for
the two legacy names. The fixture was recorded by building each file with that
commit's classes and loading it with that commit's code; it is not regenerated.
"""

import dataclasses
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

from fibsem.codec.upgrades.correlation import (
    CORRELATION_FILE_VERSION,
    OLD_STATUS,
    upgrade,
    upgrade_input_data,
    upgrade_result,
)
from fibsem.correlation.history import LamellaCorrelation, load_run, save_run
from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    CorrelationPointOfInterest,
    CorrelationResult,
    CorrelationState,
    PointStatus,
    PointType,
    PointXYZ,
    load_correlation_file,
)
from fibsem.structures.settings import Point
from fibsem.util.timestamps import to_datetime

PINS_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "correlation_file_pins.json"
)
with open(PINS_PATH) as _f:
    CASES = json.load(_f)["cases"]
LOADABLE = [c for c in CASES if "loaded" in c["load_correlation_file"]]

LOADERS = {
    "load_correlation_file": load_correlation_file,
    "CorrelationInputData.load": CorrelationInputData.load,
    "CorrelationResult.load": CorrelationResult.load,
}
# The type-specific loaders were pinned on the two legacy files only.
PINNED = [(c, name) for c in CASES for name in LOADERS if name in c]

# What the current format never writes.
LEGACY_KEYS = {
    "computed_from",
    "rotation_quaternion",
    "fm_image_shape",
    "fib_image_shape",
    "fib_image_pixel_size",
    "fm_pixel_size_z",
    "fib_image_filename",
    "fm_image_filename",
}


def snapshot(obj):
    """``obj`` as plain JSON, every type that matters to equality tagged."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = {"__class__": type(obj).__name__}
        for f in dataclasses.fields(obj):
            out[f.name] = snapshot(getattr(obj, f.name))
        return out
    if isinstance(obj, Enum):
        return {"__enum__": f"{type(obj).__name__}.{obj.name}"}
    if isinstance(obj, datetime):
        # A file without updated_at loads with the time it was read.
        if obj.tzinfo and abs(datetime.now(timezone.utc) - obj) < timedelta(hours=1):
            return {"__datetime__": "now"}
        return {"__datetime__": obj.isoformat()}
    if isinstance(obj, tuple):
        return {"__tuple__": [snapshot(v) for v in obj]}
    if isinstance(obj, list):
        return [snapshot(v) for v in obj]
    if isinstance(obj, dict):
        return {k: snapshot(v) for k, v in obj.items()}
    return obj


def _without_version(snap):
    """A state's snapshot without its format version, which is now always current."""
    if snap.get("__class__") == "CorrelationState":
        snap = dict(snap)
        snap.pop("version")
    return snap


def _parts(obj) -> set:
    """The ids of every mutable object inside ``obj``, ``obj`` included."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        children = [getattr(obj, f.name) for f in dataclasses.fields(obj)]
    elif isinstance(obj, dict):
        children = list(obj.values())
    elif isinstance(obj, list):
        children = obj
    else:
        return set()
    found = {id(obj)}
    for child in children:
        found |= _parts(child)
    return found


def _keys(value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from _keys(v)
    elif isinstance(value, list):
        for v in value:
            yield from _keys(v)


def _point_types(value):
    if isinstance(value, dict):
        if "point_type" in value:
            yield value["point_type"]
        for v in value.values():
            yield from _point_types(v)
    elif isinstance(value, list):
        for v in value:
            yield from _point_types(v)


def _reads_local_time(case) -> bool:
    """Whether a pin holds a POSIX ``updated_at``, which reads in local time."""
    return bool(re.search(r'"updated_at": -?[0-9]', json.dumps(case["file"])))


@pytest.fixture(autouse=True)
def _zone_of_the_pins(request, monkeypatch):
    """The zone the pins were recorded in: a POSIX updated_at reads in local time."""
    if not hasattr(time, "tzset"):
        params = getattr(getattr(request.node, "callspec", None), "params", {})
        case = params.get("case")
        if case is not None and _reads_local_time(case):
            pytest.skip("time.tzset is POSIX-only")
        yield
        return
    monkeypatch.setenv("TZ", "XST+6")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def _write(tmp_path, case):
    path = tmp_path / case["file_name"]
    path.write_text(json.dumps(case["file"], indent=4))
    return path


@pytest.mark.parametrize(
    "case, loader", PINNED, ids=[f"{c['name']}-{name}" for c, name in PINNED]
)
def test_every_old_file_loads_as_it_did(tmp_path, case, loader):
    expected = case[loader]
    path = _write(tmp_path, case)
    if "error" in expected:
        with pytest.raises(Exception) as raised:
            LOADERS[loader](str(path))
        assert type(raised.value).__name__ == expected["error"]
        assert (
            str(raised.value).replace(repr(str(path)), "'<path>'")
            == expected["message"]
        )
        return
    loaded = LOADERS[loader](str(path))
    if isinstance(loaded, CorrelationState):
        assert loaded.version == CORRELATION_FILE_VERSION
        if loaded.result is not None and loaded.result.input_data is not None:
            # the points are a copy of the snapshot, never shared with it (FIB-295)
            shared = _parts(loaded.input_data) & _parts(loaded.result.input_data)
            assert not shared
    assert _without_version(snapshot(loaded)) == _without_version(expected["loaded"])


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_loading_never_rewrites_the_file(tmp_path, case):
    path = _write(tmp_path, case)
    before = path.read_bytes()
    mtime = path.stat().st_mtime_ns
    for loader in LOADERS.values():
        try:
            loader(str(path))
        except Exception:
            pass
    run = tmp_path / "run"
    run.mkdir()
    (run / case["file_name"]).write_bytes(before)
    LamellaCorrelation.discover(str(tmp_path))
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == mtime
    assert (run / case["file_name"]).read_bytes() == before


@pytest.mark.parametrize("case", LOADABLE, ids=[c["name"] for c in LOADABLE])
def test_an_old_run_is_saved_back_in_the_current_format(tmp_path, case):
    run = tmp_path / "run"
    run.mkdir()
    (run / case["file_name"]).write_text(json.dumps(case["file"]))
    state = load_run(str(run))

    written = save_run(str(run), state)
    assert os.path.basename(written) == "correlation.json"
    raw = json.loads(Path(written).read_text())
    assert raw["version"] == CORRELATION_FILE_VERSION
    assert not LEGACY_KEYS & set(_keys(raw))
    assert set(_point_types(raw)) <= set(PointType.__members__)
    # and it reads back as what was loaded
    assert snapshot(load_run(str(run))) == snapshot(state)


@pytest.mark.parametrize("case", LOADABLE, ids=[c["name"] for c in LOADABLE])
def test_upgrade_leaves_its_input_and_a_current_file_alone(case):
    original = json.loads(json.dumps(case["file"]))
    once = upgrade(original)
    assert original == case["file"]
    assert upgrade(once) == once


def _state() -> CorrelationState:
    def coord(x, pt, **kw):
        return Coordinate(point=PointXYZ(x, x + 1.0, x / 10), point_type=pt, **kw)

    current = CorrelationInputData(
        fib_coordinates=[
            coord(float(i), PointType.FIB, status=PointStatus.PLACED, provenance="user")
            for i in range(4)
        ],
        fm_coordinates=[
            coord(float(i) + 0.5, PointType.FM, status=PointStatus.FITTED, fitted=True)
            for i in range(4)
        ],
        poi_coordinates=[coord(9.0, PointType.POI)],
        fm_surface_coordinate=coord(3.0, PointType.SURFACE_FM),
        ri_pre_correction_factor=0.8,
        ri_pre_correction_params={"depth_um": 4.0, "source": "computed"},
        stored_fib_image_shape=(1024, 1536),
        stored_fib_image_pixel_size=6.5e-8,
        stored_fm_pixel_size_z=5e-7,
        stored_fib_image_filename="ref_ib.tif",
        stored_fm_image_filename="stack.ome.tiff",
    )
    then = CorrelationInputData.from_dict(current.to_dict())
    then.fib_coordinates[0].point.x = 42.0
    result = CorrelationResult(
        poi=[
            CorrelationPointOfInterest(
                Point(1.0, 2.0), Point(3.0, 4.0), Point(5e-6, 6e-6)
            )
        ],
        scale=0.6,
        rotation_matrix=[[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        rms_error=0.7,
        delta_2d=[Point(0.1, 0.2)],
        reprojected_3d=[PointXYZ(1.0, 2.0, 3.0)],
        input_data=then,
        updated_at=to_datetime("2026-10-01T09:30:00.250000+02:00"),
        seed={"source": "geometry"},
        placement_offset=[0.5, -0.25],
    )
    return CorrelationState(input_data=current, result=result)


def test_a_new_file_round_trips(tmp_path):
    state = _state()
    path = save_run(str(tmp_path), state)
    loaded = load_correlation_file(path)
    assert loaded == state
    assert snapshot(loaded) == snapshot(state)
    # the points and the snapshot are separate objects after a load, too
    assert loaded.input_data is not loaded.result.input_data


def test_a_new_file_has_no_legacy_spelling(tmp_path):
    state = _state()
    state.input_data.fm_coordinates[0].suggested = True
    raw = json.loads(Path(save_run(str(tmp_path), state)).read_text())

    assert raw["version"] == CORRELATION_FILE_VERSION == 2
    assert not LEGACY_KEYS & set(_keys(raw))
    assert "suggested" not in set(_keys(raw))
    assert raw["result"]["input_data"]["fib_coordinates"][0]["point"]["x"] == 42.0
    assert raw["result"]["updated_at"] == "2026-10-01T09:30:00.250000+02:00"
    for side in (raw["input_data"], raw["result"]["input_data"]):
        assert side["stored_fib_image_shape"] == [1024, 1536]
        assert side["stored_fib_image_pixel_size"] == 6.5e-8
        assert side["stored_fm_pixel_size_z"] == 5e-7
        assert side["stored_fib_image_filename"] == "ref_ib.tif"
        assert side["stored_fm_image_filename"] == "stack.ome.tiff"
        # enums by name: SURFACE_FM, never its value FM-SURFACE
        assert side["fm_surface_coordinate"]["point_type"] == "SURFACE_FM"
    assert set(_point_types(raw)) == {"FIB", "FM", "POI", "SURFACE_FM"}


def test_the_stored_values_follow_the_live_image():
    data = CorrelationInputData(
        stored_fib_image_shape=(10, 20), stored_fib_image_pixel_size=1e-9
    )
    data.fib_image = type(
        "Image",
        (),
        {
            "data": np.zeros((30, 40)),
            "metadata": type("M", (), {"pixel_size": Point(2e-8, 2e-8)})(),
            "filepath": "/somewhere/live_ib.tif",
        },
    )()
    written = data.to_dict()
    assert written["stored_fib_image_shape"] == [30, 40]
    assert written["stored_fib_image_pixel_size"] == 2e-8
    assert written["stored_fib_image_filename"] == "live_ib.tif"


def test_old_status_strings_and_nulls_read_into_the_vocabulary():
    for old, new in OLD_STATUS.items():
        assert new in PointStatus.USABLE | PointStatus.TENTATIVE
        data = upgrade_input_data(
            {
                "fib_coordinates": [
                    {
                        "point": {"x": 1, "y": 2, "z": 3},
                        "point_type": "FIB",
                        "status": old,
                    }
                ]
            }
        )
        c = CorrelationInputData.from_dict(data).fib_coordinates[0]
        assert c.status == new
        assert not c.suggested
    data = upgrade_input_data(
        {
            "fib_coordinates": [
                {
                    "point": {"x": 1, "y": 2, "z": 3},
                    "point_type": "FIB",
                    "status": None,
                    "provenance": None,
                }
            ]
        }
    )
    c = CorrelationInputData.from_dict(data).fib_coordinates[0]
    assert (c.status, c.provenance) == ("", "")


def test_a_point_type_written_by_value_reads_by_name():
    for point_type in PointType:
        written = {"point": {"x": 0, "y": 0, "z": 0}, "point_type": point_type.value}
        data = upgrade_input_data({"fib_coordinates": [written]})
        assert data["fib_coordinates"][0]["point_type"] == point_type.name
        assert Coordinate.from_dict(data["fib_coordinates"][0]).point_type is point_type


def test_a_result_reads_its_old_keys():
    old = {
        "poi": [],
        "input_data": {"fib_coordinates": [], "fib_image_pixel_size": 1e-8},
        "rotation_quaternion": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "updated_at": 1725000000.25,
    }
    new = upgrade_result(old)
    assert set(new) == {"poi", "input_data", "rotation_matrix", "updated_at"}
    assert new["input_data"] == {
        "fib_coordinates": [],
        "stored_fib_image_pixel_size": 1e-8,
    }
    assert isinstance(new["updated_at"], str)
    assert to_datetime(new["updated_at"]) == to_datetime(1725000000.25)


def test_a_file_from_a_newer_fibsem_is_refused_and_left_alone(tmp_path):
    path = tmp_path / "correlation.json"
    text = json.dumps({"version": CORRELATION_FILE_VERSION + 1, "input_data": {}})
    path.write_text(text)
    with pytest.raises(ValueError, match="newer fibsem"):
        load_correlation_file(str(path))
    assert path.read_text() == text
