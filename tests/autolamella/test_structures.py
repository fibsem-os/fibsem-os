import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List

import pytest

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    DefectState,
    DefectType,
    Experiment,
    LamellaDefaultConfig,
    Lamella,
)
from fibsem.applications.autolamella.workflows.tasks.basic_milling import BasicMillingTaskConfig
from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.patterning.patterns2 import RectanglePattern
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import DEFAULT_ALIGNMENT_AREA, FibsemMillingSettings, FibsemRectangle, ImageSettings, MicroscopeState, Point, ReferenceImageParameters
from psygnal.containers import EventedDict

# ── DefectState tests ─────────────────────────────────────────────────────────

def test_defect_state_defaults():
    d = DefectState()
    assert d.state == DefectType.NONE
    assert d.last_completed_task == ""
    assert d.description == ""
    assert d.updated_at is None


def test_defect_state_to_dict_and_from_dict():
    d = DefectState(state=DefectType.REWORK, last_completed_task="Mill Rough", description="thin spot")
    data = d.to_dict()
    assert data["state"] == "REWORK"
    assert data["last_completed_task"] == "Mill Rough"
    assert data["description"] == "thin spot"

    d2 = DefectState.from_dict(data)
    assert d2.state == DefectType.REWORK
    assert d2.last_completed_task == "Mill Rough"
    assert d2.description == "thin spot"


def test_defect_state_from_dict_all_types():
    for dt in DefectType:
        d = DefectState.from_dict({"state": dt.name})
        assert d.state == dt


def test_defect_state_from_dict_empty():
    d = DefectState.from_dict({})
    assert d.state == DefectType.NONE
    assert d.last_completed_task == ""


def test_defect_state_backwards_compat_no_defect():
    """Old format: has_defect=False → NONE"""
    d = DefectState.from_dict({"has_defect": False, "requires_rework": False, "description": "", "updated_at": None})
    assert d.state == DefectType.NONE


def test_defect_state_backwards_compat_failure():
    """Old format: has_defect=True, requires_rework=False → FAILURE"""
    d = DefectState.from_dict({"has_defect": True, "requires_rework": False, "description": "bad mill", "updated_at": 1.0})
    assert d.state == DefectType.FAILURE
    assert d.description == "bad mill"
    assert d.updated_at == 1.0


def test_defect_state_backwards_compat_rework():
    """Old format: has_defect=True, requires_rework=True → REWORK"""
    d = DefectState.from_dict({"has_defect": True, "requires_rework": True, "description": "thin", "updated_at": None})
    assert d.state == DefectType.REWORK


def test_defect_state_backwards_compat_missing_fields():
    """Old format with only has_defect present (requires_rework absent) → FAILURE"""
    d = DefectState.from_dict({"has_defect": True})
    assert d.state == DefectType.FAILURE


def test_defect_state_clear():
    d = DefectState(state=DefectType.FAILURE, last_completed_task="Mill Rough", description="oops")
    d.clear()
    assert d.state == DefectType.NONE
    assert d.last_completed_task == ""
    assert d.description == ""
    assert d.updated_at is None


def test_defect_state_set_defect_default():
    d = DefectState()
    d.set_defect(description="cracked")
    assert d.state == DefectType.FAILURE
    assert d.description == "cracked"
    assert d.updated_at is not None


def test_defect_state_set_defect_rework():
    d = DefectState()
    d.set_defect(description="thin spot", state=DefectType.REWORK)
    assert d.state == DefectType.REWORK


def test_lamella_is_failure():
    from pathlib import Path
    lam = Lamella(path=Path("/tmp/test/lam"), number=1, petname="test-lam")
    assert lam.is_failure is False
    lam.defect.state = DefectType.FAILURE
    assert lam.is_failure is True
    lam.defect.state = DefectType.REWORK
    assert lam.is_failure is False
    lam.defect.state = DefectType.NONE
    assert lam.is_failure is False


# ── LamellaDefaultConfig ──────────────────────────────────────────────────────

def test_lamella_defaults_defaults():
    t = LamellaDefaultConfig()
    assert t.use_petname is True
    assert t.name_prefix == ""
    assert t.alignment_area is None
    assert t.poi is None


def test_lamella_defaults_to_dict_minimal():
    d = LamellaDefaultConfig().to_dict()
    assert d["use_petname"] is True
    assert d["name_prefix"] == ""
    assert "alignment_area" not in d
    assert "poi" not in d


def test_lamella_defaults_to_dict_full():
    rect = FibsemRectangle(left=0.1, top=0.2, width=0.3, height=0.4)
    poi = Point(x=1.0, y=2.0)
    t = LamellaDefaultConfig(use_petname=False, name_prefix="GridA-", alignment_area=rect, poi=poi)
    d = t.to_dict()
    assert d["use_petname"] is False
    assert d["name_prefix"] == "GridA-"
    assert d["alignment_area"] == {"left": 0.1, "top": 0.2, "width": 0.3, "height": 0.4}
    assert d["poi"]["x"] == 1.0 and d["poi"]["y"] == 2.0


def test_lamella_defaults_from_dict_empty():
    t = LamellaDefaultConfig.from_dict({})
    assert t.use_petname is True
    assert t.name_prefix == ""
    assert t.alignment_area is None
    assert t.poi is None


def test_lamella_defaults_from_dict_full():
    d = {
        "use_petname": False,
        "name_prefix": "GridA-",
        "alignment_area": {"left": 0.1, "top": 0.2, "width": 0.3, "height": 0.4},
        "poi": {"x": 1.0, "y": 2.0},
    }
    t = LamellaDefaultConfig.from_dict(d)
    assert t.use_petname is False
    assert t.name_prefix == "GridA-"
    assert isinstance(t.alignment_area, FibsemRectangle)
    assert t.alignment_area.left == 0.1
    assert isinstance(t.poi, Point)
    assert t.poi.x == 1.0


def test_lamella_defaults_round_trip():
    rect = FibsemRectangle(left=0.05, top=0.1, width=0.2, height=0.5)
    poi = Point(x=3.0, y=-1.0)
    t = LamellaDefaultConfig(use_petname=False, name_prefix="X-", alignment_area=rect, poi=poi)
    t2 = LamellaDefaultConfig.from_dict(t.to_dict())
    assert t2.use_petname is False
    assert t2.name_prefix == "X-"
    assert t2.alignment_area == rect
    assert t2.poi == poi


# ── Experiment.add_new_lamella respects LamellaDefaultConfig ─────────────────

def _make_experiment(tmp_path: Path, template: LamellaDefaultConfig = None) -> Experiment:
    exp = Experiment(path=tmp_path, name="test-exp")
    protocol = AutoLamellaTaskProtocol()
    if template is not None:
        protocol.lamella_defaults = template
    exp.task_protocol = protocol
    return exp


def test_add_new_lamella_default_template(tmp_path):
    exp = _make_experiment(tmp_path)
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    lamella = exp.positions[0]
    assert lamella.alignment_area == FibsemRectangle.from_dict(DEFAULT_ALIGNMENT_AREA)
    assert lamella.poi == Point(0, 0)
    assert lamella.name.startswith("01-")


def test_add_new_lamella_no_petname(tmp_path):
    exp = _make_experiment(tmp_path, LamellaDefaultConfig(use_petname=False))
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    assert exp.positions[0].name == "Lamella-01"


def test_add_new_lamella_name_prefix(tmp_path):
    exp = _make_experiment(tmp_path, LamellaDefaultConfig(use_petname=False, name_prefix="GridA"))
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    assert exp.positions[0].name == "GridA-Lamella-01"


def test_add_new_lamella_name_prefix_with_petname(tmp_path):
    exp = _make_experiment(tmp_path, LamellaDefaultConfig(use_petname=True, name_prefix="GridA"))
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    assert exp.positions[0].name.startswith("GridA-01-")


def test_add_new_lamella_custom_alignment_area(tmp_path):
    rect = FibsemRectangle(left=0.05, top=0.1, width=0.3, height=0.5)
    exp = _make_experiment(tmp_path, LamellaDefaultConfig(alignment_area=rect))
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    assert exp.positions[0].alignment_area == rect


def test_add_new_lamella_custom_poi(tmp_path):
    poi = Point(x=1.5, y=-2.0)
    exp = _make_experiment(tmp_path, LamellaDefaultConfig(poi=poi))
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    assert exp.positions[0].poi == poi


def test_add_new_lamella_defaults_values_are_independent(tmp_path):
    """Each lamella gets its own copy, not a shared reference."""
    rect = FibsemRectangle(left=0.05, top=0.1, width=0.3, height=0.5)
    exp = _make_experiment(tmp_path, LamellaDefaultConfig(alignment_area=rect))
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    exp.positions[0].alignment_area = FibsemRectangle(left=0.9, top=0.9, width=0.05, height=0.05)
    assert exp.positions[1].alignment_area == rect


def test_add_new_lamella_seeds_milling_angle_from_setup_task(tmp_path):
    """Every new lamella should be seeded with the target milling angle so it is
    not saved as null before the SetupLamella workflow runs."""
    from fibsem.applications.autolamella.workflows.tasks.select_position import (
        SelectMillingPositionTaskConfig,
    )

    exp = _make_experiment(tmp_path)
    setup = SelectMillingPositionTaskConfig(
        task_name="Setup Lamella Position", milling=EventedDict(), milling_angle=27.0
    )
    task_config = EventedDict()
    task_config["Setup Lamella Position"] = setup

    exp.add_new_lamella(MicroscopeState(), task_config)
    exp.add_new_lamella(MicroscopeState(), task_config)

    assert exp.positions[0].milling_angle == 27.0
    assert exp.positions[1].milling_angle == 27.0


def test_add_new_lamella_milling_angle_none_without_setup_task(tmp_path):
    """Without a SetupLamella task config there is no target angle to seed, so
    milling_angle stays None (and adding must not raise)."""
    exp = _make_experiment(tmp_path)
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    assert exp.positions[0].milling_angle is None


def test_add_new_lamella_no_name_collision_after_delete(tmp_path):
    """Deleting a lamella must not cause the next one to reuse its name/number."""
    exp = _make_experiment(tmp_path, LamellaDefaultConfig(use_petname=False))
    exp.add_new_lamella(MicroscopeState(), EventedDict())  # Lamella-01
    exp.add_new_lamella(MicroscopeState(), EventedDict())  # Lamella-02
    exp.add_new_lamella(MicroscopeState(), EventedDict())  # Lamella-03
    del exp.positions[1]                                   # remove Lamella-02
    exp.add_new_lamella(MicroscopeState(), EventedDict())  # must be Lamella-04, not Lamella-03
    names = [lam.name for lam in exp.positions]
    assert len(names) == len(set(names)), f"Duplicate names after delete: {names}"
    assert "Lamella-04" in names


# ── AutoLamellaTaskConfig.estimated_time ─────────────────────────────────────

def _make_task_config(n_stages: int = 1) -> BasicMillingTaskConfig:
    stage = FibsemMillingStage(
        milling=FibsemMillingSettings(milling_current=2e-9),
        pattern=RectanglePattern(width=10e-6, height=5e-6, depth=1e-6),
    )
    milling_task = FibsemMillingTaskConfig(stages=[stage] * n_stages)
    config = BasicMillingTaskConfig(milling={"mill": milling_task})
    return config


def test_task_config_estimated_time_is_float():
    config = _make_task_config()
    assert isinstance(config.estimated_time, float)
    assert config.estimated_time >= 0.0


def test_task_config_estimated_time_includes_milling():
    config = _make_task_config(n_stages=1)
    milling_time = sum(t.estimated_time for t in config.milling.values())
    assert config.estimated_time >= milling_time


def test_task_config_estimated_time_includes_reference_imaging():
    img = ImageSettings(resolution=(1536, 1024), dwell_time=1e-6)
    ref = ReferenceImageParameters(
        imaging=img, acquire_sem=True, acquire_fib=True, acquire_image1=True, acquire_image2=True
    )
    config = _make_task_config()
    config.reference_imaging = ref
    milling_time = sum(t.estimated_time for t in config.milling.values())
    assert config.estimated_time == pytest.approx(milling_time + ref.estimated_time)


def test_task_config_estimated_time_no_imaging():
    ref = ReferenceImageParameters(
        acquire_sem=False, acquire_fib=False, acquire_image1=False, acquire_image2=False
    )
    config = _make_task_config()
    config.reference_imaging = ref
    milling_time = sum(t.estimated_time for t in config.milling.values())
    assert config.estimated_time == pytest.approx(milling_time)


def test_task_protocol_estimated_time_per_task():
    from fibsem import config as fcfg
    protocol = AutoLamellaTaskProtocol.load(fcfg.AUTOLAMELLA_TASK_PROTOCOL_PATH)
    for task in protocol.task_config.values():
        assert isinstance(task.estimated_time, float)
        assert task.estimated_time >= 0.0

