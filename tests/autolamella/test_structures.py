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
    LamellaTemplateConfig,
    Lamella,
)
from fibsem.structures import DEFAULT_ALIGNMENT_AREA, FibsemRectangle, MicroscopeState, Point
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


# ── LamellaTemplateConfig ──────────────────────────────────────────────────────

def test_lamella_template_defaults():
    t = LamellaTemplateConfig()
    assert t.use_petname is True
    assert t.name_prefix == ""
    assert t.alignment_area is None
    assert t.poi is None


def test_lamella_template_to_dict_minimal():
    d = LamellaTemplateConfig().to_dict()
    assert d["use_petname"] is True
    assert d["name_prefix"] == ""
    assert "alignment_area" not in d
    assert "poi" not in d


def test_lamella_template_to_dict_full():
    rect = FibsemRectangle(left=0.1, top=0.2, width=0.3, height=0.4)
    poi = Point(x=1.0, y=2.0)
    t = LamellaTemplateConfig(use_petname=False, name_prefix="GridA-", alignment_area=rect, poi=poi)
    d = t.to_dict()
    assert d["use_petname"] is False
    assert d["name_prefix"] == "GridA-"
    assert d["alignment_area"] == {"left": 0.1, "top": 0.2, "width": 0.3, "height": 0.4}
    assert d["poi"]["x"] == 1.0 and d["poi"]["y"] == 2.0


def test_lamella_template_from_dict_empty():
    t = LamellaTemplateConfig.from_dict({})
    assert t.use_petname is True
    assert t.name_prefix == ""
    assert t.alignment_area is None
    assert t.poi is None


def test_lamella_template_from_dict_full():
    d = {
        "use_petname": False,
        "name_prefix": "GridA-",
        "alignment_area": {"left": 0.1, "top": 0.2, "width": 0.3, "height": 0.4},
        "poi": {"x": 1.0, "y": 2.0},
    }
    t = LamellaTemplateConfig.from_dict(d)
    assert t.use_petname is False
    assert t.name_prefix == "GridA-"
    assert isinstance(t.alignment_area, FibsemRectangle)
    assert t.alignment_area.left == 0.1
    assert isinstance(t.poi, Point)
    assert t.poi.x == 1.0


def test_lamella_template_round_trip():
    rect = FibsemRectangle(left=0.05, top=0.1, width=0.2, height=0.5)
    poi = Point(x=3.0, y=-1.0)
    t = LamellaTemplateConfig(use_petname=False, name_prefix="X-", alignment_area=rect, poi=poi)
    t2 = LamellaTemplateConfig.from_dict(t.to_dict())
    assert t2.use_petname is False
    assert t2.name_prefix == "X-"
    assert t2.alignment_area == rect
    assert t2.poi == poi


# ── Experiment.add_new_lamella respects LamellaTemplateConfig ─────────────────

def _make_experiment(tmp_path: Path, template: LamellaTemplateConfig = None) -> Experiment:
    exp = Experiment(path=tmp_path, name="test-exp")
    protocol = AutoLamellaTaskProtocol()
    if template is not None:
        protocol.lamella_template = template
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
    exp = _make_experiment(tmp_path, LamellaTemplateConfig(use_petname=False))
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    assert exp.positions[0].name == "Lamella-01"


def test_add_new_lamella_name_prefix(tmp_path):
    exp = _make_experiment(tmp_path, LamellaTemplateConfig(use_petname=False, name_prefix="GridA"))
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    assert exp.positions[0].name == "GridA-Lamella-01"


def test_add_new_lamella_name_prefix_with_petname(tmp_path):
    exp = _make_experiment(tmp_path, LamellaTemplateConfig(use_petname=True, name_prefix="GridA"))
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    assert exp.positions[0].name.startswith("GridA-01-")


def test_add_new_lamella_custom_alignment_area(tmp_path):
    rect = FibsemRectangle(left=0.05, top=0.1, width=0.3, height=0.5)
    exp = _make_experiment(tmp_path, LamellaTemplateConfig(alignment_area=rect))
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    assert exp.positions[0].alignment_area == rect


def test_add_new_lamella_custom_poi(tmp_path):
    poi = Point(x=1.5, y=-2.0)
    exp = _make_experiment(tmp_path, LamellaTemplateConfig(poi=poi))
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    assert exp.positions[0].poi == poi


def test_add_new_lamella_template_values_are_independent(tmp_path):
    """Each lamella gets its own copy, not a shared reference."""
    rect = FibsemRectangle(left=0.05, top=0.1, width=0.3, height=0.5)
    exp = _make_experiment(tmp_path, LamellaTemplateConfig(alignment_area=rect))
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    exp.positions[0].alignment_area = FibsemRectangle(left=0.9, top=0.9, width=0.05, height=0.05)
    assert exp.positions[1].alignment_area == rect


def test_add_new_lamella_no_name_collision_after_delete(tmp_path):
    """Deleting a lamella must not cause the next one to reuse its name/number."""
    exp = _make_experiment(tmp_path, LamellaTemplateConfig(use_petname=False))
    exp.add_new_lamella(MicroscopeState(), EventedDict())  # Lamella-01
    exp.add_new_lamella(MicroscopeState(), EventedDict())  # Lamella-02
    exp.add_new_lamella(MicroscopeState(), EventedDict())  # Lamella-03
    del exp.positions[1]                                   # remove Lamella-02
    exp.add_new_lamella(MicroscopeState(), EventedDict())  # must be Lamella-04, not Lamella-03
    names = [lam.name for lam in exp.positions]
    assert len(names) == len(set(names)), f"Duplicate names after delete: {names}"
    assert "Lamella-04" in names

