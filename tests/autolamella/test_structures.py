import os
from dataclasses import dataclass
from enum import Enum
from typing import List

import pytest

from fibsem.applications.autolamella.structures import (
    DefectState,
    DefectType,
    Lamella,
)

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

