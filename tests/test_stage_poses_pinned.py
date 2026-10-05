"""Today's stage poses and every reader of `rotation_180`, pinned per stage type (FIB-1101).

FIB-1101 moves the poses onto the stage device: the stage declares SEM, FIB and FM, the
compustage branch leaves `_update_orientations`, classification becomes "nearest
declared pose", and each reader of `rotation_180` moves to the whole FIB pose. None of
that should change a value on any stage this repository ships. This file is the
before-picture those changes are checked against, so it changes only in a PR that says
which value moves and why.

One `_measure` per stage type, compared against the literal table `PINNED`. The table
was produced by running `_measure` on main before any of that work, and is written out
rather than recomputed so that it cannot follow the code it guards.

The stage types are the shipped configurations, on the Demo backend. The orientation
table is built by `FibsemMicroscope` from the system settings and two facts the
instrument reports at connect (`stage_is_compustage`, `system.stage.rotation`), and no
backend overrides it, so Demo builds the same table a Thermo, Tescan or Odemis
connection would. `tfs-arctis` is the exception that needs help: its compustage is
reported by AutoScript, so it is pinned here as connect leaves it on an Arctis, with
those two facts set the way `ThermoMicroscope` sets them.

Readers measured, with the stage at each declared pose:
- `vertical_move`, which reverses dy once the stage is tilted past -90 degrees (a
  compustage at its FIB and FM poses; FIB-1124 step 2);
- the geometry stamped onto images (`hardware_geometry`);
- `transformations._projection_terms`, the pre-tilt sign of the view-corrected move;
- `reprojection._tescan_pose_angles`, the same sign for reprojection;
- `coincidence.geometry_from_images`, whether it warns about the flipped side;
- `correlation.geometry._complete_fm_pose`, the FM pose for a position without r or t.

The configuration window's "Facing the ion beam" row reads `stage.rotation_180`, which
is pinned directly.
"""

import logging
import os
from typing import Any, Dict

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import BeamType, FibsemStagePosition

# The probes `get_stage_orientation` is asked about, relative to each declared pose:
# the pose itself, then nudged in r and t either side of today's tolerances (5 degrees
# in r, 0.1 rad = 5.73 degrees in t).
PROBES = {
    "": (0.0, 0.0),
    " r+4": (4.0, 0.0),
    " r-4": (-4.0, 0.0),
    " r+6": (6.0, 0.0),
    " t+5": (0.0, 5.0),
    " t-5": (0.0, -5.0),
    " t+7": (0.0, 7.0),
    " t-7": (0.0, -7.0),
}


def _microscope(stage_type: str):
    filename = stage_type.split(" ")[0]
    microscope, _ = utils.setup_session(
        config_path=os.path.join(cfg.CONFIG_PATH, filename),
        manufacturer="Demo",
    )
    if stage_type.endswith("as an Arctis"):
        # What ThermoMicroscope's connect reads from an Arctis: AutoScript says the
        # compustage is installed, and its axes have no r.
        microscope.stage_is_compustage = True
        microscope.system.stage.rotation = False
    microscope._update_orientations()
    return microscope


def _deg(angle: float) -> float:
    return float(np.degrees(angle))


def _image(geometry, pose: FibsemStagePosition, beam_type: BeamType):
    from fibsem.structures import (
        BeamSettings,
        FibsemImage,
        FibsemImageMetadata,
        ImageSettings,
        MicroscopeState,
        Point,
    )

    pixel_size = 65e-9
    shape = (64, 96)
    metadata = FibsemImageMetadata(
        image_settings=ImageSettings(
            hfw=pixel_size * shape[1],
            resolution=(shape[1], shape[0]),
            beam_type=beam_type,
        ),
        microscope_state=MicroscopeState(
            stage_position=FibsemStagePosition(x=0, y=0, z=0, r=pose.r, t=pose.t),
            electron_beam=BeamSettings(beam_type=BeamType.ELECTRON, scan_rotation=0.0),
            ion_beam=BeamSettings(beam_type=BeamType.ION, scan_rotation=0.0),
        ),
        pixel_size=Point(pixel_size, pixel_size),
        hardware_geometry=geometry,
    )
    return FibsemImage(data=np.zeros(shape, dtype=np.uint8), metadata=metadata)


class _Warnings(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _warns_flipped_side(geometry, pose: FibsemStagePosition) -> bool:
    from fibsem.alignment.coincidence import geometry_from_images

    handler = _Warnings()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        geometry_from_images(
            _image(geometry, pose, BeamType.ELECTRON),
            _image(geometry, pose, BeamType.ION),
        )
    finally:
        root.removeHandler(handler)
    return any("flipped (FIB-orientation) side" in m for m in handler.messages)


def _measure(stage_type: str) -> Dict[str, Any]:
    """Every pinned value for one stage type, flattened to "what: value"."""
    from fibsem.correlation.geometry import _complete_fm_pose
    from fibsem.imaging.tiling.reprojection import _tescan_pose_angles
    from fibsem.transformations import _projection_terms

    microscope = _microscope(stage_type)
    out: Dict[str, Any] = {}

    orientations = dict(microscope.orientations)
    out["orientations"] = ",".join(orientations)
    for name, pose in orientations.items():
        out[f"{name}.r"] = _deg(pose.r)
        out[f"{name}.t"] = _deg(pose.t)

    out["stage.rotation_180"] = microscope.system.stage.rotation_180

    for name, pose in orientations.items():
        for suffix, (dr, dt) in PROBES.items():
            probe = FibsemStagePosition(
                r=pose.r + np.radians(dr), t=pose.t + np.radians(dt)
            )
            out[f"classify {name}{suffix}"] = microscope.get_stage_orientation(probe)
    # MILLING is a range, not a pose: any tilt above -45 degrees at the SEM rotation.
    for tilt in (-44.0, -46.0):
        probe = FibsemStagePosition(r=orientations["SEM"].r, t=np.radians(tilt))
        out[f"classify SEM.r t={tilt:g}"] = microscope.get_stage_orientation(probe)

    # vertical_move reads the stage tilt and moves relatively: give it each pose and
    # record the move, in micrometres per micrometre of dy in the FIB view.
    relative = []
    microscope.move_stage_relative = relative.append
    for name, pose in orientations.items():
        microscope.get_stage_position = lambda pose=pose: FibsemStagePosition(
            x=0, y=0, z=0, r=pose.r, t=pose.t
        )
        microscope.vertical_move(dy=1e-6)
        out[f"vertical_move {name}"] = (
            relative[-1].x * 1e6,
            relative[-1].y * 1e6,
            relative[-1].z * 1e6,
        )
    del microscope.get_stage_position

    geometry = microscope.hardware_geometry()
    out["stamped rotation_180"] = geometry.rotation_180
    out["stamped is_compustage"] = geometry.is_compustage

    for name, pose in orientations.items():
        sign, pretilt, tilt = _projection_terms(geometry, pose.r, pose.t)
        out[f"projection_terms {name}"] = (sign, _deg(pretilt), _deg(tilt))
        tilt, pretilt, inclination = _tescan_pose_angles(geometry, pose)
        out[f"tescan_pose_angles {name}"] = (
            _deg(tilt),
            _deg(pretilt),
            _deg(inclination),
        )
        out[f"coincidence warns {name}"] = _warns_flipped_side(geometry, pose)

    fm_pose = _complete_fm_pose(FibsemStagePosition(x=0, y=0, z=0), geometry)
    out["complete_fm_pose.r"] = _deg(fm_pose.r)
    out["complete_fm_pose.t"] = _deg(fm_pose.t)
    return out


def _same(measured: Any, pinned: Any) -> bool:
    if isinstance(pinned, tuple):
        return len(measured) == len(pinned) and all(
            _same(m, p) for m, p in zip(measured, pinned)
        )
    if isinstance(pinned, (bool, str)):
        return measured == pinned
    return measured == pytest.approx(pinned, abs=1e-6)


# Which pinned table each stage type must reproduce. Several shipped files describe the
# same geometry, so they share one table rather than repeating it.
STAGE_TYPES = {
    "microscope-configuration.yaml": "rotating, pre-tilt 35",
    "tfs-aquilos2-configuration.yaml": "rotating, pre-tilt 35",
    "tfs-hydra-configuration.yaml": "rotating, pre-tilt 35",
    "odemis-configuration.yaml": "rotating, pre-tilt 35",
    "sim-iflm-configuration.yaml": "rotating, pre-tilt 35",
    "tescan-configuration.yaml": "rotating, reference 180",
    "sim-arctis-configuration.yaml": "compustage",
    "tfs-arctis-configuration.yaml as an Arctis": "compustage",
}

# Generated on main at a63fd87 by running `_measure`; see the module docstring.
# Angles in degrees.
PINNED: Dict[str, Dict[str, Any]] = {
    "rotating, pre-tilt 35": {
        "orientations": "SEM,FIB,MILLING",
        "SEM.r": 0.0,
        "SEM.t": 35.0,
        "FIB.r": 180.0,
        "FIB.t": 17.0,
        "MILLING.r": 0.0,
        "MILLING.t": 12.0,
        "stage.rotation_180": 180.0,
        "classify SEM": "SEM",
        "classify SEM r+4": "SEM",
        "classify SEM r-4": "SEM",
        "classify SEM r+6": "NONE",
        "classify SEM t+5": "SEM",
        "classify SEM t-5": "SEM",
        "classify SEM t+7": "MILLING",
        "classify SEM t-7": "MILLING",
        "classify FIB": "FIB",
        "classify FIB r+4": "FIB",
        "classify FIB r-4": "FIB",
        "classify FIB r+6": "NONE",
        "classify FIB t+5": "FIB",
        "classify FIB t-5": "FIB",
        "classify FIB t+7": "NONE",
        "classify FIB t-7": "NONE",
        "classify MILLING": "MILLING",
        "classify MILLING r+4": "MILLING",
        "classify MILLING r-4": "MILLING",
        "classify MILLING r+6": "NONE",
        "classify MILLING t+5": "MILLING",
        "classify MILLING t-5": "MILLING",
        "classify MILLING t+7": "MILLING",
        "classify MILLING t-7": "MILLING",
        "classify SEM.r t=-44": "MILLING",
        "classify SEM.r t=-46": "NONE",
        "vertical_move SEM": (0.0, 0.727879, 1.039519),
        "vertical_move FIB": (0.0, 0.371025, 1.213568),
        "vertical_move MILLING": (0.0, 0.263844, 1.241287),
        "stamped rotation_180": 180.0,
        "stamped is_compustage": False,
        "projection_terms SEM": (1.0, 35.0, 35.0),
        "tescan_pose_angles SEM": (35.0, 35.0, 0.0),
        "coincidence warns SEM": False,
        "projection_terms FIB": (1.0, -35.0, 17.0),
        "tescan_pose_angles FIB": (17.0, -35.0, 52.0),
        "coincidence warns FIB": True,
        "projection_terms MILLING": (1.0, 35.0, 12.0),
        "tescan_pose_angles MILLING": (12.0, 35.0, -23.0),
        "coincidence warns MILLING": False,
        "complete_fm_pose.r": 180.0,
        "complete_fm_pose.t": 17.0,
    },
    "rotating, reference 180": {
        "orientations": "SEM,FIB,MILLING",
        "SEM.r": 180.0,
        "SEM.t": 0.0,
        "FIB.r": 0.0,
        "FIB.t": 55.0,
        "MILLING.r": 180.0,
        "MILLING.t": -20.0,
        "stage.rotation_180": 0.0,
        "classify SEM": "SEM",
        "classify SEM r+4": "SEM",
        "classify SEM r-4": "SEM",
        "classify SEM r+6": "NONE",
        "classify SEM t+5": "SEM",
        "classify SEM t-5": "SEM",
        "classify SEM t+7": "MILLING",
        "classify SEM t-7": "MILLING",
        "classify FIB": "FIB",
        "classify FIB r+4": "FIB",
        "classify FIB r-4": "FIB",
        "classify FIB r+6": "NONE",
        "classify FIB t+5": "FIB",
        "classify FIB t-5": "FIB",
        "classify FIB t+7": "NONE",
        "classify FIB t-7": "NONE",
        "classify MILLING": "MILLING",
        "classify MILLING r+4": "MILLING",
        "classify MILLING r-4": "MILLING",
        "classify MILLING r+6": "NONE",
        "classify MILLING t+5": "MILLING",
        "classify MILLING t-5": "MILLING",
        "classify MILLING t+7": "MILLING",
        "classify MILLING t-7": "MILLING",
        "classify SEM.r t=-44": "MILLING",
        "classify SEM.r t=-46": "NONE",
        "vertical_move SEM": (0.0, 0.0, 1.220775),
        "vertical_move FIB": (0.0, 1.0, 0.700208),
        "vertical_move MILLING": (0.0, -0.417529, 1.147153),
        "stamped rotation_180": 0.0,
        "stamped is_compustage": False,
        "projection_terms SEM": (1.0, 0.0, 0.0),
        "tescan_pose_angles SEM": (0.0, 0.0, 0.0),
        "coincidence warns SEM": False,
        "projection_terms FIB": (1.0, 0.0, 55.0),
        "tescan_pose_angles FIB": (55.0, 0.0, 55.0),
        "coincidence warns FIB": True,
        "projection_terms MILLING": (1.0, 0.0, -20.0),
        "tescan_pose_angles MILLING": (-20.0, 0.0, -20.0),
        "coincidence warns MILLING": False,
        "complete_fm_pose.r": 0.0,
        "complete_fm_pose.t": 55.0,
    },
    "compustage": {
        "orientations": "SEM,FIB,MILLING,FM",
        "SEM.r": 0.0,
        "SEM.t": 0.0,
        "FIB.r": 0.0,
        "FIB.t": -128.0,
        "MILLING.r": 0.0,
        "MILLING.t": -23.0,
        "FM.r": 0.0,
        "FM.t": -180.0,
        "stage.rotation_180": 0.0,
        "classify SEM": "SEM",
        "classify SEM r+4": "SEM",
        "classify SEM r-4": "SEM",
        "classify SEM r+6": "NONE",
        "classify SEM t+5": "SEM",
        "classify SEM t-5": "SEM",
        "classify SEM t+7": "MILLING",
        "classify SEM t-7": "MILLING",
        "classify FIB": "FIB",
        "classify FIB r+4": "FIB",
        "classify FIB r-4": "FIB",
        "classify FIB r+6": "NONE",
        "classify FIB t+5": "FIB",
        "classify FIB t-5": "FIB",
        "classify FIB t+7": "NONE",
        "classify FIB t-7": "NONE",
        "classify MILLING": "MILLING",
        "classify MILLING r+4": "MILLING",
        "classify MILLING r-4": "MILLING",
        "classify MILLING r+6": "NONE",
        "classify MILLING t+5": "MILLING",
        "classify MILLING t-5": "MILLING",
        "classify MILLING t+7": "MILLING",
        "classify MILLING t-7": "MILLING",
        "classify FM": "FM",
        "classify FM r+4": "FM",
        "classify FM r-4": "FM",
        "classify FM r+6": "NONE",
        "classify FM t+5": "FM",
        "classify FM t-5": "FM",
        "classify FM t+7": "NONE",
        "classify FM t-7": "NONE",
        "classify SEM.r t=-44": "MILLING",
        "classify SEM.r t=-46": "NONE",
        "vertical_move SEM": (0.0, 0.0, 1.269018),
        "vertical_move FIB": (0.0, 1.0, 0.781286),
        "vertical_move MILLING": (0.0, -0.495845, 1.168137),
        "vertical_move FM": (0.0, 0.0, 1.269018),
        "stamped rotation_180": 0.0,
        "stamped is_compustage": True,
        "projection_terms SEM": (-1.0, 0.0, 180.0),
        "tescan_pose_angles SEM": (0.0, 0.0, 0.0),
        "coincidence warns SEM": False,
        "projection_terms FIB": (1.0, 0.0, 52.0),
        "tescan_pose_angles FIB": (-128.0, 0.0, -128.0),
        "coincidence warns FIB": False,
        "projection_terms MILLING": (-1.0, 0.0, 157.0),
        "tescan_pose_angles MILLING": (-23.0, 0.0, -23.0),
        "coincidence warns MILLING": False,
        "projection_terms FM": (-1.0, 0.0, 0.0),
        "tescan_pose_angles FM": (-180.0, 0.0, -180.0),
        "coincidence warns FM": False,
        "complete_fm_pose.r": 0.0,
        "complete_fm_pose.t": -180.0,
    },
}


@pytest.mark.parametrize("stage_type", sorted(STAGE_TYPES))
def test_stage_poses_and_their_readers_are_unchanged(stage_type: str):
    measured = _measure(stage_type)
    pinned = PINNED[STAGE_TYPES[stage_type]]

    assert sorted(measured) == sorted(pinned)
    different = {
        key: (measured[key], pinned[key])
        for key in pinned
        if not _same(measured[key], pinned[key])
    }
    assert different == {}, "(measured, pinned): " + repr(different)
