"""Can this device see the sample from here -- one question, both mountings.

`get_device_imaging_state` is the conjunction FIB-839 measured, promoted to a method
on `FibsemMicroscope` -- the top-level system, which owns the stage and so owns the
question. It returns a state rather than a bool because the reason is the useful
part: each failing value names the remedy, and callers act on the value by policy
(acquisition gates permit NEEDS_REPOSE; planning sites require READY).

Nothing calls it yet. The gates move onto it in the next change; this file pins what
they will find when they do.
"""

import logging
import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import DeviceImagingState, FibsemStagePosition

IFLM_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml")
ARCTIS_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")


def _microscope(config_path: str = IFLM_CONFIG):
    microscope, _ = utils.setup_session(config_path=config_path)
    return microscope


def _offset_at_the_fm():
    microscope = _microscope()
    microscope.move_to_orientation("FIB")
    microscope.move_to_microscope("FM")
    return microscope


# ── READY: the one cell per mounting ─────────────────────────────────


def test_ready_at_the_fm_on_an_offset_mount():
    assert (
        _offset_at_the_fm().get_device_imaging_state("FM") is DeviceImagingState.READY
    )


def test_ready_at_the_fm_on_a_compustage():
    microscope = _microscope(ARCTIS_CONFIG)
    microscope.move_to_microscope("FM")

    assert microscope.get_device_imaging_state("FM") is DeviceImagingState.READY


def test_the_failing_term_names_the_remedy_on_an_offset_mount():
    """At the beams in FIB pose: the pose is right, the place is wrong. Travel."""
    microscope = _microscope()
    microscope.move_to_orientation("FIB")

    assert microscope.get_device_imaging_state("FM") is DeviceImagingState.NEEDS_TRAVEL


def test_the_failing_term_names_the_remedy_on_a_compustage():
    """Looking at the sample with a beam: the place is right (it always is -- the FM
    shares the beams' origin), the pose is wrong. Re-pose, which is exactly how a
    compustage reaches its FM."""
    microscope = _microscope(ARCTIS_CONFIG)
    microscope.move_to_orientation("SEM")

    assert microscope.get_device_imaging_state("FM") is DeviceImagingState.NEEDS_REPOSE


def test_wrong_on_both_axes_brackets_the_repose_first():
    """Offset at the beams in SEM: not at the FM, and SEM is not a pose the objective
    images from. The remedy order is re-pose THEN travel, so the rotation happens at
    the beams and never under the objective (FIB-841)."""
    microscope = _microscope()
    microscope.move_to_orientation("SEM")

    assert (
        microscope.get_device_imaging_state("FM")
        is DeviceImagingState.NEEDS_REPOSE_THEN_TRAVEL
    )


def test_a_compustage_can_never_need_travel():
    """The geometry, not a branch, is what preserves acquire-anywhere on an Arctis:
    its FM shares the beams' origin, so the place term is true at every pose and no
    travel state is reachable. An acquisition gate that refuses travel states
    therefore never refuses on a compustage -- the ALLOW_UNKNOWN_ORIENTATIONS
    behaviour, derived rather than flagged."""
    microscope = _microscope(ARCTIS_CONFIG)

    for orientation in ("SEM", "FIB", "MILLING"):
        microscope.move_to_orientation(orientation)
        assert microscope.get_device_imaging_state("FM") in (
            DeviceImagingState.READY,
            DeviceImagingState.NEEDS_REPOSE,
        )


# ── the beams as a device ────────────────────────────────────────────


def test_the_beams_are_ready_in_every_pose():
    """FIBSEM declares no acquisition orientations: the pose term is vacuously true,
    and the place term alone decides."""
    microscope = _microscope()

    for orientation in ("SEM", "FIB", "MILLING"):
        microscope.move_to_orientation(orientation)
        assert microscope.get_device_imaging_state("FIBSEM") is DeviceImagingState.READY


def test_the_beams_need_travel_from_the_fm():
    microscope = _offset_at_the_fm()

    assert (
        microscope.get_device_imaging_state("FIBSEM") is DeviceImagingState.NEEDS_TRAVEL
    )


# ── no device ────────────────────────────────────────────────────────


def test_a_system_with_no_fluorescence_microscope_says_so():
    """The tri-state's whole point: NO_DEVICE is terminal, distinct from every
    remedy state. Today's callers collapse this into the same False as "away", so a
    refusal cannot say whether to move the stage or give up."""
    microscope = _microscope(cfg.MICROSCOPE_CONFIGURATION_PATH)
    assert microscope.fm is None

    assert microscope.get_device_imaging_state("FM") is DeviceImagingState.NO_DEVICE


# ── stored poses ─────────────────────────────────────────────────────


def test_a_stored_pose_is_answered_without_moving_anything():
    """Both workflow tasks ask about a lamella's saved position, not the stage's."""
    microscope = _microscope()
    at_the_fm_in_fib = FibsemStagePosition(
        x=48.8e-3,
        y=0.0,
        z=0.0,
        r=microscope.get_orientation("FIB").r,
        t=microscope.get_orientation("FIB").t,
    )
    before = microscope.get_stage_position()

    state = microscope.get_device_imaging_state("FM", at_the_fm_in_fib)

    assert state is DeviceImagingState.READY
    assert microscope.get_stage_position().is_close(before, tol=1e-9)


# ── the connect-time geometry warnings ───────────────────────────────


def _write_config(tmp_path, source, edit):
    data = utils.load_yaml(source)
    edit(data)
    path = os.path.join(tmp_path, "config.yaml")
    utils.save_yaml(path, data)
    return path


# `setup_session` reconfigures logging with `force=True`, which strips caplog's
# handler, so these capture with a plain handler and call
# `_warn_on_fluorescence_geometry` directly -- it is the same method the connect path
# runs (one line in each __init__), and it is idempotent.


def _warnings_from(microscope) -> str:
    captured: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record.getMessage())

    handler = _Capture(level=logging.WARNING)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        microscope._warn_on_fluorescence_geometry()
    finally:
        root.removeHandler(handler)
    return "\n".join(captured)


def test_an_offset_fm_with_no_declared_geometry_warns_at_connect(tmp_path):
    """The silent case the default flip created: `fm.enabled` without a `devices:`
    block inherits the objective-under-the-grid default, and every place-term answer
    is about somewhere its FM is not. No error ever follows -- the conjunction is
    just never true at the FM -- so connect is the one loud moment available."""

    def drop_devices(data):
        del data["stage"]["devices"]
        del data["stage"]["device_range"]

    path = _write_config(tmp_path, IFLM_CONFIG, drop_devices)

    assert "no `stage.devices` block is declared" in _warnings_from(_microscope(path))


def test_a_compustage_with_a_phantom_offset_fm_warns_at_connect(tmp_path):
    """The pre-flip world, declared explicitly: a compustage whose FM origin is away
    from the beams describes a place its stage never travels to."""

    def add_phantom(data):
        data["stage"]["devices"] = {
            "FIBSEM": {"origin": {"x": 0.0}},
            "FM": {"origin": {"x": 48.8e-3}},
        }

    path = _write_config(tmp_path, ARCTIS_CONFIG, add_phantom)

    assert "a place the stage never travels to" in _warnings_from(_microscope(path))


@pytest.mark.parametrize(
    "config_path",
    [IFLM_CONFIG, ARCTIS_CONFIG, cfg.MICROSCOPE_CONFIGURATION_PATH],
    ids=["declared-offset", "default-compustage", "beam-only"],
)
def test_a_coherent_configuration_does_not_warn(config_path):
    assert _warnings_from(_microscope(config_path)) == ""
