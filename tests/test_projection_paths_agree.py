"""One implementation of the inverse projection, reached three ways.

`FibsemMicroscope._inverse_view_corrected_stage_movement` and
`imaging/tiling/reprojection._inverse_y_corrected_stage_movement` each used to carry
their own copy of the trigonometry that recovers an in-image y-displacement from a y/z
stage movement. Three copies of one decision, kept consistent only by everyone
remembering to edit all three -- and they had already drifted (FIB-500).

Both now defer to `transformations.inverse_view_corrected_dy`. These tests are what makes
that worth having: they fail if a copy ever comes back, and they pin the reachability
argument that lets the drift be closed without a hardware check.

Run directly:
    python -m pytest tests/test_projection_paths_agree.py
"""

from __future__ import annotations

import ast
import io
import os

import numpy as np
import pytest

from fibsem import utils
from fibsem.imaging.tiling.reprojection import _inverse_y_corrected_stage_movement
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemStagePosition,
    ImageSettings,
)
from fibsem.transformations import inverse_view_corrected_dy

# Poses worth asking about, in degrees. The configured orientations plus the two the
# geometry is exact at, because a sweep of round numbers misses -128 and -23 entirely.
TILTS = (-180.0, -160.0, -128.0, -90.0, -23.0, 0.0, 20.0, 38.0)
DELTAS = ((12e-6, -5e-6), (-40e-6, 22e-6), (0.0, 30e-6), (30e-6, 0.0), (0.0, 0.0))


@pytest.fixture(scope="module")
def microscope():
    """A simulated Arctis -- a compustage, which is where the three paths differed."""
    import fibsem.config as fibsem_config

    path = os.path.join(
        os.path.dirname(fibsem_config.__file__),
        "config",
        "sim-arctis-configuration.yaml",
    )
    scope, _ = utils.setup_session(manufacturer="Demo", config_path=path)
    assert scope.stage_is_compustage, "the config stopped being a compustage"
    return scope


def _image_at(
    microscope, beam_type: BeamType, pose: FibsemStagePosition
) -> FibsemImage:
    """An image carrying *pose* and the instrument's geometry, for the metadata path."""
    image = FibsemImage.generate_blank_image(resolution=(64, 64), hfw=100e-6)
    image.data = np.zeros((64, 64), dtype=np.uint8)
    state = microscope.get_microscope_state(beam_type=beam_type)
    state.stage_position = pose
    image.metadata.image_settings = ImageSettings(hfw=100e-6, beam_type=beam_type)
    image.metadata.microscope_state = state
    image.metadata.system_info = microscope.system.info
    image.metadata.hardware_geometry = microscope.hardware_geometry()
    return image


def _shared(microscope, beam_type, pose, dy, dz) -> float:
    """The one implementation, called directly."""
    return inverse_view_corrected_dy(
        dy=dy,
        dz=dz,
        view_tilt=microscope._beam_view_tilt(beam_type),
        geometry=microscope.hardware_geometry(),
        stage_rotation=pose.r if pose.r is not None else 0.0,
        stage_tilt=pose.t if pose.t is not None else 0.0,
    )


class TestTheThreePathsAgree:
    """The contract the delegation exists to make unbreakable.

    Not "they happen to agree today" -- these run the full pose matrix, so a
    reintroduced copy that is right at the flat poses and wrong at the tilted ones
    (which is what both geometry defects here looked like) fails rather than passing.
    """

    @pytest.mark.parametrize("tilt_deg", TILTS)
    @pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
    def test_the_image_metadata_path_matches(self, microscope, beam_type, tilt_deg):
        pose = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=0.0, t=np.deg2rad(tilt_deg))
        image = _image_at(microscope, beam_type, pose)
        for dy, dz in DELTAS:
            assert _inverse_y_corrected_stage_movement(
                image, dy=dy, dz=dz, beam_type=beam_type
            ) == pytest.approx(
                _shared(microscope, beam_type, pose, dy, dz), abs=1e-15
            ), f"tiled path differs at t={tilt_deg}, {beam_type.name}, ({dy}, {dz})"

    @pytest.mark.parametrize("tilt_deg", TILTS)
    @pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
    def test_the_live_microscope_path_matches(self, microscope, beam_type, tilt_deg):
        """The live path reads the pose off the instrument, so the stage has to be there."""
        pose = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=0.0, t=np.deg2rad(tilt_deg))
        microscope.move_stage_absolute(pose)
        live = microscope.get_stage_position()
        for dy, dz in DELTAS:
            assert microscope._inverse_y_corrected_stage_movement(
                dy=dy, dz=dz, beam_type=beam_type
            ) == pytest.approx(
                _shared(microscope, beam_type, live, dy, dz), abs=1e-15
            ), f"live path differs at t={tilt_deg}, {beam_type.name}, ({dy}, {dz})"

    def test_the_live_path_does_not_read_the_orientation_from_hardware(
        self, microscope
    ):
        """It used to ask `get_stage_orientation()` to decide the compustage FIB pose.

        The shared implementation derives that from the pose it is handed, which is one
        fewer instrument round trip on a path the canvas calls per marker.
        """
        calls = []
        original = type(microscope).get_stage_orientation

        def counted(self, *args, **kwargs):
            calls.append(1)
            return original(self, *args, **kwargs)

        type(microscope).get_stage_orientation = counted
        try:
            microscope._inverse_y_corrected_stage_movement(
                dy=10e-6, dz=5e-6, beam_type=BeamType.ELECTRON
            )
        finally:
            type(microscope).get_stage_orientation = original
        assert calls == [], "the orientation is being read from the instrument again"


class TestTheCompustageFibDivergenceIsClosed:
    """FIB-500. The tiled copy decided the FIB orientation from tilt alone, where the
    other two also require the rotation to match the reference -- so at tilt -128 with a
    non-zero rotation the sign inverted between them."""

    @pytest.mark.parametrize("rotation_deg", [90.0, 180.0, 270.0])
    @pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
    def test_the_paths_agree_at_a_rotated_fib_pose(
        self, microscope, beam_type, rotation_deg
    ):
        pose = FibsemStagePosition(
            x=0.0,
            y=0.0,
            z=0.0,
            r=np.deg2rad(rotation_deg),
            t=np.deg2rad(-128.0),
        )
        image = _image_at(microscope, beam_type, pose)
        assert _inverse_y_corrected_stage_movement(
            image, dy=12e-6, dz=-5e-6, beam_type=beam_type
        ) == pytest.approx(
            _shared(microscope, beam_type, pose, 12e-6, -5e-6), abs=1e-15
        )


class TestARotatedCompustagePoseIsUnreachable:
    """Why closing that divergence needed no hardware confirmation.

    The poses that changed cannot be produced: a compustage has no rotation axis. That
    argument is the whole justification, so it is pinned here rather than left in a
    comment where it can quietly expire.
    """

    def test_a_compustage_has_no_rotation_axis(self):
        from fibsem.microscopes.simulator import STAGE_LIMITS_COMPUSTAGE

        assert "r" not in STAGE_LIMITS_COMPUSTAGE
        assert set(STAGE_LIMITS_COMPUSTAGE) == {"x", "y", "z", "t"}

    def test_a_compustage_position_is_converted_with_rotation_hardcoded_to_zero(self):
        """Asserted through the AST, not by calling it.

        `stage_position_from_autoscript` raises without the AutoScript libraries, which
        neither CI nor a development machine has -- so the only way to hold this is to
        read the source. Located through the module rather than by relative path, which
        would depend on pytest's working directory.
        """
        from fibsem.microscopes import autoscript as autoscript_module

        with io.open(autoscript_module.__file__, encoding="utf-8") as handle:
            source = handle.read()
        tree = ast.parse(source)
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "stage_position_from_autoscript"
        )
        # The CompustagePosition branch is the one guarded by an isinstance check.
        branch = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Call)
            and getattr(node.test.func, "id", None) == "isinstance"
            and any(
                getattr(arg, "id", None) == "CompustagePosition"
                for arg in node.test.args
            )
        )
        call = next(
            node
            for node in ast.walk(branch)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "FibsemStagePosition"
        )
        rotation = next(kw.value for kw in call.keywords if kw.arg == "r")
        assert isinstance(rotation, ast.Constant), (
            "rotation is no longer a literal -- a compustage may now carry one, and the "
            "FIB-500 reachability argument no longer holds"
        )
        assert rotation.value == 0.0
