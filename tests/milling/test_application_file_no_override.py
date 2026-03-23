"""
Test suite: Application file is NOT overridden by draw_rectangle().

Background
----------
On ThermoFisher Scios microscopes (and potentially other models), the
application file "Si-ccs" is not installed — or has a different name such
as "Si-ccs New".  Previously, ``ThermoMicroscope.draw_rectangle()`` and
``OdemisMicroscope.draw_rectangle()`` **unconditionally overrode** the
application file to "Si-ccs" (for ``CleaningCrossSection`` patterns) or
"Si-multipass" (for ``RegularCrossSection`` patterns), ignoring whatever
the user configured in ``FibsemMillingSettings.application_file``.

This caused two problems:

1. **Hard failure**: On microscopes without "Si-ccs" and no fuzzy match,
   a ``ValueError`` was raised (application file not available).
2. **Silent wrong match**: On microscopes with "Si-ccs New" (Scios),
   ``difflib.get_close_matches`` resolved "Si-ccs" → "Si-ccs New" — a
   possibly unintended application file with different dose/dwell
   parameters.

The fix removes the hardcoded ``set_application_file()`` calls from
``draw_rectangle()`` so that the user's configured application file
(set via ``setup_milling()``) is respected. Legacy application file names
are retained only as **fallbacks** (try/except), mirroring the pattern
already used by ``draw_circle()`` and ``draw_bitmap_pattern()``.

These tests verify that:

- ``setup_milling()`` correctly sets the application file.
- ``draw_rectangle()`` does NOT override the application file, regardless
  of cross-section type.
- The full ``mill_stages()`` pipeline respects non-default application
  files for cleaning-cross-section patterns (the most common case).
"""

import pytest

from fibsem import utils
from fibsem.milling import FibsemMillingStage, mill_stages
from fibsem.milling.patterning.patterns2 import RectanglePattern, TrenchPattern
from fibsem.structures import (
    CrossSectionPattern,
    FibsemMillingSettings,
    FibsemRectangleSettings,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def demo_microscope():
    """Create a DemoMicroscope (simulator) for testing."""
    microscope, settings = utils.setup_session(manufacturer="Demo")
    return microscope


# ---------------------------------------------------------------------------
# Unit tests: setup_milling sets the application file
# ---------------------------------------------------------------------------


class TestSetupMillingSetsApplicationFile:
    """Verify setup_milling() stores the user's application file."""

    def test_default_application_file(self, demo_microscope):
        """Default application_file='Si' should be set after setup_milling."""
        mill_settings = FibsemMillingSettings()
        assert mill_settings.application_file == "Si"

        demo_microscope.setup_milling(mill_settings)
        assert demo_microscope.milling_system.default_application_file == "Si"

    def test_custom_application_file(self, demo_microscope):
        """A non-default application file should propagate through setup_milling."""
        mill_settings = FibsemMillingSettings(application_file="Si-ccs")
        demo_microscope.setup_milling(mill_settings)
        assert demo_microscope.milling_system.default_application_file == "Si-ccs"

    def test_custom_application_file_multipass(self, demo_microscope):
        mill_settings = FibsemMillingSettings(application_file="Si-multipass")
        demo_microscope.setup_milling(mill_settings)
        assert demo_microscope.milling_system.default_application_file == "Si-multipass"


# ---------------------------------------------------------------------------
# Unit tests: draw_rectangle does NOT override the application file
# ---------------------------------------------------------------------------


class TestDrawRectangleNoApplicationFileOverride:
    """
    Verify draw_rectangle() does not change the application file.

    The DemoMicroscope's draw_rectangle is a simple passthrough (no
    application file logic), so these tests confirm the *contract*: the
    application file set by setup_milling() must be the same before and
    after draw_rectangle(), for all cross-section types.
    """

    @pytest.mark.parametrize(
        "cross_section",
        [
            CrossSectionPattern.Rectangle,
            CrossSectionPattern.CleaningCrossSection,
            CrossSectionPattern.RegularCrossSection,
        ],
        ids=["Rectangle", "CleaningCrossSection", "RegularCrossSection"],
    )
    def test_application_file_unchanged_after_draw(
        self, demo_microscope, cross_section
    ):
        """
        After draw_rectangle() with any cross-section type, the
        application file must remain the user-configured value.
        """
        custom_app_file = "Si"
        mill_settings = FibsemMillingSettings(application_file=custom_app_file)
        demo_microscope.setup_milling(mill_settings)

        pattern = FibsemRectangleSettings(
            width=10e-6,
            height=5e-6,
            depth=1e-6,
            centre_x=0,
            centre_y=0,
            rotation=0,
            cross_section=cross_section,
        )

        demo_microscope.draw_rectangle(pattern)

        # The application file must NOT have been overridden.
        assert (
            demo_microscope.milling_system.default_application_file
            == custom_app_file
        ), (
            f"Expected application file '{custom_app_file}' but got "
            f"'{demo_microscope.milling_system.default_application_file}' "
            f"after draw_rectangle with cross_section={cross_section.name}. "
            f"draw_rectangle() should not override the user's application file."
        )


# ---------------------------------------------------------------------------
# Integration tests: full mill_stages pipeline
# ---------------------------------------------------------------------------


class TestMillStagesRespectsApplicationFile:
    """
    End-to-end tests through mill_stages() to verify that custom
    application files propagate correctly when using cross-section
    patterns (CleaningCrossSection / RegularCrossSection).
    """

    def test_mill_stages_with_cleaning_cross_section_and_default_app_file(
        self, demo_microscope
    ):
        """
        A milling stage using CleaningCrossSection with the default 'Si'
        application file should complete without errors.

        This simulates a Scios-style workflow where "Si-ccs" is NOT
        available and the user configures their protocol with
        application_file='Si'.
        """
        stage = FibsemMillingStage(name="test-ccs-default-app-file")
        stage.milling = FibsemMillingSettings(
            application_file="Si",
            milling_current=20e-12,
        )
        stage.pattern = RectanglePattern(
            width=10e-6,
            height=5e-6,
            depth=1e-6,
            cross_section=CrossSectionPattern.CleaningCrossSection,
        )
        # Should NOT raise ValueError about missing "Si-ccs".
        mill_stages(demo_microscope, [stage])

    def test_mill_stages_with_cleaning_cross_section_and_si_ccs(
        self, demo_microscope
    ):
        """
        A milling stage using CleaningCrossSection with 'Si-ccs' (the
        traditional Aquilos/Arctis application file) should also work.
        """
        stage = FibsemMillingStage(name="test-ccs-si-ccs")
        stage.milling = FibsemMillingSettings(
            application_file="Si-ccs",
            milling_current=20e-12,
        )
        stage.pattern = RectanglePattern(
            width=10e-6,
            height=5e-6,
            depth=1e-6,
            cross_section=CrossSectionPattern.CleaningCrossSection,
        )
        mill_stages(demo_microscope, [stage])

    def test_mill_stages_with_regular_cross_section_and_default_app_file(
        self, demo_microscope
    ):
        """
        RegularCrossSection with default 'Si' application file should
        work without requiring "Si-multipass".
        """
        stage = FibsemMillingStage(name="test-rcs-default")
        stage.milling = FibsemMillingSettings(
            application_file="Si",
            milling_current=20e-12,
        )
        stage.pattern = RectanglePattern(
            width=10e-6,
            height=5e-6,
            depth=1e-6,
            cross_section=CrossSectionPattern.RegularCrossSection,
        )
        mill_stages(demo_microscope, [stage])

    def test_mill_stages_trench_with_cleaning_cross_section(
        self, demo_microscope
    ):
        """
        TrenchPattern with CleaningCrossSection passes the cross_section
        to both sub-rectangles. Verify the full pipeline works with a
        non-"Si-ccs" application file.
        """
        stage = FibsemMillingStage(name="test-trench-ccs")
        stage.milling = FibsemMillingSettings(
            application_file="Si",
            milling_current=20e-12,
        )
        stage.pattern = TrenchPattern(
            width=10e-6,
            upper_trench_height=3e-6,
            lower_trench_height=3e-6,
            depth=1e-6,
            cross_section=CrossSectionPattern.CleaningCrossSection,
        )
        mill_stages(demo_microscope, [stage])

    def test_mill_stages_multiple_stages_mixed_app_files(
        self, demo_microscope
    ):
        """
        Multiple milling stages with different application files should
        each use their own configured value.

        This simulates a realistic lamella protocol:
          Stage 1: Coarse trench removal    -> "Si" + Rectangle
          Stage 2: Rough cleaning pass      -> "Si-ccs" + CleaningCrossSection
          Stage 3: Fine polishing pass      -> "Si-ccs" + CleaningCrossSection
        """
        coarse = FibsemMillingStage(name="coarse")
        coarse.milling = FibsemMillingSettings(
            application_file="Si",
            milling_current=7.6e-9,
        )
        coarse.pattern = RectanglePattern(
            width=20e-6,
            height=10e-6,
            depth=5e-6,
        )

        rough = FibsemMillingStage(name="rough")
        rough.milling = FibsemMillingSettings(
            application_file="Si-ccs",
            milling_current=740e-12,
        )
        rough.pattern = RectanglePattern(
            width=15e-6,
            height=1e-6,
            depth=0.5e-6,
            cross_section=CrossSectionPattern.CleaningCrossSection,
        )

        polish = FibsemMillingStage(name="polish")
        polish.milling = FibsemMillingSettings(
            application_file="Si-ccs",
            milling_current=60e-12,
        )
        polish.pattern = RectanglePattern(
            width=15e-6,
            height=0.3e-6,
            depth=0.2e-6,
            cross_section=CrossSectionPattern.CleaningCrossSection,
        )

        mill_stages(demo_microscope, [coarse, rough, polish])


# ---------------------------------------------------------------------------
# Unit tests: application file name resolution (prefix / fuzzy matching)
# ---------------------------------------------------------------------------


# Scios-style application files: "Si New", "Si-ccs New", "Si-multipass New"
SCIOS_APPLICATION_FILES = [
    "Si New",
    "Si-multipass New",
    "Si-ccs New",
    "autolamella",
    "cryo_Pt_dep",
]


@pytest.fixture
def scios_microscope():
    """Create a DemoMicroscope with Scios-style application file names.

    On a real Scios, the application files have " New" appended to their
    names (e.g. "Si New" instead of "Si").  This fixture simulates that
    by replacing the simulator's application file list.
    """
    microscope, settings = utils.setup_session(manufacturer="Demo")
    microscope.milling_system.application_files = list(SCIOS_APPLICATION_FILES)
    return microscope


class TestApplicationFileResolution:
    """Verify get_application_file resolves standard names to Scios names.

    On Scios microscopes, application files are named "Si New",
    "Si-ccs New", etc. instead of the standard "Si", "Si-ccs" used in
    protocols.  The resolution chain in get_application_file() must
    handle this transparently so existing protocols work without changes.
    """

    # --- Exact matches (should work on any microscope) ---

    def test_exact_match(self, scios_microscope):
        """An exact match should be returned as-is."""
        resolved = scios_microscope.set_default_application_file(
            "Si New", strict=False
        )
        assert resolved == "Si New"

    # --- Prefix matches (the core Scios fix) ---

    def test_prefix_match_si(self, scios_microscope):
        """'Si' should resolve to 'Si New' via prefix match."""
        resolved = scios_microscope.set_default_application_file(
            "Si", strict=False
        )
        assert resolved == "Si New"

    def test_prefix_match_si_ccs(self, scios_microscope):
        """'Si-ccs' should resolve to 'Si-ccs New' via prefix match."""
        resolved = scios_microscope.set_default_application_file(
            "Si-ccs", strict=False
        )
        assert resolved == "Si-ccs New"

    def test_prefix_match_si_multipass(self, scios_microscope):
        """'Si-multipass' should resolve to 'Si-multipass New' via prefix."""
        resolved = scios_microscope.set_default_application_file(
            "Si-multipass", strict=False
        )
        assert resolved == "Si-multipass New"

    def test_prefix_match_picks_shortest(self, scios_microscope):
        """When multiple files share a prefix, the shortest wins.

        For example, if both 'Si New' and 'Si New Extended' existed,
        requesting 'Si' should resolve to 'Si New' (shortest).
        """
        scios_microscope.milling_system.application_files = [
            "Si New Extended",
            "Si New",
            "Si-ccs New",
        ]
        resolved = scios_microscope.set_default_application_file(
            "Si", strict=False
        )
        assert resolved == "Si New"

    # --- Strict mode ---

    def test_strict_mode_rejects_non_exact(self, scios_microscope):
        """strict=True should reject 'Si' when only 'Si New' exists."""
        with pytest.raises(ValueError, match="not available"):
            scios_microscope.set_default_application_file(
                "Si", strict=True
            )

    # --- No match ---

    def test_no_match_raises(self, scios_microscope):
        """A completely unrelated name should raise ValueError."""
        with pytest.raises(ValueError, match="not available"):
            scios_microscope.set_default_application_file(
                "NonExistentFile", strict=False
            )

    # --- setup_milling integration ---

    def test_setup_milling_resolves_si_on_scios(self, scios_microscope):
        """setup_milling with 'Si' should resolve to 'Si New' on Scios.

        Previously setup_milling used strict=True, causing a ValueError.
        Now it uses strict=False with prefix matching.
        """
        mill_settings = FibsemMillingSettings(application_file="Si")
        scios_microscope.setup_milling(mill_settings)
        assert scios_microscope.milling_system.default_application_file == "Si New"

    def test_setup_milling_resolves_si_ccs_on_scios(self, scios_microscope):
        """setup_milling with 'Si-ccs' should resolve to 'Si-ccs New'."""
        mill_settings = FibsemMillingSettings(application_file="Si-ccs")
        scios_microscope.setup_milling(mill_settings)
        assert scios_microscope.milling_system.default_application_file == "Si-ccs New"

    def test_mill_stages_full_pipeline_scios(self, scios_microscope):
        """Full mill_stages pipeline with Scios-style app file names.

        A realistic lamella protocol using standard names ("Si", "Si-ccs")
        should work on a Scios where only "Si New" and "Si-ccs New" exist.
        """
        coarse = FibsemMillingStage(name="coarse")
        coarse.milling = FibsemMillingSettings(
            application_file="Si",
            milling_current=7.6e-9,
        )
        coarse.pattern = RectanglePattern(
            width=20e-6, height=10e-6, depth=5e-6,
        )

        rough = FibsemMillingStage(name="rough")
        rough.milling = FibsemMillingSettings(
            application_file="Si-ccs",
            milling_current=740e-12,
        )
        rough.pattern = RectanglePattern(
            width=15e-6, height=1e-6, depth=0.5e-6,
            cross_section=CrossSectionPattern.CleaningCrossSection,
        )

        # Should complete without ValueError.
        mill_stages(scios_microscope, [coarse, rough])
