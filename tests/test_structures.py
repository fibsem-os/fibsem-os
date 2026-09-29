import datetime
import os

import pytest

from fibsem.microscopes.autoscript import THERMO_API_AVAILABLE
from fibsem.structures import (
    AutoFocusMode,
    BeamSettings,
    BeamType,
    FibsemDetectorSettings,
    FibsemGasInjectionSettings,
    FibsemImage,
    FibsemRectangle,
    FibsemStagePosition,
    ImageSettings,
    MicroscopeState,
    OverviewAcquisitionSettings,
    ReferenceImageParameters,
    TileOrderStrategy,
)

# microscope state
# electron_beam, electron_detector, ion_beam, ion_detector are now optional


def test_microscope_state():

    state = MicroscopeState()

    state.to_dict()

    state.electron_beam = None
    state.electron_detector = None

    state.to_dict()

    state.ion_beam = None
    state.ion_detector = None

    state.to_dict()


def test_gas_injection_settings():

    # gis
    gis_settings = FibsemGasInjectionSettings(
        gas="Pt",
        port=0,
        duration=30,
    )

    # to dict
    gdict = gis_settings.to_dict()
    assert gdict["gas"] == gis_settings.gas
    assert gdict["port"] == gis_settings.port
    assert gdict["duration"] == gis_settings.duration
    assert gdict["insert_position"] == gis_settings.insert_position

    # from dict
    gis_settings2 = FibsemGasInjectionSettings.from_dict(gdict)
    assert gis_settings2.gas == gis_settings.gas
    assert gis_settings2.port == gis_settings.port
    assert gis_settings2.duration == gis_settings.duration
    assert gis_settings2.insert_position == gis_settings.insert_position

    multichem_settings = FibsemGasInjectionSettings(
        gas="Pt", port=0, duration=30, insert_position="ELECTRON_DEFAULT"
    )

    # to dict
    gdict = multichem_settings.to_dict()
    assert gdict["gas"] == multichem_settings.gas
    assert gdict["port"] == multichem_settings.port
    assert gdict["duration"] == multichem_settings.duration
    assert gdict["insert_position"] == multichem_settings.insert_position

    # from dict
    multichem_settings2 = FibsemGasInjectionSettings.from_dict(gdict)
    assert multichem_settings2.gas == multichem_settings.gas
    assert multichem_settings2.port == multichem_settings.port
    assert multichem_settings2.duration == multichem_settings.duration
    assert multichem_settings2.insert_position == multichem_settings.insert_position


def test_fibsem_image_extract_region():
    """Test FibsemImage.extract_region returns cropped data with updated reduced_area metadata."""
    image = FibsemImage.generate_blank_image(resolution=(100, 100), hfw=100e-6)
    import numpy as np

    image.data[:] = np.arange(image.data.size, dtype=image.data.dtype).reshape(
        image.data.shape
    )

    rect = FibsemRectangle(left=0.25, top=0.25, width=0.5, height=0.5)
    result = image.extract_region(rect)

    # data shape reflects the crop
    assert result.data.shape == (50, 50)

    # resolution and hfw are unchanged from the original
    assert (
        result.metadata.image_settings.resolution
        == image.metadata.image_settings.resolution
    )
    assert result.metadata.image_settings.hfw == image.metadata.image_settings.hfw

    # pixel size is unchanged
    assert result.metadata.pixel_size.x == image.metadata.pixel_size.x
    assert result.metadata.pixel_size.y == image.metadata.pixel_size.y

    # reduced_area is set to the extracted rect
    assert result.metadata.image_settings.reduced_area == rect


def test_fibsem_image_resize():
    """resize() returns correct shape and updates pixel_size; hfw and resolution are updated."""
    image = FibsemImage.generate_blank_image(resolution=(100, 100), hfw=100e-6)
    orig_px = image.metadata.pixel_size.x  # 100e-6 / 100 = 1e-6

    result = image.resize((50, 50))

    assert result.data.shape == (50, 50)
    assert result.metadata.image_settings.resolution == (50, 50)
    # pixel size doubles when halving resolution at fixed HFW
    assert abs(result.metadata.pixel_size.x - orig_px * 2) < 1e-12
    assert abs(result.metadata.pixel_size.y - orig_px * 2) < 1e-12
    # original is unchanged
    assert image.data.shape == (100, 100)


def test_fibsem_image_resize_no_metadata():
    """resize() raises ValueError when image has no metadata."""
    import numpy as np

    image = FibsemImage(data=np.zeros((100, 100), dtype=np.uint8))
    with pytest.raises(ValueError):
        image.resize((50, 50))


def test_fibsem_image_brightness():
    """brightness property returns mean pixel value."""
    import numpy as np

    data = np.full((10, 10), 128, dtype=np.uint8)
    image = FibsemImage(data=data)
    assert image.brightness == 128.0


def test_fibsem_image_apply_gamma():
    """apply_gamma returns a new image with pixel values adjusted and metadata preserved."""
    import numpy as np

    from fibsem.autofunctions.gamma import apply_gamma

    data = np.full((10, 10), 128, dtype=np.uint8)
    image = FibsemImage.generate_blank_image(resolution=(10, 10), hfw=10e-6)
    image.data[:] = data

    result = image.apply_gamma(1.0)
    assert result.data.shape == image.data.shape
    assert result.data.dtype == image.data.dtype
    # gamma=1 is identity
    assert np.array_equal(result.data, image.data)

    # gamma < 1 should brighten (increase values)
    bright = image.apply_gamma(0.5)
    assert bright.data.mean() > image.data.mean()

    # gamma > 1 should darken (decrease values)
    dark = image.apply_gamma(2.0)
    assert dark.data.mean() < image.data.mean()

    # standalone function raises on invalid gamma
    with pytest.raises(ValueError):
        apply_gamma(data, 0.0)


def test_fibsem_image_extract_region_invalid_rect():
    """extract_region raises ValueError for out-of-bounds rectangles."""
    image = FibsemImage.generate_blank_image(resolution=(100, 100), hfw=100e-6)

    with pytest.raises(ValueError):
        image.extract_region(FibsemRectangle(left=0.8, top=0.0, width=0.5, height=0.5))


def test_fibsem_image_extract_region_no_metadata():
    """extract_region raises ValueError when image has no metadata."""
    import numpy as np

    image = FibsemImage(data=np.zeros((100, 100), dtype=np.uint8))

    with pytest.raises(ValueError):
        image.extract_region(FibsemRectangle(left=0.0, top=0.0, width=0.5, height=0.5))


def test_there_is_only_one_autofocus_settings_now():
    """The mode-only class is gone, and the name means one thing again.

    `fibsem.structures` used to define an `AutoFocusSettings` whose sole member was the
    mode, sharing a name with the real sweep config in `autofunctions.autofocus`.
    `AutoFocusMode` had already been through the identical mistake -- two enums of one
    name, so `NONE is NONE` was False across the two import paths with no type error to
    catch it -- so this asserts the duplicate stayed dead (FIB-646).
    """
    import fibsem.structures as structures
    from fibsem.autofunctions.autofocus import AutoFocusSettings as Canonical

    assert not hasattr(structures, "AutoFocusSettings")
    # And the one that survives is the sweep config, not a mode holder.
    assert not hasattr(Canonical(), "mode")
    assert Canonical().passes


def test_overview_acquisition_settings_defaults():
    s = OverviewAcquisitionSettings()
    assert s.nrows == 3
    assert s.ncols == 3
    # 10%, matching `OverviewParameters` on the fluorescence side. The two defaulted
    # differently for one setting with one meaning, so a run configured on one tab and
    # read on the other started from a different grid (FIB-696).
    assert s.overlap == 0.1
    assert s.focus_stack_settings.enabled is False
    assert s.focus_stack_settings.n_steps == 3
    assert s.focus_stack_settings.auto_focus is True
    assert s.autofocus_mode is AutoFocusMode.NONE
    # The sweep is the library default rather than a pinned copy, so an improvement to
    # the default reaches overviews too.
    from fibsem.autofunctions.autofocus import AutoFocusSettings

    assert s.autofocus_settings.to_dict() == AutoFocusSettings().to_dict()


def test_overview_acquisition_settings_round_trip():
    from fibsem.structures import FocusStackSettings

    for mode in AutoFocusMode:
        s = OverviewAcquisitionSettings(
            image_settings=ImageSettings(resolution=(1536, 1024), hfw=150e-6),
            nrows=2,
            ncols=3,
            overlap=0.1,
            focus_stack_settings=FocusStackSettings(
                enabled=True, n_steps=5, auto_focus=False
            ),
            autofocus_mode=mode,
        )
        restored = OverviewAcquisitionSettings.from_dict(s.to_dict())
        assert restored.nrows == s.nrows
        assert restored.ncols == s.ncols
        assert restored.overlap == s.overlap
        assert restored.focus_stack_settings.enabled == s.focus_stack_settings.enabled
        assert restored.focus_stack_settings.n_steps == s.focus_stack_settings.n_steps
        assert (
            restored.focus_stack_settings.auto_focus
            == s.focus_stack_settings.auto_focus
        )
        assert restored.autofocus_mode is mode


def test_overview_acquisition_settings_from_dict_legacy():
    """Dicts without autofocus_settings key (old format) default to NONE."""
    d = {
        "image_settings": ImageSettings().to_dict(),
        "nrows": 3,
        "ncols": 3,
        "overlap": 0.0,
        "use_focus_stack": False,
        # no autofocus_settings key, no focus_stack_settings key (old format)
    }
    s = OverviewAcquisitionSettings.from_dict(d)
    assert s.autofocus_mode is AutoFocusMode.NONE
    assert s.focus_stack_settings.enabled is False


def test_a_file_written_before_the_split_still_loads_its_mode():
    """The shape this replaced put the mode *inside* `autofocus_settings`.

    Every protocol and overview-parameters file written before FIB-646 looks like this,
    and the mode is the only thing in them worth keeping -- so losing it silently would
    turn a saved EACH_TILE run into a run that never focuses, which produces a plausible
    mosaic rather than an error.
    """
    from fibsem.autofunctions.autofocus import AutoFocusSettings

    for mode in AutoFocusMode:
        d = {
            "image_settings": ImageSettings().to_dict(),
            "nrows": 3,
            "ncols": 3,
            "overlap": 0.1,
            "autofocus_settings": {"mode": mode.value},
        }
        s = OverviewAcquisitionSettings.from_dict(d)
        assert s.autofocus_mode is mode
        # An old file carries no sweep, so it gets the default -- which is what it was
        # running under anyway, the mode having been all it could configure.
        assert s.autofocus_settings.to_dict() == AutoFocusSettings().to_dict()


def test_an_explicit_null_autofocus_settings_loads():
    """A third shape, and not a hypothetical one -- it is what real runs wrote.

    `overview-parameters.json` under a log directory contains
    `"autofocus_settings": null`, so the key is present and empty rather than absent.
    A `d.get("autofocus_settings", {})` would hand `None` to the branch below and raise
    on `"mode" in None`; `or {}` is what makes the present-but-null case behave like the
    absent one.
    """
    d = {
        "image_settings": ImageSettings().to_dict(),
        "nrows": 3,
        "ncols": 3,
        "overlap": 0.1,
        "autofocus_settings": None,
    }
    s = OverviewAcquisitionSettings.from_dict(d)
    assert s.autofocus_mode is AutoFocusMode.NONE
    assert s.autofocus_settings.passes


def test_the_old_and_new_shapes_are_told_apart_by_a_key_the_new_one_cannot_write():
    """`"mode"` is the discriminator, and this pins why that is safe rather than lucky.

    The sweep config writes method / passes / probe_resolution / probe_dwell_time /
    reduced_area / use_autocontrast / channel_name. If it ever grows a `mode` key, every
    new file starts being read as an old one -- silently, and as `AutoFocusMode(...)` of
    something that is not a mode.
    """
    from fibsem.autofunctions.autofocus import AutoFocusSettings

    assert "mode" not in AutoFocusSettings().to_dict()


def test_a_new_shape_file_keeps_its_sweep():
    """The other direction: a real sweep must survive, not be replaced by the default."""
    from fibsem.autofunctions.autofocus import AutoFocusSettings, FocusMethod

    sweep = AutoFocusSettings.from_coarse_fine(
        coarse_range=123e-6,
        coarse_step=7e-6,
        fine_enabled=False,
        method=FocusMethod.SOBEL,
    )
    s = OverviewAcquisitionSettings(
        autofocus_mode=AutoFocusMode.EACH_TILE, autofocus_settings=sweep
    )
    restored = OverviewAcquisitionSettings.from_dict(s.to_dict())

    assert restored.autofocus_mode is AutoFocusMode.EACH_TILE
    assert restored.autofocus_settings.method is FocusMethod.SOBEL
    assert restored.autofocus_settings.passes[0].search_range == 123e-6
    assert restored.autofocus_settings.passes[1].enabled is False


def test_overview_acquisition_settings_from_dict_legacy_focus_stack_enabled():
    """Old use_focus_stack=True maps to focus_stack_settings.enabled=True."""
    d = {
        "image_settings": ImageSettings().to_dict(),
        "nrows": 3,
        "ncols": 3,
        "overlap": 0.0,
        "use_focus_stack": True,
    }
    s = OverviewAcquisitionSettings.from_dict(d)
    assert s.focus_stack_settings.enabled is True


def test_overview_acquisition_total_fov_square_no_overlap():
    """Square tiles, no overlap: total FOV == n * hfw."""
    s = OverviewAcquisitionSettings(
        image_settings=ImageSettings(resolution=(1024, 1024), hfw=100e-6),
        nrows=3,
        ncols=4,
        overlap=0.0,
    )
    assert s.total_fov_x == pytest.approx(4 * 100e-6)
    assert s.total_fov_y == pytest.approx(3 * 100e-6)


def test_overview_acquisition_total_fov_with_overlap():
    """Overlap reduces effective step; total FOV = (n-1)*step + hfw."""
    s = OverviewAcquisitionSettings(
        image_settings=ImageSettings(resolution=(1024, 1024), hfw=100e-6),
        nrows=3,
        ncols=3,
        overlap=0.1,
    )
    step = 100e-6 * 0.9
    expected = 2 * step + 100e-6
    assert s.total_fov_x == pytest.approx(expected)
    assert s.total_fov_y == pytest.approx(expected)


def test_overview_acquisition_total_fov_non_square():
    """Non-square tiles: total_fov_y scaled by aspect ratio."""
    s = OverviewAcquisitionSettings(
        image_settings=ImageSettings(resolution=(1536, 1024), hfw=150e-6),
        nrows=3,
        ncols=3,
        overlap=0.1,
    )
    step_x = 150e-6 * 0.9
    tile_fov_y = 150e-6 * (1024 / 1536)
    step_y = tile_fov_y * 0.9
    assert s.total_fov_x == pytest.approx(2 * step_x + 150e-6)
    assert s.total_fov_y == pytest.approx(2 * step_y + tile_fov_y)


def test_overview_acquisition_total_fov_single_tile():
    """Single tile (1x1): total FOV equals tile FOV regardless of overlap."""
    s = OverviewAcquisitionSettings(
        image_settings=ImageSettings(resolution=(1024, 1024), hfw=200e-6),
        nrows=1,
        ncols=1,
        overlap=0.2,
    )
    assert s.total_fov_x == pytest.approx(200e-6)
    assert s.total_fov_y == pytest.approx(200e-6)


def test_overview_acquisition_tile_order_round_trip():
    for strategy in TileOrderStrategy:
        s = OverviewAcquisitionSettings(tile_order=strategy)
        restored = OverviewAcquisitionSettings.from_dict(s.to_dict())
        assert restored.tile_order is strategy


def test_overview_acquisition_tile_order_legacy_default():
    """Dicts without tile_order key default to TYPEWRITER."""
    d = {
        "image_settings": ImageSettings().to_dict(),
        "nrows": 3,
        "ncols": 3,
        "overlap": 0.0,
        "use_focus_stack": False,
    }
    s = OverviewAcquisitionSettings.from_dict(d)
    assert s.tile_order is TileOrderStrategy.TYPEWRITER


def test_scan_time_is_dwell_over_every_pixel():
    """The whole of it: dwell time, times width, times height."""
    settings = ImageSettings(resolution=(1024, 1024), dwell_time=1e-6)
    assert settings.scan_time == pytest.approx(1.048576)


def test_scan_time_counts_the_passes_each_pixel_gets():
    """Line and frame integration both re-scan, so both multiply.

    `scan_interlacing` changes the order lines are visited in, not how many are
    visited -- a scan time that grew with it would be wrong by that factor.
    """
    base = ImageSettings(resolution=(512, 512), dwell_time=2e-6)
    assert base.scan_time == pytest.approx(0.524288)

    lines = ImageSettings(resolution=(512, 512), dwell_time=2e-6, line_integration=4)
    assert lines.scan_time == pytest.approx(base.scan_time * 4)

    frames = ImageSettings(resolution=(512, 512), dwell_time=2e-6, frame_integration=3)
    assert frames.scan_time == pytest.approx(base.scan_time * 3)

    both = ImageSettings(
        resolution=(512, 512), dwell_time=2e-6, line_integration=4, frame_integration=3
    )
    assert both.scan_time == pytest.approx(base.scan_time * 12)

    interlaced = ImageSettings(
        resolution=(512, 512), dwell_time=2e-6, scan_interlacing=8
    )
    assert interlaced.scan_time == pytest.approx(base.scan_time)


def test_overview_scan_time_counts_the_tiles_it_will_acquire():
    """The enabled ones, not the grid's shape.

    A masked run scans only what is selected; quoting the full grid would overstate a
    typical sparse selection roughly threefold -- and this number is shown to someone
    deciding whether to start it.
    """
    image = ImageSettings(resolution=(1024, 1024), dwell_time=1e-6)
    dense = OverviewAcquisitionSettings(image_settings=image, nrows=3, ncols=3)
    assert dense.scan_time == pytest.approx(image.scan_time * 9)

    sparse = OverviewAcquisitionSettings(
        image_settings=image,
        nrows=3,
        ncols=3,
        tile_mask=[[True, False, False], [False, True, False], [False, False, True]],
    )
    assert sparse.scan_time == pytest.approx(image.scan_time * 3)


if THERMO_API_AVAILABLE:
    from fibsem.structures import CompustagePosition, CoordinateSystem, StagePosition

    def test_to_autoscript_position():

        stage_position = FibsemStagePosition(
            x=1, y=2, z=3, r=4, t=5, coordinate_system="RAW"
        )

        # test conversion to StagePosition
        autoscript_stage_position = stage_position.to_autoscript_position()

        assert autoscript_stage_position.x == stage_position.x
        assert autoscript_stage_position.y == stage_position.y
        assert autoscript_stage_position.z == stage_position.z
        assert autoscript_stage_position.r == stage_position.r
        assert autoscript_stage_position.t == stage_position.t
        assert autoscript_stage_position.coordinate_system == CoordinateSystem.RAW

        # test convesion to CompuStagePosition
        autoscript_compustage_position = stage_position.to_autoscript_position(
            compustage=True
        )

        assert autoscript_compustage_position.x == stage_position.x
        assert autoscript_compustage_position.y == stage_position.y
        assert autoscript_compustage_position.z == stage_position.z
        assert autoscript_compustage_position.a == stage_position.t
        assert (
            autoscript_compustage_position.coordinate_system
            == CoordinateSystem.SPECIMEN
        )

    def test_from_autoscript_position():

        autoscript_stage_position = StagePosition(
            x=1, y=2, z=3, r=4, t=5, coordinate_system=CoordinateSystem.RAW
        )
        autoscript_compustage_position = CompustagePosition(
            x=1, y=2, z=3, a=5, coordinate_system=CoordinateSystem.RAW
        )

        # test conversion from StagePosition
        stage_position = FibsemStagePosition.from_autoscript_position(
            autoscript_stage_position
        )

        assert stage_position.x == autoscript_stage_position.x
        assert stage_position.y == autoscript_stage_position.y
        assert stage_position.z == autoscript_stage_position.z
        assert stage_position.r == autoscript_stage_position.r
        assert stage_position.t == autoscript_stage_position.t
        assert stage_position.coordinate_system == "RAW"

        # test conversion from CompuStagePosition
        stage_position = FibsemStagePosition.from_autoscript_position(
            autoscript_compustage_position
        )

        assert stage_position.x == autoscript_compustage_position.x
        assert stage_position.y == autoscript_compustage_position.y
        assert stage_position.z == autoscript_compustage_position.z
        assert stage_position.r == 0
        assert stage_position.t == autoscript_compustage_position.a
        assert stage_position.coordinate_system == "SPECIMEN"


# ── ImageSettings.estimated_time ─────────────────────────────────────────────


def test_image_settings_estimated_time_basic():
    img = ImageSettings(resolution=(1536, 1024), dwell_time=1e-6)
    expected = 1536 * 1024 * 1e-6
    assert img.estimated_time == pytest.approx(expected)


def test_image_settings_estimated_time_frame_integration():
    img = ImageSettings(resolution=(1536, 1024), dwell_time=1e-6, frame_integration=4)
    expected = 1536 * 1024 * 1e-6 * 4
    assert img.estimated_time == pytest.approx(expected)


def test_image_settings_estimated_time_line_integration():
    img = ImageSettings(resolution=(1536, 1024), dwell_time=1e-6, line_integration=2)
    expected = 1536 * 1024 * 1e-6 * 2
    assert img.estimated_time == pytest.approx(expected)


def test_image_settings_estimated_time_both_integrations():
    img = ImageSettings(
        resolution=(1536, 1024),
        dwell_time=1e-6,
        frame_integration=4,
        line_integration=2,
    )
    expected = 1536 * 1024 * 1e-6 * 4 * 2
    assert img.estimated_time == pytest.approx(expected)


# ── ReferenceImageParameters.estimated_time ───────────────────────────────────


def test_reference_image_parameters_estimated_time_both_beams_both_fovs():
    img = ImageSettings(resolution=(1536, 1024), dwell_time=1e-6)
    ref = ReferenceImageParameters(
        imaging=img,
        acquire_sem=True,
        acquire_fib=True,
        acquire_image1=True,
        acquire_image2=True,
    )
    # 2 FOVs × 2 beams = 4 images
    assert ref.estimated_time == pytest.approx(img.estimated_time * 4)


def test_reference_image_parameters_estimated_time_sem_only():
    img = ImageSettings(resolution=(1536, 1024), dwell_time=1e-6)
    ref = ReferenceImageParameters(
        imaging=img,
        acquire_sem=True,
        acquire_fib=False,
        acquire_image1=True,
        acquire_image2=True,
    )
    # 2 FOVs × 1 beam = 2 images
    assert ref.estimated_time == pytest.approx(img.estimated_time * 2)


def test_reference_image_parameters_estimated_time_one_fov():
    img = ImageSettings(resolution=(1536, 1024), dwell_time=1e-6)
    ref = ReferenceImageParameters(
        imaging=img,
        acquire_sem=True,
        acquire_fib=True,
        acquire_image1=True,
        acquire_image2=False,
    )
    # 1 FOV × 2 beams = 2 images
    assert ref.estimated_time == pytest.approx(img.estimated_time * 2)


def test_reference_image_parameters_estimated_time_no_images():
    img = ImageSettings(resolution=(1536, 1024), dwell_time=1e-6)
    ref = ReferenceImageParameters(
        imaging=img,
        acquire_sem=False,
        acquire_fib=False,
        acquire_image1=True,
        acquire_image2=True,
    )
    assert ref.estimated_time == 0.0


# FibsemImage.filepath — the file an image is associated with on disk


def test_fibsem_image_filepath_is_none_until_written_or_read():
    """An image that has never been saved or loaded has no file."""
    import numpy as np

    image = FibsemImage(data=np.zeros((10, 10), dtype=np.uint8))
    assert image.filepath is None


def test_fibsem_image_save_sets_filepath_to_the_resolved_file(tmp_path):
    """save() records the path it actually wrote, extension included."""
    import numpy as np

    image = FibsemImage(data=np.zeros((10, 10), dtype=np.uint8))

    # deliberately passed without a suffix: save() resolves it to .tif, and the
    # recorded path must be the resolved one, not the argument.
    returned = image.save(str(tmp_path / "ref_image"))

    assert image.filepath == returned
    assert image.filepath.endswith(".tif")
    assert os.path.isfile(image.filepath)


def test_fibsem_image_load_sets_filepath(tmp_path):
    """load() records where the image came from."""
    import numpy as np

    written = FibsemImage(data=np.zeros((10, 10), dtype=np.uint8)).save(
        str(tmp_path / "ref_image")
    )

    loaded = FibsemImage.load(written)

    assert loaded.filepath == written


# fibsem_revision — the running commit, recorded alongside fibsem_version


def test_only_system_info_carries_the_software_versions():
    """SystemInfo owns what software is running; the experiment reference carries
    identity only. They both did until FIB-448, with no rule about which won if
    they disagreed -- and they could, being built at different moments."""
    from fibsem.structures import FibsemExperimentRef, SystemInfo

    ref = FibsemExperimentRef().to_dict()
    info = SystemInfo.from_dict({}).to_dict()

    for key in ("fibsem_version", "fibsem_revision", "application"):
        assert key in info, f"SystemInfo should own {key}"
        assert key not in ref, f"the experiment reference should not duplicate {key}"

    # Identity, at both rates: which experiment (set once at registration) and where
    # in it (set per task, FIB-466). Nothing about the software or the instrument.
    assert set(ref) == {
        "id",
        "name",
        "date",
        "item_id",
        "item_name",
        "task_id",
        "task_name",
    }
    # Neither, since v5: it was declared in both and populated in neither. An
    # application inside fibsem has no version of its own, and fibsem_revision
    # already pins the commit doing the work.
    assert "application_version" not in info
    assert "application_version" not in ref


def test_metadata_from_dict_accepts_pre_change_files():
    """Saved data written before fibsem_revision existed must still load."""
    from fibsem.structures import FibsemExperimentRef, SystemInfo

    legacy_experiment = {
        "id": "exp-1",
        "method": "autolamella",
        "date": 1700000000.0,
        "application": "fibsemOS",
        "fibsem_version": "0.5.1",
        "application_version": "0.5.1",
    }
    # Keys this build no longer reads must not stop the file loading. The versions
    # are still in the file; a reader wanting them looks at system.info, or at the
    # raw dict for a file old enough to lack one. See FIB-448.
    experiment = FibsemExperimentRef.from_dict(legacy_experiment)
    assert experiment.id == "exp-1"
    assert experiment.date == 1700000000.0

    # `application_version` is here because SystemInfo declared it up to v4. It is
    # constructed field by field, so a key it no longer knows is ignored rather than
    # raising -- which is what makes dropping the field safe for existing files.
    legacy_info = {
        "name": "Test",
        "manufacturer": "Demo",
        "fibsem_version": "0.5.1",
        "application_version": "0.5.1",
    }
    info = SystemInfo.from_dict(legacy_info)
    assert info.name == "Test"
    assert info.fibsem_version == "0.5.1"


def test_metadata_round_trips_fibsem_revision():
    from fibsem.structures import SystemInfo

    info = SystemInfo.from_dict({"fibsem_revision": "v0.5.1-48-g4cd11d9c"})
    assert info.fibsem_revision == "v0.5.1-48-g4cd11d9c"
    assert SystemInfo.from_dict(info.to_dict()).fibsem_revision == "v0.5.1-48-g4cd11d9c"


def test_experiment_date_is_creation_time_not_import_time():
    """A plain dataclass default would freeze this at module-import time."""
    import time

    from fibsem.structures import FibsemExperimentRef

    before = datetime.datetime.timestamp(datetime.datetime.now())
    time.sleep(0.01)
    experiment = FibsemExperimentRef()
    time.sleep(0.01)
    after = datetime.datetime.timestamp(datetime.datetime.now())

    assert before < experiment.date < after
