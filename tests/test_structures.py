import datetime

import pytest

from fibsem.structures import (
    THERMO_API_AVAILABLE,
    AutoFocusMode,
    AutoFocusSettings,
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
        gas="Pt",
        port=0,
        duration=30,
        insert_position="ELECTRON_DEFAULT"
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
    image.data[:] = np.arange(image.data.size, dtype=image.data.dtype).reshape(image.data.shape)

    rect = FibsemRectangle(left=0.25, top=0.25, width=0.5, height=0.5)
    result = image.extract_region(rect)

    # data shape reflects the crop
    assert result.data.shape == (50, 50)

    # resolution and hfw are unchanged from the original
    assert result.metadata.image_settings.resolution == image.metadata.image_settings.resolution
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
    from fibsem.imaging.autogamma import apply_gamma
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


def test_autofocus_settings_defaults():
    s = AutoFocusSettings()
    assert s.mode is AutoFocusMode.NONE


def test_autofocus_settings_round_trip():
    for mode in AutoFocusMode:
        s = AutoFocusSettings(mode=mode)
        d = s.to_dict()
        assert d["mode"] == mode.name
        restored = AutoFocusSettings.from_dict(d)
        assert restored.mode is mode


def test_autofocus_settings_from_dict_missing_key():
    """Missing key in dict falls back to NONE."""
    s = AutoFocusSettings.from_dict({})
    assert s.mode is AutoFocusMode.NONE


def test_overview_acquisition_settings_defaults():
    s = OverviewAcquisitionSettings()
    assert s.nrows == 3
    assert s.ncols == 3
    assert s.overlap == 0.0
    assert s.use_focus_stack is False
    assert s.autofocus_settings.mode is AutoFocusMode.NONE


def test_overview_acquisition_settings_round_trip():
    for mode in AutoFocusMode:
        s = OverviewAcquisitionSettings(
            image_settings=ImageSettings(resolution=(1536, 1024), hfw=150e-6),
            nrows=2, ncols=3, overlap=0.1, use_focus_stack=True,
            autofocus_settings=AutoFocusSettings(mode=mode),
        )
        restored = OverviewAcquisitionSettings.from_dict(s.to_dict())
        assert restored.nrows == s.nrows
        assert restored.ncols == s.ncols
        assert restored.overlap == s.overlap
        assert restored.use_focus_stack == s.use_focus_stack
        assert restored.autofocus_settings.mode is mode


def test_overview_acquisition_settings_from_dict_legacy():
    """Dicts without autofocus_settings key (old format) default to NONE."""
    d = {
        "image_settings": ImageSettings().to_dict(),
        "nrows": 3, "ncols": 3, "overlap": 0.0, "use_focus_stack": False,
        # no autofocus_settings key
    }
    s = OverviewAcquisitionSettings.from_dict(d)
    assert s.autofocus_settings.mode is AutoFocusMode.NONE


def test_overview_acquisition_total_fov_square_no_overlap():
    """Square tiles, no overlap: total FOV == n * hfw."""
    s = OverviewAcquisitionSettings(
        image_settings=ImageSettings(resolution=(1024, 1024), hfw=100e-6),
        nrows=3, ncols=4, overlap=0.0,
    )
    assert s.total_fov_x == pytest.approx(4 * 100e-6)
    assert s.total_fov_y == pytest.approx(3 * 100e-6)


def test_overview_acquisition_total_fov_with_overlap():
    """Overlap reduces effective step; total FOV = (n-1)*step + hfw."""
    s = OverviewAcquisitionSettings(
        image_settings=ImageSettings(resolution=(1024, 1024), hfw=100e-6),
        nrows=3, ncols=3, overlap=0.1,
    )
    step = 100e-6 * 0.9
    expected = 2 * step + 100e-6
    assert s.total_fov_x == pytest.approx(expected)
    assert s.total_fov_y == pytest.approx(expected)


def test_overview_acquisition_total_fov_non_square():
    """Non-square tiles: total_fov_y scaled by aspect ratio."""
    s = OverviewAcquisitionSettings(
        image_settings=ImageSettings(resolution=(1536, 1024), hfw=150e-6),
        nrows=3, ncols=3, overlap=0.1,
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
        nrows=1, ncols=1, overlap=0.2,
    )
    assert s.total_fov_x == pytest.approx(200e-6)
    assert s.total_fov_y == pytest.approx(200e-6)


if THERMO_API_AVAILABLE:

    from fibsem.structures import CompustagePosition, CoordinateSystem, StagePosition

    def test_to_autoscript_position():
        
        stage_position = FibsemStagePosition(x=1, y=2, z=3, r=4, t=5, coordinate_system="RAW")

        # test conversion to StagePosition
        autoscript_stage_position = stage_position.to_autoscript_position()

        assert autoscript_stage_position.x == stage_position.x
        assert autoscript_stage_position.y == stage_position.y
        assert autoscript_stage_position.z == stage_position.z
        assert autoscript_stage_position.r == stage_position.r
        assert autoscript_stage_position.t == stage_position.t
        assert autoscript_stage_position.coordinate_system == CoordinateSystem.RAW

        # test convesion to CompuStagePosition
        autoscript_compustage_position = stage_position.to_autoscript_position(compustage=True)

        assert autoscript_compustage_position.x == stage_position.x
        assert autoscript_compustage_position.y == stage_position.y
        assert autoscript_compustage_position.z == stage_position.z
        assert autoscript_compustage_position.a == stage_position.t
        assert autoscript_compustage_position.coordinate_system == CoordinateSystem.SPECIMEN


    def test_from_autoscript_position():
        
        autoscript_stage_position = StagePosition(x=1, y=2, z=3, r=4, t=5, coordinate_system=CoordinateSystem.RAW)
        autoscript_compustage_position = CompustagePosition(x=1, y=2, z=3, a=5, coordinate_system=CoordinateSystem.RAW)

        # test conversion from StagePosition
        stage_position = FibsemStagePosition.from_autoscript_position(autoscript_stage_position)

        assert stage_position.x == autoscript_stage_position.x
        assert stage_position.y == autoscript_stage_position.y
        assert stage_position.z == autoscript_stage_position.z
        assert stage_position.r == autoscript_stage_position.r
        assert stage_position.t == autoscript_stage_position.t
        assert stage_position.coordinate_system == "RAW"

        # test conversion from CompuStagePosition
        stage_position = FibsemStagePosition.from_autoscript_position(autoscript_compustage_position)

        assert stage_position.x == autoscript_compustage_position.x
        assert stage_position.y == autoscript_compustage_position.y
        assert stage_position.z == autoscript_compustage_position.z
        assert stage_position.r == 0
        assert stage_position.t == autoscript_compustage_position.a
        assert stage_position.coordinate_system == "SPECIMEN"
