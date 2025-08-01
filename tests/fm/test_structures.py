import tempfile
import pytest
import numpy as np
from datetime import datetime

from fibsem.fm.structures import (
    ChannelSettings,
    ZParameters,
    FluorescenceImage,
    FluorescenceChannelMetadata,
    FluorescenceImageMetadata,
)
from fibsem.structures import FibsemStagePosition


def test_channel_settings():
    channel_settings = ChannelSettings(
        name="test-channel",
        excitation_wavelength=488.0,
        emission_wavelength=520.0,
        exposure_time=0.1,
        power=0.3,
    )

    # test to_dict method
    cs_dict = channel_settings.to_dict()
    assert cs_dict["name"] == "test-channel"
    assert cs_dict["excitation_wavelength"] == 488.0
    assert cs_dict["emission_wavelength"] == 520.0
    assert cs_dict["exposure_time"] == 0.1
    assert cs_dict["power"] == 0.3

    # test from_dict method
    cs_from_dict = ChannelSettings.from_dict(cs_dict)
    assert cs_from_dict.name == "test-channel"
    assert cs_from_dict.excitation_wavelength == 488.0
    assert cs_from_dict.emission_wavelength == 520.0
    assert cs_from_dict.exposure_time == 0.1
    assert cs_from_dict.power == 0.3

    # test with emission set to None (reflection)
    channel_settings.emission_wavelength = None
    cs_dict_none = channel_settings.to_dict()
    assert cs_dict_none["emission_wavelength"] is None
    cs_from_dict_none = ChannelSettings.from_dict(cs_dict_none)
    assert cs_from_dict_none.emission_wavelength is None


def test_z_parameters():
    zinit = 0.0
    zmin, zmax, zstep = -10e-6, 10e-6, 1e-6
    zparams = ZParameters(zmin=zmin, zmax=zmax, zstep=zstep)

    # test to_dict method
    zparams_dict = zparams.to_dict()
    assert zparams_dict["zmin"] == zmin
    assert zparams_dict["zmax"] == zmax
    assert zparams_dict["zstep"] == zstep
    # test from_dict method
    zparams_from_dict = ZParameters.from_dict(zparams_dict)
    assert zparams_from_dict.zmin == zparams.zmin
    assert zparams_from_dict.zmax == zparams.zmax
    assert zparams_from_dict.zstep == zparams.zstep

    # test generate_z_positions
    z_positions = zparams.generate_positions(z_init=zinit)
    assert len(z_positions) == 21
    assert np.isclose(z_positions[0], zmin)
    assert np.isclose(z_positions[-1], zmax)
    assert any([np.isclose(zinit, pos) for pos in z_positions])

    # test with even step
    zparams.zstep = 2.0e-6
    z_positions_even = zparams.generate_positions(z_init=zinit)
    assert len(z_positions_even) == 11
    assert np.isclose(z_positions_even[0], zmin)
    assert np.isclose(z_positions_even[-1], zmax)
    assert any([np.isclose(zinit, pos) for pos in z_positions])

    # test with asymmetric range
    zparams.zmin = -10e-6
    zparams.zmax = 5e-6
    zparams.zstep = 1.0e-6
    z_positions = zparams.generate_positions(z_init=zinit)
    assert len(z_positions) == 16
    assert np.isclose(z_positions[0], -10e-6)
    assert np.isclose(z_positions[-1], 5e-6)
    assert any([np.isclose(zinit, pos) for pos in z_positions])

    # test with zero step
    zparams.zstep = 0.0
    with pytest.raises(ValueError):
        zparams.generate_positions(z_init=zinit)

    # test with negative step
    zparams.zstep = -1.0e-6
    with pytest.raises(ValueError):
        zparams.generate_positions(z_init=zinit)


def test_fluorescence_channel_metadata():
    """Test FluorescenceChannelMetadata creation and serialization."""
    # Test with all parameters
    channel = FluorescenceChannelMetadata(
        name="DAPI",
        excitation_wavelength=365.0,
        emission_wavelength=447.0,
        power=0.5,
        exposure_time=0.1,
        gain=2.0,
        offset=100.0,
        binning=2,
        objective_position=1e-3,
        objective_magnification=63.0,
        objective_numerical_aperture=1.4,
    )

    # Test required fields
    assert channel.name == "DAPI"
    assert channel.excitation_wavelength == 365.0
    assert channel.power == 0.5
    assert channel.exposure_time == 0.1
    assert channel.gain == 2.0
    assert channel.offset == 100.0

    # Test optional fields
    assert channel.emission_wavelength == 447.0
    assert channel.binning == 2
    assert channel.objective_position == 1e-3
    assert channel.objective_magnification == 63.0
    assert channel.objective_numerical_aperture == 1.4

    # Test serialization
    channel_dict = channel.to_dict()
    assert channel_dict["channel"]["name"] == "DAPI"
    assert channel_dict["filter_set"]["excitation_wavelength"] == 365.0
    assert channel_dict["filter_set"]["emission_wavelength"] == 447.0
    assert channel_dict["light_source"]["power"] == 0.5
    assert channel_dict["camera"]["exposure_time"] == 0.1
    assert channel_dict["camera"]["gain"] == 2.0
    assert channel_dict["camera"]["offset"] == 100.0
    assert channel_dict["camera"]["binning"] == 2
    assert channel_dict["objective"]["position"] == 1e-3
    assert channel_dict["objective"]["magnification"] == 63.0
    assert channel_dict["objective"]["numerical_aperture"] == 1.4

    # Test deserialization
    channel_from_dict = FluorescenceChannelMetadata.from_dict(channel_dict)
    assert channel_from_dict.name == channel.name
    assert channel_from_dict.excitation_wavelength == channel.excitation_wavelength
    assert channel_from_dict.emission_wavelength == channel.emission_wavelength
    assert channel_from_dict.power == channel.power
    assert channel_from_dict.exposure_time == channel.exposure_time
    assert channel_from_dict.gain == channel.gain
    assert channel_from_dict.offset == channel.offset
    assert channel_from_dict.binning == channel.binning
    assert channel_from_dict.objective_position == channel.objective_position
    assert channel_from_dict.objective_magnification == channel.objective_magnification
    assert (
        channel_from_dict.objective_numerical_aperture
        == channel.objective_numerical_aperture
    )


def test_fluorescence_channel_metadata_defaults():
    """Test FluorescenceChannelMetadata with default values."""
    # Test minimal creation
    channel = FluorescenceChannelMetadata(
        name="GFP",
        excitation_wavelength=488.0,
        power=1.0,
        exposure_time=0.05,
        gain=1.5,
        offset=50.0,
    )

    # Test defaults
    assert channel.emission_wavelength is None
    assert channel.binning == 1
    assert channel.objective_position == 0.0
    assert channel.objective_magnification is None
    assert channel.objective_numerical_aperture is None


def test_fluorescence_channel_metadata_validation():
    """Test FluorescenceChannelMetadata validation."""
    # Test invalid binning
    with pytest.raises(ValueError, match="Invalid binning factor"):
        FluorescenceImageMetadata(
            acquisition_date=datetime.now().isoformat(),
            pixel_size_x=1e-6,
            pixel_size_y=1e-6,
            resolution=(1024, 1024),
            channels=[
                FluorescenceChannelMetadata(
                    name="Test",
                    excitation_wavelength=488.0,
                    power=1.0,
                    exposure_time=0.1,
                    gain=1.0,
                    offset=0.0,
                    binning=3,  # Invalid binning
                )
            ],
        )


def test_fluorescence_image_metadata():
    """Test FluorescenceImageMetadata creation and methods."""
    # Create test channels
    channel1 = FluorescenceChannelMetadata(
        name="DAPI",
        excitation_wavelength=365.0,
        emission_wavelength=447.0,
        power=0.5,
        exposure_time=0.1,
        gain=2.0,
        offset=100.0,
    )

    channel2 = FluorescenceChannelMetadata(
        name="GFP",
        excitation_wavelength=488.0,
        emission_wavelength=520.0,
        power=0.3,
        exposure_time=0.05,
        gain=1.5,
        offset=50.0,
    )

    # Create stage position
    stage_pos = FibsemStagePosition(
        x=1e-3, y=2e-3, z=0.0, r=0.0, t=0.0, coordinate_system="RAW"
    )

    # Create metadata
    metadata = FluorescenceImageMetadata(
        acquisition_date="2025-01-01T12:00:00",
        pixel_size_x=100e-9,
        pixel_size_y=100e-9,
        pixel_size_z=200e-9,
        resolution=(2048, 2048),
        channels=[channel1, channel2],
        z_positions=[0.0, 200e-9, 400e-9],
        stage_position=stage_pos,
        system_info={"microscope": "Test Scope"},
    )

    # Test basic properties
    assert metadata.acquisition_date == "2025-01-01T12:00:00"
    assert metadata.pixel_size_x == 100e-9
    assert metadata.pixel_size_y == 100e-9
    assert metadata.pixel_size_z == 200e-9
    assert metadata.resolution == (2048, 2048)
    assert len(metadata.channels) == 2
    assert metadata.z_positions == [0.0, 200e-9, 400e-9]
    assert metadata.stage_position == stage_pos
    assert metadata.system_info == {"microscope": "Test Scope"}

    # Test helper methods
    assert metadata.get_channel_count() == 2
    assert metadata.get_z_count() == 3

    # Test add_channel
    channel3 = FluorescenceChannelMetadata(
        name="Cy5",
        excitation_wavelength=635.0,
        power=0.8,
        exposure_time=0.2,
        gain=3.0,
        offset=200.0,
    )
    metadata.add_channel(channel3)
    assert metadata.get_channel_count() == 3
    assert metadata.channels[2].name == "Cy5"


def test_fluorescence_image_metadata_validation():
    """Test FluorescenceImageMetadata validation."""
    # Test no channels
    with pytest.raises(ValueError, match="At least one channel must be specified"):
        FluorescenceImageMetadata(
            acquisition_date=datetime.now().isoformat(),
            pixel_size_x=1e-6,
            pixel_size_y=1e-6,
            resolution=(1024, 1024),
            channels=[],
        )

    # Test z-stack validation
    with pytest.raises(ValueError, match="Z-stack must have at least 2 positions"):
        FluorescenceImageMetadata(
            acquisition_date=datetime.now().isoformat(),
            pixel_size_x=1e-6,
            pixel_size_y=1e-6,
            resolution=(1024, 1024),
            channels=[
                FluorescenceChannelMetadata(
                    name="Test",
                    excitation_wavelength=488.0,
                    power=1.0,
                    exposure_time=0.1,
                    gain=1.0,
                    offset=0.0,
                )
            ],
            z_positions=[0.0],  # Only one position
        )


def test_fluorescence_image_metadata_auto_pixel_size_z():
    """Test automatic pixel_size_z calculation."""
    metadata = FluorescenceImageMetadata(
        acquisition_date=datetime.now().isoformat(),
        pixel_size_x=1e-6,
        pixel_size_y=1e-6,
        resolution=(1024, 1024),
        channels=[
            FluorescenceChannelMetadata(
                name="Test",
                excitation_wavelength=488.0,
                power=1.0,
                exposure_time=0.1,
                gain=1.0,
                offset=0.0,
            )
        ],
        z_positions=[0.0, 500e-9, 1000e-9],
    )

    # Should auto-calculate pixel_size_z
    assert metadata.pixel_size_z == 500e-9


def test_fluorescence_image_metadata_to_dict():
    """Test FluorescenceImageMetadata to_dict conversion."""
    channel = FluorescenceChannelMetadata(
        name="DAPI",
        excitation_wavelength=365.0,
        emission_wavelength=447.0,
        power=0.5,
        exposure_time=0.1,
        gain=2.0,
        offset=100.0,
        objective_magnification=63.0,
        objective_numerical_aperture=1.4,
    )

    stage_pos = FibsemStagePosition(
        x=1e-3, y=2e-3, z=0.0, r=0.0, t=0.0, coordinate_system="RAW"
    )

    metadata = FluorescenceImageMetadata(
        acquisition_date="2025-01-01T12:00:00",
        pixel_size_x=100e-9,
        pixel_size_y=100e-9,
        resolution=(2048, 2048),
        channels=[channel],
        stage_position=stage_pos,
        system_info={"test": "data"},
    )

    metadata_dict = metadata.to_dict()

    # Test top-level fields
    assert metadata_dict["acquisition_date"] == "2025-01-01T12:00:00"
    assert metadata_dict["pixel_size_x"] == 100e-9
    assert metadata_dict["pixel_size_y"] == 100e-9
    assert metadata_dict["resolution"] == (2048, 2048)
    assert metadata_dict["channel_count"] == 1
    assert metadata_dict["z_count"] == 1
    assert metadata_dict["system_info"] == {"test": "data"}

    # Test stage position
    assert metadata_dict["stage_position"]["x"] == 1e-3
    assert metadata_dict["stage_position"]["y"] == 2e-3

    # Test channel data
    assert len(metadata_dict["channels"]) == 1
    ch_dict = metadata_dict["channels"][0]
    assert ch_dict["name"] == "DAPI"
    assert ch_dict["excitation_wavelength"] == 365.0
    assert ch_dict["emission_wavelength"] == 447.0
    assert ch_dict["objective_magnification"] == 63.0
    assert ch_dict["objective_numerical_aperture"] == 1.4


def test_fluorescence_image_metadata_from_dict():
    """Test FluorescenceImageMetadata from_dict conversion."""
    metadata_dict = {
        "acquisition_date": "2025-01-01T12:00:00",
        "pixel_size_x": 100e-9,
        "pixel_size_y": 100e-9,
        "pixel_size_z": 200e-9,
        "resolution": (2048, 2048),
        "z_positions": [0.0, 200e-9, 400e-9],
        "stage_position": {
            "x": 1e-3,
            "y": 2e-3,
            "z": 0.0,
            "r": 0.0,
            "t": 0.0,
            "coordinate_system": "RAW",
        },
        "system_info": {"test": "data"},
        "channels": [
            {
                "name": "DAPI",
                "excitation_wavelength": 365.0,
                "emission_wavelength": 447.0,
                "power": 0.5,
                "exposure_time": 0.1,
                "gain": 2.0,
                "offset": 100.0,
                "binning": 1,
                "objective_position": 0.0,
                "objective_magnification": 63.0,
                "objective_numerical_aperture": 1.4,
            }
        ],
    }

    metadata = FluorescenceImageMetadata.from_dict(metadata_dict)

    # Test reconstruction
    assert metadata.acquisition_date == "2025-01-01T12:00:00"
    assert metadata.pixel_size_x == 100e-9
    assert metadata.pixel_size_y == 100e-9
    assert metadata.pixel_size_z == 200e-9
    assert metadata.resolution == (2048, 2048)
    assert metadata.z_positions == [0.0, 200e-9, 400e-9]
    assert metadata.system_info == {"test": "data"}

    # Test stage position reconstruction
    assert metadata.stage_position.x == 1e-3
    assert metadata.stage_position.y == 2e-3
    assert metadata.stage_position.coordinate_system == "RAW"

    # Test channel reconstruction
    assert len(metadata.channels) == 1
    channel = metadata.channels[0]
    assert channel.name == "DAPI"
    assert channel.excitation_wavelength == 365.0
    assert channel.emission_wavelength == 447.0
    assert channel.objective_magnification == 63.0
    assert channel.objective_numerical_aperture == 1.4


def test_fluorescence_image_creation():
    """Test FluorescenceImage creation with structured metadata."""
    # Create test data
    data = np.random.randint(0, 255, (1, 3, 512, 512), dtype=np.uint8)  # C, Z, Y, X

    # Create metadata
    channel = FluorescenceChannelMetadata(
        name="GFP",
        excitation_wavelength=488.0,
        emission_wavelength=520.0,
        power=0.3,
        exposure_time=0.05,
        gain=1.5,
        offset=50.0,
    )

    metadata = FluorescenceImageMetadata(
        acquisition_date=datetime.now().isoformat(),
        pixel_size_x=100e-9,
        pixel_size_y=100e-9,
        resolution=(512, 512),
        channels=[channel],
    )

    # Create image
    image = FluorescenceImage(data=data, metadata=metadata)

    # Test properties
    assert image.data.shape == (1, 3, 512, 512)
    assert image.data.dtype == np.uint8
    assert image.metadata == metadata
    assert image.metadata.channels[0].name == "GFP"


def test_fluorescence_image_save_load_roundtrip():
    """Test save/load roundtrip with structured annotations."""
    # Create test data
    data = np.random.randint(0, 255, (2, 1, 256, 256), dtype=np.uint8)  # C, Z, Y, X

    # Create complex metadata
    channel1 = FluorescenceChannelMetadata(
        name="DAPI",
        excitation_wavelength=365.0,
        emission_wavelength=447.0,
        power=0.5,
        exposure_time=0.1,
        gain=2.0,
        offset=100.0,
        objective_magnification=63.0,
        objective_numerical_aperture=1.4,
    )

    channel2 = FluorescenceChannelMetadata(
        name="GFP",
        excitation_wavelength=488.0,
        emission_wavelength=520.0,
        power=0.3,
        exposure_time=0.05,
        gain=1.5,
        offset=50.0,
    )

    stage_pos = FibsemStagePosition(
        x=1e-3, y=2e-3, z=0.0, r=0.0, t=0.0, coordinate_system="RAW"
    )

    metadata = FluorescenceImageMetadata(
        acquisition_date="2025-01-01T12:00:00",
        pixel_size_x=100e-9,
        pixel_size_y=100e-9,
        resolution=(256, 256),
        channels=[channel1, channel2],
        stage_position=stage_pos,
        system_info={"microscope": "Test Scope", "version": "1.0"},
    )

    # Create and save image
    original_image = FluorescenceImage(data=data, metadata=metadata)

    with tempfile.NamedTemporaryFile(suffix=".ome.tiff", delete=False) as tmp_file:
        filename = tmp_file.name

    try:
        original_image.save(filename)

        # Load image back
        loaded_image = FluorescenceImage.load(filename)

        # Test data preservation
        np.testing.assert_array_equal(loaded_image.data, original_image.data)

        # Test metadata preservation
        assert (
            loaded_image.metadata.acquisition_date
            == original_image.metadata.acquisition_date
        )
        assert (
            loaded_image.metadata.pixel_size_x == original_image.metadata.pixel_size_x
        )
        assert (
            loaded_image.metadata.pixel_size_y == original_image.metadata.pixel_size_y
        )
        assert loaded_image.metadata.resolution == original_image.metadata.resolution
        assert loaded_image.metadata.system_info == original_image.metadata.system_info

        # Test stage position preservation
        assert (
            loaded_image.metadata.stage_position.x
            == original_image.metadata.stage_position.x
        )
        assert (
            loaded_image.metadata.stage_position.y
            == original_image.metadata.stage_position.y
        )
        assert (
            loaded_image.metadata.stage_position.coordinate_system
            == original_image.metadata.stage_position.coordinate_system
        )

        # Test channel preservation
        assert len(loaded_image.metadata.channels) == len(
            original_image.metadata.channels
        )
        for orig_ch, loaded_ch in zip(
            original_image.metadata.channels, loaded_image.metadata.channels
        ):
            assert loaded_ch.name == orig_ch.name
            assert loaded_ch.excitation_wavelength == orig_ch.excitation_wavelength
            assert loaded_ch.emission_wavelength == orig_ch.emission_wavelength
            assert loaded_ch.power == orig_ch.power
            assert loaded_ch.exposure_time == orig_ch.exposure_time
            assert loaded_ch.gain == orig_ch.gain
            assert loaded_ch.offset == orig_ch.offset
            assert loaded_ch.objective_magnification == orig_ch.objective_magnification
            assert (
                loaded_ch.objective_numerical_aperture
                == orig_ch.objective_numerical_aperture
            )

    finally:
        import os

        if os.path.exists(filename):
            os.unlink(filename)


def test_fluorescence_image_load_fallback():
    """Test FluorescenceImage load fallback to basic metadata."""
    # Create simple TIFF without OME metadata
    data = np.random.randint(0, 255, (512, 512), dtype=np.uint8)

    with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as tmp_file:
        filename = tmp_file.name

    try:
        import tifffile

        tifffile.imwrite(filename, data)

        # Load should fall back to basic metadata
        loaded_image = FluorescenceImage.load(filename)

        # Test data (should be reshaped to CZYX format)
        expected_data = data[
            np.newaxis, np.newaxis, :, :
        ]  # Reshape to (1, 1, 512, 512)
        np.testing.assert_array_equal(loaded_image.data, expected_data)

        # Test basic metadata
        assert loaded_image.metadata is not None
        assert len(loaded_image.metadata.channels) == 1
        assert loaded_image.metadata.channels[0].name == "Channel_01"
        assert loaded_image.metadata.channels[0].excitation_wavelength == 488.0
        assert loaded_image.metadata.resolution == (512, 512)
        assert loaded_image.metadata.pixel_size_x == 1e-6
        assert loaded_image.metadata.pixel_size_y == 1e-6

    finally:
        import os

        if os.path.exists(filename):
            os.unlink(filename)


def test_fluorescence_image_multi_channel_stacking():
    """Test multi-channel image creation."""
    # Create individual channel images
    data1 = np.random.randint(0, 255, (1, 256, 256), dtype=np.uint8)  # Z, Y, X
    data2 = np.random.randint(0, 255, (1, 256, 256), dtype=np.uint8)

    channel1 = FluorescenceChannelMetadata(
        name="DAPI",
        excitation_wavelength=365.0,
        power=0.5,
        exposure_time=0.1,
        gain=2.0,
        offset=100.0,
    )
    channel2 = FluorescenceChannelMetadata(
        name="GFP",
        excitation_wavelength=488.0,
        power=0.3,
        exposure_time=0.05,
        gain=1.5,
        offset=50.0,
    )

    metadata1 = FluorescenceImageMetadata(
        acquisition_date=datetime.now().isoformat(),
        pixel_size_x=100e-9,
        pixel_size_y=100e-9,
        resolution=(256, 256),
        channels=[channel1],
    )
    metadata2 = FluorescenceImageMetadata(
        acquisition_date=datetime.now().isoformat(),
        pixel_size_x=100e-9,
        pixel_size_y=100e-9,
        resolution=(256, 256),
        channels=[channel2],
    )

    image1 = FluorescenceImage(data=data1, metadata=metadata1)
    image2 = FluorescenceImage(data=data2, metadata=metadata2)

    # Create multi-channel image
    multi_channel_image = FluorescenceImage.create_multi_channel_image([image1, image2])

    # Test data stacking
    assert multi_channel_image.data.shape == (2, 1, 256, 256)  # C, Z, Y, X
    np.testing.assert_array_equal(multi_channel_image.data[0], data1)
    np.testing.assert_array_equal(multi_channel_image.data[1], data2)

    # Test metadata combination
    assert len(multi_channel_image.metadata.channels) == 2
    assert multi_channel_image.metadata.channels[0].name == "DAPI"
    assert multi_channel_image.metadata.channels[1].name == "GFP"


def test_fluorescence_image_z_stack_creation():
    """Test Z-stack creation."""
    # Create images at different z positions
    data = np.random.randint(0, 255, (256, 256), dtype=np.uint8)  # Y, X

    images = []
    z_positions = [0.0, 500e-9, 1000e-9]

    for z_pos in z_positions:
        channel = FluorescenceChannelMetadata(
            name="GFP",
            excitation_wavelength=488.0,
            power=0.3,
            exposure_time=0.05,
            gain=1.5,
            offset=50.0,
            objective_position=z_pos,
        )
        metadata = FluorescenceImageMetadata(
            acquisition_date=datetime.now().isoformat(),
            pixel_size_x=100e-9,
            pixel_size_y=100e-9,
            resolution=(256, 256),
            channels=[channel],
        )
        # Add singleton Z dimension for create_z_stack
        data_3d = data[np.newaxis, :, :]  # Z, Y, X
        images.append(FluorescenceImage(data=data_3d, metadata=metadata))

    # Create z-stack
    z_stack = FluorescenceImage.create_z_stack(images)

    # Test data stacking
    assert z_stack.data.shape == (3, 256, 256)  # Z, Y, X

    # Test metadata
    assert z_stack.metadata.z_positions == z_positions
    assert z_stack.metadata.pixel_size_z == 500e-9  # Auto-calculated
    assert len(z_stack.metadata.channels) == 1
    assert z_stack.metadata.channels[0].name == "GFP"


def test_fluorescence_image():
    """Legacy test placeholder - now covered by specific tests above."""
    pass


def test_max_intensity_projection_single_z():
    """Test maximum intensity projection with single z-plane (no projection needed)."""
    # Create test data with single z-plane
    data = np.random.randint(0, 255, (2, 1, 256, 256), dtype=np.uint8)  # C, Z, Y, X
    
    # Create metadata
    channel1 = FluorescenceChannelMetadata(
        name="DAPI",
        excitation_wavelength=365.0,
        power=0.5,
        exposure_time=0.1,
        gain=2.0,
        offset=100.0,
    )
    channel2 = FluorescenceChannelMetadata(
        name="GFP",
        excitation_wavelength=488.0,
        power=0.3,
        exposure_time=0.05,
        gain=1.5,
        offset=50.0,
    )
    
    metadata = FluorescenceImageMetadata(
        acquisition_date=datetime.now().isoformat(),
        pixel_size_x=100e-9,
        pixel_size_y=100e-9,
        resolution=(256, 256),
        channels=[channel1, channel2],
    )
    
    image = FluorescenceImage(data=data, metadata=metadata)
    
    # Test projection of all channels (should return a copy)
    projected = image.max_intensity_projection()
    assert projected.data.shape == (2, 1, 256, 256)
    np.testing.assert_array_equal(projected.data, data)
    assert len(projected.metadata.channels) == 2
    assert projected.metadata.z_positions is None
    assert projected.metadata.pixel_size_z is None
    
    # Test projection of single channel
    projected_ch0 = image.max_intensity_projection(channel=0)
    assert projected_ch0.data.shape == (1, 1, 256, 256)
    np.testing.assert_array_equal(projected_ch0.data[0], data[0])
    assert len(projected_ch0.metadata.channels) == 1
    assert projected_ch0.metadata.channels[0].name == "DAPI"


def test_max_intensity_projection_multi_z():
    """Test maximum intensity projection with multiple z-planes."""
    # Create test data with known maximum values
    data = np.zeros((2, 3, 4, 4), dtype=np.uint16)  # C, Z, Y, X
    
    # Channel 0: max values in different z-planes for different pixels
    data[0, 0, 0, 0] = 100  # max at z=0
    data[0, 1, 0, 0] = 50
    data[0, 2, 0, 0] = 75
    
    data[0, 0, 1, 1] = 200
    data[0, 1, 1, 1] = 300  # max at z=1
    data[0, 2, 1, 1] = 150
    
    data[0, 0, 2, 2] = 80
    data[0, 1, 2, 2] = 120
    data[0, 2, 2, 2] = 400  # max at z=2
    
    # Channel 1: different pattern
    data[1, 0, 0, 0] = 500  # max at z=0
    data[1, 1, 0, 0] = 300
    data[1, 2, 0, 0] = 200
    
    data[1, 0, 1, 1] = 100
    data[1, 1, 1, 1] = 150
    data[1, 2, 1, 1] = 600  # max at z=2
    
    # Create metadata
    channel1 = FluorescenceChannelMetadata(
        name="DAPI",
        excitation_wavelength=365.0,
        power=0.5,
        exposure_time=0.1,
        gain=2.0,
        offset=100.0,
    )
    channel2 = FluorescenceChannelMetadata(
        name="GFP", 
        excitation_wavelength=488.0,
        power=0.3,
        exposure_time=0.05,
        gain=1.5,
        offset=50.0,
    )
    
    metadata = FluorescenceImageMetadata(
        acquisition_date=datetime.now().isoformat(),
        pixel_size_x=100e-9,
        pixel_size_y=100e-9,
        pixel_size_z=200e-9,
        resolution=(4, 4),
        channels=[channel1, channel2],
        z_positions=[0.0, 200e-9, 400e-9],
    )
    
    image = FluorescenceImage(data=data, metadata=metadata)
    
    # Test projection of all channels
    projected = image.max_intensity_projection()
    assert projected.data.shape == (2, 1, 4, 4)
    
    # Check maximum values are correct
    assert projected.data[0, 0, 0, 0] == 100  # max of [100, 50, 75]
    assert projected.data[0, 0, 1, 1] == 300  # max of [200, 300, 150] 
    assert projected.data[0, 0, 2, 2] == 400  # max of [80, 120, 400]
    
    assert projected.data[1, 0, 0, 0] == 500  # max of [500, 300, 200]
    assert projected.data[1, 0, 1, 1] == 600  # max of [100, 150, 600]
    
    # Check metadata
    assert len(projected.metadata.channels) == 2
    assert projected.metadata.z_positions is None
    assert projected.metadata.pixel_size_z is None
    assert projected.metadata.pixel_size_x == 100e-9
    assert projected.metadata.pixel_size_y == 100e-9
    
    # Test projection of single channel (channel 0)
    projected_ch0 = image.max_intensity_projection(channel=0)
    assert projected_ch0.data.shape == (1, 1, 4, 4)
    assert projected_ch0.data[0, 0, 0, 0] == 100
    assert projected_ch0.data[0, 0, 1, 1] == 300
    assert projected_ch0.data[0, 0, 2, 2] == 400
    assert len(projected_ch0.metadata.channels) == 1
    assert projected_ch0.metadata.channels[0].name == "DAPI"
    
    # Test projection of single channel (channel 1)
    projected_ch1 = image.max_intensity_projection(channel=1)
    assert projected_ch1.data.shape == (1, 1, 4, 4)
    assert projected_ch1.data[0, 0, 0, 0] == 500
    assert projected_ch1.data[0, 0, 1, 1] == 600
    assert len(projected_ch1.metadata.channels) == 1
    assert projected_ch1.metadata.channels[0].name == "GFP"


def test_max_intensity_projection_error_cases():
    """Test error cases for maximum intensity projection."""
    # Create test data
    data = np.random.randint(0, 255, (2, 3, 256, 256), dtype=np.uint8)  # C, Z, Y, X
    
    channel1 = FluorescenceChannelMetadata(
        name="DAPI",
        excitation_wavelength=365.0,
        power=0.5,
        exposure_time=0.1,
        gain=2.0,
        offset=100.0,
    )
    channel2 = FluorescenceChannelMetadata(
        name="GFP",
        excitation_wavelength=488.0,
        power=0.3,
        exposure_time=0.05,
        gain=1.5,
        offset=50.0,
    )
    
    metadata = FluorescenceImageMetadata(
        acquisition_date=datetime.now().isoformat(),
        pixel_size_x=100e-9,
        pixel_size_y=100e-9,
        resolution=(256, 256),
        channels=[channel1, channel2],
    )
    
    image = FluorescenceImage(data=data, metadata=metadata)
    
    # Test invalid channel index (too high)
    with pytest.raises(ValueError, match="Channel index 2 out of range"):
        image.max_intensity_projection(channel=2)
    
    # Test invalid channel index (negative)
    with pytest.raises(ValueError, match="Channel index -1 out of range"):
        image.max_intensity_projection(channel=-1)
    
    # Test with insufficient dimensions
    data_2d = np.random.randint(0, 255, (256, 256), dtype=np.uint8)  # Y, X only
    image_2d = FluorescenceImage(data=data_2d, metadata=metadata)
    
    with pytest.raises(ValueError, match="Image must have at least 3 dimensions"):
        image_2d.max_intensity_projection()


def test_max_intensity_projection_metadata_preservation():
    """Test that metadata is properly preserved and updated in projections."""
    # Create test data
    data = np.random.randint(0, 255, (1, 5, 128, 128), dtype=np.uint16)  # C, Z, Y, X
    
    # Create comprehensive metadata
    channel = FluorescenceChannelMetadata(
        name="Cy5",
        excitation_wavelength=635.0,
        emission_wavelength=680.0,
        power=0.8,
        exposure_time=0.2,
        gain=3.0,
        offset=200.0,
        binning=2,
        objective_position=1e-3,
        objective_magnification=63.0,
        objective_numerical_aperture=1.4,
    )
    
    stage_pos = FibsemStagePosition(
        x=1e-3, y=2e-3, z=0.0, r=0.0, t=0.0, coordinate_system="RAW"
    )
    
    metadata = FluorescenceImageMetadata(
        acquisition_date="2025-01-01T12:00:00",
        pixel_size_x=50e-9,
        pixel_size_y=50e-9,
        pixel_size_z=100e-9,
        resolution=(128, 128),
        channels=[channel],
        z_positions=[0.0, 100e-9, 200e-9, 300e-9, 400e-9],
        stage_position=stage_pos,
        system_info={"microscope": "Test Scope", "version": "1.0"},
        dimension_order="CZYX",
    )
    
    image = FluorescenceImage(data=data, metadata=metadata)
    
    # Test projection
    projected = image.max_intensity_projection()
    
    # Test preserved metadata
    assert projected.metadata.acquisition_date == "2025-01-01T12:00:00"
    assert projected.metadata.pixel_size_x == 50e-9
    assert projected.metadata.pixel_size_y == 50e-9
    assert projected.metadata.resolution == (128, 128)
    assert projected.metadata.stage_position == stage_pos
    assert projected.metadata.system_info == {"microscope": "Test Scope", "version": "1.0"}
    assert projected.metadata.dimension_order == "CZYX"
    
    # Test updated metadata (z-related fields should be cleared)
    assert projected.metadata.pixel_size_z is None
    assert projected.metadata.z_positions is None
    
    # Test channel metadata preservation
    assert len(projected.metadata.channels) == 1
    proj_channel = projected.metadata.channels[0]
    assert proj_channel.name == "Cy5"
    assert proj_channel.excitation_wavelength == 635.0
    assert proj_channel.emission_wavelength == 680.0
    assert proj_channel.power == 0.8
    assert proj_channel.exposure_time == 0.2
    assert proj_channel.gain == 3.0
    assert proj_channel.offset == 200.0
    assert proj_channel.binning == 2
    assert proj_channel.objective_position == 1e-3
    assert proj_channel.objective_magnification == 63.0
    assert proj_channel.objective_numerical_aperture == 1.4


def test_max_intensity_projection_data_types():
    """Test maximum intensity projection with different data types."""
    # Test with different data types
    shapes = (1, 4, 64, 64)  # C, Z, Y, X
    
    # Create metadata
    channel = FluorescenceChannelMetadata(
        name="Test",
        excitation_wavelength=488.0,
        power=1.0,
        exposure_time=0.1,
        gain=1.0,
        offset=0.0,
    )
    
    metadata = FluorescenceImageMetadata(
        acquisition_date=datetime.now().isoformat(),
        pixel_size_x=100e-9,
        pixel_size_y=100e-9,
        resolution=(64, 64),
        channels=[channel],
    )
    
    # Test uint8
    data_uint8 = np.random.randint(0, 255, shapes, dtype=np.uint8)
    image_uint8 = FluorescenceImage(data=data_uint8, metadata=metadata)
    projected_uint8 = image_uint8.max_intensity_projection()
    assert projected_uint8.data.dtype == np.uint8
    assert projected_uint8.data.shape == (1, 1, 64, 64)
    
    # Test uint16
    data_uint16 = np.random.randint(0, 65535, shapes, dtype=np.uint16)
    image_uint16 = FluorescenceImage(data=data_uint16, metadata=metadata)
    projected_uint16 = image_uint16.max_intensity_projection()
    assert projected_uint16.data.dtype == np.uint16
    assert projected_uint16.data.shape == (1, 1, 64, 64)
    
    # Test float32
    data_float32 = np.random.random(shapes).astype(np.float32)
    image_float32 = FluorescenceImage(data=data_float32, metadata=metadata)
    projected_float32 = image_float32.max_intensity_projection()
    assert projected_float32.data.dtype == np.float32
    assert projected_float32.data.shape == (1, 1, 64, 64)


def test_max_intensity_projection_large_stack():
    """Test maximum intensity projection with larger z-stack."""
    # Create larger z-stack to test performance
    nc, nz, ny, nx = 3, 20, 512, 512
    data = np.random.randint(0, 4095, (nc, nz, ny, nx), dtype=np.uint16)  # C, Z, Y, X
    
    # Set some known maximum values to verify correctness
    data[0, 5, 100, 100] = 4095  # max for channel 0 at (100, 100)
    data[1, 10, 200, 200] = 4095  # max for channel 1 at (200, 200)
    data[2, 15, 300, 300] = 4095  # max for channel 2 at (300, 300)
    
    # Create metadata
    channels = []
    for i in range(nc):
        channel = FluorescenceChannelMetadata(
            name=f"Channel_{i+1}",
            excitation_wavelength=400.0 + i * 100,
            power=0.5,
            exposure_time=0.1,
            gain=1.0,
            offset=0.0,
        )
        channels.append(channel)
    
    metadata = FluorescenceImageMetadata(
        acquisition_date=datetime.now().isoformat(),
        pixel_size_x=100e-9,
        pixel_size_y=100e-9,
        pixel_size_z=50e-9,
        resolution=(nx, ny),
        channels=channels,
        z_positions=[i * 50e-9 for i in range(nz)],
    )
    
    image = FluorescenceImage(data=data, metadata=metadata)
    
    # Test projection of all channels
    projected = image.max_intensity_projection()
    assert projected.data.shape == (nc, 1, ny, nx)
    
    # Verify known maximum values
    assert projected.data[0, 0, 100, 100] == 4095
    assert projected.data[1, 0, 200, 200] == 4095
    assert projected.data[2, 0, 300, 300] == 4095
    
    # Test single channel projection
    projected_ch1 = image.max_intensity_projection(channel=1)
    assert projected_ch1.data.shape == (1, 1, ny, nx)
    assert projected_ch1.data[0, 0, 200, 200] == 4095
    assert len(projected_ch1.metadata.channels) == 1
    assert projected_ch1.metadata.channels[0].name == "Channel_2"


def test_max_intensity_projection_2d_return():
    """Test max_intensity_projection with return_2d=True option."""
    # Create test data with multiple channels and z-planes
    ny, nx = 256, 256
    nc, nz = 2, 5
    data = np.random.randint(0, 1000, (nc, nz, ny, nx), dtype=np.uint16)
    
    # Set known maximum values at specific locations
    data[0, 2, 50, 50] = 4095  # Channel 0, z=2
    data[1, 4, 100, 100] = 4095  # Channel 1, z=4
    
    # Create metadata
    channels = []
    for i in range(nc):
        channel = FluorescenceChannelMetadata(
            name=f"Channel_{i+1}",
            excitation_wavelength=488.0 + i * 50,
            power=1.0,
            exposure_time=0.1,
            gain=1.0,
            offset=0.0
        )
        channels.append(channel)
    
    metadata = FluorescenceImageMetadata(
        acquisition_date=datetime.now().isoformat(),
        pixel_size_x=100e-9,
        pixel_size_y=100e-9,
        resolution=(nx, ny),
        channels=channels,
        z_positions=[i * 100e-9 for i in range(nz)]
    )
    
    image = FluorescenceImage(data=data, metadata=metadata)
    
    # Test 2D return for all channels
    projected_2d = image.max_intensity_projection(return_2d=True)
    
    # Should return numpy array, not FluorescenceImage
    assert isinstance(projected_2d, np.ndarray)
    assert not hasattr(projected_2d, 'metadata')
    
    # Shape should be (C, Y, X) after squeezing singleton Z dimension
    assert projected_2d.shape == (nc, ny, nx)
    
    # Verify maximum values are preserved
    assert projected_2d[0, 50, 50] == 4095
    assert projected_2d[1, 100, 100] == 4095
    
    # Test 2D return for single channel
    projected_2d_ch0 = image.max_intensity_projection(channel=0, return_2d=True)
    
    # Should return 2D array (Y, X) after squeezing C and Z dimensions
    assert projected_2d_ch0.shape == (ny, nx)
    assert projected_2d_ch0[50, 50] == 4095
    
    # Test 2D return for single z-plane (no projection needed)
    single_z_data = data[:, :1, :, :]  # Take only first z-plane
    single_z_image = FluorescenceImage(data=single_z_data, metadata=metadata)
    
    projected_2d_single_z = single_z_image.max_intensity_projection(return_2d=True)
    assert projected_2d_single_z.shape == (nc, ny, nx)
    
    # Verify data is identical (no projection needed)
    np.testing.assert_array_equal(projected_2d_single_z, single_z_data.squeeze(axis=1))


def test_max_intensity_projection_2d_vs_metadata_return():
    """Test that 2D and metadata returns produce equivalent data."""
    # Create test data
    ny, nx = 128, 128
    nc, nz = 1, 3
    data = np.random.randint(0, 1000, (nc, nz, ny, nx), dtype=np.uint16)
    
    # Create metadata
    channel = FluorescenceChannelMetadata(
        name="Test",
        excitation_wavelength=488.0,
        power=1.0,
        exposure_time=0.1,
        gain=1.0,
        offset=0.0
    )
    
    metadata = FluorescenceImageMetadata(
        acquisition_date=datetime.now().isoformat(),
        pixel_size_x=100e-9,
        pixel_size_y=100e-9,
        resolution=(nx, ny),
        channels=[channel]
    )
    
    image = FluorescenceImage(data=data, metadata=metadata)
    
    # Get projections with both methods
    projected_metadata = image.max_intensity_projection(return_2d=False)
    projected_2d = image.max_intensity_projection(return_2d=True)
    
    # Verify that the squeezed metadata version equals the 2D version
    np.testing.assert_array_equal(
        np.squeeze(projected_metadata.data), 
        projected_2d
    )
    
    # Test with specific channel
    projected_metadata_ch = image.max_intensity_projection(channel=0, return_2d=False)
    projected_2d_ch = image.max_intensity_projection(channel=0, return_2d=True)
    
    np.testing.assert_array_equal(
        np.squeeze(projected_metadata_ch.data), 
        projected_2d_ch
    )


def test_focus_stack_basic():
    """Test basic focus stacking functionality."""
    # Create test data with simulated focus variation
    ny, nx = 64, 64
    nc, nz = 1, 5
    data = np.zeros((nc, nz, ny, nx), dtype=np.uint16)
    
    # Create synthetic focus pattern - each z-plane has peak sharpness in different regions
    for z in range(nz):
        # Create a gaussian blob that's sharpest at different z-planes for different regions
        y_center = ny // 2
        x_center = nx // 2 + (z - nz//2) * 8  # Shift center for each z-plane
        
        y, x = np.ogrid[:ny, :nx]
        # Create a focused region for this z-plane
        focused_region = np.exp(-((y - y_center)**2 + (x - x_center)**2) / (2 * 10**2))
        
        # Add noise and make this z-plane sharpest in its region
        base_blur = 0.3  # Base blur level
        sharpness = np.where(focused_region > 0.5, 1.0, base_blur)
        
        # Simulate image with varying sharpness
        data[0, z, :, :] = (focused_region * sharpness * 1000 + np.random.normal(0, 50, (ny, nx))).astype(np.uint16)
    
    # Create metadata
    channel = FluorescenceChannelMetadata(
        name="Test",
        excitation_wavelength=488.0,
        power=1.0,
        exposure_time=0.1,
        gain=1.0,
        offset=0.0
    )
    
    metadata = FluorescenceImageMetadata(
        acquisition_date=datetime.now().isoformat(),
        pixel_size_x=100e-9,
        pixel_size_y=100e-9,
        resolution=(nx, ny),
        channels=[channel],
        z_positions=[i * 100e-9 for i in range(nz)]
    )
    
    image = FluorescenceImage(data=data, metadata=metadata)
    
    # Test focus stacking with different methods
    for method in ['laplacian', 'sobel', 'variance', 'tenengrad']:
        stacked = image.focus_stack(method=method)
        
        # Check return type and shape
        assert isinstance(stacked, FluorescenceImage)
        assert stacked.data.shape == (1, 1, ny, nx)
        
        # Check metadata preservation
        assert len(stacked.metadata.channels) == 1
        assert stacked.metadata.channels[0].name == "Test"
        assert stacked.metadata.z_positions is None
        assert stacked.metadata.pixel_size_z is None
        
        # Check that the stacked image has reasonable values
        assert stacked.data.dtype == data.dtype
        assert np.all(stacked.data >= 0)


def test_focus_stack_2d_return():
    """Test focus stacking with 2D return option."""
    # Create simple test data
    ny, nx = 32, 32
    nc, nz = 2, 3
    data = np.random.randint(100, 1000, (nc, nz, ny, nx), dtype=np.uint16)
    
    # Create sharp features at different z-planes
    data[0, 0, 10:15, 10:15] = 2000  # Sharp feature in channel 0, z=0
    data[0, 2, 20:25, 20:25] = 2000  # Sharp feature in channel 0, z=2
    data[1, 1, 5:10, 25:30] = 2000   # Sharp feature in channel 1, z=1
    
    # Create metadata
    channels = [
        FluorescenceChannelMetadata(name="CH1", excitation_wavelength=488, power=1.0, exposure_time=0.1, gain=1.0, offset=0.0),
        FluorescenceChannelMetadata(name="CH2", excitation_wavelength=555, power=1.0, exposure_time=0.1, gain=1.0, offset=0.0)
    ]
    
    metadata = FluorescenceImageMetadata(
        acquisition_date=datetime.now().isoformat(),
        pixel_size_x=100e-9,
        pixel_size_y=100e-9,
        resolution=(nx, ny),
        channels=channels
    )
    
    image = FluorescenceImage(data=data, metadata=metadata)
    
    # Test 2D return for all channels
    stacked_2d = image.focus_stack(return_2d=True)
    assert isinstance(stacked_2d, np.ndarray)
    assert stacked_2d.shape == (nc, ny, nx)
    
    # Test 2D return for single channel
    stacked_2d_ch0 = image.focus_stack(channel=0, return_2d=True)
    assert isinstance(stacked_2d_ch0, np.ndarray)
    assert stacked_2d_ch0.shape == (ny, nx)
    
    # Test that 2D and metadata returns are equivalent
    stacked_metadata = image.focus_stack(return_2d=False)
    np.testing.assert_array_equal(np.squeeze(stacked_metadata.data), stacked_2d)


def test_focus_stack_error_cases():
    """Test focus stacking error handling."""
    # Create minimal test data
    ny, nx = 16, 16
    
    # Test insufficient z-planes
    data_2d = np.random.randint(0, 255, (1, 1, ny, nx), dtype=np.uint8)
    channel = FluorescenceChannelMetadata(name="Test", excitation_wavelength=488, power=1.0, exposure_time=0.1, gain=1.0, offset=0.0)
    metadata = FluorescenceImageMetadata(
        acquisition_date=datetime.now().isoformat(),
        pixel_size_x=100e-9, pixel_size_y=100e-9, resolution=(nx, ny), channels=[channel]
    )
    image_2d = FluorescenceImage(data=data_2d, metadata=metadata)
    
    with pytest.raises(ValueError, match="Focus stacking requires at least 2 z-planes"):
        image_2d.focus_stack()
    
    # Create valid multi-z data for other tests
    data_3d = np.random.randint(0, 255, (2, 3, ny, nx), dtype=np.uint8)
    metadata_3d = FluorescenceImageMetadata(
        acquisition_date=datetime.now().isoformat(),
        pixel_size_x=100e-9, pixel_size_y=100e-9, resolution=(nx, ny), 
        channels=[channel, channel]
    )
    image_3d = FluorescenceImage(data=data_3d, metadata=metadata_3d)
    
    # Test invalid channel index
    with pytest.raises(ValueError, match="Channel index 5 out of range"):
        image_3d.focus_stack(channel=5)
    
    # Test invalid method
    with pytest.raises(ValueError, match="Method 'invalid' not supported"):
        image_3d.focus_stack(method='invalid')


def test_focus_stack_methods_comparison():
    """Test that different focus measure methods produce reasonable results."""
    # Create test data with clear focus variation
    ny, nx = 32, 32
    nc, nz = 1, 4
    data = np.zeros((nc, nz, ny, nx), dtype=np.uint16)
    
    # Create different focus patterns for each z-plane
    for z in range(nz):
        # Different regions are in focus at different z-planes
        region_size = 8
        start_y = z * 6
        start_x = z * 6
        end_y = min(start_y + region_size, ny)
        end_x = min(start_x + region_size, nx)
        
        # Base noise
        data[0, z, :, :] = np.random.randint(50, 150, (ny, nx))
        
        # Sharp feature in specific region
        data[0, z, start_y:end_y, start_x:end_x] = 1000
    
    # Create metadata
    channel = FluorescenceChannelMetadata(name="Test", excitation_wavelength=488, power=1.0, exposure_time=0.1, gain=1.0, offset=0.0)
    metadata = FluorescenceImageMetadata(
        acquisition_date=datetime.now().isoformat(),
        pixel_size_x=100e-9, pixel_size_y=100e-9, resolution=(nx, ny), channels=[channel]
    )
    image = FluorescenceImage(data=data, metadata=metadata)
    
    # Test all methods
    results = {}
    for method in ['laplacian', 'sobel', 'variance', 'tenengrad']:
        stacked = image.focus_stack(method=method, return_2d=True)
        results[method] = stacked
        
        # All methods should preserve the high-intensity regions
        assert np.max(stacked) >= 800  # Should preserve most of the sharp features
        assert stacked.shape == (ny, nx)
    
    # Results should be similar but not identical
    # (Different methods may select slightly different pixels)
    for method1 in results:
        for method2 in results:
            if method1 != method2:
                # Results should be correlated but not identical
                correlation = np.corrcoef(results[method1].flatten(), results[method2].flatten())[0, 1]
                assert correlation > 0.7  # Should be reasonably similar


def test_focus_stack_block_based():
    """Test block-based focus stacking."""
    # Create test data with clear focus variation by blocks
    ny, nx = 128, 128  # Large enough for multiple 64x64 blocks
    nc, nz = 1, 4
    data = np.random.randint(50, 150, (nc, nz, ny, nx), dtype=np.uint16)
    
    # Create focused regions in different blocks and z-planes
    # Top-left block (0-64, 0-64) is sharpest at z=0
    data[0, 0, 0:64, 0:64] = np.random.randint(800, 1000, (64, 64))
    
    # Top-right block (0-64, 64-128) is sharpest at z=1
    data[0, 1, 0:64, 64:128] = np.random.randint(800, 1000, (64, 64))
    
    # Bottom-left block (64-128, 0-64) is sharpest at z=2
    data[0, 2, 64:128, 0:64] = np.random.randint(800, 1000, (64, 64))
    
    # Bottom-right block (64-128, 64-128) is sharpest at z=3
    data[0, 3, 64:128, 64:128] = np.random.randint(800, 1000, (64, 64))
    
    # Create metadata
    channel = FluorescenceChannelMetadata(
        name="Test", excitation_wavelength=488, power=1.0, 
        exposure_time=0.1, gain=1.0, offset=0.0
    )
    
    metadata = FluorescenceImageMetadata(
        acquisition_date=datetime.now().isoformat(),
        pixel_size_x=100e-9, pixel_size_y=100e-9, 
        resolution=(nx, ny), channels=[channel]
    )
    
    image = FluorescenceImage(data=data, metadata=metadata)
    
    # Test block-based focus stacking
    stacked_blocks = image.focus_stack(use_blocks=True, block_size=64, return_2d=True)
    stacked_pixels = image.focus_stack(use_blocks=False, return_2d=True)
    
    # Both should have same shape
    assert stacked_blocks.shape == stacked_pixels.shape == (ny, nx)
    
    # Block-based should preserve the high-intensity regions we created
    assert np.max(stacked_blocks[0:64, 0:64]) >= 700      # Top-left
    assert np.max(stacked_blocks[0:64, 64:128]) >= 700    # Top-right
    assert np.max(stacked_blocks[64:128, 0:64]) >= 700    # Bottom-left
    assert np.max(stacked_blocks[64:128, 64:128]) >= 700  # Bottom-right
    
    # Test with different block sizes
    stacked_small_blocks = image.focus_stack(use_blocks=True, block_size=32, return_2d=True)
    assert stacked_small_blocks.shape == (ny, nx)
    
    # Test metadata preservation
    stacked_with_metadata = image.focus_stack(use_blocks=True, return_2d=False)
    assert isinstance(stacked_with_metadata, FluorescenceImage)
    assert stacked_with_metadata.metadata.channels[0].name == "Test"
    
    # Test smoothing options
    stacked_smooth = image.focus_stack(use_blocks=True, smooth_transitions=True, return_2d=True)
    stacked_no_smooth = image.focus_stack(use_blocks=True, smooth_transitions=False, return_2d=True)
    
    assert stacked_smooth.shape == stacked_no_smooth.shape == (ny, nx)
    
    # Both should preserve high-intensity regions
    assert np.max(stacked_smooth) >= 500
    assert np.max(stacked_no_smooth) >= 500
