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
