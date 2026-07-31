import threading
from unittest.mock import Mock, call, patch

import numpy as np
import pytest

from fibsem import utils
from fibsem.cancellation import OperationCancelledError
from fibsem.fm import acquisition
from fibsem.fm.acquisition import (
    acquire_and_stitch_tileset,
    acquire_at_positions,
    acquire_channels,
    acquire_image,
    acquire_multiple_overviews,
    acquire_z_stack,
    calculate_grid_coverage_area,
    calculate_grid_dimensions,
    calculate_grid_overlap,
    calculate_grid_size_for_area,
    generate_grid_positions,
    plot_grid_positions,
    stitch_tileset,
)
from fibsem.fm.structures import (
    AutoFocusMode,
    AutoFocusSettings,
    ChannelSettings,
    FluorescenceImage,
    FMStagePosition,
    OverviewParameters,
    ZParameters,
)
from fibsem.structures import FibsemStagePosition, TileOrderStrategy


@pytest.fixture
def demo_microscope():
    """Fixture providing a demo microscope for testing."""
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    return microscope

@pytest.fixture
def fm_microscope(demo_microscope):
    """Fixture providing a demo microscope configured for FM operations."""
    microscope = demo_microscope
    microscope.system.stage.shuttle_pre_tilt = 0
    microscope.stage_is_compustage = True
    microscope.move_to_microscope("FM")
    return microscope

def test_generate_grid_positions_odd_dimensions():
    """Test grid position generation with odd numbers of columns and rows."""
    # Test 3x3 grid
    positions = generate_grid_positions(ncols=3, nrows=3, fov_x=10.0, fov_y=10.0, overlap=0.0)

    # Should have 9 positions
    assert len(positions) == 9

    # Extract x and y coordinates
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]

    # For 3x3 grid with no overlap, step = 10.0
    # Positions should be at offsets: -1, 0, 1 (centered around 0)
    expected_coords = [-10.0, 0.0, 10.0]

    # Check that we have the expected coordinate values
    assert sorted(set(x_coords)) == expected_coords
    assert sorted(set(y_coords)) == expected_coords

    # Check specific positions. Row 0 is the top of the mosaic, so the top-left
    # corner is at positive y -- these were labelled the other way round while the
    # row direction was inverted, and the labels were the only thing that noticed.
    assert (-10.0, 10.0) in positions   # Top-left
    assert (0.0, 0.0) in positions      # Center
    assert (10.0, -10.0) in positions   # Bottom-right

def test_generate_grid_positions_even_dimensions():
    """Test grid position generation with even numbers of columns and rows."""
    # Test 4x4 grid
    positions = generate_grid_positions(ncols=4, nrows=4, fov_x=10.0, fov_y=10.0, overlap=0.0)

    # Should have 16 positions
    assert len(positions) == 16

    # Extract x and y coordinates
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]

    # For 4x4 grid with no overlap, step = 10.0
    # Positions should be at offsets: -1.5, -0.5, 0.5, 1.5 (centered around 0)
    expected_coords = [-15.0, -5.0, 5.0, 15.0]

    # Check that we have the expected coordinate values
    assert sorted(set(x_coords)) == expected_coords
    assert sorted(set(y_coords)) == expected_coords

    # Check that the grid is centered (mean should be close to 0)
    assert abs(np.mean(x_coords)) < 1e-10
    assert abs(np.mean(y_coords)) < 1e-10

def test_generate_grid_positions_with_overlap():
    """Test grid position generation with tile overlap."""
    # Test 3x3 grid with 20% overlap
    positions = generate_grid_positions(ncols=3, nrows=3, fov_x=10.0, fov_y=10.0, overlap=0.2)

    # Should have 9 positions
    assert len(positions) == 9

    # Extract x and y coordinates
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]

    # With 20% overlap, step = 10.0 * (1 - 0.2) = 8.0
    # Positions should be at offsets: -1, 0, 1 → -8.0, 0.0, 8.0
    expected_coords = [-8.0, 0.0, 8.0]

    # Check that we have the expected coordinate values
    assert sorted(set(x_coords)) == expected_coords
    assert sorted(set(y_coords)) == expected_coords

    # Check that the grid is centered
    assert abs(np.mean(x_coords)) < 1e-10
    assert abs(np.mean(y_coords)) < 1e-10

def test_generate_grid_positions_centering():
    """Test that grid positions are properly centered for various dimensions."""
    test_cases = [
        (1, 1),   # Single position
        (2, 2),   # Even x even
        (3, 3),   # Odd x odd
        (2, 3),   # Even x odd
        (3, 2),   # Odd x even
        (4, 6),   # Even x even (different sizes)
        (5, 7),   # Odd x odd (different sizes)
    ]

    for ncols, nrows in test_cases:
        positions = generate_grid_positions(ncols=ncols, nrows=nrows, fov_x=10.0, fov_y=10.0, overlap=0.0)

        # Should have correct number of positions
        assert len(positions) == ncols * nrows

        # Extract coordinates
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]

        # Grid should be centered (mean should be 0)
        assert abs(np.mean(x_coords)) < 1e-10, f"Failed for {ncols}x{nrows}: x_mean = {np.mean(x_coords)}"
        assert abs(np.mean(y_coords)) < 1e-10, f"Failed for {ncols}x{nrows}: y_mean = {np.mean(y_coords)}"

def test_generate_grid_positions_single_dimension():
    """Test grid position generation with single row or column."""
    # Test 1x3 grid (single row)
    positions = generate_grid_positions(ncols=1, nrows=3, fov_x=10.0, fov_y=10.0, overlap=0.0)

    assert len(positions) == 3
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]

    # Only one column, so all x should be 0
    assert all(x == 0.0 for x in x_coords)
    # Y coordinates should be -10, 0, 10
    assert sorted(y_coords) == [-10.0, 0.0, 10.0]

    # Test 3x1 grid (single column)
    positions = generate_grid_positions(ncols=3, nrows=1, fov_x=10.0, fov_y=10.0, overlap=0.0)

    assert len(positions) == 3
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]

    # Only one row, so all y should be 0
    assert all(y == 0.0 for y in y_coords)
    # X coordinates should be -10, 0, 10
    assert sorted(x_coords) == [-10.0, 0.0, 10.0]

def test_generate_grid_positions_return_type():
    """Test that function returns the correct data types."""
    positions = generate_grid_positions(ncols=2, nrows=2, fov_x=10.0, fov_y=10.0, overlap=0.1)

    # Should return list of tuples
    assert isinstance(positions, list)
    assert len(positions) == 4

    for pos in positions:
        assert isinstance(pos, tuple)
        assert len(pos) == 2
        assert isinstance(pos[0], float)
        assert isinstance(pos[1], float)

def test_generate_grid_positions_different_fov():
    """Test grid position generation with different horizontal and vertical FOV values."""
    # Test 2x2 grid with wider horizontal FOV
    positions = generate_grid_positions(ncols=2, nrows=2, fov_x=20.0, fov_y=10.0, overlap=0.0)

    # Should have 4 positions
    assert len(positions) == 4

    # Extract x and y coordinates
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]

    # For 2x2 grid with no overlap:
    # X step = 20.0, positions at offsets: -0.5, 0.5 → -10.0, 10.0
    # Y step = 10.0, positions at offsets: -0.5, 0.5 → -5.0, 5.0
    expected_x_coords = [-10.0, 10.0]
    expected_y_coords = [-5.0, 5.0]

    assert sorted(set(x_coords)) == expected_x_coords
    assert sorted(set(y_coords)) == expected_y_coords

    # Check that grid is centered
    assert abs(np.mean(x_coords)) < 1e-10
    assert abs(np.mean(y_coords)) < 1e-10

    # Check specific positions
    assert (-10.0, -5.0) in positions
    assert (10.0, -5.0) in positions
    assert (-10.0, 5.0) in positions
    assert (10.0, 5.0) in positions

def test_generate_grid_positions_rectangular_fov():
    """Test grid position generation with rectangular FOV (taller than wide)."""
    # Test 3x3 grid with taller vertical FOV
    positions = generate_grid_positions(ncols=3, nrows=3, fov_x=8.0, fov_y=12.0, overlap=0.0)

    # Should have 9 positions
    assert len(positions) == 9

    # Extract x and y coordinates
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]

    # For 3x3 grid with no overlap:
    # X step = 8.0, positions at offsets: -1, 0, 1 → -8.0, 0.0, 8.0
    # Y step = 12.0, positions at offsets: -1, 0, 1 → -12.0, 0.0, 12.0
    expected_x_coords = [-8.0, 0.0, 8.0]
    expected_y_coords = [-12.0, 0.0, 12.0]

    assert sorted(set(x_coords)) == expected_x_coords
    assert sorted(set(y_coords)) == expected_y_coords

    # Check that grid is centered
    assert abs(np.mean(x_coords)) < 1e-10
    assert abs(np.mean(y_coords)) < 1e-10

    # Check corner positions
    assert (-8.0, -12.0) in positions  # Top-left
    assert (0.0, 0.0) in positions     # Center
    assert (8.0, 12.0) in positions    # Bottom-right

def test_generate_grid_positions_asymmetric_fov_with_overlap():
    """Test grid position generation with asymmetric FOV and tile overlap."""
    # Test 2x3 grid with different FOV values and overlap
    positions = generate_grid_positions(ncols=2, nrows=3, fov_x=15.0, fov_y=9.0, overlap=0.2)

    # Should have 6 positions
    assert len(positions) == 6

    # Extract x and y coordinates
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]

    # With 20% overlap:
    # X step = 15.0 * (1 - 0.2) = 12.0, positions at offsets: -0.5, 0.5 → -6.0, 6.0
    # Y step = 9.0 * (1 - 0.2) = 7.2, positions at offsets: -1, 0, 1 → -7.2, 0.0, 7.2
    expected_x_coords = [-6.0, 6.0]
    expected_y_coords = [-7.2, 0.0, 7.2]

    assert sorted(set(x_coords)) == expected_x_coords
    assert sorted(set(y_coords)) == expected_y_coords

    # Check that grid is centered
    assert abs(np.mean(x_coords)) < 1e-10
    assert abs(np.mean(y_coords)) < 1e-10

    # Check some specific positions
    assert (-6.0, -7.2) in positions
    assert (6.0, 0.0) in positions
    assert (-6.0, 7.2) in positions

def test_generate_grid_positions_extreme_aspect_ratio():
    """Test grid position generation with extreme aspect ratio FOV."""
    # Test 4x1 grid with very wide horizontal FOV
    positions = generate_grid_positions(ncols=4, nrows=1, fov_x=40.0, fov_y=5.0, overlap=0.0)

    # Should have 4 positions
    assert len(positions) == 4

    # Extract x and y coordinates
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]

    # For 4x1 grid with no overlap:
    # X step = 40.0, positions at offsets: -1.5, -0.5, 0.5, 1.5 → -60.0, -20.0, 20.0, 60.0
    # Y step = 5.0, only one row so all y = 0.0
    expected_x_coords = [-60.0, -20.0, 20.0, 60.0]
    expected_y_coords = [0.0]

    assert sorted(set(x_coords)) == expected_x_coords
    assert sorted(set(y_coords)) == expected_y_coords

    # Check that all y coordinates are 0 (single row)
    assert all(y == 0.0 for y in y_coords)

    # Check that grid is centered
    assert abs(np.mean(x_coords)) < 1e-10
    assert abs(np.mean(y_coords)) < 1e-10

def test_calculate_grid_overlap():
    """Test overlap calculation from grid positions."""
    # Test with known overlap
    positions = generate_grid_positions(ncols=3, nrows=3, fov_x=10.0, fov_y=8.0, overlap=0.2)
    overlap_x, overlap_y = calculate_grid_overlap(positions, 10.0, 8.0)

    # Should recover the original overlap (within small tolerance)
    assert abs(overlap_x - 0.2) < 1e-10
    assert abs(overlap_y - 0.2) < 1e-10

    # Test with no overlap
    positions_no_overlap = generate_grid_positions(ncols=2, nrows=2, fov_x=5.0, fov_y=5.0, overlap=0.0)
    overlap_x, overlap_y = calculate_grid_overlap(positions_no_overlap, 5.0, 5.0)

    assert abs(overlap_x - 0.0) < 1e-10
    assert abs(overlap_y - 0.0) < 1e-10

    # Test with different horizontal and vertical overlaps
    positions_asym = generate_grid_positions(ncols=2, nrows=3, fov_x=10.0, fov_y=6.0, overlap=0.15)
    overlap_x, overlap_y = calculate_grid_overlap(positions_asym, 10.0, 6.0)

    assert abs(overlap_x - 0.15) < 1e-10
    assert abs(overlap_y - 0.15) < 1e-10

    # Test with single position (should return 0, 0)
    single_position = [(0.0, 0.0)]
    overlap_x, overlap_y = calculate_grid_overlap(single_position, 10.0, 10.0)

    assert overlap_x == 0.0
    assert overlap_y == 0.0

def test_calculate_grid_dimensions():
    """Test grid dimension calculation from positions."""
    # Test 3x4 grid
    positions = generate_grid_positions(ncols=3, nrows=4, fov_x=10.0, fov_y=8.0, overlap=0.1)
    ncols, nrows = calculate_grid_dimensions(positions)

    assert ncols == 3
    assert nrows == 4

    # Test 2x2 grid
    positions_2x2 = generate_grid_positions(ncols=2, nrows=2, fov_x=5.0, fov_y=5.0, overlap=0.0)
    ncols, nrows = calculate_grid_dimensions(positions_2x2)

    assert ncols == 2
    assert nrows == 2

    # Test 1x5 grid (single row)
    positions_1x5 = generate_grid_positions(ncols=1, nrows=5, fov_x=10.0, fov_y=10.0, overlap=0.0)
    ncols, nrows = calculate_grid_dimensions(positions_1x5)

    assert ncols == 1
    assert nrows == 5

    # Test 7x1 grid (single column)
    positions_7x1 = generate_grid_positions(ncols=7, nrows=1, fov_x=10.0, fov_y=10.0, overlap=0.0)
    ncols, nrows = calculate_grid_dimensions(positions_7x1)

    assert ncols == 7
    assert nrows == 1

    # Test empty positions
    empty_positions = []
    ncols, nrows = calculate_grid_dimensions(empty_positions)

    assert ncols == 0
    assert nrows == 0

    # Test single position
    single_position = [(0.0, 0.0)]
    ncols, nrows = calculate_grid_dimensions(single_position)

    assert ncols == 1
    assert nrows == 1

    # Test with different overlaps (should still give same dimensions)
    positions_overlap = generate_grid_positions(ncols=4, nrows=3, fov_x=10.0, fov_y=10.0, overlap=0.25)
    ncols, nrows = calculate_grid_dimensions(positions_overlap)

    assert ncols == 4
    assert nrows == 3

def test_calculate_grid_size_for_area():
    """Test calculation of grid size needed to cover an area."""
    # Test exact fit (no overlap)
    ncols, nrows = calculate_grid_size_for_area(30.0, 20.0, 10.0, 10.0, 0.0)
    assert ncols == 3
    assert nrows == 2

    # Test with overlap
    ncols, nrows = calculate_grid_size_for_area(30.0, 20.0, 10.0, 10.0, 0.1)
    # With 10% overlap, step = 9.0, so we need more tiles
    assert ncols == 4  # (30 - 10) / 9 + 1 = 3.22... -> 4
    assert nrows == 3  # (20 - 10) / 9 + 1 = 2.11... -> 3

    # Test area smaller than FOV
    ncols, nrows = calculate_grid_size_for_area(5.0, 8.0, 10.0, 10.0, 0.0)
    assert ncols == 1
    assert nrows == 1

    # Test area exactly equal to FOV
    ncols, nrows = calculate_grid_size_for_area(10.0, 10.0, 10.0, 10.0, 0.0)
    assert ncols == 1
    assert nrows == 1

    # Test with different FOV dimensions
    ncols, nrows = calculate_grid_size_for_area(100.0, 60.0, 15.0, 12.0, 0.2)
    # step_x = 15 * 0.8 = 12, step_y = 12 * 0.8 = 9.6
    # ncols = ceil((100 - 15) / 12) + 1 = ceil(85/12) + 1 = 8 + 1 = 9
    # nrows = ceil((60 - 12) / 9.6) + 1 = ceil(48/9.6) + 1 = 5 + 1 = 6
    assert ncols == 9
    assert nrows == 6

    # Test error conditions
    import pytest

    # Negative area dimensions
    with pytest.raises(ValueError, match="Area dimensions must be positive"):
        calculate_grid_size_for_area(-10.0, 20.0, 10.0, 10.0, 0.1)

    # Zero FOV
    with pytest.raises(ValueError, match="FOV dimensions must be positive"):
        calculate_grid_size_for_area(30.0, 20.0, 0.0, 10.0, 0.1)

    # Invalid overlap
    with pytest.raises(ValueError, match="Overlap must be between 0.0 and 1.0"):
        calculate_grid_size_for_area(30.0, 20.0, 10.0, 10.0, 1.0)

    # Test single row/column scenarios
    ncols, nrows = calculate_grid_size_for_area(50.0, 8.0, 10.0, 10.0, 0.0)
    assert ncols == 5
    assert nrows == 1

    ncols, nrows = calculate_grid_size_for_area(8.0, 50.0, 10.0, 10.0, 0.0)
    assert ncols == 1
    assert nrows == 5

def test_calculate_grid_coverage_area():
    """Test calculation of total area covered by a grid."""
    # Test no overlap
    width, height = calculate_grid_coverage_area(3, 2, 10.0, 10.0, 0.0)
    # 3 tiles: (3-1) * 10 + 10 = 30, 2 tiles: (2-1) * 10 + 10 = 20
    assert width == 30.0
    assert height == 20.0

    # Test with overlap
    width, height = calculate_grid_coverage_area(3, 2, 10.0, 10.0, 0.1)
    # step = 10 * 0.9 = 9
    # 3 tiles: (3-1) * 9 + 10 = 28, 2 tiles: (2-1) * 9 + 10 = 19
    assert width == 28.0
    assert height == 19.0

    # Test single tile
    width, height = calculate_grid_coverage_area(1, 1, 15.0, 12.0, 0.2)
    assert width == 15.0
    assert height == 12.0

    # Test single row
    width, height = calculate_grid_coverage_area(4, 1, 10.0, 8.0, 0.0)
    # 4 tiles: (4-1) * 10 + 10 = 40, 1 tile: 8
    assert width == 40.0
    assert height == 8.0

    # Test single column
    width, height = calculate_grid_coverage_area(1, 5, 10.0, 8.0, 0.0)
    # 1 tile: 10, 5 tiles: (5-1) * 8 + 8 = 40
    assert width == 10.0
    assert height == 40.0

    # Test different FOV dimensions with overlap
    width, height = calculate_grid_coverage_area(3, 4, 15.0, 12.0, 0.2)
    # step_x = 15 * 0.8 = 12, step_y = 12 * 0.8 = 9.6
    # width = (3-1) * 12 + 15 = 39, height = (4-1) * 9.6 + 12 = 40.8
    assert np.isclose(width, 39.0)
    assert np.isclose(height, 40.8)

    # Test reciprocal relationship with calculate_grid_size_for_area
    # Generate a grid size for an area, then calculate the area it covers
    target_width, target_height = 100.0, 80.0
    fov_x, fov_y = 10.0, 8.0
    overlap = 0.15

    # Calculate required grid size
    ncols, nrows = calculate_grid_size_for_area(target_width, target_height, fov_x, fov_y, overlap)

    # Calculate area covered by that grid
    actual_width, actual_height = calculate_grid_coverage_area(ncols, nrows, fov_x, fov_y, overlap)

    # Should cover at least the target area
    assert actual_width >= target_width
    assert actual_height >= target_height

    # Should not be too much larger (within one tile's worth)
    assert actual_width <= target_width + fov_x
    assert actual_height <= target_height + fov_y

    # Test error conditions
    import pytest

    # Invalid grid dimensions
    with pytest.raises(ValueError, match="Grid dimensions must be positive"):
        calculate_grid_coverage_area(0, 2, 10.0, 10.0, 0.1)

    # Invalid FOV
    with pytest.raises(ValueError, match="FOV dimensions must be positive"):
        calculate_grid_coverage_area(3, 2, -10.0, 10.0, 0.1)

    # Invalid overlap
    with pytest.raises(ValueError, match="Overlap must be between 0.0 and 1.0"):
        calculate_grid_coverage_area(3, 2, 10.0, 10.0, 1.0)

    # Test extreme overlap (high but valid)
    width, height = calculate_grid_coverage_area(3, 2, 10.0, 10.0, 0.9)
    # step = 10 * 0.1 = 1
    # width = (3-1) * 1 + 10 = 12, height = (2-1) * 1 + 10 = 11
    assert width == 12.0
    assert height == 11.0

# Tests for acquire_at_positions function
def test_acquire_at_positions_single_position(fm_microscope):
    """Test acquire_at_positions with a single position using demo microscope."""
    microscope = fm_microscope

    # Create test position
    stage_pos = FibsemStagePosition(x=0.001, y=0.002, z=0.003, r=0, t=np.radians(-180))
    fm_position = FMStagePosition(
        name="test_pos_1",
        stage_position=stage_pos,
        objective_position=0.012
    )

    # Create test channel settings
    channel = ChannelSettings(
        name="DAPI",
        excitation_wavelength=358,
        emission_wavelength=461,
        power=0.1,
        exposure_time=0.5
    )

    # Call function
    result = acquire_at_positions(
        microscope=microscope,
        positions=[fm_position],
        channel_settings=channel
    )

    # Verify result
    assert len(result) == 1
    assert isinstance(result[0], FluorescenceImage)

def test_acquire_at_positions_multiple_positions(fm_microscope):
    """Test acquire_at_positions with multiple positions using demo microscope."""
    microscope = fm_microscope

    # Create test positions
    positions = [
        FMStagePosition(
            name="pos_1",
            stage_position=FibsemStagePosition(x=0.001, y=0.002, z=0.003, r=0, t=np.radians(-180)),
            objective_position=0.010
        ),
        FMStagePosition(
            name="pos_2", 
            stage_position=FibsemStagePosition(x=0.004, y=0.005, z=0.006, r=0, t=np.radians(-180)),
            objective_position=0.015
        )
    ]

    # Create test channel
    channel = ChannelSettings(
        name="GFP",
        excitation_wavelength=488,
        emission_wavelength=509,
        power=0.2,
        exposure_time=0.3
    )

    # Call function
    result = acquire_at_positions(
        microscope=microscope,
        positions=positions,
        channel_settings=channel
    )

    # Verify result
    assert len(result) == 2
    for image in result:
        assert isinstance(image, FluorescenceImage)

def test_acquire_at_positions_with_z_stack(fm_microscope):
    """Test acquire_at_positions with Z-stack parameters using demo microscope."""
    microscope = fm_microscope

    # Create test position
    position = FMStagePosition(
        name="z_stack_pos",
        stage_position=FibsemStagePosition(x=0.001, y=0.002, z=0.003, r=0, t=np.radians(-180)),
        objective_position=0.012
    )

    # Create test channel
    channel = ChannelSettings(
        name="DAPI",
        excitation_wavelength=358,
        emission_wavelength=461,
        power=0.1,
        exposure_time=0.5
    )

    # Create Z-stack parameters
    z_params = ZParameters(
        zmin=-0.005,
        zmax=0.005,
        zstep=0.001
    )

    # Call function
    result = acquire_at_positions(
        microscope=microscope,
        positions=[position],
        channel_settings=channel,
        zparams=z_params
    )

    # Verify result
    assert len(result) == 1
    assert isinstance(result[0], FluorescenceImage)

def test_acquire_at_positions_empty_list(fm_microscope):
    """Test acquire_at_positions with empty positions list raises ValueError."""
    microscope = fm_microscope

    # Create test channel
    channel = ChannelSettings(
        name="DAPI",
        excitation_wavelength=358,
        emission_wavelength=461,
        power=0.1,
        exposure_time=0.5
    )

    # Call function with empty list should raise ValueError
    with pytest.raises(ValueError, match="Positions list cannot be empty"):
        acquire_at_positions(
            microscope=microscope,
            positions=[],
            channel_settings=channel
        )

def test_acquire_channels(demo_microscope):
    """Test acquire_channels with multiple channels using demo microscope."""
    microscope = demo_microscope

    # Create multiple test channels
    channels = [
        ChannelSettings(
            name="DAPI",
            excitation_wavelength=358,
            emission_wavelength=461,
            power=0.1,
            exposure_time=0.5
        ),
        ChannelSettings(
            name="GFP",
            excitation_wavelength=488,
            emission_wavelength=509,
            power=0.2,
            exposure_time=0.3
        )
    ]

    # Call function
    result = acquire_channels(
        microscope=microscope.fm,
        channel_settings=channels
    )

    # Verify result
    assert isinstance(result, FluorescenceImage)

    # Check metadata
    assert len(result.metadata.channels) == 2
    assert result.metadata.channels[0].name == "DAPI"
    assert result.metadata.channels[0].excitation_wavelength == 358
    assert result.metadata.channels[0].emission_wavelength == 461
    assert result.metadata.channels[0].power == 0.1
    assert result.metadata.channels[0].exposure_time == 0.5

    assert result.metadata.channels[1].name == "GFP"
    assert result.metadata.channels[1].excitation_wavelength == 488
    assert result.metadata.channels[1].emission_wavelength == 509
    assert result.metadata.channels[1].power == 0.2
    assert result.metadata.channels[1].exposure_time == 0.3

    # Check image data has correct number of channels
    assert result.data.shape[0] == 2  # Should have 2 channels (CZYX format)

    # Check that acquisition date is set
    assert result.metadata.acquisition_date is not None

def test_acquire_z_stack(demo_microscope):
    """Test acquire_z_stack with Z-stack parameters using demo microscope."""
    microscope = demo_microscope

    # Create test channel
    channel = ChannelSettings(
        name="DAPI",
        excitation_wavelength=358,
        emission_wavelength=461,
        power=0.1,
        exposure_time=0.1
    )

    # Create Z-stack parameters
    z_params = ZParameters(
        zmin=-0.005,
        zmax=0.005,
        zstep=0.001
    )

    # Call function
    result = acquire_z_stack(
        microscope=microscope.fm,
        channel_settings=channel,
        zparams=z_params
    )

    # Verify result
    assert isinstance(result, FluorescenceImage)

    # Check metadata
    assert len(result.metadata.channels) == 1
    assert result.metadata.channels[0].name == "DAPI"
    assert result.metadata.channels[0].excitation_wavelength == 358
    assert result.metadata.channels[0].emission_wavelength == 461
    assert result.metadata.channels[0].power == 0.1
    assert result.metadata.channels[0].exposure_time == 0.1

    # Check Z-stack parameters are preserved
    assert result.metadata.z_positions is not None
    expected_nz = len(z_params.generate_positions(microscope.fm.objective.position))

    assert len(result.metadata.z_positions) == expected_nz

    # Check image data has correct Z dimension
    assert result.data.shape[1] == expected_nz  # Z dimension should match expected slices (CZYX format)

    # Check that acquisition date is set
    assert result.metadata.acquisition_date is not None


def test_acquire_image(fm_microscope):
    """Test acquire_image function using demo microscope."""
    microscope = fm_microscope

    # Create test channel
    channel = ChannelSettings(
        name="GFP",
        excitation_wavelength=488,
        emission_wavelength=509,
        power=0.2,
        exposure_time=0.3,
    )

    # Call function without Z-stack
    result = acquire_image(microscope=microscope.fm, channel_settings=channel)

    # Verify result
    assert isinstance(result, FluorescenceImage)

    # Check metadata
    assert len(result.metadata.channels) == 1
    assert result.metadata.channels[0].name == "GFP"
    assert result.metadata.channels[0].excitation_wavelength == 488
    assert result.metadata.channels[0].emission_wavelength == 509
    assert result.metadata.channels[0].power == 0.2
    assert result.metadata.channels[0].exposure_time == 0.3

    # Check that no Z-stack parameters are set for regular image
    assert result.metadata.z_positions is None

    # Check image data dimensions (should be 2D for single channel, single Z)
    assert len(result.data.shape) >= 2  # At least height and width
    assert result.data.shape[0] == 1  # Single channel (TCZYX format)
    assert result.data.shape[1] == 1

    # Check that acquisition date is set
    assert result.metadata.acquisition_date is not None


def test_acquire_and_stitch_tileset(fm_microscope):
    """Test acquire_and_stitch_tileset auto-converts single channel to list."""
    microscope = fm_microscope

    # Create single test channel (not in a list)
    channel = ChannelSettings(
        name="DAPI",
        excitation_wavelength=358,
        emission_wavelength=461,
        power=0.1,
        exposure_time=0.1,
    )

    # Test with 2x2 grid
    overview_params = OverviewParameters(
        rows=2, cols=2, overlap=0.1, use_zstack=False, autofocus_mode=AutoFocusMode.NONE
    )

    # Call function with single channel (should auto-convert to list)
    result = acquire_and_stitch_tileset(
        microscope=microscope,
        channel_settings=channel,  # Single channel, not list
        overview_parameters=overview_params,
    )

    # Verify result
    assert isinstance(result, FluorescenceImage)

    # Check dimensions: should be (nc_channel=1, nz=1, ny, nx)
    assert len(result.data.shape) == 4
    assert result.data.shape[0] == 1  # 1 channel
    assert result.data.shape[1] == 1  # 1 z-plane

    # Check metadata
    assert len(result.metadata.channels) == 1
    assert result.metadata.channels[0].name == "DAPI"

    # Create multiple test channels
    channel_settings = [
        ChannelSettings(
            name="DAPI",
            excitation_wavelength=358,
            emission_wavelength=461,
            power=0.1,
            exposure_time=0.1,
        ),
        ChannelSettings(
            name="GFP",
            excitation_wavelength=488,
            emission_wavelength=509,
            power=0.2,
            exposure_time=0.1,
        ),
    ]

    # Test with 3x3 grid
    overview_params_multi = OverviewParameters(
        rows=3, cols=3, overlap=0.1, use_zstack=False, autofocus_mode=AutoFocusMode.NONE
    )

    # Call function with multiple channels
    result = acquire_and_stitch_tileset(
        microscope=microscope,
        channel_settings=channel_settings,  # List of channels
        overview_parameters=overview_params_multi,
    )

    # Verify result
    assert isinstance(result, FluorescenceImage)

    # Check dimensions: should be (nc_channel=1, nz=1, ny, nx)
    assert len(result.data.shape) == 4
    assert result.data.shape[0] == 2  # 2 channels
    assert result.data.shape[1] == 1  # 1 z-plane

    # Check metadata
    assert len(result.metadata.channels) == 2
    assert result.metadata.channels[0].name == "DAPI"
    assert result.metadata.channels[1].name == "GFP"

    # z-stack parameters
    z_params = ZParameters(zmin=-0.005, zmax=0.005, zstep=0.001)
    result = acquire_and_stitch_tileset(
        microscope=microscope,
        channel_settings=channel_settings,  # List of channels
        overview_parameters=overview_params_multi,
        zparams=z_params,
    )

    # Verify result
    assert isinstance(result, FluorescenceImage)

    # Check dimensions: should be (nc_channel=1, nz=1, ny, nx)
    assert len(result.data.shape) == 4
    assert result.data.shape[0] == 2  # 2 channels
    assert result.data.shape[1] == 1  # 1 z-plane

    # Check metadata
    assert len(result.metadata.channels) == 2
    assert result.metadata.channels[0].name == "DAPI"
    assert result.metadata.channels[1].name == "GFP"


def test_acquire_multiple_overviews(fm_microscope):
    """Test acquire_multiple_overviews function with multiple FM positions."""
    microscope = fm_microscope

    # Create test positions
    positions = [
        FMStagePosition(
            name="Region1",
            stage_position=FibsemStagePosition(x=0.001, y=0.002, z=0.003),
            objective_position=0.012,
        ),
        FMStagePosition(
            name="Region2",
            stage_position=FibsemStagePosition(x=0.004, y=0.005, z=0.006),
            objective_position=0.015,
        ),
        FMStagePosition(
            name="Region3",
            stage_position=FibsemStagePosition(x=0.007, y=0.008, z=0.009),
            objective_position=0.018,
        ),
    ]

    # Create test channel
    channel = ChannelSettings(
        name="DAPI",
        excitation_wavelength=358,
        emission_wavelength=461,
        power=0.1,
        exposure_time=0.1,
    )

    # Test with 2x2 grid
    overview_params = OverviewParameters(
        rows=2, cols=2, overlap=0.1, use_zstack=False, autofocus_mode=AutoFocusMode.NONE
    )

    # Call function
    result = acquire_multiple_overviews(
        microscope=microscope,
        positions=positions,
        channel_settings=channel,
        overview_parameters=overview_params,
    )

    # Verify result
    assert len(result) == 3  # Should have 3 overview images

    for i, overview in enumerate(result):
        assert isinstance(overview, FluorescenceImage)

        # Check dimensions: should be (nc_channel=1, nz=1, ny, nx)
        assert len(overview.data.shape) == 4
        assert overview.data.shape[0] == 1  # 1 channel
        assert overview.data.shape[1] == 1  # 1 z-plane

        # Check metadata
        if overview.metadata is None:
            raise ValueError("Overview metadata is None")
        assert len(overview.metadata.channels) == 1
        assert overview.metadata.channels[0].name == "DAPI"

        # Check that description contains position name
        assert positions[i].name in overview.metadata.description

        # Check that stage position is set correctly using isclose
        assert overview.metadata.stage_position is not None
        assert np.isclose(
            overview.metadata.stage_position.x, positions[i].stage_position.x, atol=1e-6
        )
        assert np.isclose(
            overview.metadata.stage_position.y, positions[i].stage_position.y, atol=1e-6
        )

    # Call function with empty positions list should raise ValueError
    with pytest.raises(ValueError, match="Positions list cannot be empty"):
        acquire_multiple_overviews(
            microscope=microscope,
            positions=[],  # Empty list
            channel_settings=channel,
            overview_parameters=overview_params,
        )

    # Call function without FM microscope should raise ValueError
    with pytest.raises(ValueError, match="Fluorescence microscope not initialized"):
        microscope.fm = None  # Remove FM microscope
        acquire_multiple_overviews(
            microscope=microscope,
            positions=positions,
            channel_settings=channel,
            overview_parameters=overview_params,
        )


# ---------------------------------------------------------------------------
# Row direction
#
# `acquire_tileset` drives the stage and `stitch_tileset` pastes tiles purely by
# index -- they never communicate, so they have to agree on which way row indices
# run or the mosaic comes out with its rows reversed. `imaging/tiled.py` already
# fixes that convention for beam tilesets (`TilePosition`: "row 0 = top",
# "dy negative = down"), and both tilers express their steps in image coordinates
# and hand them to the same projection. So the FM tiler has to match it.
# ---------------------------------------------------------------------------


def _record_tile_positions(microscope, monkeypatch):
    """Run a tileset with the stage and camera stubbed, returning where tiles land.

    Records the absolute stage positions the acquisition visits, in acquisition order,
    rather than the movement calls it makes to get there. An earlier version of these
    tests recorded `fm_stable_move` calls, which pinned the stepping *mechanism*: they
    broke the moment stepping moved to precomputed absolute positions, even though
    every tile still landed in exactly the same place. Where the tiles end up is the
    thing worth asserting, and it survives the implementation changing underneath.
    """
    positions = []

    def fake_move(position):
        positions.append(position)

    stub = Mock(spec=FluorescenceImage)
    monkeypatch.setattr(microscope, "safe_absolute_stage_movement", fake_move)
    monkeypatch.setattr("fibsem.fm.acquisition.acquire_image", lambda *a, **k: stub)
    return positions


def _run_tileset(microscope, rows, cols, overlap=0.1):
    acquisition.acquire_tileset(
        microscope=microscope,
        channel_settings=ChannelSettings(
            name="DAPI", excitation_wavelength=358, emission_wavelength=461,
            power=0.1, exposure_time=0.1,
        ),
        overview_parameters=OverviewParameters(
            rows=rows, cols=cols, overlap=overlap,
            use_zstack=False, autofocus_mode=AutoFocusMode.NONE,
        ),
    )


def _by_tile(positions, rows, cols):
    """Tiles are visited row-major, so index k is (k // cols, k % cols).

    The acquisition also returns to its starting position when it finishes, which
    lands in the same recording; the tile moves are the leading `rows * cols`.
    """
    assert len(positions) == rows * cols + 1, (
        f"expected one move per tile plus the restore, got {len(positions)} "
        f"for a {rows}x{cols} grid"
    )
    return {(k // cols, k % cols): p for k, p in enumerate(positions[:rows * cols])}


@pytest.mark.parametrize("rows,cols", [(2, 2), (3, 2), (1, 4), (4, 1)])
def test_acquire_tileset_visits_rows_top_to_bottom(fm_microscope, monkeypatch, rows, cols):
    """Row 0 is the top of the mosaic, so it sits at the highest stage y."""
    microscope = fm_microscope
    positions = _record_tile_positions(microscope, monkeypatch)
    _run_tileset(microscope, rows, cols)
    tiles = _by_tile(positions, rows, cols)

    ys = [tiles[(row, 0)].y for row in range(rows)]
    assert ys == sorted(ys, reverse=True), f"rows must descend in y, got {ys}"
    if rows > 1:
        # ... and the grid stays centred on the starting position.
        assert ys[0] == pytest.approx(-ys[-1])

    xs = [tiles[(0, col)].x for col in range(cols)]
    assert xs == sorted(xs), f"columns must ascend in x, got {xs}"


def test_acquire_tileset_uses_absolute_positions_not_accumulated_steps(
    fm_microscope, monkeypatch
):
    """Every tile is projected from the same base, so error cannot accumulate.

    With relative stepping the y pitch between rows is a sum of separately-projected
    steps; with absolute positions it is a difference of two projections from one
    origin. This asserts the pitch is uniform, which is what the difference buys.
    """
    rows, cols = 4, 1
    microscope = fm_microscope
    positions = _record_tile_positions(microscope, monkeypatch)
    _run_tileset(microscope, rows, cols)
    tiles = _by_tile(positions, rows, cols)

    pitches = [tiles[(r + 1, 0)].y - tiles[(r, 0)].y for r in range(rows - 1)]
    assert all(p == pytest.approx(pitches[0]) for p in pitches), (
        f"row pitch must be uniform, got {pitches}"
    )


def test_acquire_tileset_row_direction_matches_the_beam_tiler(fm_microscope, monkeypatch):
    """The FM tiler and the beam tiler must agree on the direction of each axis.

    Pins the two against each other rather than against a hard-coded sign, so the
    convention can only ever be changed in both places at once.
    """
    from fibsem.imaging.tiled import compute_tile_grid
    from fibsem.structures import ImageSettings, OverviewAcquisitionSettings

    rows, cols = 3, 3
    microscope = fm_microscope
    positions = _record_tile_positions(microscope, monkeypatch)
    _run_tileset(microscope, rows, cols)
    tiles = _by_tile(positions, rows, cols)

    beam_tiles = compute_tile_grid(
        OverviewAcquisitionSettings(
            image_settings=ImageSettings(hfw=100e-6, resolution=[1024, 1024]),
            nrows=rows, ncols=cols, overlap=0.1,
        )
    )

    # compute_tile_grid measures from the top-left tile, the acquisition from the
    # centre, so compare directions between tiles rather than absolute offsets.
    origin = tiles[(0, 0)]
    for tile in beam_tiles:
        pos = tiles[(tile.row, tile.col)]
        fm_dx, fm_dy = pos.x - origin.x, pos.y - origin.y
        assert np.sign(fm_dx) == np.sign(tile.dx), (
            f"x direction disagrees at row {tile.row} col {tile.col}: "
            f"FM {fm_dx:+.3e} vs beam {tile.dx:+.3e}"
        )
        assert np.sign(fm_dy) == np.sign(tile.dy), (
            f"y direction disagrees at row {tile.row} col {tile.col}: "
            f"FM {fm_dy:+.3e} vs beam {tile.dy:+.3e}"
        )


def test_acquire_tileset_returns_to_the_starting_position(fm_microscope, monkeypatch):
    """The grid is centred on wherever the stage was, and it is left there."""
    rows, cols = 2, 2
    microscope = fm_microscope
    start = microscope.get_stage_position()
    positions = _record_tile_positions(microscope, monkeypatch)
    _run_tileset(microscope, rows, cols)

    final = positions[-1]
    assert final.x == pytest.approx(start.x)
    assert final.y == pytest.approx(start.y)
    assert final.z == pytest.approx(start.z)


# ---------------------------------------------------------------------------
# Cancellation
#
# A cancel raises OperationCancelledError -- the same type milling, autofocus and
# (now) the beam tiler use -- rather than returning a short tileset. The partial
# tiles stay on the runner, which the function form could not offer.
#
# Previously acquire_and_stitch_tileset inferred a cancel from
#     if not tileset or any(tile is None for row in tileset for tile in row)
# which cannot distinguish a user cancel from a tile that genuinely failed.
# ---------------------------------------------------------------------------


def _runner(microscope, rows, cols, stop_event=None, overlap=0.1):
    return acquisition.FMTiledAcquisitionRunner(
        microscope=microscope,
        channel_settings=ChannelSettings(
            name="DAPI", excitation_wavelength=358, emission_wavelength=461,
            power=0.1, exposure_time=0.1,
        ),
        overview_parameters=OverviewParameters(
            rows=rows, cols=cols, overlap=overlap,
            use_zstack=False, autofocus_mode=AutoFocusMode.NONE,
        ),
        stop_event=stop_event,
    )


def _cancelling_after(n_tiles, stop_event, stub):
    """An acquire_image stand-in that trips the stop event after n tiles."""
    calls = {"n": 0}

    def fake_acquire(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] >= n_tiles:
            stop_event.set()
        return stub

    return fake_acquire


def test_cancel_before_the_first_tile_raises(fm_microscope, monkeypatch):
    microscope = fm_microscope
    _record_tile_positions(microscope, monkeypatch)
    stop_event = threading.Event()
    stop_event.set()
    runner = _runner(microscope, 2, 2, stop_event)

    with pytest.raises(OperationCancelledError):
        runner.run()

    # The grid is allocated up front, so the shape is the grid's whatever happened.
    assert [[t for t in row] for row in runner.tileset] == [[None, None], [None, None]]


def test_cancel_midway_keeps_every_acquired_tile_in_its_grid_position(
    fm_microscope, monkeypatch
):
    """A partial row survives, and each tile stays at its own (row, col).

    Tiles used to be appended a whole row at a time, so a mid-row cancel threw that
    row away. They are now written by index into a pre-allocated grid: nothing
    acquired is lost, and nothing shifts position to fill a gap -- which matters
    because the canvas coordinates a tile stitches to come from its indices.
    """
    microscope = fm_microscope
    _record_tile_positions(microscope, monkeypatch)
    stub = Mock(spec=FluorescenceImage)
    stop_event = threading.Event()
    monkeypatch.setattr(
        "fibsem.fm.acquisition.acquire_image", _cancelling_after(3, stop_event, stub)
    )
    runner = _runner(microscope, 3, 2, stop_event)

    with pytest.raises(OperationCancelledError):
        runner.run()

    # 3 tiles into a 3x2 typewriter traversal: row 0 complete, row 1 half done.
    acquired = [
        (r, c)
        for r, row in enumerate(runner.tileset)
        for c, tile in enumerate(row)
        if tile is not None
    ]
    assert acquired == [(0, 0), (0, 1), (1, 0)]
    assert len(runner.tileset) == 3 and all(len(row) == 2 for row in runner.tileset)


def test_cancel_still_restores_the_starting_position(fm_microscope, monkeypatch):
    """The restore is in a finally block and must survive the raise."""
    microscope = fm_microscope
    start = microscope.get_stage_position()
    positions = _record_tile_positions(microscope, monkeypatch)
    stub = Mock(spec=FluorescenceImage)
    stop_event = threading.Event()
    monkeypatch.setattr(
        "fibsem.fm.acquisition.acquire_image", _cancelling_after(2, stop_event, stub)
    )

    with pytest.raises(OperationCancelledError):
        _runner(microscope, 3, 3, stop_event).run()

    final = positions[-1]
    assert final.x == pytest.approx(start.x)
    assert final.y == pytest.approx(start.y)
    assert final.z == pytest.approx(start.z)


def test_a_cancelled_tile_image_raises(fm_microscope, monkeypatch):
    """acquire_image returning None is itself the cancellation signal."""
    microscope = fm_microscope
    _record_tile_positions(microscope, monkeypatch)
    monkeypatch.setattr("fibsem.fm.acquisition.acquire_image", lambda *a, **k: None)

    with pytest.raises(OperationCancelledError):
        _runner(microscope, 2, 2).run()


def test_stitching_reports_a_cancel_as_none_not_an_error(fm_microscope, monkeypatch):
    """The top-level behaviour callers see is unchanged by the new contract."""
    microscope = fm_microscope
    _record_tile_positions(microscope, monkeypatch)
    monkeypatch.setattr("fibsem.fm.acquisition.acquire_image", lambda *a, **k: None)

    result = acquire_and_stitch_tileset(
        microscope=microscope,
        channel_settings=ChannelSettings(
            name="DAPI", excitation_wavelength=358, emission_wavelength=461,
            power=0.1, exposure_time=0.1,
        ),
        overview_parameters=OverviewParameters(
            rows=2, cols=2, overlap=0.1,
            use_zstack=False, autofocus_mode=AutoFocusMode.NONE,
        ),
    )

    assert result is None


def test_a_genuine_failure_is_not_reported_as_a_cancel(fm_microscope, monkeypatch):
    """The distinction the old heuristic could not make.

    A tile that fails for a real reason must propagate, not be silently reported as
    a user cancel. Previously both produced a short tileset and were indistinguishable.
    """
    microscope = fm_microscope
    _record_tile_positions(microscope, monkeypatch)

    def exploding_acquire(*args, **kwargs):
        raise RuntimeError("camera fell over")

    monkeypatch.setattr("fibsem.fm.acquisition.acquire_image", exploding_acquire)

    with pytest.raises(RuntimeError, match="camera fell over"):
        _runner(microscope, 2, 2).run()


# ── sparse tilesets (FIB-392) ────────────────────────────────────────────
#
# A masked run acquires a subset of the grid. What must not change is *where* the
# acquired tiles go: same stage positions and same canvas coordinates as the dense
# run, so a sparse overview can be compared with, or later filled in from, a full one.


def _sparse_runner(microscope, rows, cols, mask=None, order=None, stop_event=None):
    return acquisition.FMTiledAcquisitionRunner(
        microscope=microscope,
        channel_settings=ChannelSettings(name="DAPI", exposure_time=0.1),
        overview_parameters=OverviewParameters(
            rows=rows, cols=cols, overlap=0.1, use_zstack=False,
            autofocus_mode=AutoFocusMode.NONE,
            tile_order=order or TileOrderStrategy.TYPEWRITER,
            tile_mask=mask,
        ),
        stop_event=stop_event,
    )


def _plus_mask(n):
    """Middle row and middle column enabled; corners skipped."""
    mid = n // 2
    return [[(i == mid or j == mid) for j in range(n)] for i in range(n)]


def test_a_masked_run_visits_only_the_enabled_tiles(fm_microscope, monkeypatch):
    positions = _record_tile_positions(fm_microscope, monkeypatch)

    _sparse_runner(fm_microscope, 5, 5, mask=_plus_mask(5)).run()

    # 9 enabled tiles of 25, plus the restore move at the end. Nothing is driven to a
    # skipped tile -- the point of a sparse overview is the travel and the exposure it
    # does not spend.
    assert len(positions) == 9 + 1


def test_skipped_tiles_stay_none_and_the_grid_keeps_its_shape(fm_microscope, monkeypatch):
    _record_tile_positions(fm_microscope, monkeypatch)
    mask = _plus_mask(5)

    runner = _sparse_runner(fm_microscope, 5, 5, mask=mask)
    tileset = runner.run()

    assert len(tileset) == 5 and all(len(row) == 5 for row in tileset)
    for r in range(5):
        for c in range(5):
            assert (tileset[r][c] is not None) is mask[r][c], f"tile ({r}, {c})"


def test_masking_changes_which_tiles_are_acquired_never_where(fm_microscope, monkeypatch):
    """The headline property, at runner level."""
    dense = _record_tile_positions(fm_microscope, monkeypatch)
    _sparse_runner(fm_microscope, 5, 5).run()
    dense_by_tile = {
        (k // 5, k % 5): p for k, p in enumerate(list(dense)[:25])
    }

    sparse = _record_tile_positions(fm_microscope, monkeypatch)
    mask = _plus_mask(5)
    _sparse_runner(fm_microscope, 5, 5, mask=mask).run()

    enabled = [(r, c) for r in range(5) for c in range(5) if mask[r][c]]
    for (row, col), position in zip(enabled, list(sparse)[:len(enabled)]):
        expected = dense_by_tile[(row, col)]
        assert position.x == pytest.approx(expected.x), f"tile ({row}, {col}) x"
        assert position.y == pytest.approx(expected.y), f"tile ({row}, {col}) y"
        assert position.z == pytest.approx(expected.z), f"tile ({row}, {col}) z"


def test_a_fully_masked_grid_is_rejected(fm_microscope, monkeypatch):
    """Rather than silently returning an all-zero mosaic."""
    _record_tile_positions(fm_microscope, monkeypatch)
    mask = [[False] * 3 for _ in range(3)]

    with pytest.raises(ValueError, match="disables every tile"):
        _sparse_runner(fm_microscope, 3, 3, mask=mask).run()


def test_progress_totals_count_enabled_tiles_not_grid_cells(fm_microscope, monkeypatch):
    """A bar that stops at 9/25 on a successful run reads as a failure."""
    _record_tile_positions(fm_microscope, monkeypatch)
    emitted = []
    fm_microscope.fm.acquisition_progress_signal.connect(emitted.append)

    _sparse_runner(fm_microscope, 5, 5, mask=_plus_mask(5)).run()

    totals = {p["total"] for p in emitted if "total" in p}
    assert totals == {9}
    counters = [p["current"] for p in emitted if p.get("state") == "acquiring"]
    assert counters[-1] == 9


def test_tile_order_drives_the_visit_sequence(fm_microscope, monkeypatch):
    """FM had no ordering at all before this; it was hard-coded row-major."""
    typewriter = _record_tile_positions(fm_microscope, monkeypatch)
    _sparse_runner(fm_microscope, 3, 3, order=TileOrderStrategy.TYPEWRITER).run()

    serpentine = _record_tile_positions(fm_microscope, monkeypatch)
    _sparse_runner(fm_microscope, 3, 3, order=TileOrderStrategy.SERPENTINE).run()

    # Row 1 runs right-to-left under serpentine, so tiles 3..5 are reversed.
    assert [p.x for p in list(serpentine)[3:6]] == [p.x for p in list(typewriter)[3:6]][::-1]
    # Same set of positions either way -- only the order differs.
    assert sorted(round(p.x, 12) for p in list(typewriter)[:9]) == \
           sorted(round(p.x, 12) for p in list(serpentine)[:9])


def test_each_row_autofocus_is_promoted_to_each_tile_for_a_spiral(fm_microscope, monkeypatch):
    """A spiral revisits rows non-sequentially, so "on each new row" is meaningless."""
    _record_tile_positions(fm_microscope, monkeypatch)
    runner = acquisition.FMTiledAcquisitionRunner(
        microscope=fm_microscope,
        channel_settings=ChannelSettings(name="DAPI", exposure_time=0.1),
        overview_parameters=OverviewParameters(
            rows=3, cols=3, overlap=0.1, use_zstack=False,
            autofocus_mode=AutoFocusMode.EACH_ROW,
            tile_order=TileOrderStrategy.SPIRAL,
        ),
        # _setup resolves the autofocus configuration, and rejects a non-NONE mode
        # without one -- so the promotion cannot be observed unless this is supplied.
        autofocus_settings=AutoFocusSettings(),
    )
    runner._setup()

    assert runner._autofocus_mode is AutoFocusMode.EACH_TILE


def test_each_row_autofocus_survives_a_row_wise_order(fm_microscope, monkeypatch):
    """The promotion is specific to SPIRAL; typewriter and serpentine keep rows."""
    _record_tile_positions(fm_microscope, monkeypatch)
    runner = acquisition.FMTiledAcquisitionRunner(
        microscope=fm_microscope,
        channel_settings=ChannelSettings(name="DAPI", exposure_time=0.1),
        overview_parameters=OverviewParameters(
            rows=3, cols=3, overlap=0.1, use_zstack=False,
            autofocus_mode=AutoFocusMode.EACH_ROW,
            tile_order=TileOrderStrategy.SERPENTINE,
        ),
        autofocus_settings=AutoFocusSettings(),
    )
    runner._setup()

    assert runner._autofocus_mode is AutoFocusMode.EACH_ROW


def test_stitching_leaves_skipped_tiles_as_zeros_at_full_canvas_size(fm_microscope):
    """The mosaic is the whole grid whatever the mask, with gaps left black."""
    mask = _plus_mask(3)
    dense = acquire_and_stitch_tileset(
        microscope=fm_microscope,
        channel_settings=ChannelSettings(name="DAPI", exposure_time=0.001),
        overview_parameters=OverviewParameters(rows=3, cols=3, overlap=0.1),
    )
    sparse = acquire_and_stitch_tileset(
        microscope=fm_microscope,
        channel_settings=ChannelSettings(name="DAPI", exposure_time=0.001),
        overview_parameters=OverviewParameters(rows=3, cols=3, overlap=0.1, tile_mask=mask),
    )

    assert sparse.data.shape == dense.data.shape
    # The corners are skipped, so the canvas there was never written.
    tile_h, tile_w = fm_microscope.fm.camera.resolution[1], fm_microscope.fm.camera.resolution[0]
    assert not sparse.data[0, 0, :tile_h // 2, :tile_w // 2].any(), "top-left corner"
    assert sparse.data[0, 0].any(), "the enabled tiles were written"


def test_a_tileset_of_nothing_but_gaps_is_rejected(fm_microscope):
    with pytest.raises(ValueError, match="no acquired tiles"):
        stitch_tileset([[None, None], [None, None]], 0.1, FibsemStagePosition())


# ── the mosaic is just a large image ─────────────────────────────────────
#
# A stitched overview is an ordinary fluorescence image that happens to be big. It
# carries the metadata a single acquisition carries, describing the whole mosaic --
# nothing tileset-specific. What that demands is that the metadata be *true*: the
# centre it reports has to be where the mosaic actually is, whatever was acquired.


@pytest.mark.parametrize(
    ("name", "predicate"),
    [
        ("dense", None),
        ("symmetric-plus", lambda i, j: i == 1 or j == 1),
        ("off-centre-block", lambda i, j: i < 2 and j < 2),
        ("single-corner-tile", lambda i, j: (i, j) == (0, 0)),
    ],
)
def test_the_mosaic_centre_is_the_run_centre_whatever_was_acquired(
    fm_microscope, name, predicate
):
    """Averaging the acquired tiles only finds the centre when they are symmetric.

    A single corner tile of a 3x3 averages to that tile -- a full tile away from the
    mosaic centre, on a canvas that is still the whole 3x3. The grid is laid out
    centred on the starting stage position, so that position is the answer, exactly,
    and independently of the mask.
    """
    mask = (None if predicate is None
            else [[predicate(i, j) for j in range(3)] for i in range(3)])
    centre = fm_microscope.get_stage_position()

    mosaic = acquire_and_stitch_tileset(
        microscope=fm_microscope,
        channel_settings=ChannelSettings(name="DAPI", exposure_time=0.001),
        overview_parameters=OverviewParameters(rows=3, cols=3, overlap=0.1, tile_mask=mask),
    )

    assert mosaic.metadata.stage_position.x == pytest.approx(centre.x, abs=1e-12)
    assert mosaic.metadata.stage_position.y == pytest.approx(centre.y, abs=1e-12)
    assert mosaic.metadata.stage_position.z == pytest.approx(centre.z, abs=1e-12)


def test_the_mosaic_records_the_objective_the_run_used(fm_microscope):
    """Every tile is taken at the run's initial objective position, so it is the
    mosaic's -- not whatever the first acquired tile happened to carry."""
    objective = fm_microscope.fm.objective.position

    mosaic = acquire_and_stitch_tileset(
        microscope=fm_microscope,
        channel_settings=ChannelSettings(name="DAPI", exposure_time=0.001),
        overview_parameters=OverviewParameters(rows=2, cols=2, overlap=0.1),
    )

    assert all(ch.objective_position == pytest.approx(objective)
               for ch in mosaic.metadata.channels)


def test_the_mosaic_carries_the_same_metadata_as_a_single_image(fm_microscope):
    """Only the resolution differs. It is a large image, not a different kind of thing."""
    channel = ChannelSettings(name="DAPI", excitation_wavelength=358, exposure_time=0.001)
    single = acquire_image(fm_microscope.fm, channel)

    mosaic = acquire_and_stitch_tileset(
        microscope=fm_microscope,
        channel_settings=channel,
        overview_parameters=OverviewParameters(rows=2, cols=2, overlap=0.1),
    )

    assert [ch.name for ch in mosaic.metadata.channels] == \
           [ch.name for ch in single.metadata.channels]
    assert mosaic.metadata.pixel_size_x == single.metadata.pixel_size_x
    assert mosaic.metadata.pixel_size_y == single.metadata.pixel_size_y
    assert mosaic.metadata.resolution[0] > single.metadata.resolution[0]
    assert mosaic.metadata.resolution == (mosaic.data.shape[-1], mosaic.data.shape[-2])


def test_stitching_demands_to_be_told_where_the_mosaic_is():
    """Not an optional refinement: it is what places the mosaic on the canvas.

    Deriving it from the tiles is available and wrong, so the parameter is required
    rather than defaulted -- a caller that does not know the centre cannot produce a
    correctly-placed mosaic, and should fail rather than guess.
    """
    with pytest.raises(TypeError, match="centre_position"):
        stitch_tileset([[Mock(spec=FluorescenceImage)]], 0.1)


# ── live mosaic preview ──────────────────────────────────────────────────
#
# The runner owns the mosaic-so-far and publishes it with every tile, the way
# `TiledAcquisitionRunner` does for the beam side, so a viewer stays stateless and
# simply redisplays what it is handed. It is decimated because a full-resolution
# multi-channel 16-bit mosaic is tens of megabytes per emission.


def _preview_frames(microscope, rows, cols, mask=None, channels=1):
    frames = []
    microscope.fm.acquisition_progress_signal.connect(
        lambda d: frames.append(d) if d.get("state") == "tile" else None
    )
    channel_settings = [
        ChannelSettings(name=f"CH{i}", exposure_time=0.001) for i in range(channels)
    ]
    acquisition.FMTiledAcquisitionRunner(
        microscope=microscope,
        channel_settings=channel_settings,
        overview_parameters=OverviewParameters(
            rows=rows, cols=cols, overlap=0.1, tile_mask=mask
        ),
    ).run()
    return frames


def test_a_preview_frame_is_published_for_every_acquired_tile(fm_microscope):
    mask = [[(i == 1 or j == 1) for j in range(3)] for i in range(3)]

    frames = _preview_frames(fm_microscope, 3, 3, mask=mask)

    assert len(frames) == 5  # the plus, not the grid
    assert [f["current"] for f in frames] == [1, 2, 3, 4, 5]
    assert {f["total"] for f in frames} == {5}


def test_the_preview_fills_in_as_tiles_land(fm_microscope):
    """The point of the preview: it has to actually change between tiles."""
    frames = _preview_frames(fm_microscope, 2, 2)

    coverage = [int((f["image"] > 0).sum()) for f in frames]
    assert all(b > a for a, b in zip(coverage, coverage[1:])), coverage


def test_each_preview_frame_is_its_own_array(fm_microscope):
    """Emitting the live buffer would be a read/write race across threads.

    The acquisition runs on a worker thread and the signal is queued, so the slot runs
    on the GUI thread after the buffer has been painted into again. The beam tiler
    emits its canvas directly; this one copies.
    """
    frames = _preview_frames(fm_microscope, 2, 2)

    first = frames[0]["image"]
    assert not any(first is f["image"] for f in frames[1:])
    # and the first frame still shows one tile, not the finished mosaic
    assert (first > 0).sum() < (frames[-1]["image"] > 0).sum()


def test_the_preview_is_decimated(fm_microscope):
    frames = _preview_frames(fm_microscope, 5, 5)

    image = frames[-1]["image"]
    assert max(image.shape[-2:]) <= acquisition.PREVIEW_MAX_DIMENSION
    assert frames[-1]["preview_stride"] > 1


def test_the_preview_leaves_skipped_tiles_blank(fm_microscope):
    """Placement follows the same grid the stitch uses, holes included."""
    mask = [[(i == 1 and j == 1) for j in range(3)] for i in range(3)]

    frames = _preview_frames(fm_microscope, 3, 3, mask=mask)

    image = frames[-1]["image"][0]
    h, w = image.shape
    assert not image[: h // 4, : w // 4].any(), "top-left corner was never acquired"
    assert image[h // 3: 2 * h // 3, w // 3: 2 * w // 3].any(), "the centre tile was"


def test_the_preview_has_one_plane_per_channel(fm_microscope):
    frames = _preview_frames(fm_microscope, 2, 2, channels=2)

    assert frames[-1]["image"].shape[0] == 2


@pytest.mark.parametrize(
    ("shape", "n_channels", "expected"),
    [
        ((8, 8), 1, (1, 8, 8)),            # plain 2D
        ((3, 8, 8), 1, (1, 8, 8)),         # single channel z-stack -> projected
        ((3, 8, 8), 3, (3, 8, 8)),         # three channels, no z -> kept
        ((2, 4, 8, 8), 2, (2, 8, 8)),      # channels + z -> projected
    ],
    ids=["2d", "single-channel-zstack", "multi-channel", "channels-and-z"],
)
def test_tile_data_is_normalised_to_channel_planes(shape, n_channels, expected):
    """The 3D case is ambiguous from the shape alone and is resolved by channel count."""
    data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)

    assert acquisition._to_channel_planes(data, n_channels).shape == expected


# ── autofocus passes and progress ────────────────────────────────────────


def _sweep_settings():
    from fibsem.autofunctions.autofocus import FocusSweepPass

    return AutoFocusSettings(passes=[
        FocusSweepPass(search_range=20e-6, step_size=5e-6),   # coarse: 5 positions
        FocusSweepPass(search_range=6e-6, step_size=1e-6),    # fine:   7 positions
    ])


def _autofocus_payloads(microscope, monkeypatch, mode):
    _record_tile_positions(microscope, monkeypatch)
    seen = []
    microscope.fm.acquisition_progress_signal.connect(
        lambda d: seen.append(d) if d.get("task") == "autofocus" else None
    )
    acquisition.FMTiledAcquisitionRunner(
        microscope=microscope,
        channel_settings=ChannelSettings(name="DAPI", exposure_time=0.001),
        overview_parameters=OverviewParameters(rows=2, cols=2, overlap=0.1,
                                               autofocus_mode=mode),
        autofocus_settings=_sweep_settings(),
    ).run()
    return seen


def test_focusing_once_runs_every_enabled_pass(fm_microscope, monkeypatch):
    """Coarse then fine, which is the whole reason to configure more than one pass.

    Every mode used to take the narrowest pass alone, so a configured coarse sweep was
    silently ignored -- including in the one case it exists for.
    """
    seen = _autofocus_payloads(fm_microscope, monkeypatch, AutoFocusMode.ONCE)

    passes = sorted({(d["pass_index"], d["total_passes"], d["total_zlevels"])
                     for d in seen})
    assert passes == [(1, 2, 5), (2, 2, 7)]


def test_per_tile_focusing_runs_only_the_narrowest_pass(fm_microscope, monkeypatch):
    """It refines an already-good position; a wide sweep at every tile would dominate."""
    seen = _autofocus_payloads(fm_microscope, monkeypatch, AutoFocusMode.EACH_TILE)

    assert sorted({(d["pass_index"], d["total_passes"]) for d in seen}) == [(1, 1)]
    assert {d["total_zlevels"] for d in seen} == {7}          # the fine pass
    assert len([d for d in seen if d["zlevel"] == 1]) == 4    # once per tile


def test_the_sweep_reports_progress_per_position(fm_microscope, monkeypatch):
    """Otherwise the bars freeze for the length of the sweep, which on per-tile
    focusing is most of the run."""
    seen = _autofocus_payloads(fm_microscope, monkeypatch, AutoFocusMode.ONCE)

    assert [d["zlevel"] for d in seen if d["pass_index"] == 1] == [1, 2, 3, 4, 5]
    assert all(d["channel"] == "DAPI" for d in seen)


def test_a_cancelled_sweep_does_not_start_the_next_pass(fm_microscope, monkeypatch):
    _record_tile_positions(fm_microscope, monkeypatch)
    stop_event = threading.Event()
    seen = []
    fm_microscope.fm.acquisition_progress_signal.connect(
        lambda d: seen.append(d) if d.get("task") == "autofocus" else None
    )

    def cancel_during_the_first_pass(*args, **kwargs):
        stop_event.set()
        return Mock(spec=FluorescenceImage)

    monkeypatch.setattr(fm_microscope.fm, "acquire_image", cancel_during_the_first_pass)

    with pytest.raises(OperationCancelledError):
        acquisition.FMTiledAcquisitionRunner(
            microscope=fm_microscope,
            channel_settings=ChannelSettings(name="DAPI", exposure_time=0.001),
            overview_parameters=OverviewParameters(rows=2, cols=2, overlap=0.1,
                                                   autofocus_mode=AutoFocusMode.ONCE),
            autofocus_settings=_sweep_settings(),
            stop_event=stop_event,
        ).run()

    assert {d["pass_index"] for d in seen} == {1}, "the second pass must not start"
