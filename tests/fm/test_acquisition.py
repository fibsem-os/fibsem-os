import numpy as np

from fibsem.fm.acquisition import generate_grid_positions, plot_grid_positions, calculate_grid_overlap, calculate_grid_dimensions, calculate_grid_size_for_area, calculate_grid_coverage_area


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
    
    # Check specific positions
    assert (-10.0, -10.0) in positions  # Top-left
    assert (0.0, 0.0) in positions      # Center
    assert (10.0, 10.0) in positions    # Bottom-right


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
