import numpy as np

from fibsem.fm.acquisition import generate_grid_positions


def test_generate_grid_positions_odd_dimensions():
    """Test grid position generation with odd numbers of columns and rows."""
    # Test 3x3 grid
    positions = generate_grid_positions(ncols=3, nrows=3, fov=10.0, tile_overlap=0.0)
    
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
    positions = generate_grid_positions(ncols=4, nrows=4, fov=10.0, tile_overlap=0.0)
    
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
    positions = generate_grid_positions(ncols=3, nrows=3, fov=10.0, tile_overlap=0.2)
    
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
        positions = generate_grid_positions(ncols=ncols, nrows=nrows, fov=10.0, tile_overlap=0.0)
        
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
    positions = generate_grid_positions(ncols=1, nrows=3, fov=10.0, tile_overlap=0.0)
    
    assert len(positions) == 3
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]
    
    # Only one column, so all x should be 0
    assert all(x == 0.0 for x in x_coords)
    # Y coordinates should be -10, 0, 10
    assert sorted(y_coords) == [-10.0, 0.0, 10.0]
    
    # Test 3x1 grid (single column)
    positions = generate_grid_positions(ncols=3, nrows=1, fov=10.0, tile_overlap=0.0)
    
    assert len(positions) == 3
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]
    
    # Only one row, so all y should be 0
    assert all(y == 0.0 for y in y_coords)
    # X coordinates should be -10, 0, 10
    assert sorted(x_coords) == [-10.0, 0.0, 10.0]


def test_generate_grid_positions_return_type():
    """Test that function returns the correct data types."""
    positions = generate_grid_positions(ncols=2, nrows=2, fov=10.0, tile_overlap=0.1)
    
    # Should return list of tuples
    assert isinstance(positions, list)
    assert len(positions) == 4
    
    for pos in positions:
        assert isinstance(pos, tuple)
        assert len(pos) == 2
        assert isinstance(pos[0], float)
        assert isinstance(pos[1], float)