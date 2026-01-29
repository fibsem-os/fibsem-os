"""Tests for napari utilities - focused on version compatibility checks."""

import sys
import os

# Set up environment variables for headless testing
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest


def test_napari_supports_projection_mode_version_check():
    """Test that the projection_mode support check works correctly."""
    # Import after setting environment variable
    from fibsem.ui.napari.utilities import _napari_supports_projection_mode
    
    # This test just verifies the function exists and returns a boolean
    # The actual version check depends on the installed napari version
    result = _napari_supports_projection_mode()
    assert isinstance(result, bool)


def test_napari_supports_border_width_version_check():
    """Test that the border_width support check works correctly."""
    from fibsem.ui.napari.utilities import _napari_supports_border_width
    
    # This test just verifies the function exists and returns a boolean
    result = _napari_supports_border_width()
    assert isinstance(result, bool)


def test_add_points_layer_removes_projection_mode_when_not_supported():
    """Test that add_points_layer handles projection_mode gracefully."""
    try:
        import napari
    except ImportError:
        pytest.skip("napari not installed")
    
    from fibsem.ui.napari.utilities import add_points_layer, _napari_supports_projection_mode
    import numpy as np
    
    # Create a minimal viewer
    viewer = napari.Viewer(show=False)
    
    try:
        # Test data
        data = np.array([[0, 0], [1, 1], [2, 2]])
        
        # Add points with projection_mode
        layer = add_points_layer(
            viewer,
            data=data,
            name="Test Points",
            projection_mode="all"
        )
        
        # Verify the layer was created
        assert layer is not None
        assert layer.name == "Test Points"
        assert len(layer.data) == 3
        
        # If projection_mode is supported, it should be set
        if _napari_supports_projection_mode():
            # This attribute might not exist in older versions, so we check conditionally
            if hasattr(layer, 'projection_mode'):
                assert layer.projection_mode == "all"
    finally:
        viewer.close()


def test_add_points_layer_without_projection_mode():
    """Test that add_points_layer works without projection_mode parameter."""
    try:
        import napari
    except ImportError:
        pytest.skip("napari not installed")
    
    from fibsem.ui.napari.utilities import add_points_layer
    import numpy as np
    
    # Create a minimal viewer
    viewer = napari.Viewer(show=False)
    
    try:
        # Test data
        data = np.array([[0, 0], [1, 1], [2, 2]])
        
        # Add points without projection_mode
        layer = add_points_layer(
            viewer,
            data=data,
            name="Test Points"
        )
        
        # Verify the layer was created
        assert layer is not None
        assert layer.name == "Test Points"
        assert len(layer.data) == 3
    finally:
        viewer.close()
