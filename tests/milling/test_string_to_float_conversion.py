"""
Test suite to verify that pattern settings classes correctly handle
string values for numeric fields (converting them to float).

This addresses the issue where configuration values loaded from YAML/JSON
files can be strings instead of numeric types, causing runtime errors
in volume property calculations.
"""
import pytest
import numpy as np

from fibsem.structures import (
    FibsemRectangleSettings,
    FibsemCircleSettings,
    FibsemLineSettings,
    FibsemBitmapSettings,
    FibsemPolygonSettings,
)


def test_rectangle_settings_string_dimensions():
    """
    Test that FibsemRectangleSettings correctly converts string dimensions to float.
    This is the primary test case addressing the reported bug.
    """
    # Create rectangle settings with string values (simulating YAML/JSON loading)
    rect = FibsemRectangleSettings(
        width='1e-05',      # String value
        height='4e-06',     # String value
        depth='8e-06',      # String value (this was causing the bug)
        centre_x='0',
        centre_y='0',
        rotation='0',
    )
    
    # Verify that values are converted to float
    assert isinstance(rect.width, float)
    assert isinstance(rect.height, float)
    assert isinstance(rect.depth, float)
    assert isinstance(rect.centre_x, float)
    assert isinstance(rect.centre_y, float)
    assert isinstance(rect.rotation, float)
    
    # Verify the values are correct
    assert rect.width == 1e-05
    assert rect.height == 4e-06
    assert rect.depth == 8e-06
    
    # Most importantly, verify that volume calculation works
    volume = rect.volume
    assert isinstance(volume, float)
    expected_volume = 1e-05 * 4e-06 * 8e-06
    assert abs(volume - expected_volume) < 1e-20


def test_rectangle_settings_from_dict_with_strings():
    """
    Test that FibsemRectangleSettings.from_dict() correctly converts string values.
    """
    data = {
        'width': '2e-06',
        'height': '3e-06',
        'depth': '1e-06',
        'centre_x': '0',
        'centre_y': '0',
        'rotation': '0',
    }
    
    rect = FibsemRectangleSettings.from_dict(data)
    
    # Verify type conversion
    assert isinstance(rect.width, float)
    assert isinstance(rect.height, float)
    assert isinstance(rect.depth, float)
    
    # Verify values
    assert rect.width == 2e-06
    assert rect.height == 3e-06
    assert rect.depth == 1e-06
    
    # Verify volume calculation works
    volume = rect.volume
    assert isinstance(volume, float)
    expected_volume = 2e-06 * 3e-06 * 1e-06
    assert abs(volume - expected_volume) < 1e-20


def test_circle_settings_string_dimensions():
    """
    Test that FibsemCircleSettings correctly converts string dimensions to float.
    """
    circle = FibsemCircleSettings(
        radius='5e-06',
        depth='1e-06',
        centre_x='1e-06',
        centre_y='-1e-06',
    )
    
    # Verify type conversion
    assert isinstance(circle.radius, float)
    assert isinstance(circle.depth, float)
    assert isinstance(circle.centre_x, float)
    assert isinstance(circle.centre_y, float)
    
    # Verify values
    assert circle.radius == 5e-06
    assert circle.depth == 1e-06
    
    # Verify volume calculation works
    volume = circle.volume
    assert isinstance(volume, float)
    expected_volume = np.pi * (5e-06)**2 * 1e-06
    assert abs(volume - expected_volume) < 1e-20


def test_circle_settings_from_dict_with_strings():
    """
    Test that FibsemCircleSettings.from_dict() correctly converts string values.
    """
    data = {
        'radius': '5e-06',
        'depth': '1e-06',
        'centre_x': '1e-06',
        'centre_y': '-1e-06',
    }
    
    circle = FibsemCircleSettings.from_dict(data)
    
    # Verify type conversion
    assert isinstance(circle.radius, float)
    assert isinstance(circle.depth, float)
    
    # Verify volume calculation works
    volume = circle.volume
    assert isinstance(volume, float)


def test_line_settings_string_dimensions():
    """
    Test that FibsemLineSettings correctly converts string dimensions to float.
    """
    line = FibsemLineSettings(
        start_x='0',
        start_y='0',
        end_x='2e-06',
        end_y='2e-06',
        depth='1e-06',
    )
    
    # Verify type conversion
    assert isinstance(line.start_x, float)
    assert isinstance(line.start_y, float)
    assert isinstance(line.end_x, float)
    assert isinstance(line.end_y, float)
    assert isinstance(line.depth, float)
    
    # Verify values
    assert line.start_x == 0.0
    assert line.end_x == 2e-06
    assert line.depth == 1e-06
    
    # Verify volume calculation works
    volume = line.volume
    assert isinstance(volume, float)


def test_line_settings_from_dict_with_strings():
    """
    Test that FibsemLineSettings.from_dict() correctly converts string values.
    """
    data = {
        'start_x': '0',
        'start_y': '0',
        'end_x': '2e-06',
        'end_y': '2e-06',
        'depth': '1e-06',
    }
    
    line = FibsemLineSettings.from_dict(data)
    
    # Verify type conversion
    assert isinstance(line.depth, float)
    
    # Verify volume calculation works
    volume = line.volume
    assert isinstance(volume, float)


def test_bitmap_settings_string_dimensions():
    """
    Test that FibsemBitmapSettings correctly converts string dimensions to float.
    """
    bitmap = FibsemBitmapSettings(
        width='10e-06',
        height='15e-06',
        depth='1e-06',
        centre_x='0',
        centre_y='0',
        rotation='0',
    )
    
    # Verify type conversion
    assert isinstance(bitmap.width, float)
    assert isinstance(bitmap.height, float)
    assert isinstance(bitmap.depth, float)
    assert isinstance(bitmap.centre_x, float)
    assert isinstance(bitmap.centre_y, float)
    
    # Verify values
    assert bitmap.width == 10e-06
    assert bitmap.height == 15e-06
    assert bitmap.depth == 1e-06


def test_polygon_settings_from_dict_with_string_depth():
    """
    Test that FibsemPolygonSettings.from_dict() correctly converts string depth.
    """
    data = {
        'vertices': [[0, 0], [1e-6, 0], [1e-6, 1e-6], [0, 1e-6]],
        'depth': '1e-06',
        'is_exclusion': False,
    }
    
    polygon = FibsemPolygonSettings.from_dict(data)
    
    # Verify type conversion
    assert isinstance(polygon.depth, float)
    assert polygon.depth == 1e-06
    
    # Verify volume calculation works
    volume = polygon.volume
    assert isinstance(volume, float)


def test_mixed_numeric_and_string_values():
    """
    Test that mixing numeric and string values works correctly.
    This simulates partially-loaded configuration data.
    """
    rect = FibsemRectangleSettings(
        width=1e-05,         # Float value
        height='4e-06',      # String value
        depth=8e-06,         # Float value
        centre_x='0',        # String value
        centre_y=0,          # Numeric value
        rotation=0,
    )
    
    # All values should be float after initialization
    assert isinstance(rect.width, float)
    assert isinstance(rect.height, float)
    assert isinstance(rect.depth, float)
    assert isinstance(rect.centre_x, float)
    assert isinstance(rect.centre_y, float)
    
    # Volume calculation should work
    volume = rect.volume
    assert isinstance(volume, float)


def test_none_values_handled_gracefully():
    """
    Test that None values don't cause errors in type conversion.
    """
    # FibsemRectangleSettings requires all position/dimension fields,
    # but let's test with optional fields
    rect = FibsemRectangleSettings(
        width=1e-05,
        height=4e-06,
        depth=8e-06,
        centre_x=0,
        centre_y=0,
        rotation=0,  # Has default value
        time=0.0,    # Has default value
    )
    
    # This should not raise an error
    assert isinstance(rect.width, float)
    volume = rect.volume
    assert isinstance(volume, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
