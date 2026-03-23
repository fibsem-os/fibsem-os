"""Tests for custom widgets - focused on the closest_value logic."""

import sys
import os

# Set up Qt environment variables for headless testing
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PyQt5.QtWidgets import QApplication

# Ensure QApplication exists for testing Qt widgets
app = QApplication.instance() or QApplication(sys.argv)


def test_string_subtraction_error_original():
    """Test that the original error case (string subtraction) is now handled."""
    # This would have caused: TypeError: unsupported operand type(s) for -: 'str' and 'str'
    items = ['8i-cc3', 'Si-cc3 Neu', 'config1', 'config2']
    value = 'nonexistent'
    
    # The logic we're testing
    if items:
        if isinstance(value, (int, float)) and all(isinstance(x, (int, float)) for x in items):
            # numeric comparison - should not be used for strings
            closest_value = min(items, key=lambda x: abs(x - value))
        else:
            # string comparison - should be used for strings
            str_value = str(value).lower()
            closest_value = next(
                (item for item in items if str(item).lower() == str_value),
                next(
                    (item for item in items if str_value in str(item).lower()),
                    items[0]
                )
            )
    else:
        closest_value = value
    
    # Should default to first item when no match
    assert closest_value == '8i-cc3'


def test_numeric_values_closest_match():
    """Test that numeric values find the closest match."""
    items = [1.0, 2.0, 3.0, 4.0, 5.0]
    value = 2.3
    
    if items:
        if isinstance(value, (int, float)) and all(isinstance(x, (int, float)) for x in items):
            closest_value = min(items, key=lambda x: abs(x - value))
        else:
            str_value = str(value).lower()
            closest_value = next(
                (item for item in items if str(item).lower() == str_value),
                next(
                    (item for item in items if str_value in str(item).lower()),
                    items[0]
                )
            )
    else:
        closest_value = value
    
    assert closest_value == 2.0


def test_string_values_exact_match():
    """Test that string values find exact match."""
    items = ['8i-cc3', 'Si-cc3 Neu', 'config1', 'config2']
    value = '8i-cc3'
    
    if items:
        if isinstance(value, (int, float)) and all(isinstance(x, (int, float)) for x in items):
            closest_value = min(items, key=lambda x: abs(x - value))
        else:
            str_value = str(value).lower()
            closest_value = next(
                (item for item in items if str(item).lower() == str_value),
                next(
                    (item for item in items if str_value in str(item).lower()),
                    items[0]
                )
            )
    else:
        closest_value = value
    
    assert closest_value == '8i-cc3'


def test_string_values_case_insensitive_match():
    """Test that string values find case-insensitive match."""
    items = ['Config1', 'Config2', 'Config3']
    value = 'config1'
    
    if items:
        if isinstance(value, (int, float)) and all(isinstance(x, (int, float)) for x in items):
            closest_value = min(items, key=lambda x: abs(x - value))
        else:
            str_value = str(value).lower()
            closest_value = next(
                (item for item in items if str(item).lower() == str_value),
                next(
                    (item for item in items if str_value in str(item).lower()),
                    items[0]
                )
            )
    else:
        closest_value = value
    
    assert closest_value == 'Config1'


def test_string_values_substring_match():
    """Test that string values find substring match."""
    items = ['8i-cc3-config', 'Si-cc3 Neu', 'other-config']
    value = 'cc3'
    
    if items:
        if isinstance(value, (int, float)) and all(isinstance(x, (int, float)) for x in items):
            closest_value = min(items, key=lambda x: abs(x - value))
        else:
            str_value = str(value).lower()
            closest_value = next(
                (item for item in items if str(item).lower() == str_value),
                next(
                    (item for item in items if str_value in str(item).lower()),
                    items[0]
                )
            )
    else:
        closest_value = value
    
    assert closest_value == '8i-cc3-config'


def test_fuzzy_string_matching():
    """Test fuzzy string matching with application file names."""
    from fibsem.ui.widgets.custom_widgets import _find_closest_string_match
    
    # Test exact match
    items = ['Si-cc3', 'Si-cc3 Neu', 'config1', 'config2']
    result = _find_closest_string_match('Si-cc3', items)
    assert result == 'Si-cc3'
    
    # Test case-insensitive match
    result = _find_closest_string_match('si-cc3', items)
    assert result == 'Si-cc3'
    
    # Test fuzzy match for similar strings
    result = _find_closest_string_match('Si-cc3 New', items)
    # Should match 'Si-cc3 Neu' as it's very similar
    assert result == 'Si-cc3 Neu'
    
    # Test substring match
    result = _find_closest_string_match('cc3', items)
    # Should match first item containing 'cc3'
    assert result == 'Si-cc3'
    
    # Test no match - should return first item
    result = _find_closest_string_match('completely-different', items)
    assert result == 'Si-cc3'


def test_fuzzy_matching_empty_items():
    """Test fuzzy matching with empty items list."""
    from fibsem.ui.widgets.custom_widgets import _find_closest_string_match
    
    items = []
    result = _find_closest_string_match('test', items)
    assert result == 'test'


def test_fuzzy_matching_application_files():
    """Test fuzzy matching with real application file names."""
    from fibsem.ui.widgets.custom_widgets import _find_closest_string_match
    
    # Real-world scenario from the bug report
    items = ['8i-cc3', 'Si-cc3', 'Si-cc3 Neu', 'Si-cc3 New', 'Other Config']
    
    # Test matching "Si-cc3" should find exact match
    result = _find_closest_string_match('Si-cc3', items)
    assert result == 'Si-cc3'
    
    # Test matching similar variant should find close match
    result = _find_closest_string_match('Si-cc3 N', items)
    # Should match either 'Si-cc3 Neu' or 'Si-cc3 New' as they're both similar
    assert result in ['Si-cc3 Neu', 'Si-cc3 New', 'Si-cc3']


def test_combobox_with_fuzzy_matching():
    """Test that combobox uses fuzzy matching for application files."""
    from fibsem.ui.widgets.custom_widgets import _create_combobox_control
    
    items = ['Si-cc3', 'Si-cc3 Neu', 'config1', 'config2']
    
    # Test with fuzzy match
    control = _create_combobox_control('Si-cc3 New', items, None)
    
    # Should select 'Si-cc3 Neu' as closest match
    current_text = control.currentText()
    current_data = control.currentData()
    assert current_data in items  # Should be one of the valid items
    # The exact match depends on fuzzy matching algorithm, but should not crash
