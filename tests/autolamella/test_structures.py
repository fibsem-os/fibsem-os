import os
from dataclasses import dataclass
from enum import Enum
from typing import List

import pytest

from autolamella.structures import (
    AutoLamellaMethod,
    AutoLamellaStage,
    Lamella,
    LamellaState,
    get_autolamella_method,
    is_ready_for,
)

# Test Properties
def test_method_names():
    assert AutoLamellaMethod.ON_GRID.name == "AutoLamella-OnGrid"
    assert AutoLamellaMethod.TRENCH.name == "AutoLamella-Trench"
    assert AutoLamellaMethod.WAFFLE.name == "AutoLamella-Waffle"
    assert AutoLamellaMethod.LIFTOUT.name == "AutoLamella-Liftout"
    assert AutoLamellaMethod.SERIAL_LIFTOUT.name == "AutoLamella-Serial-Liftout"

def test_workflow_lengths():
    assert len(AutoLamellaMethod.ON_GRID.workflow) == 3
    assert len(AutoLamellaMethod.TRENCH.workflow) == 1
    assert len(AutoLamellaMethod.WAFFLE.workflow) == 5
    assert len(AutoLamellaMethod.LIFTOUT.workflow) == 7
    assert len(AutoLamellaMethod.SERIAL_LIFTOUT.workflow) == 7

def test_workflow_contents():
    # Test ON_GRID workflow
    assert AutoLamellaMethod.ON_GRID.workflow == [
        AutoLamellaStage.SetupLamella,
        AutoLamellaStage.MillRough,
        AutoLamellaStage.MillPolishing,
    ]
    
    # Test TRENCH workflow
    assert AutoLamellaMethod.TRENCH.workflow == [
        AutoLamellaStage.MillTrench,
    ]

@pytest.mark.parametrize("method,expected", [
    (AutoLamellaMethod.ON_GRID, True),
    (AutoLamellaMethod.TRENCH, False),
    (AutoLamellaMethod.WAFFLE, True),
    (AutoLamellaMethod.LIFTOUT, False),
    (AutoLamellaMethod.SERIAL_LIFTOUT, False),
])
def test_is_on_grid(method, expected):
    assert method.is_on_grid == expected

@pytest.mark.parametrize("method,expected", [
    (AutoLamellaMethod.ON_GRID, False),
    (AutoLamellaMethod.TRENCH, True),
    (AutoLamellaMethod.WAFFLE, True),
    (AutoLamellaMethod.LIFTOUT, True),
    (AutoLamellaMethod.SERIAL_LIFTOUT, True),
])
def test_is_trench(method, expected):
    assert method.is_trench == expected

@pytest.mark.parametrize("method,expected", [
    (AutoLamellaMethod.ON_GRID, False),
    (AutoLamellaMethod.TRENCH, False),
    (AutoLamellaMethod.WAFFLE, False),
    (AutoLamellaMethod.LIFTOUT, True),
    (AutoLamellaMethod.SERIAL_LIFTOUT, True),
])
def test_is_liftout(method, expected):
    assert method.is_liftout == expected

# Test Navigation Methods
@pytest.mark.parametrize("method,current_stage,expected", [
    # Test ON_GRID navigation
    (AutoLamellaMethod.ON_GRID, AutoLamellaStage.Created, AutoLamellaStage.SetupLamella),
    (AutoLamellaMethod.ON_GRID, AutoLamellaStage.SetupLamella, AutoLamellaStage.MillRough),
    (AutoLamellaMethod.ON_GRID, AutoLamellaStage.MillRough, AutoLamellaStage.MillPolishing),
    (AutoLamellaMethod.ON_GRID, AutoLamellaStage.MillPolishing, None),
    
    # Test TRENCH navigation
    (AutoLamellaMethod.TRENCH, AutoLamellaStage.Created, AutoLamellaStage.MillTrench),
    (AutoLamellaMethod.TRENCH, AutoLamellaStage.MillTrench, None),
    
    # Test edge cases
    (AutoLamellaMethod.ON_GRID, AutoLamellaStage.Finished, AutoLamellaStage.Finished),
    (AutoLamellaMethod.ON_GRID, AutoLamellaStage.PositionReady, AutoLamellaStage.SetupLamella),
])
def test_get_next(method, current_stage, expected):
    assert method.get_next(current_stage) == expected

@pytest.mark.parametrize("method,current_stage,expected", [
    # Test ON_GRID navigation
    (AutoLamellaMethod.ON_GRID, AutoLamellaStage.MillPolishing, AutoLamellaStage.MillRough),
    (AutoLamellaMethod.ON_GRID, AutoLamellaStage.MillRough, AutoLamellaStage.SetupLamella),
    (AutoLamellaMethod.ON_GRID, AutoLamellaStage.SetupLamella, AutoLamellaStage.PositionReady),
    
    # Test TRENCH navigation
    (AutoLamellaMethod.TRENCH, AutoLamellaStage.MillTrench, AutoLamellaStage.PositionReady),
    
    # Test edge cases
    (AutoLamellaMethod.ON_GRID, AutoLamellaStage.Created, AutoLamellaStage.Created),
    (AutoLamellaMethod.ON_GRID, AutoLamellaStage.PositionReady, AutoLamellaStage.Created),
    (AutoLamellaMethod.ON_GRID, AutoLamellaStage.Finished, AutoLamellaStage.MillPolishing),
])
def test_get_previous(method, current_stage, expected):
    assert method.get_previous(current_stage) == expected

# Test Complete Workflow Navigation
def test_complete_workflow_navigation():
    method = AutoLamellaMethod.LIFTOUT
    current_stage = AutoLamellaStage.Created
    
    # Forward navigation through entire workflow
    expected_stages = [
        AutoLamellaStage.MillTrench,
        AutoLamellaStage.MillUndercut,
        AutoLamellaStage.LiftoutLamella,
        AutoLamellaStage.LandLamella,
        AutoLamellaStage.SetupLamella,
        AutoLamellaStage.MillRough,
        AutoLamellaStage.MillPolishing,
    ]
    
    for expected in expected_stages:
        next_stage = method.get_next(current_stage)
        assert next_stage == expected
        current_stage = next_stage
    
    # Verify end of workflow
    assert method.get_next(current_stage) is None
    
    # Backward navigation
    for expected in reversed(expected_stages[:-1]):
        previous_stage = method.get_previous(current_stage)
        assert previous_stage == expected
        current_stage = previous_stage

def test_is_ready_for():

    lamella = Lamella(path = os.getcwd(), 
                      petname="Lamella-01", 
                      number=1, 
                      protocol={},  
                      state=LamellaState(stage=AutoLamellaStage.Created))
    lamella.states[AutoLamellaStage.Created] = LamellaState(stage=AutoLamellaStage.Created)

    method = AutoLamellaMethod.ON_GRID

    # not ready
    for workflow in [AutoLamellaStage.MillRough, AutoLamellaStage.MillPolishing]:
        is_ready = is_ready_for(lamella=lamella, method=method, workflow=workflow)
        assert is_ready is False, f"Workflow: {workflow}: is_ready: {is_ready}"

    lamella.states[AutoLamellaStage.PositionReady] = LamellaState(stage=AutoLamellaStage.PositionReady)

    # ready for setupLamella
    workflow: AutoLamellaStage = AutoLamellaStage.SetupLamella
    is_ready = is_ready_for(lamella=lamella, method=method, workflow=workflow)
    assert is_ready is True, f"Workflow: {workflow}: is_ready: {is_ready}"

    # after completing setup, ready for Rough Milling
    lamella.states[AutoLamellaStage.SetupLamella] = LamellaState(stage=AutoLamellaStage.SetupLamella)

    workflow = AutoLamellaStage.MillRough
    is_ready = is_ready_for(lamella=lamella, method=method, workflow=workflow)
    assert is_ready is True, f"Workflow: {workflow}: is_ready: {is_ready}"

    workflow = AutoLamellaStage.MillPolishing
    is_ready = is_ready_for(lamella=lamella, method=method, workflow=workflow)
    assert is_ready is False, f"Workflow: {workflow}: is_ready: {is_ready}"

    # after completing Rough Milling, ready for Polishing
    lamella.states[AutoLamellaStage.MillRough] = LamellaState(stage=AutoLamellaStage.MillRough)

    workflow = AutoLamellaStage.MillPolishing
    is_ready = is_ready_for(lamella=lamella, method=method, workflow=workflow)
    assert is_ready is True, f"Workflow: {workflow}: is_ready: {is_ready}"
    
    # if lamella is a failure, not ready for any workflow
    lamella.is_failure = True
    workflow: AutoLamellaStage = AutoLamellaStage.SetupLamella
    is_ready = is_ready_for(lamella=lamella, method=method, workflow=workflow)
    assert is_ready is False, f"Workflow: {workflow}: is_ready: {is_ready}"




def test_valid_method_names():
    """Test all official method names resolve correctly"""
    test_cases = [
        ("AutoLamella-OnGrid", AutoLamellaMethod.ON_GRID),
        ("AutoLamella-Waffle", AutoLamellaMethod.WAFFLE),
        ("AutoLamella-Trench", AutoLamellaMethod.TRENCH),
        ("AutoLamella-Liftout", AutoLamellaMethod.LIFTOUT),
        ("AutoLamella-Serial-Liftout", AutoLamellaMethod.SERIAL_LIFTOUT),
    ]
    
    for name, expected_method in test_cases:
        assert get_autolamella_method(name) == expected_method

def test_alias_resolution():
    """Test that all aliases resolve to correct methods"""
    test_cases = [
        # ON_GRID aliases
        ("autolamella-on-grid", AutoLamellaMethod.ON_GRID),
        ("on-grid", AutoLamellaMethod.ON_GRID),
        ("AutoLiftout", AutoLamellaMethod.ON_GRID),
        
        # WAFFLE aliases
        ("autolamella-waffle", AutoLamellaMethod.WAFFLE),
        ("waffle", AutoLamellaMethod.WAFFLE),
        
        # TRENCH aliases
        ("autolamella-trench", AutoLamellaMethod.TRENCH),
        ("trench", AutoLamellaMethod.TRENCH),
        
        # LIFTOUT aliases
        ("autolamella-liftout", AutoLamellaMethod.LIFTOUT),
        ("liftout", AutoLamellaMethod.LIFTOUT),
        
        # SERIAL_LIFTOUT aliases
        ("autolamella-serial-liftout", AutoLamellaMethod.SERIAL_LIFTOUT),
        ("serial-liftout", AutoLamellaMethod.SERIAL_LIFTOUT),
    ]
    
    for alias, expected_method in test_cases:
        assert get_autolamella_method(alias) == expected_method

def test_case_insensitivity():
    """Test that method resolution is case-insensitive"""
    test_cases = [
        ("AUTOLAMELLA-ON-GRID", AutoLamellaMethod.ON_GRID),
        ("autolamella-on-grid", AutoLamellaMethod.ON_GRID),
        ("AutoLamella-OnGrid", AutoLamellaMethod.ON_GRID),
        ("ON-GRID", AutoLamellaMethod.ON_GRID),
        ("on-grid", AutoLamellaMethod.ON_GRID),
        ("WAFFLE", AutoLamellaMethod.WAFFLE),
        ("waffle", AutoLamellaMethod.WAFFLE),
    ]
    
    for name, expected_method in test_cases:
        assert get_autolamella_method(name) == expected_method

def test_invalid_method_names():
    """Test that invalid method names raise ValueError with proper message"""
    invalid_names = [
        "",  # Empty string
        "unknown",  # Non-existent method
        "auto-lamella",  # Partial match
        "grid",  # Partial match
        "serial",  # Partial match
        "auto",  # Partial match
        "lamella",  # Partial match
    ]
    
    for invalid_name in invalid_names:
        with pytest.raises(ValueError) as exc_info:
            get_autolamella_method(invalid_name)
        
        # Verify error message contains the invalid name
        assert invalid_name in str(exc_info.value)
        # Verify error message contains "Valid methods are:"
        assert "Valid methods are:" in str(exc_info.value)

def test_whitespace_handling():
    """Test handling of whitespace in method names"""
    test_cases = [
        (" on-grid ", AutoLamellaMethod.ON_GRID),  # Leading/trailing spaces
        ("\ton-grid\n", AutoLamellaMethod.ON_GRID),  # Tabs and newlines
        ("  AutoLamella-OnGrid  ", AutoLamellaMethod.ON_GRID),  # Multiple spaces
    ]
    
    for name, expected_method in test_cases:
        with pytest.raises(ValueError):
            get_autolamella_method(name)

@pytest.mark.parametrize("method", list(AutoLamellaMethod))
def test_bidirectional_resolution(method):
    """Test that each method's name resolves back to itself"""
    resolved_method = get_autolamella_method(method.name)
    assert resolved_method == method

def test_all_aliases_listed():
    """Test that all aliases in the error message match the actual aliases"""
    try:
        get_autolamella_method("invalid_method")
    except ValueError as e:
        error_message = str(e)
        # Extract the list of valid methods from the error message
        valid_methods_str = error_message.split("Valid methods are: ")[1]
        valid_methods_list = eval(valid_methods_str)  # Safe since we control the input
        
        # Check that all actual aliases are in the error message
        method_aliases = {
            AutoLamellaMethod.ON_GRID: ["autolamella-on-grid", "on-grid", "AutoLamella-OnGrid", "AutoLiftout"],
            AutoLamellaMethod.WAFFLE: ["autolamella-waffle", "waffle", "AutoLamella-Waffle"],
            AutoLamellaMethod.TRENCH: ["autolamella-trench", "trench", "AutoLamella-Trench"],
            AutoLamellaMethod.LIFTOUT: ["autolamella-liftout", "liftout", "AutoLamella-Liftout"],
            AutoLamellaMethod.SERIAL_LIFTOUT: ["autolamella-serial-liftout", "serial-liftout", "AutoLamella-Serial-Liftout"],
        }
        
        all_aliases = set(alias for aliases in method_aliases.values() for alias in aliases)
        assert set(valid_methods_list) == all_aliases