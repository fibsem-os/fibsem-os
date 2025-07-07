# Milling Task Architecture Refactor

## Overview

This refactor introduces a task-based architecture for milling operations, consolidating parameters that had 'soft' requirements to be the same across all stages into a higher-level `FibsemMillingTask` wrapper.

## Problem Statement

Previously, milling stages had several parameters that:
- **Had 'soft' requirements to be the same** across all stages (e.g., field of view, milling channel, alignment settings)
- **Should be set consistently** for all stages in a sequence (e.g., imaging settings, post-milling acquisition)
- **Created parameter duplication** and potential inconsistency between stages

## Changes Made

### New Architecture

1. **FibsemMillingTaskConfig** - Encapsulates task-level parameters:
   - `hfw` (field of view) - Should be consistent across stages
   - `milling_channel` - Beam type for milling operations
   - `acquire_after_milling` - Whether to acquire images after milling
   - `alignment` - Alignment settings for drift correction
   - `imaging` - Settings for post-milling image acquisition

2. **FibsemMillingTask** - Higher-level wrapper that orchestrates stage execution:
   - Contains task-level configuration
   - Manages list of stages
   - Owns reference image for alignment
   - Provides unique task ID for tracking

### Key Benefits

- **Eliminates parameter duplication** - Task-level settings defined once
- **Ensures consistency** - All stages use the same task-level parameters
- **Clean separation of concerns** - Task orchestration vs. stage execution
- **Better dependency management** - Strategies receive config without knowing full task structure

## Usage Example

### Before (Old Architecture)
```python
# Had to set parameters on each stage individually
stage1 = FibsemMillingStage(
    name="Rough Mill",
    milling=FibsemMillingSettings(hfw=150e-6, milling_channel=BeamType.ION),
    alignment=MillingAlignment(enabled=True),
    imaging=ImageSettings(path="/data/images")
)

stage2 = FibsemMillingStage(
    name="Fine Mill", 
    milling=FibsemMillingSettings(hfw=150e-6, milling_channel=BeamType.ION),  # Duplication!
    alignment=MillingAlignment(enabled=True),  # Duplication!
    imaging=ImageSettings(path="/data/images")  # Duplication!
)
```

### After (New Architecture)
```python
from fibsem.milling.base import FibsemMillingTask, FibsemMillingTaskConfig

# Define task-level configuration once
config = FibsemMillingTaskConfig(
    hfw=150e-6,
    milling_channel=BeamType.ION,
    acquire_after_milling=True,
    alignment=MillingAlignment(enabled=True),
    imaging=ImageSettings(path="/data/images")
)

# Create stages with only stage-specific parameters
stage1 = FibsemMillingStage(
    name="Rough Mill",
    milling=FibsemMillingSettings(milling_current=1e-9),
    pattern=RectanglePattern(width=10e-6, height=5e-6)
)

stage2 = FibsemMillingStage(
    name="Fine Mill",
    milling=FibsemMillingSettings(milling_current=100e-12),
    pattern=RectanglePattern(width=8e-6, height=4e-6)
)

# Create and run task
task = FibsemMillingTask(
    name="Lamella Preparation",
    config=config,
    stages=[stage1, stage2]
)

# Execute the task
task.run(microscope=microscope)
```

## Technical Implementation

### Parameter Application
Task-level parameters are applied to stages during the `setup_milling` phase:
```python
def setup_milling(microscope, milling_stage, config, reference_image=None):
    # Apply task-level configuration to stage
    milling_stage.milling.hfw = config.hfw
    milling_stage.milling.milling_channel = config.milling_channel
    
    # Setup microscope with updated settings
    microscope.setup_milling(mill_settings=milling_stage.milling)
```

### Strategy Integration
Milling strategies now receive task configuration directly:
```python
def run(self, microscope, stage, config: FibsemMillingTaskConfig, 
        reference_image=None, **kwargs):
    # Strategy has access to task-level config without knowing about full task
    setup_milling(microscope, stage, config, reference_image)
```

## Migration Path

The refactor maintains backward compatibility through a wrapper function:
```python
def mill_stages(microscope, stages, parent_ui=None):
    """Backwards compatible wrapper function to mill stages."""
    # Automatically creates task config from first stage
    config = FibsemMillingTaskConfig(
        hfw=stages[0].milling.hfw,
        alignment=stages[0].alignment,
        # ... other parameters
    )
    
    task = FibsemMillingTask(config=config, stages=stages)
    task.run(microscope=microscope, parent_ui=parent_ui)
```

## Additional Improvements

- **Unique Task IDs** - Each task gets a UUID for tracking and logging
- **Better Error Handling** - Clear error messages for missing reference images
- **Cleaner Serialization** - Task and config objects support `to_dict()`/`from_dict()`
- **Reference Image Ownership** - Task owns and manages reference images for alignment