from typing import Dict, Optional

from fibsem.applications.autolamella.protocol.constants import (
    FIDUCIAL_KEY,
    MICROEXPANSION_KEY,
    MILL_POLISHING_KEY,
    MILL_ROUGH_KEY,
    TRENCH_KEY,
    UNDERCUT_KEY,
)
from fibsem.milling import FibsemMillingStage
from fibsem.milling.patterning.patterns2 import (
    FiducialPattern,
    MicroExpansionPattern,
    RectanglePattern,
    TrenchPattern,
    WaffleNotchPattern,
)

from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import (
    BeamType,
    CrossSectionPattern,
    FibsemMillingSettings,
    Point,
)

DEFAULT_MILLING_CONFIG: Dict[str, FibsemMillingTaskConfig] = {}
DEFAULT_MILLING_CONFIG[TRENCH_KEY] = FibsemMillingTaskConfig(
    name="Trench Milling",
    field_of_view=180e-6,
    stages=[
        FibsemMillingStage(
            name="Trench Milling 01",
            milling=FibsemMillingSettings(milling_current=7.6e-9),
            pattern=TrenchPattern(
                width=22e-6,
                upper_trench_height=32e-6,
                lower_trench_height=16e-6,
                spacing=25e-6,
                depth=1.25e-6,
            ),
        )
    ],
)

DEFAULT_MILLING_CONFIG[UNDERCUT_KEY] = FibsemMillingTaskConfig(
    name="Undercut Milling",
    field_of_view=150e-6,
    stages=[
        FibsemMillingStage(
            name="Undercut Milling 01",
            milling=FibsemMillingSettings(milling_current=7.6e-9),
            pattern=RectanglePattern(width=22e-6, height=16e-6, depth=1.25e-6),
        )
    ],
)


DEFAULT_MILLING_CONFIG[FIDUCIAL_KEY] = FibsemMillingTaskConfig(
    name="Fiducial Milling",
    field_of_view=80e-6,
    stages=[
        FibsemMillingStage(
            milling=FibsemMillingSettings(milling_current=1.0e-9),
            pattern=FiducialPattern(point=Point(25e-6, 0)),
        )
    ],
)

DEFAULT_MILLING_CONFIG[MILL_ROUGH_KEY] = FibsemMillingTaskConfig(
    name="Rough Milling",
    field_of_view=80e-6,
    stages=[
        FibsemMillingStage(
            name="Rough Milling 01",
            milling=FibsemMillingSettings(
                milling_current=0.74e-9, application_file="Si-ccs"
            ),
            pattern=TrenchPattern(
                width=10e-6,
                upper_trench_height=3.5e-6,
                lower_trench_height=3.5e-6,
                spacing=4.6e-6,
                depth=0.65e-6,
                cross_section=CrossSectionPattern.CleaningCrossSection,
            ),
        ),
        FibsemMillingStage(
            name="Rough Milling 02",
            milling=FibsemMillingSettings(
                milling_current=0.2e-9, application_file="Si-ccs"
            ),
            pattern=TrenchPattern(
                width=9.5e-6,
                upper_trench_height=2.0e-6,
                lower_trench_height=2.0e-6,
                spacing=1.6e-6,
                depth=0.65e-6,
                cross_section=CrossSectionPattern.CleaningCrossSection,
            ),
        ),
        FibsemMillingStage(
            name="Stress Relief Feature",
            milling=FibsemMillingSettings(
                milling_current=1.0e-9, application_file="Si"
            ),
            pattern=MicroExpansionPattern(),
        ),
    ],
)

DEFAULT_MILLING_CONFIG[MILL_POLISHING_KEY] = FibsemMillingTaskConfig(
    name="Polishing Milling",
    field_of_view=80e-6,
    stages=[
        FibsemMillingStage(
            name="Polishing Milling 01",
            milling=FibsemMillingSettings(milling_current=60e-12, application_file="Si-ccs"),
            pattern=TrenchPattern(width=9.0e-6, depth=4.0e-7, spacing=3.0e-7, 
                                  upper_trench_height=0.7e-6, lower_trench_height=0.7e-6, 
                                  cross_section=CrossSectionPattern.CleaningCrossSection),
        )
    ],
)
