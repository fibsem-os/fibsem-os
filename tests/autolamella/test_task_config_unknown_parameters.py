"""Loading a task config warns for a parameter it does not know, and only for that.

The spot burn and both coincidence-milling configs read their own parameters, and
used to start by calling the base loader, which checked those parameters against
the base class's fields: every one of the task's own settings logged "Unknown
parameter", so a key that really was unknown was lost among them (FIB-1107).
"""

import logging

import pytest

from fibsem.applications.autolamella.workflows.tasks.mill_coincident import (
    MillCoincidentTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.setup_coincidence_milling import (
    SetupCoincidenceMillingTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.spot_burn import (
    SpotBurnFiducialTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.trench import (
    MillTrenchTaskConfig,
)

# a config of each kind with its parameters off their defaults, so a value that
# was dropped on load and replaced by the default would show
CONFIGS = [
    SpotBurnFiducialTaskConfig(
        milling_current=200e-12, exposure_time=5, autofocus=True
    ),
    MillCoincidentTaskConfig(
        setup_task="Another Setup", acquire_fluorescence_images=False
    ),
    SetupCoincidenceMillingTaskConfig(
        field_of_view=120e-6,
        intensity_drop_fraction=0.25,
        align_coincidence=True,
        objective_position=1.5e-3,
    ),
]


def _unknown_parameter_warnings(caplog):
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "Unknown parameter" in r.getMessage()
    ]


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.task_type)
def test_a_saved_config_loads_without_unknown_parameter_warnings(config, caplog):
    ddict = config.to_dict()

    with caplog.at_level(logging.WARNING):
        loaded = type(config).from_dict(ddict)

    assert _unknown_parameter_warnings(caplog) == []
    for name in config.parameters:
        assert getattr(loaded, name) == getattr(config, name), name


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.task_type)
def test_an_unknown_parameter_warns_once_naming_the_task(config, caplog):
    ddict = config.to_dict()
    ddict["parameters"]["exposure_tme"] = 3

    with caplog.at_level(logging.WARNING):
        type(config).from_dict(ddict)

    warnings = _unknown_parameter_warnings(caplog)
    assert len(warnings) == 1
    assert "'exposure_tme'" in warnings[0]
    assert config.task_type in warnings[0]


def test_a_retired_parameter_is_dropped_without_a_warning(caplog):
    """Every saved Trench config carries ``align_reference``, retired with the
    alignment to the legacy ref_PositionReady.tif. Loading one is not a problem
    to report; a key nobody recognises still is."""
    ddict = MillTrenchTaskConfig(charge_neutralisation=False).to_dict()
    ddict["parameters"]["align_reference"] = True
    ddict["parameters"]["charge_neutralistion"] = True

    with caplog.at_level(logging.WARNING):
        loaded = MillTrenchTaskConfig.from_dict(ddict)

    warnings = _unknown_parameter_warnings(caplog)
    assert len(warnings) == 1 and "'charge_neutralistion'" in warnings[0]
    assert not hasattr(loaded, "align_reference")
    assert loaded.charge_neutralisation is False
