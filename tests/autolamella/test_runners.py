import os
from copy import deepcopy

import pytest
from fibsem import utils
from fibsem.milling import get_protocol_from_stages

from fibsem.applications.autolamella import config as cfg
from fibsem.applications.autolamella.structures import (
    AutoLamellaMethod,
    AutoLamellaProtocol,
    AutoLamellaStage,
    LamellaState,
    create_new_experiment,
    create_new_lamella,
)
from fibsem.applications.autolamella.workflows.runners import run_autolamella


def test_run_autolamella():
    """Basic smoke test for the run_autolamella function."""
    microscope, settings = utils.setup_session()
    protocol = AutoLamellaProtocol.load(cfg.PROTOCOL_PATH)

    path = os.getcwd()
    exp = create_new_experiment(path=path, name="test-experiment")

    protocol.configuration = settings
    mprotocol = protocol.milling
    tmp_protocol = deepcopy({k: get_protocol_from_stages(v) for k, v in mprotocol.items()})

    state = LamellaState(stage=AutoLamellaStage.PositionReady, microscope_state=microscope.get_microscope_state())
    lamella = create_new_lamella(experiment_path=exp.path, 
                                 number=1, 
                                 state=state, 
                                 protocol=tmp_protocol)
    exp.positions.append(lamella)

    run_autolamella(microscope=microscope, experiment=exp, protocol=protocol)

    # assert lamella status
    assert lamella.state.stage is AutoLamellaStage.Finished

    ongrid_method = AutoLamellaMethod.ON_GRID
    for stage in ongrid_method.workflow:
        assert stage in lamella.workflow_stages_completed

    # TODO: remove temp experiment directory