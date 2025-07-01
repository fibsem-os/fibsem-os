import pytest
from fibsem.milling import get_protocol_from_stages
from autolamella.structures import Lamella, create_new_lamella, AutoLamellaProtocol, LamellaState
from autolamella import config as cfg
from copy import deepcopy

# test lamella class
import os
TMP_EXPERIMENT_PATH = os.path.join(os.getcwd(), "tmp-AutoLamella")
os.makedirs(TMP_EXPERIMENT_PATH, exist_ok=True)

@pytest.fixture
def lamella() -> Lamella:
    """Create a lamella object for testing."""
    protocol = AutoLamellaProtocol.load(cfg.PROTOCOL_PATH)
    tmp_protocol = deepcopy({k: get_protocol_from_stages(v) for k, v in protocol.milling.items()})

    lamella: Lamella = create_new_lamella(experiment_path=TMP_EXPERIMENT_PATH, 
                                 number=1,
                                 state=LamellaState(),
                                 protocol=tmp_protocol)
    return lamella


def test_lamella_init(lamella: Lamella):
    """Test the initialization of the Lamella class."""
    
    assert os.path.exists(lamella.path)
    assert TMP_EXPERIMENT_PATH in lamella.path
    assert lamella.number == 1

    # check that the imaging path is set correctly
    for k, v in lamella.protocol.items():
        for stage in v:
            assert stage["imaging"]["path"] == lamella.path

# remove the tmp directory after the test
@pytest.fixture(autouse=True)
def cleanup():
    yield
    if os.path.exists(TMP_EXPERIMENT_PATH):
        for root, dirs, files in os.walk(TMP_EXPERIMENT_PATH, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(TMP_EXPERIMENT_PATH)