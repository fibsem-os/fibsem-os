"""The Demo boots each column focused at its configured eucentric height (FIB-819).

A beam whose working distance starts away from its eucentric height images a
sample at the eucentric point out of focus until something autofocuses it.
"""

import pytest

from fibsem import utils
from fibsem.drivers.demo.simulator import initial_demo_parts
from fibsem.structures.settings import BeamType
from tests.fixtures.microscopes import demo_session


@pytest.mark.parametrize(
    "config",
    [None, "tfs-arctis-configuration.yaml", "jeol-configuration.yaml"],
)
def test_initial_parts_take_each_beams_eucentric_height(config):
    from fibsem import config as cfg

    path = None if config is None else f"{cfg.SHIPPED_CONFIG_PATH}/{config}"
    system = utils.load_microscope_configuration(path).system

    parts = initial_demo_parts(system)

    assert parts.electron_system.beam.working_distance == pytest.approx(
        system.electron.eucentric_height
    )
    assert parts.ion_system.beam.working_distance == pytest.approx(
        system.ion.eucentric_height
    )


def test_a_connected_demo_reads_its_eucentric_working_distance():
    microscope = demo_session("tfs-arctis-configuration.yaml")
    try:
        for beam_type, settings in (
            (BeamType.ELECTRON, microscope.system.electron),
            (BeamType.ION, microscope.system.ion),
        ):
            assert microscope.get_working_distance(beam_type) == pytest.approx(
                settings.eucentric_height
            ), beam_type
    finally:
        microscope.disconnect()
