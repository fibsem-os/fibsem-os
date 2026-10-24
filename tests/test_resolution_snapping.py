"""A Thermo acquire at an unlisted resolution scans the nearest listed one."""

import pytest

from fibsem.devices.beam import nearest_resolution

# what a JEOL-like scan lists: 4:3 frames only
FOUR_THREE = [(640, 480), (1280, 960), (2560, 1920)]


@pytest.mark.parametrize(
    "requested, expected",
    [
        ((1280, 960), (1280, 960)),
        ((1536, 1024), (1280, 960)),
        ((3072, 2048), (2560, 1920)),
        ((384, 256), (640, 480)),
    ],
)
def test_the_nearest_listed_resolution(requested, expected):
    assert nearest_resolution(requested, FOUR_THREE) == expected


def test_nothing_listed_keeps_the_resolution():
    assert nearest_resolution((1000, 700), None) == (1000, 700)


def _task_on(choices):
    from types import SimpleNamespace

    from fibsem.structures.settings import BeamType

    parameter = SimpleNamespace(choices=choices)
    ion = SimpleNamespace(parameters={"resolution": parameter})
    return SimpleNamespace(microscope=SimpleNamespace(beams={BeamType.ION: ion}))


def test_the_alignment_reference_is_taken_at_a_listed_resolution():
    from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask

    resolution = AutoLamellaTask._alignment_resolution(_task_on(FOUR_THREE))
    assert resolution == (1280, 960)


def test_an_ion_beam_that_lists_none_keeps_the_alignment_resolution():
    from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask

    assert AutoLamellaTask._alignment_resolution(_task_on(None)) == (1536, 1024)
