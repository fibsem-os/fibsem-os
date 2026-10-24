"""Both overview tabs edit their grid in one panel, which reads the FM overview's
declaration: the beam overview's must say the same."""

from fibsem.fm.structures import OverviewParameters
from fibsem.structures.settings import OverviewAcquisitionSettings


def test_the_beam_and_fm_overviews_declare_the_same_grid():
    fm, beam = OverviewParameters.params, OverviewAcquisitionSettings.params
    for fm_name, beam_name in (
        ("rows", "nrows"),
        ("cols", "ncols"),
        ("overlap", "overlap"),
    ):
        assert fm[fm_name].limits == beam[beam_name].limits
        assert fm[fm_name].display == beam[beam_name].display
