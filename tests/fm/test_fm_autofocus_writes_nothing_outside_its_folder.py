"""An FM autofocus run writes only where its caller says.

The diagnostic plot used to fall back to ``autofocus/autofocus_<time>.png`` under
the working directory whenever no path was given -- which is how every autofocus
run from the fluorescence control widget saved it, so the plot landed wherever the
app had been started from, outside the experiment. Now no path means no file, and
a run saves its result (plot included) only into the folder it is handed, as
``run_auto_focus(..., path=, name=)`` does for the beams.

On the Demo microscope, with the working directory pointed at an empty folder.
"""

import pytest

from fibsem.fm.calibration import plot_autofocus, run_autofocus
from fibsem.fm.structures import ZParameters

ZPARAMS = ZParameters(zmin=-2e-6, zmax=2e-6, zstep=1e-6)


@pytest.fixture
def fm():
    from fibsem import utils

    scope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    assert scope.fm is not None
    scope.fm.objective.insert()
    return scope.fm


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    folder = tmp_path / "cwd"
    folder.mkdir()
    monkeypatch.chdir(folder)
    return folder


def test_a_run_and_its_plot_without_a_path_write_nothing(fm, cwd):
    result = run_autofocus(fm, z_parameters=ZPARAMS)
    assert result is not None

    plot_autofocus(result)  # as the control widget called it

    assert list(cwd.iterdir()) == []


def test_a_run_given_a_path_saves_there_and_only_there(fm, cwd, tmp_path):
    out = tmp_path / "experiment" / "autofunctions"

    run_autofocus(fm, z_parameters=ZPARAMS, path=str(out), name="af")

    assert (out / "af" / "data.json").is_file()
    assert (out / "af" / "plot.png").is_file()
    assert list(cwd.iterdir()) == []
