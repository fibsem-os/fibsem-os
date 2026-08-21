"""The tab-corner header swaps its create/load buttons for the experiment name.

Before an experiment exists, "Create Experiment" and "Load Experiment" are the only
thing to do and are shown in the primary colour. Once one is loaded they used to stay
put, greyed out, next to a label repeating the name -- three controls and roughly a
third of the tab bar spent on actions the File menu already carries. Now the name
itself becomes the button, and it holds those actions in a menu.

The window cannot be constructed offscreen -- it builds a napari viewer and a vispy
quad view, neither of which survives `QT_QPA_PLATFORM=offscreen` -- so its handler is
borrowed onto a host carrying the three real buttons, the same way
`test_status_bar_tile_outcome.py` does. The buttons are the real ones and so is the
Experiment; only the widget that owns them is stood in for.
"""

from __future__ import annotations

import ast
import inspect
import os
import textwrap
from datetime import datetime

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QPushButton  # noqa: E402

from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    DefectType,
    Experiment,
    Lamella,
)
from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (  # noqa: E402
    BUTTON_ICON_SIZE,
    EXPERIMENT_MENU_MAX_WIDTH,
    AutoLamellaSingleWindowUI,
    experiment_tooltip,
    set_button_icon,
)


class _UIStub:
    """Stands in for AutoLamellaUI: the header reads exactly these two attributes."""

    def __init__(self, experiment=None, microscope=None):
        self.experiment = experiment
        self.microscope = microscope


class _HeaderHost:
    _update_experiment_header = AutoLamellaSingleWindowUI._update_experiment_header

    def __init__(self):
        self.btn_create_experiment = QPushButton("Create Experiment")
        self.btn_load_experiment = QPushButton("Load Experiment")
        self.btn_experiment_menu = QPushButton()
        self.autolamella_ui = _UIStub()

    def close(self):
        for btn in (
            self.btn_create_experiment,
            self.btn_load_experiment,
            self.btn_experiment_menu,
        ):
            btn.deleteLater()


@pytest.fixture
def host(qapp):
    host = _HeaderHost()
    yield host
    host.close()


@pytest.fixture
def experiment(tmp_path):
    return Experiment(path=str(tmp_path), name="AutoLamella-2026-08-21-headers")


def test_buttons_shown_and_disabled_before_connecting(host):
    host._update_experiment_header()

    assert not host.btn_create_experiment.isHidden()
    assert not host.btn_load_experiment.isHidden()
    assert not host.btn_create_experiment.isEnabled()
    assert not host.btn_load_experiment.isEnabled()
    # Nothing to name yet, so the name button stays out of the way.
    assert host.btn_experiment_menu.isHidden()


def test_buttons_enabled_once_connected(host):
    host.autolamella_ui.microscope = object()

    host._update_experiment_header()

    assert host.btn_create_experiment.isEnabled()
    assert host.btn_load_experiment.isEnabled()
    assert host.btn_experiment_menu.isHidden()


def test_name_button_replaces_the_pair_once_loaded(host, experiment):
    host.autolamella_ui.microscope = object()
    host.autolamella_ui.experiment = experiment

    host._update_experiment_header()

    assert host.btn_create_experiment.isHidden()
    assert host.btn_load_experiment.isHidden()
    assert not host.btn_experiment_menu.isHidden()
    assert experiment.name in host.btn_experiment_menu.text()
    assert experiment.name in host.btn_experiment_menu.toolTip()


def test_default_length_name_is_not_elided(host, tmp_path):
    """A date-stamped default name has to fit whole -- naming the experiment is the
    button's entire job, and an allowance guessed for the icon, padding and chevron
    once cut one short."""
    name = "AutoLamella-2026-08-21-11-04"
    host.autolamella_ui.microscope = object()
    host.autolamella_ui.experiment = Experiment(path=str(tmp_path), name=name)

    host._update_experiment_header()

    assert host.btn_experiment_menu.text() == name


def test_long_name_is_elided_not_stretched(host, tmp_path):
    long_name = "AutoLamella-2026-08-21-" + "grid" * 20
    host.autolamella_ui.microscope = object()
    host.autolamella_ui.experiment = Experiment(path=str(tmp_path), name=long_name)

    host._update_experiment_header()

    text = host.btn_experiment_menu.text()
    assert text != long_name
    assert "…" in text
    # The full name is still reachable, just not at the cost of the tab bar.
    assert long_name in host.btn_experiment_menu.toolTip()
    assert host.btn_experiment_menu.sizeHint().width() <= EXPERIMENT_MENU_MAX_WIDTH


def test_menu_reuses_the_file_menu_actions():
    """The header menu must add the File menu's own QActions, not rebuild them.

    Two separate sets would drift: a shortcut, a rename or a new guard added in one
    place would silently miss the other. The constructor cannot be run here, so the
    source is read for which actions it adds.
    """
    tree = ast.parse(
        textwrap.dedent(
            inspect.getsource(AutoLamellaSingleWindowUI.create_notification_button)
        )
    )
    added = {
        node.args[0].attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "addAction"
        and node.args
        and isinstance(node.args[0], ast.Attribute)
    }
    assert added == {
        "action_new_experiment",
        "action_load_experiment",
        "action_open_experiment_directory",
    }


class TestExperimentTooltip:
    """What the hover card over the experiment name says.

    Rows come off the experiment, and a row whose fact is missing is dropped rather
    than rendered blank: a protocol is attached after construction, and `created_at`
    is absent from experiments written before it was recorded.
    """

    def test_names_dates_counts_and_path(self, tmp_path):
        experiment = Experiment(path=str(tmp_path), name="AutoLamella-2026-08-21-11-04")
        experiment.created_at = datetime(2026, 8, 21, 11, 4).timestamp()
        experiment.task_protocol = AutoLamellaTaskProtocol(name="Waffle Method")
        for i in range(3):
            experiment.positions.append(
                Lamella(petname=f"{i:02d}-lam", path=str(tmp_path / str(i)), number=i)
            )

        tip = experiment_tooltip(experiment)

        assert "AutoLamella-2026-08-21-11-04" in tip
        assert "21 Aug 2026, 11:04" in tip
        assert ">3<" in tip
        assert "Waffle Method" in tip
        assert str(experiment.path) in tip

    def test_defective_count_only_when_there_is_one(self, tmp_path):
        experiment = Experiment(path=str(tmp_path), name="defects")
        for i in range(2):
            experiment.positions.append(
                Lamella(petname=f"{i:02d}-lam", path=str(tmp_path / str(i)), number=i)
            )
        # Matched with the closing tag: pytest's own tmp_path is named after this
        # test, and the path is one of the rows.
        assert "defective</span>" not in experiment_tooltip(experiment)

        experiment.positions[0].defect.state = DefectType.FAILURE

        tip = experiment_tooltip(experiment)
        # "defective", not "failed": is_failure is a human's judgement about the
        # lamella, not a record of a task that failed.
        assert "1 defective</span>" in tip
        assert "failed" not in tip

    def test_missing_facts_drop_out(self, tmp_path):
        experiment = Experiment(path=str(tmp_path), name="fresh")
        experiment.task_protocol = None
        experiment.created_at = None

        tip = experiment_tooltip(experiment)

        assert "Protocol" not in tip
        assert "Created" not in tip
        assert "Lamella" in tip

    def test_name_is_escaped(self, tmp_path):
        """The name is user-supplied and lands in a rich-text tooltip."""
        experiment = Experiment(path=str(tmp_path), name="a<b>&c")

        tip = experiment_tooltip(experiment)

        assert "a&lt;b&gt;&amp;c" in tip
        assert "<b>&c" not in tip


def test_button_icon_size_grows_with_its_padding(qapp):
    """The icon box has to widen by the gap, or the glyph shrinks instead.

    Qt scales a button's icon into `iconSize`, so padding the pixmap without
    widening that box squeezes the drawing back down rather than pushing the text
    across -- a failure that looks like a slightly small icon and nothing else.
    """
    button = QPushButton("Experiment")

    set_button_icon(button, "mdi:flask-outline", gap=5)

    assert button.iconSize().width() == BUTTON_ICON_SIZE + 5
    assert button.iconSize().height() == BUTTON_ICON_SIZE
    # The glyph itself keeps its square, with the gap as transparent padding.
    drawn = button.icon().pixmap(button.iconSize())
    ratio = drawn.devicePixelRatio() or 1.0
    assert round(drawn.width() / ratio) == BUTTON_ICON_SIZE + 5
    assert round(drawn.height() / ratio) == BUTTON_ICON_SIZE
    button.deleteLater()
