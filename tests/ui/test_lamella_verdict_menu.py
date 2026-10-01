"""A lamella can be judged good, and a good lamella is not drawn as a defective one.

The lamella's menu offered "No defect", which wrote ``Verdict.NONE`` -- an alias of
``UNASSESSED``. So "I looked and it is fine" was stored as "nobody has looked", and
nothing in the app could record a good lamella. The model had ``GOOD`` all along (and
the agent server could set it); every display then treated anything but ``NONE`` as a
defect, so a good lamella would have been drawn with the failure icon.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_lamella_verdict_menu.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

# CI installs `.[test]`, not `.[ui]`, so PyQt5 is absent there.
pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QMenu  # noqa: E402

from fibsem.applications.autolamella.structures import (  # noqa: E402
    Lamella,
    QualityRecord,
    Verdict,
)
from fibsem.applications.autolamella.ui.lamella_card_widget import (  # noqa: E402
    LamellaCardContainer,
)
from fibsem.applications.autolamella.ui.lamella_list_widget import (  # noqa: E402
    LamellaListWidget,
    add_defect_menu,
    has_defect,
    has_verdict,
)
from fibsem.applications.autolamella.ui.lamella_name_list_widget import (  # noqa: E402
    LamellaNameListWidget,
)

_app = QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def lamella(tmp_path) -> Lamella:
    return Lamella(path=tmp_path / "01-a", number=1, petname="01-a")


def _menu(lamella, changed=None):
    parent = QMenu()
    sub = add_defect_menu(parent, lamella, changed or (lambda: None))
    actions = {a.text(): a for a in sub.actions() if a.text()}
    return parent, sub, actions


def test_the_menu_offers_good_as_well_as_the_defects(lamella):
    _, sub, actions = _menu(lamella)
    assert sub.title() == "Verdict"
    assert list(actions) == ["Good", "Rework required", "Failed", "Not assessed"]


def test_good_is_stored_as_good_and_dated(lamella):
    """The defect: the only "healthy" choice stored UNASSESSED."""
    changed = []
    parent, _sub, actions = _menu(lamella, lambda: changed.append(True))

    actions["Good"].trigger()

    assert lamella.defect.verdict is Verdict.GOOD
    assert lamella.defect.updated_at is not None
    assert changed == [True]


def test_not_assessed_takes_the_judgement_back(lamella):
    lamella.defect = QualityRecord(verdict=Verdict.FAILED, reason="cracked")
    parent, _sub, actions = _menu(lamella)

    actions["Not assessed"].trigger()

    assert lamella.defect.verdict is Verdict.UNASSESSED
    assert lamella.defect.reason == ""


def test_the_current_verdict_is_ticked(lamella):
    lamella.defect = QualityRecord(verdict=Verdict.REWORK)
    _, sub, actions = _menu(lamella)
    sub.aboutToShow.emit()

    ticked = [text for text, action in actions.items() if action.isChecked()]
    assert ticked == ["Rework required"]


def test_choosing_the_current_verdict_writes_nothing(lamella):
    """No new record, so no save and no redraw for a click that changed nothing."""
    lamella.defect = QualityRecord(verdict=Verdict.GOOD, author="human:Ada")
    changed = []
    parent, _sub, actions = _menu(lamella, lambda: changed.append(True))

    actions["Good"].trigger()

    assert changed == []
    assert lamella.defect.author == "human:Ada"


def test_a_good_lamella_is_judged_but_not_defective(lamella):
    lamella.defect = QualityRecord(verdict=Verdict.GOOD)
    assert has_verdict(lamella)
    assert not has_defect(lamella)


class TestEveryDisplay:
    """The row, the card and the experiment list all draw a good lamella as good."""

    def test_the_workflow_row(self, lamella):
        widget = LamellaListWidget()
        widget.set_lamellae([lamella])
        row = widget._row(0)
        assert not row.btn_defect.isVisibleTo(row), "nobody has judged it yet"

        lamella.defect = QualityRecord(verdict=Verdict.GOOD)
        row.refresh()

        assert row.btn_defect.isVisibleTo(row)
        assert row.btn_defect.toolTip() == "Good"

    def test_the_card(self, lamella):
        container = LamellaCardContainer(columns=1, mode="standard")
        card = container.add_lamella(lamella)

        lamella.defect = QualityRecord(verdict=Verdict.GOOD)

        assert card._btn_defect.isVisibleTo(card)
        assert card._btn_defect.toolTip() == "Good"

    def test_the_experiment_list_row(self, lamella):
        widget = LamellaNameListWidget()
        widget.enable_defect_button(True)
        widget.set_lamella([lamella])
        widget.show()
        row = list(widget._rows())[0]

        lamella.defect = QualityRecord(verdict=Verdict.GOOD, reason="thin, clean")

        assert row.btn_defect.isVisible()
        assert row.btn_defect.toolTip() == "Good: thin, clean"
        widget.close()
