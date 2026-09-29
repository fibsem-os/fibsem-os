"""One ElidedLabel, shared (was two: `custom_widgets` and `fm_overview_widget`).

The two had drifted into different contracts. The overview's kept `_full_text` across
`setText`, returned it from `text()` and set a tooltip; the other did none of that, so a
label built empty and filled later would have painted itself blank -- latent, since its
callers all passed the text to the constructor and never set it again. The surviving
class is the overview's, whose behaviour these pin, plus the other's paint-time elide.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from fibsem.ui.widgets.custom_widgets import ElidedLabel

LONG = "Overview acquisition failed: the stage refused to move to tile 14 of 25"


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def shown(qapp):
    """A label in a holder narrow enough that it has to elide."""

    def _build(text: str = "", width: int = 200) -> ElidedLabel:
        holder = QWidget()
        QVBoxLayout(holder).addWidget(label := ElidedLabel(text))
        holder.resize(width, 60)
        holder.show()
        label._holder = holder  # keep it alive for the length of the test
        for _ in range(3):
            qapp.processEvents()
        return label

    return _build


def drawn(label: ElidedLabel) -> str:
    """What QLabel will actually paint, as opposed to what was set."""
    return QLabel.text(label)


class TestTheContract:
    def test_text_returns_what_was_set_not_what_is_drawn(self, shown):
        """Callers compare against it to decide whether a line is theirs to overwrite --
        the overview's status label does -- and eliding is presentation."""
        label = shown(LONG)

        assert label.text() == LONG
        assert drawn(label) != LONG
        assert drawn(label).endswith("…")

    def test_the_tooltip_carries_the_whole_string(self, shown):
        """The only way to read one that does not fit."""
        assert shown(LONG).toolTip() == LONG

    def test_a_caller_can_override_the_tooltip_afterwards(self, shown):
        """The overview's result message does, with the save path."""
        label = shown(LONG)

        label.setToolTip("/data/experiment/overview-01.ome.tiff")

        assert label.toolTip() == "/data/experiment/overview-01.ome.tiff"

    def test_a_label_built_empty_keeps_what_it_is_given_later(self, shown, qapp):
        """How the overview builds both of its labels. The version that did not override
        `setText` elided from the constructor's text forever, so this went blank."""
        label = shown("")

        label.setText(LONG)
        for _ in range(3):
            qapp.processEvents()
            label.repaint()

        assert label.text() == LONG
        painted = drawn(label)
        # Not a fixed prefix: how much fits in 200px is the platform's font, not this
        # widget's contract. macOS fits "Overview acquisition"; the Linux CI runner
        # elides a character earlier, to "Overview acquisiti…", and the test failed
        # there for a difference it was never about. What it is about is that the label
        # paints the text it was given after construction rather than the empty string
        # it was built with, so assert that: something was painted, and it is the start
        # of the new text.
        assert painted, "painted nothing: the label kept its empty constructor text"
        assert LONG.startswith(painted.rstrip("…")), (
            f"painted something that is not the start of the new text: {painted!r}"
        )

    def test_none_is_not_the_string_none(self, shown):
        label = shown(LONG)

        label.setText(None)

        assert label.text() == ""


class TestElidingSettles:
    def test_repeated_paints_do_not_eat_the_text(self, shown, qapp):
        """`paintEvent` puts the elided string on the label, so it must not be taken as
        the real one -- each pass would elide the ellipsis a little further."""
        label = shown(LONG)
        first = drawn(label)

        for _ in range(10):
            qapp.processEvents()
            label.repaint()

        assert drawn(label) == first

    def test_a_wider_label_shows_more(self, shown, qapp):
        """`text()` returning the full string is what makes this possible: the elided
        one is recomputed from it, never from itself.

        Also the trap in merging the two classes. `paintEvent` used to compare its
        result against `self.text()`, which now returns the full string -- so the
        comparison was "elided vs full", never equal for an elided label and always
        equal for one that fits, which is exactly backwards.
        """
        label = shown(LONG, width=200)
        assert drawn(label) != LONG

        label._holder.resize(900, 60)
        for _ in range(3):
            qapp.processEvents()
            label.repaint()

        assert drawn(label) == LONG


class TestTheOtherCallers:
    """`plugins_dialog` and `script_manager_dialog` build theirs with the text and never
    set it again. That path has to keep working, since it was the surviving class's."""

    def test_the_constructor_path_elides(self, shown):
        assert drawn(shown(LONG)).endswith("…")

    def test_short_text_is_left_alone(self, shown):
        assert drawn(shown("Ready", width=400)) == "Ready"
