"""The merged Overview tab: one tab, two modalities, both alive at once (FIB-780).

Built against a real simulator rather than a stub host. What is worth checking here is
not that a stack raises the page you clicked -- Qt does that -- but the three things the
merge could plausibly have broken, none of which a stub would exercise:

* **Both widgets stay built.** Switching modality must not tear the other one down, or
  each switch pays a rebuild and the view you were on is lost.
* **Availability is per modality, not per tab.** A system with no camera has a perfectly
  usable Overview tab; it is one chip on it that cannot be reached.
* **The lock still has two inputs.** FIB-706 stops one overview driving the stage while
  the other acquires, and both are still able to acquire from under a single tab.

Run directly:
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_overview_container_tab.py
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.ui.overview_container_tab import (  # noqa: E402
    MODALITY_FIBSEM,
    MODALITY_FLUORESCENCE,
    AutoLamellaOverviewContainerTab,
)


@pytest.fixture
def qapp():
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def _ui(microscope=None, experiment=None):
    return type("_UI", (), {"microscope": microscope, "experiment": experiment})()


def _fm_microscope():
    from fibsem.ui.fm.overview_app import build_microscope

    return build_microscope()


def _plain_microscope():
    """A Demo microscope with no fluorescence detector attached."""
    from fibsem import utils

    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.fm = None
    return microscope


def _built(qapp, microscope):
    tab = AutoLamellaOverviewContainerTab(_ui(microscope))
    tab.refresh_microscope()
    return tab


def _teardown(tab):
    for host in (tab.beam_tab, tab.fm_tab):
        host._drop_overview()
    tab.deleteLater()


# ── both sides stay alive ────────────────────────────────────────────────


def test_switching_modality_does_not_tear_the_other_side_down(qapp):
    """The whole reason the view state survives a switch. If the hidden page were
    dropped, every switch would rebuild a widget and lose the view it was left on --
    which is the thing FIB-780 asked for and the thing a stack gets for free."""
    tab = _built(qapp, _fm_microscope())
    assert tab.beam_tab.overview is not None
    assert tab.fm_tab.overview is not None

    assert tab.set_modality(MODALITY_FLUORESCENCE)
    assert tab.beam_tab.overview is not None, "the beam widget was dropped on a switch"

    assert tab.set_modality(MODALITY_FIBSEM)
    assert tab.fm_tab.overview is not None, "the FM widget was dropped on a switch"

    _teardown(tab)


def test_the_shown_page_is_the_one_the_chip_names(qapp):
    tab = _built(qapp, _fm_microscope())

    tab.set_modality(MODALITY_FLUORESCENCE)
    assert tab.stack.currentWidget() is tab.fm_tab
    assert tab.modality == MODALITY_FLUORESCENCE

    tab.set_modality(MODALITY_FIBSEM)
    assert tab.stack.currentWidget() is tab.beam_tab

    _teardown(tab)


def test_a_switch_announces_itself_once(qapp):
    """Once per actual change. Re-picking the modality already showing is a no-op, or a
    host listening to this would act on a click that changed nothing."""
    tab = _built(qapp, _fm_microscope())
    seen = []
    tab.modality_changed.connect(seen.append)

    tab.set_modality(MODALITY_FLUORESCENCE)
    tab.set_modality(MODALITY_FLUORESCENCE)

    assert seen == [MODALITY_FLUORESCENCE]

    _teardown(tab)


# ── availability is per modality ─────────────────────────────────────────


def test_a_system_with_no_camera_still_has_a_usable_tab(qapp):
    """The merge's most visible consequence, and the one worth pinning: what used to
    grey out a whole tab now greys out one chip."""
    tab = _built(qapp, _plain_microscope())

    assert tab.is_available, "the tab is dead on a system that has beams"
    assert tab.available_modalities() == [MODALITY_FIBSEM]
    assert not tab.modality_chip(MODALITY_FLUORESCENCE).isEnabled()
    assert tab.modality_chip(MODALITY_FIBSEM).isEnabled()

    _teardown(tab)


def test_an_unreachable_modality_cannot_be_raised(qapp):
    """A guard rather than a path a click takes -- the chip for it is disabled. It
    matters because the page behind an unavailable modality has no widget at all, so
    raising it would put a bare container on screen, which reads as a canvas that failed
    to draw rather than as a system without a camera."""
    tab = _built(qapp, _plain_microscope())

    assert not tab.set_modality(MODALITY_FLUORESCENCE)
    assert tab.modality == MODALITY_FIBSEM
    assert tab.stack.currentWidget() is tab.beam_tab

    _teardown(tab)


def test_no_microscope_leaves_the_tab_unavailable_with_a_reason(qapp):
    tab = AutoLamellaOverviewContainerTab(_ui(None))
    tab.refresh_microscope()

    assert not tab.is_available
    available, reason = tab.unavailable_summary()
    assert not available
    assert reason == "Connect a microscope to use the Overview"

    tab.deleteLater()


def test_availability_is_announced_for_the_tab_not_the_page(qapp):
    """The window enables the tab off this signal. It has to mean "either page works",
    or a system with no camera would have its Overview tab greyed out entirely."""
    tab = AutoLamellaOverviewContainerTab(_ui(_plain_microscope()))
    seen = []
    tab.availability_changed.connect(seen.append)

    tab.refresh_microscope()

    assert seen and seen[-1] is True
    _teardown(tab)


def test_an_unknown_modality_is_refused(qapp):
    tab = AutoLamellaOverviewContainerTab(_ui(None))
    with pytest.raises(ValueError):
        tab.set_modality("XRAY")
    tab.deleteLater()


# ── the lock still has two inputs ────────────────────────────────────────


def test_acquiring_is_true_while_either_side_runs(qapp):
    """FIB-706 unchanged by the merge: both pages stay alive, so both can still be
    mid-tileset, and the answer the window locks on has to cover both.

    Re-derived from the tabs rather than forwarded, so a bool arriving from the side that
    just *stopped* cannot unlock the stage while the other is still placing tiles.
    """
    tab = _built(qapp, _fm_microscope())
    seen = []
    tab.acquiring_changed.connect(seen.append)

    tab.beam_tab.overview._set_running(True)
    assert tab.is_acquiring
    assert seen[-1] is True

    tab.fm_tab.overview._set_running(True)
    tab.beam_tab.overview._set_running(False)
    assert tab.is_acquiring, "unlocked while the fluorescence side was still running"
    assert seen[-1] is True

    tab.fm_tab.overview._set_running(False)
    assert not tab.is_acquiring
    assert seen[-1] is False

    _teardown(tab)


# ── the shared bar: lending the page's view chips a place in it ──────────


def test_the_active_pages_view_chips_move_into_the_bar(qapp):
    """One row, not two. The strip is the page's own -- it discovers views as they are
    acquired in -- so it is moved rather than copied."""
    tab = _built(qapp, _fm_microscope())

    strip = tab.beam_tab.overview.view_strip
    assert tab._view_slot.isAncestorOf(strip), "the beam view chips are not in the bar"
    assert not tab._divider.isHidden()

    _teardown(tab)


def test_the_fluorescence_page_has_no_view_chips_and_no_divider(qapp):
    """One camera, one view. A divider with nothing after it is a rule pointing at
    empty space."""
    tab = _built(qapp, _fm_microscope())
    tab.set_modality(MODALITY_FLUORESCENCE)

    assert getattr(tab.fm_tab.overview, "view_strip", None) is None
    assert tab._divider.isHidden()

    _teardown(tab)


def test_switching_back_and_forth_keeps_lending_the_strip(qapp):
    """The strip is handed back on the way out, so coming back has to re-mount it.
    Left un-mounted the beam page would silently lose its view chips on the second
    visit -- and there is no error to notice, just chips that stop appearing."""
    tab = _built(qapp, _fm_microscope())
    strip = tab.beam_tab.overview.view_strip

    tab.set_modality(MODALITY_FLUORESCENCE)
    assert not tab._view_slot.isAncestorOf(strip), "the strip was never handed back"

    tab.set_modality(MODALITY_FIBSEM)
    assert tab._view_slot.isAncestorOf(strip), "the strip was not lent back"

    _teardown(tab)


def test_a_rebuild_does_not_leave_the_bar_holding_a_dead_strip(qapp):
    """The reason `_unmount_view_strip` runs *before* the rebuild rather than after.

    While mounted the strip is parented here, so it does not go when Qt destroys the
    widget that owns it -- and the bar would be left holding a deleted C++ object, which
    is a segfault on the next paint rather than an exception. Reconnecting to a different
    microscope is the path that does this.
    """
    tab = _built(qapp, _fm_microscope())
    first = tab.beam_tab.overview.view_strip

    # A different microscope object is what makes `refresh_microscope` rebuild.
    tab.autolamella_ui.microscope = _fm_microscope()
    tab.refresh_microscope()

    second = tab.beam_tab.overview.view_strip
    assert second is not first, "the widget was not rebuilt; this test proves nothing"
    assert tab._view_slot.isAncestorOf(second)

    _teardown(tab)


def test_dropping_a_page_out_from_under_the_bar_is_survivable(qapp):
    """Nothing in production drops a widget without going through `refresh_microscope`,
    but tests do, and the guard costs one `except`. What it must not do is raise while
    tearing down."""
    tab = _built(qapp, _fm_microscope())
    tab.beam_tab._drop_overview()

    tab._unmount_view_strip()  # must not raise
    tab._mount_view_strip()  # must not raise
    assert tab._divider.isHidden()

    _teardown(tab)


# ── the strip itself ─────────────────────────────────────────────────────


def test_a_disabled_chip_does_not_look_like_a_live_one(qapp):
    """It read as clickable until `VIEW_CHIP_STYLE` grew a `:disabled` rule -- the greyed
    Fluorescence chip on a system with no camera rendered pixel-identical to the enabled
    one, so the only thing saying "not this one" was a tooltip nobody hovers.

    Compared by rendering rather than by reading the stylesheet: the sheet is shared with
    the view chips and a rule can be present and still lose to another selector.
    """
    live = _built(qapp, _fm_microscope())
    dead = AutoLamellaOverviewContainerTab(_ui(_plain_microscope()))
    dead.refresh_microscope()

    enabled = live.modality_chip(MODALITY_FLUORESCENCE)
    disabled = dead.modality_chip(MODALITY_FLUORESCENCE)
    assert enabled.isEnabled() and not disabled.isEnabled()

    enabled.resize(120, 24)
    disabled.resize(120, 24)
    assert enabled.grab().toImage() != disabled.grab().toImage(), (
        "a disabled chip renders identically to a live one"
    )

    _teardown(live)
    _teardown(dead)


def test_a_chip_is_the_same_size_whichever_state_it_is_in(qapp):
    """The active view chip carries a 1px border and the resting one used `border: none`,
    so a chip grew by 2px the moment it became the acquisition view -- and sat 2px taller
    than the modality chips beside it once the two shared a row."""
    from PyQt5.QtWidgets import QPushButton

    from fibsem.ui.widgets.overview_widget import (
        VIEW_CHIP_STYLE,
        VIEW_CHIP_STYLE_ACTIVE,
    )

    sizes = set()
    for sheet in (VIEW_CHIP_STYLE, VIEW_CHIP_STYLE_ACTIVE):
        chip = QPushButton("SEM @ SEM")
        chip.setStyleSheet(sheet)
        chip.adjustSize()
        sizes.add((chip.sizeHint().width(), chip.sizeHint().height()))
    assert len(sizes) == 1, f"the chip changes size with its state: {sizes}"


def test_every_chip_says_something(qapp):
    """Enabled or not. A greyed control with an empty tooltip is the failure this whole
    pattern exists to avoid, and it is the one that appears by omission when a new
    unavailable state is added later."""
    for microscope in (None, _plain_microscope(), _fm_microscope()):
        tab = AutoLamellaOverviewContainerTab(_ui(microscope))
        tab.refresh_microscope()
        for modality in (MODALITY_FIBSEM, MODALITY_FLUORESCENCE):
            chip = tab.modality_chip(modality)
            assert chip.toolTip(), f"{modality} chip is silent"
        _teardown(tab)


def test_the_checked_chip_is_the_page_on_screen(qapp):
    """Two ways to be wrong: a chip checked for a page that is not showing, and none
    checked at all. Both read as the strip having lost track of the tab."""
    tab = _built(qapp, _fm_microscope())

    for modality in (MODALITY_FLUORESCENCE, MODALITY_FIBSEM):
        tab.set_modality(modality)
        checked = [
            m
            for m in (MODALITY_FIBSEM, MODALITY_FLUORESCENCE)
            if tab.modality_chip(m).isChecked()
        ]
        assert checked == [modality]

    _teardown(tab)
