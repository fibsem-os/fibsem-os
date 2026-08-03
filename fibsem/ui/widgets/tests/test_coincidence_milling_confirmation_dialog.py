"""Headless tests for the coincidence milling confirmation dialog (FIB-430).

This is the last gate before an irreversible, sample-destroying action, so the tests
are mostly about it telling the truth: the mode it reports must be the mode that will
actually run, and starting must never be reachable by a stray keypress.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python fibsem/ui/widgets/tests/test_coincidence_milling_confirmation_dialog.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QPushButton

from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.patterning.patterns2 import RectanglePattern, TrenchPattern
from fibsem.milling.strategy.coincidence import (
    CoincidenceMillingStrategy,
    CoincidenceMillingStrategyConfig,
)
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import FibsemMillingSettings
from fibsem.ui.widgets.coincidence_milling_confirmation_dialog import (
    CoincidenceMillingConfirmationDialog,
    _format_current,
    _pattern_summary,
)

_app = QApplication.instance() or QApplication(sys.argv)


def _stage(name="Polish", current=60e-12, pattern=None, enabled=True, **cfg):
    stage = FibsemMillingStage(
        name=name,
        enabled=enabled,
        milling=FibsemMillingSettings(milling_current=current),
        pattern=pattern or RectanglePattern(width=12e-6, height=1.5e-6, depth=0.5e-6),
    )
    base = dict(
        intensity_drop_fraction=0.20,
        rolling_window=10,
        consecutive_triggers=3,
        warmup_duration=30.0,
        timeout=1800,
        supervised=True,
        save_fm_images=True,
        save_rate_limit=2,
        acquire_fib_image=False,
    )
    base.update(cfg)
    stage.strategy = CoincidenceMillingStrategy(CoincidenceMillingStrategyConfig(**base))
    return stage


def _dialog(stages=None, channel_name="GFP (488 nm)", lamella_name="01-calm-muskox"):
    config = FibsemMillingTaskConfig(stages=stages or [_stage()])
    config.field_of_view = 80e-6
    return CoincidenceMillingConfirmationDialog(
        config, lamella_name=lamella_name, channel_name=channel_name
    )


def _text(dialog) -> str:
    """Everything the dialog puts on screen, flattened."""
    from PyQt5.QtWidgets import QLabel

    return "\n".join(label.text() for label in dialog.findChildren(QLabel))


# --- the safety property -------------------------------------------------------


def test_start_is_never_the_default_button():
    """The dialog opens on the click that would start the mill. If Start were the
    default, Enter or Space would burn the sample."""
    dialog = _dialog()
    assert not dialog.button_start.isDefault(), "Start Milling is the default button"
    assert not dialog.button_start.autoDefault(), "Start Milling would take Enter"
    cancel = next(
        b for b in dialog.findChildren(QPushButton) if b.text() == "Cancel"
    )
    assert cancel.isDefault(), "Cancel is not the default"


def test_start_is_disabled_with_nothing_to_mill():
    dialog = _dialog(stages=[_stage(enabled=False)])
    assert not dialog.button_start.isEnabled()
    assert dialog.button_start.toolTip()


# --- it must report the mode that will actually run ----------------------------


def _row(dialog, label: str) -> str:
    """One row's value by label. Asserting on the flattened text instead let the Mode
    row's wording be satisfied by the Drop trigger caveat, which happens to contain the
    same phrase — the assertion passed while the row under test was wrong."""
    for row_label, value in dialog._rows():
        if row_label == label:
            return value
    raise AssertionError(f"no {label!r} row; rows are {[r[0] for r in dialog._rows()]}")


def test_supervised_says_you_stop_it():
    """Supervised is the mode where the drop trigger stops *nothing* — the strategy
    only acts on _drop_detected when not supervised. Reporting it as auto-stopping
    would tell the operator the mill halts itself when it will not."""
    dialog = _dialog(stages=[_stage(supervised=True)])
    assert "Supervised" in _text(dialog)
    assert _row(dialog, "Mode") == "supervised — you stop the mill manually"
    assert "monitoring only" in _row(dialog, "Drop trigger"), (
        "drop trigger not marked inert while supervised"
    )


def test_automated_says_it_stops_itself():
    dialog = _dialog(stages=[_stage(supervised=False)])
    assert "Automated" in _text(dialog)
    assert _row(dialog, "Mode") == "automated — stops itself on the drop trigger"
    assert "monitoring only" not in _row(dialog, "Drop trigger")


def test_mode_wording_matches_the_supervision_button():
    """The toggle a few pixels away sets its label to exactly these words
    (`_update_supervised_button`). A third synonym for one mode is worse than an
    imperfect one, so this pins them together."""
    import inspect

    from fibsem.ui.widgets import fluorescence_coincidence_viewer_widget as viewer

    source = inspect.getsource(viewer.FluorescenceCoincidenceViewerWidget._update_supervised_button)
    assert 'setText("Supervised")' in source
    assert 'setText("Automated")' in source


# --- the facts the operator is checking ----------------------------------------


def test_shows_per_stage_current_pattern_and_duration():
    stages = [
        _stage("Rough Mill", 1.0e-9, TrenchPattern(width=20e-6, depth=2e-6, spacing=3e-6,
                                                   upper_trench_height=8e-6,
                                                   lower_trench_height=6e-6)),
        _stage("Polish", 60e-12),
    ]
    text = _text(_dialog(stages=stages))
    for expected in ("Rough Mill", "Polish", "1.0 nA", "60 pA", "Trench", "Rectangle"):
        assert expected in text, f"missing {expected!r}"


def test_current_chip_spans_the_range_when_stages_differ():
    """A task stepping down rough → polish is the normal case; showing only the first
    stage's current would misreport the number most likely being checked."""
    text = _text(_dialog(stages=[_stage("Rough", 1.0e-9), _stage("Polish", 60e-12)]))
    assert "60 pA – 1.0 nA" in text

    single = _text(_dialog(stages=[_stage("Polish", 60e-12)]))
    assert "60 pA" in single and "–" not in single


def test_shows_channel_and_says_so_when_there_is_none():
    assert "GFP (488 nm)" in _text(_dialog())
    assert "none selected" in _text(_dialog(channel_name=None))


def test_shows_a_total_estimated_time():
    """The number the old QMessageBox omitted entirely."""
    config = FibsemMillingTaskConfig(stages=[_stage()])
    config.field_of_view = 80e-6
    dialog = CoincidenceMillingConfirmationDialog(config, lamella_name="x")
    assert config.estimated_time > 0
    assert "Estimated time" in _text(dialog)


def test_disabled_stages_are_counted_but_not_listed():
    stages = [_stage("Polish", 60e-12), _stage("Skipped", 1e-9, enabled=False)]
    text = _text(_dialog(stages=stages))
    assert "1 disabled" in text
    assert "Skipped" not in text, "a disabled stage was listed as if it would run"


# --- formatting helpers --------------------------------------------------------


def test_current_formats_nano_above_a_nanoamp():
    assert _format_current(1.0e-9) == "1.0 nA"
    assert _format_current(60e-12) == "60 pA"
    assert _format_current(2.5e-9) == "2.5 nA"


def test_pattern_summary_omits_dimensions_a_pattern_lacks():
    rect = _pattern_summary(_stage(pattern=RectanglePattern(width=12e-6, height=1.5e-6, depth=0.5e-6)))
    assert "12.0 wide" in rect and "1.5 tall" in rect and "0.5 deep" in rect

    trench = _pattern_summary(_stage(pattern=TrenchPattern(
        width=20e-6, depth=2e-6, spacing=3e-6,
        upper_trench_height=8e-6, lower_trench_height=6e-6)))
    assert "20.0 wide" in trench and "2.0 deep" in trench
    # secondary trench dimensions are deliberately left to the task config editor
    assert "gap" not in trench and "upper" not in trench


# --- the call site -------------------------------------------------------------


def test_run_milling_uses_this_dialog():
    import inspect

    from fibsem.ui.widgets import fluorescence_coincidence_viewer_widget as viewer

    source = inspect.getsource(viewer.FluorescenceCoincidenceViewerWidget._run_milling)
    assert "CoincidenceMillingConfirmationDialog" in source
    assert "QMessageBox(" not in source, "the old stock dialog is still built"
    assert "pformat" not in source, "raw config dump is back"


# --- FIB-443: the toggle must survive to the run ------------------------------


class _FakeStageList:
    """Stands in for MillingStageListWidget: holds the live stage objects that
    `get_stages()` hands out by reference, which is what makes a write stick."""

    def __init__(self, stages):
        self._stages = stages

    def get_stages(self):
        return self._stages


class _FakeConfigWidget:
    def __init__(self, stages):
        self.milling_stages_widget = _FakeStageList(stages)


class _FakeMillingWidget:
    def __init__(self, stages):
        self.config_widget = _FakeConfigWidget(stages)


def _viewer_with(stages):
    """A bare viewer instance with just the attributes these methods touch —
    constructing the real widget needs a microscope and an experiment."""
    from fibsem.ui.widgets.fluorescence_coincidence_viewer_widget import (
        FluorescenceCoincidenceViewerWidget as Viewer,
    )

    viewer = Viewer.__new__(Viewer)
    viewer.milling_viewer_widget = _FakeMillingWidget(stages)
    viewer._active_strategies = []  # as it is before the first mill of a session
    return viewer


def test_supervision_toggle_reaches_the_run_before_the_first_mill():
    """The operator sequence from FIB-443: toggle to automated, then start milling.

    _active_strategies is empty until _connect_coincidence_strategies runs inside
    _run_milling, so a toggle that only wrote through it was discarded — and the
    failure was in the unsafe direction, since supervised is the mode where the drop
    trigger stops nothing.
    """
    from fibsem.ui.widgets.fluorescence_coincidence_viewer_widget import (
        FluorescenceCoincidenceViewerWidget as Viewer,
    )

    stages = [_stage("Rough", 1e-9, supervised=True), _stage("Polish", 60e-12, supervised=True)]
    viewer = _viewer_with(stages)

    Viewer._on_supervised_toggled(viewer, False)

    assert [s.strategy.config.supervised for s in stages] == [False, False], (
        "toggle did not reach the strategies a run would use"
    )


def test_drop_threshold_reaches_the_run_before_the_first_mill():
    """Same root cause, same fix — the sibling control was equally inert."""
    from fibsem.ui.widgets.fluorescence_coincidence_viewer_widget import (
        FluorescenceCoincidenceViewerWidget as Viewer,
    )

    stages = [_stage("Polish", 60e-12, intensity_drop_fraction=0.20)]
    viewer = _viewer_with(stages)

    Viewer._on_drop_threshold_changed(viewer, 35)

    assert stages[0].strategy.config.intensity_drop_fraction == 0.35


def test_seeding_no_longer_reverts_a_toggle():
    """_seed_controls_from_strategy runs when a mill starts. It must now read back what
    the button already wrote, not the stale config value that used to overwrite it."""
    from fibsem.ui.widgets.fluorescence_coincidence_viewer_widget import (
        FluorescenceCoincidenceViewerWidget as Viewer,
    )

    stages = [_stage("Polish", 60e-12, supervised=True)]
    viewer = _viewer_with(stages)
    Viewer._on_supervised_toggled(viewer, False)

    # stand in for the button/spinbox the real seed would drive
    viewer._supervised = False
    viewer._update_supervised_button = lambda: None

    class _Spin:
        def blockSignals(self, _): pass
        def setValue(self, _): pass

    viewer.spin_drop_threshold = _Spin()
    Viewer._seed_controls_from_strategy(viewer)

    assert viewer._supervised is False, "seeding reverted the operator's toggle"


def _main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001 - standalone runner
            failures += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'FAILED' if failures else 'OK'} — {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
