"""The bar under each view: the displayed image's metadata, in the export bar's words.

FIB-1186. The bar replaces the title row above each SEM, FIB and FM panel, in the quad
view and the lamella editor alike, and reads the image through the same
`image_fields` the export does -- so the two can never disagree about an image.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QLabel

from fibsem import config as cfg
from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.imaging.export import (
    BEAM_FIELD_KEYS,
    FM_FIELD_KEYS,
    from_fibsem_image,
    image_fields,
    z_value,
)
from fibsem.structures import (
    BeamSettings,
    BeamType,
    FibsemDetectorSettings,
    FibsemImage,
    MicroscopeState,
)
from fibsem.ui.widgets.canvas.quad_view import (
    LamellaEditorView,
    MicroscopeViewController,
    QuadViewWidget,
)
from fibsem.ui.widgets.canvas.view_info_bar import BAR_DEFAULT_FIELDS, ViewInfoBar

_app = QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(autouse=True)
def _preferences(tmp_path, monkeypatch):
    """Every bar reads and writes a throwaway preferences file, never the real one."""
    path = tmp_path / "user-preferences.yaml"
    monkeypatch.setattr(cfg, "USER_PREFERENCES_PATH", str(path))
    monkeypatch.setattr(cfg, "CONFIG_PATH", str(tmp_path))
    return path


def _beam_image(beam_type=BeamType.ELECTRON, voltage=2e3) -> FibsemImage:
    image = FibsemImage.generate_blank_image(resolution=(1536, 1024), hfw=150e-6)
    image.metadata.image_settings.beam_type = beam_type
    settings = BeamSettings(
        beam_type, voltage=voltage, beam_current=50e-12, working_distance=4e-3
    )
    detector = FibsemDetectorSettings(type="ETD", mode="SE")
    if beam_type is BeamType.ION:
        state = MicroscopeState(ion_beam=settings, ion_detector=detector)
    else:
        state = MicroscopeState(electron_beam=settings, electron_detector=detector)
    image.metadata.microscope_state = state
    return image


def _fm_image(slices=3) -> FluorescenceImage:
    data = np.zeros((1, slices, 64, 64), dtype=np.uint16)
    data[0, :, 20:30, 20:30] = 4000
    channel = FluorescenceChannelMetadata(
        name="GFP",
        excitation_wavelength=470,
        power=0.5,
        exposure_time=0.1,
        gain=1.0,
        offset=0.0,
        color="green",
        objective_magnification=100,
        objective_numerical_aperture=0.75,
    )
    return FluorescenceImage(
        data=data,
        metadata=FluorescenceImageMetadata(
            acquisition_date="2026-10-07T14:00:00",
            pixel_size_x=110e-9,
            pixel_size_y=110e-9,
            pixel_size_z=500e-9,
            channels=[channel],
        ),
    )


def _shown(bar: ViewInfoBar):
    return [(f.label, f.value) for f in bar.visible_fields()]


@pytest.fixture
def controller():
    return MicroscopeViewController(view=QuadViewWidget())


def test_the_bar_says_what_the_export_says(controller):
    image = _beam_image()
    controller.set_image(BeamType.ELECTRON, image)
    bar = controller.widget.sem_bar
    bar.resize(2000, bar.height())  # room for everything

    exported = {f.key: f.value for f in from_fibsem_image(image).fields}
    assert bar.kind == "SEM"
    assert bar.header_label.text() == exported["detector"]
    assert _shown(bar) == [
        ("HFW", exported["hfw"]),
        ("HV", exported["voltage"]),
        ("I", exported["current"]),
        ("WD", exported["working_distance"]),
    ]


def test_each_beam_fills_its_own_bar(controller):
    controller.set_image(BeamType.ION, _beam_image(BeamType.ION, voltage=30e3))
    widget = controller.widget
    widget.fib_bar.resize(2000, 26)
    assert ("HV", "30 kV") in _shown(widget.fib_bar)
    assert _shown(widget.sem_bar) == []


def test_the_bar_sits_under_the_canvas_and_the_title_row_is_gone(controller):
    widget = controller.widget
    for canvas, bar in (
        (widget.sem_canvas, widget.sem_bar),
        (widget.fib_canvas, widget.fib_bar),
        (widget.fm_widget, widget.fm_bar),
    ):
        panel = bar.parentWidget()
        layout = panel.layout()
        assert layout.indexOf(canvas) == 0
        assert layout.indexOf(bar) == layout.count() - 1
        titles = [
            label
            for label in panel.findChildren(QLabel)
            if label.text() in ("SEM", "FIB", "FM") and label is not bar.kind_label
        ]
        assert titles == []


def test_the_fm_bar_sits_below_the_z_row(controller):
    """Every bar on its panel's bottom edge, so a row of views lines up."""
    fm_panel = controller.widget.fm_bar.parentWidget()
    assert fm_panel.layout().indexOf(controller.widget.fm_widget) == 0


def test_the_pixel_size_is_off_by_default_but_can_be_shown(controller):
    """HFW says it better under a live view; the pixel size is one setting away."""
    controller.set_image(BeamType.ELECTRON, _beam_image())
    bar = controller.widget.sem_bar
    bar.resize(2000, 26)
    assert "px" not in dict(_shown(bar))
    bar.set_field_keys(("detector", "hfw", "pixel_size"))
    assert dict(_shown(bar))["px"] == "97.7 nm"


def test_full_names_on_hover(controller):
    controller.set_image(BeamType.ELECTRON, _beam_image())
    bar = controller.widget.sem_bar
    bar.resize(2000, 26)
    tips = [label.toolTip() for label in bar.field_labels]
    assert "Horizontal field width: 150 µm" in tips
    assert "Accelerating voltage: 2 kV" in tips
    assert bar.header_label.toolTip() == "Detector · mode: ETD · SE"


def _sized(bar: ViewInfoBar, width: int) -> None:
    """Resize a shown bar and let Qt deliver it: a hidden widget's resize is deferred."""
    bar.resize(width, bar.height())
    _app.processEvents()


def test_a_narrow_bar_drops_whole_fields_behind_a_chip():
    bar = ViewInfoBar("SEM")
    bar.set_image_fields(image_fields(_beam_image()))
    bar.show()

    _sized(bar, 2000)
    assert len(bar.visible_fields()) == 4
    assert bar.more_chip.isHidden()

    _sized(bar, 280)
    visible, hidden = bar.visible_fields(), bar.hidden_fields()
    assert hidden, "280 px cannot hold four fields beside the header"
    assert len(visible) + len(hidden) == 4
    # dropped from the right, whole
    assert [f.key for f in visible + hidden] == [
        "hfw",
        "voltage",
        "current",
        "working_distance",
    ]
    # the time goes first, and is counted and listed with the fields
    assert bar.time_label.isHidden()
    assert not bar.more_chip.isHidden()
    assert bar.more_chip.text() == f"+{len(hidden) + 1}"
    for item in hidden:
        assert item.value in bar.more_chip.toolTip()
    assert "Acquired at: " in bar.more_chip.toolTip()

    _sized(bar, 2000)
    assert bar.hidden_fields() == []
    assert not bar.time_label.isHidden()
    assert bar.more_chip.isHidden()
    bar.close()


def test_the_time_drops_before_any_field():
    bar = ViewInfoBar("SEM")
    bar.set_image_fields(image_fields(_beam_image()))
    bar.show()
    _sized(bar, 2000)
    fields = sum(label.sizeHint().width() for label in bar.field_labels) + 12 * 3
    exact = bar.width() - bar._room(with_time=True) + fields
    _sized(bar, exact - 1)  # one pixel short of fitting the time as well
    assert bar.hidden_fields() == []
    assert bar.time_label.isHidden()
    assert bar.more_chip.text() == "+1"
    bar.close()


def test_the_bar_never_widens_its_view(controller):
    """A full bar must not hold the splitter open: its fields drop instead."""
    panel = controller.widget.sem_bar.parentWidget()
    before = panel.minimumSizeHint().width()
    controller.set_image(BeamType.ELECTRON, _beam_image())
    assert panel.minimumSizeHint().width() == before


def test_a_live_frame_with_the_same_metadata_does_not_rebuild(controller):
    """Every live frame lands here: an unchanged bar keeps its labels."""
    controller.set_image(BeamType.ELECTRON, _beam_image())
    labels = list(controller.widget.sem_bar.field_labels)
    controller.set_image(BeamType.ELECTRON, _beam_image())
    assert controller.widget.sem_bar.field_labels == labels


def test_an_image_without_metadata_leaves_just_the_name(controller):
    image = _beam_image()
    image.metadata = None
    controller.set_image(BeamType.ELECTRON, image)
    bar = controller.widget.sem_bar
    assert bar.kind == "SEM"
    assert bar.field_labels == []
    assert bar.header_label.isHidden()


def test_the_fm_z_follows_the_plane_on_screen(controller):
    controller.set_fm_image(_fm_image(slices=3))
    bar = controller.widget.fm_bar
    bar.resize(2000, 26)
    fm = controller.widget.fm_widget

    def z():
        return dict(_shown(bar))["Z"]

    assert bar.header_label.text() == "100× · 0.75 NA"
    assert z() == "MIP · 3 × 500 nm"
    fm.set_max_projection(False)
    assert z() == "1 of 3 × 500 nm"
    labels = list(bar.field_labels)
    fm.step_z(1)
    assert z() == "2 of 3 × 500 nm"
    # a new value in the same field: the label is reused, never stacked on a new one
    assert bar.field_labels == labels
    fm.set_max_projection(True)
    assert z() == "MIP · 3 × 500 nm"


def test_clear_empties_every_bar(controller):
    controller.set_image(BeamType.ELECTRON, _beam_image())
    controller.set_fm_image(_fm_image())
    controller.clear()
    widget = controller.widget
    for bar in (widget.sem_bar, widget.fib_bar, widget.fm_bar):
        assert bar.field_labels == []


def test_the_lamella_editor_gets_the_same_bars():
    controller = MicroscopeViewController(view=LamellaEditorView())
    controller.set_image(BeamType.ION, _beam_image(BeamType.ION, voltage=30e3))
    bar = controller.widget.fib_bar
    bar.resize(2000, 26)
    assert bar.kind == "FIB"
    assert ("HV", "30 kV") in _shown(bar)
    assert controller.widget.fm_bar.kind == "FM"


class _Objective:
    position = 0.0


class _FM:
    objective = _Objective()


class _Microscope:
    """What `update_info` reads, and an FM whose objective nobody should read."""

    fm = _FM()
    stage = object()  # a stage is fitted, so update_info shows it
    current_grid = "GRID-01"

    def __init__(self):
        from fibsem.structures import FibsemStagePosition

        self._stage_position = FibsemStagePosition(x=0, y=0, z=0, r=0, t=0)

    def get_stage_orientation(self, stage_position=None):
        return "SEM"

    def get_current_milling_angle(self, stage_position=None):
        return 10.0


def _obj(bar):
    return {f.label: f for f in bar.visible_fields()}.get("OBJ")


def test_the_fm_bar_shows_the_objective_it_is_told(controller):
    controller.update_info(_Microscope(), objective_position=200e-6)
    bar = controller.widget.fm_bar
    bar.resize(2000, 26)
    obj = _obj(bar)
    assert obj.value == "200.0 µm"
    # live, and the hover says so: this is not the stack's recorded position
    assert "Objective position (now): 200.0 µm" in [
        l.toolTip() for l in bar.field_labels
    ]

    controller.update_info(_Microscope(), objective_position=-1.5e-6)
    assert _obj(bar).value == "-1.5 µm"


def test_the_objective_is_off_the_canvas_text(controller):
    controller.update_info(_Microscope(), objective_position=200e-6)
    info = dict(controller._states[controller.widget.fm_canvas].info)
    assert "objective" not in info
    assert "stage" in info  # the stage stays there until it has a home of its own


def test_a_new_stack_or_a_clear_keeps_the_objective(controller):
    """Live values did not come from the image, so an image change leaves them."""
    controller.update_info(_Microscope(), objective_position=200e-6)
    controller.set_fm_image(_fm_image())
    bar = controller.widget.fm_bar
    bar.resize(2000, 26)
    labels = [f.label for f in bar.visible_fields()]
    assert labels[-1] == "OBJ", "live values sit after the image's fields"
    controller.clear()
    assert _obj(bar).value == "200.0 µm"


def test_a_live_value_can_be_removed():
    bar = ViewInfoBar("FM")
    bar.set_live_field("objective_position", "OBJ", "1.0 µm", name="Objective")
    assert [f.label for f in bar.visible_fields()] == ["OBJ"]
    bar.set_live_field("objective_position", "OBJ", None, name="Objective")
    assert bar.visible_fields() == []


def _chosen():
    return cfg.load_user_preferences().display.info_bar_fields


def test_only_the_selected_view_shows_its_field_button(controller):
    """Like the canvas toolbar: three idle buttons would only distract."""
    widget = controller.widget
    bars = {"SEM": widget.sem_bar, "FIB": widget.fib_bar, "FM": widget.fm_bar}
    assert widget.selected is BeamType.ELECTRON
    assert {k for k, b in bars.items() if not b.fields_button.isHidden()} == {"SEM"}
    widget.set_selected(BeamType.ION)
    assert {k for k, b in bars.items() if not b.fields_button.isHidden()} == {"FIB"}
    widget.set_selected("fm")
    assert {k for k, b in bars.items() if not b.fields_button.isHidden()} == {"FM"}


def test_the_lamella_editor_s_bars_keep_their_button():
    """It has no selection, so every bar keeps its button."""
    view = LamellaEditorView()
    for bar in (view.sem_bar, view.fib_bar, view.fm_bar):
        assert not bar.fields_button.isHidden()


def test_the_picker_offers_every_field_the_kind_records():
    sem = ViewInfoBar("SEM").open_picker()
    assert tuple(sem.checkboxes) == BEAM_FIELD_KEYS + ("date",)
    assert sem.checkboxes["pixel_size"].text() == "Pixel size (px)"
    assert sem.checkboxes["hfw"].text() == "HFW"
    assert sem.checkboxes["date"].text() == "Acquired at"
    fm = ViewInfoBar("FM").open_picker()
    assert tuple(fm.checkboxes) == FM_FIELD_KEYS + ("date",)
    sem.close()
    fm.close()


def test_the_button_opens_the_picker_ticked_as_the_bar_is():
    bar = ViewInfoBar("SEM")
    bar.fields_button.click()
    picker = bar._picker
    assert picker is not None and picker.isVisible()
    ticked = {k for k, cb in picker.checkboxes.items() if cb.isChecked()}
    assert ticked == set(BAR_DEFAULT_FIELDS) & set(picker.checkboxes)
    picker.close()


def test_ticking_a_field_shows_it_and_remembers_it(controller):
    controller.set_image(BeamType.ELECTRON, _beam_image())
    bar = controller.widget.sem_bar
    bar.resize(2000, 26)
    picker = bar.open_picker()

    picker.checkboxes["pixel_size"].setChecked(True)

    assert dict(_shown(bar))["px"] == "97.7 nm"
    assert "pixel_size" in _chosen()["SEM"]
    # a bar built later -- after a restart -- starts from the choice
    assert "pixel_size" in ViewInfoBar("SEM").field_keys()
    picker.close()


def test_every_open_bar_of_the_kind_follows_at_once(controller):
    """The quad view and the lamella editor each have an SEM bar."""
    editor = MicroscopeViewController(view=LamellaEditorView())
    picker = controller.widget.sem_bar.open_picker()
    picker.checkboxes["dwell_time"].setChecked(True)
    assert "dwell_time" in editor.widget.sem_bar.field_keys()
    assert "dwell_time" not in editor.widget.fib_bar.field_keys()
    assert "SEM" in _chosen() and "FIB" not in _chosen()
    picker.close()


def test_a_closed_window_s_bar_does_not_stop_the_others():
    """A bar can outlive its widget in the set of open bars (a closed lamella
    editor); reaching it from the picker's slot would abort the application."""
    from PyQt5 import sip

    from fibsem.ui.widgets.canvas.view_info_bar import choose_field_keys

    gone, kept = ViewInfoBar("SEM"), ViewInfoBar("SEM")
    sip.delete(gone)
    choose_field_keys("SEM", ("hfw",))
    assert kept.field_keys() == ("hfw",)


def test_unticking_the_time_hides_it(controller):
    controller.set_image(BeamType.ELECTRON, _beam_image())
    bar = controller.widget.sem_bar
    assert not bar.time_label.text() == ""
    picker = bar.open_picker()
    picker.checkboxes["date"].setChecked(False)
    assert bar.time_label.text() == ""
    picker.close()


def test_defaults_forgets_the_choice(controller):
    bar = controller.widget.sem_bar
    picker = bar.open_picker()
    picker.checkboxes["pixel_size"].setChecked(True)
    assert "SEM" in _chosen()

    picker.defaults_button.click()

    assert "SEM" not in _chosen(), "a default is not stored, so a new one reaches you"
    assert bar.field_keys() == BAR_DEFAULT_FIELDS
    assert not picker.checkboxes["pixel_size"].isChecked()
    picker.close()


def test_saving_one_setting_keeps_what_another_window_saved():
    """The main window held its preferences from start-up and saved them whole,
    which would put back a bar's fields chosen in between."""
    cfg.update_user_preferences(
        lambda p: p.display.info_bar_fields.__setitem__("FIB", ["hfw"])
    )
    cfg.update_user_preferences(lambda p: setattr(p.display, "sound_enabled", True))
    preferences = cfg.load_user_preferences()
    assert preferences.display.info_bar_fields == {"FIB": ["hfw"]}
    assert preferences.display.sound_enabled


def test_the_choice_round_trips_and_an_old_file_has_none():
    preferences = cfg.UserPreferences()
    preferences.display.info_bar_fields = {"FM": ["objective", "z"]}
    again = cfg.UserPreferences.from_dict(preferences.to_dict())
    assert again.display.info_bar_fields == {"FM": ["objective", "z"]}
    assert cfg.UserPreferences.from_dict({"display": {}}).display.info_bar_fields == {}


@pytest.mark.parametrize(
    "plane, expected",
    [(None, "MIP · 21 × 568 nm"), (0, "1 of 21 × 568 nm"), (10, "11 of 21 × 568 nm")],
)
def test_z_value(plane, expected):
    assert z_value(21, 568e-9, plane=plane) == expected
