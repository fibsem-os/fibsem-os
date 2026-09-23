"""How an image aligned over the Overview tab is shown (FIB-1030).

Each channel is kept as its own layer, so a colour, a hidden channel or a contrast edit
re-blends what is in hand and never moves the image. By default the image is drawn
signal only -- brightness as coverage, so the overview shows through where the image
holds nothing. How it is shown is plain values a record can keep, set back without
announcing, and announced once an edit settles rather than per slider step.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import yaml

pytest.importorskip("PyQt5")

import fibsem.config as fibsem_config
from fibsem import utils
from fibsem.fm.composite import composite_fm_layers
from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemStagePosition,
    ImageSettings,
)
from fibsem.ui.widgets.overview_widget import FibsemOverviewWidget

SIZE = 64
GREEN_SPOT = (12, 12)  # row, col inside the first channel's square
RED_SPOT = (48, 48)  # inside the second's
DARK = (30, 5)  # in neither


@pytest.fixture(scope="module")
def microscope():
    path = os.path.join(
        os.path.dirname(fibsem_config.__file__),
        "config",
        "sim-arctis-configuration.yaml",
    )
    scope, _ = utils.setup_session(manufacturer="Demo", config_path=path)
    return scope


@pytest.fixture(autouse=True)
def _destroy_widgets(destroy_widgets_after_test):
    """See tests/ui/test_overview_widget.py: the widget leaves top-levels behind."""


@pytest.fixture
def widget(microscope):
    w = FibsemOverviewWidget(microscope)
    w.resize(900, 700)
    w.set_image(_sem_image(microscope))
    yield w
    w.close()


def _sem_image(scope):
    pose = scope.get_orientation("SEM")
    hfw = 128 * 2e-6
    image = FibsemImage.generate_blank_image(resolution=(128, 128), hfw=hfw)
    state = scope.get_microscope_state(beam_type=BeamType.ELECTRON)
    state.stage_position = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=pose.r, t=pose.t)
    image.metadata.image_settings = ImageSettings(hfw=hfw, beam_type=BeamType.ELECTRON)
    image.metadata.microscope_state = state
    image.metadata.system_info = scope.system.info
    image.metadata.hardware_geometry = scope.hardware_geometry()
    return image


def _channel(name, color):
    return FluorescenceChannelMetadata(
        name=name,
        excitation_wavelength=488.0,
        power=0.5,
        exposure_time=0.1,
        gain=1.0,
        offset=0.0,
        color=color,
    )


def _two_channels(scope, names=("GFP", "RFP")):
    """Dark everywhere but a bright square per channel: green top-left, red
    bottom-right. Each square is 6% of the frame, so auto contrast spans it."""
    data = np.zeros((2, 1, SIZE, SIZE), dtype=np.uint16)
    data[0, 0, 4:20, 4:20] = 4000
    data[1, 0, 40:56, 40:56] = 4000
    pose = scope.get_orientation("FM")
    image = FluorescenceImage(
        data=data,
        metadata=FluorescenceImageMetadata(
            acquisition_date="2026-09-23T10:00:00",
            pixel_size_x=1e-6,
            pixel_size_y=1e-6,
            stage_position=FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=pose.r, t=pose.t),
            channels=[_channel(names[0], "green"), _channel(names[1], "red")],
        ),
    )
    image.metadata.geometry = scope.fm_image_geometry()
    return image


def _add(widget, label="fm"):
    return widget.add_aligned_image(_two_channels(widget.microscope), label=label)


def _record(widget, key):
    return widget.aligned_images.get(key)


def _wait_until(condition, timeout_ms=2000):
    from PyQt5.QtTest import QTest

    waited = 0
    while not condition() and waited < timeout_ms:
        QTest.qWait(20)
        waited += 20
    return condition()


def _wait(ms):
    from PyQt5.QtTest import QTest

    QTest.qWait(ms)


class TestEachChannelIsKept:
    def test_as_a_layer_in_its_own_colour(self, widget):
        record = _record(widget, _add(widget))
        assert [(l.name, l.color) for l in record.layers] == [
            ("GFP", "green"),
            ("RFP", "red"),
        ]
        assert record.layers[0].data.shape == (SIZE, SIZE)
        assert np.array_equal(record.rgb, composite_fm_layers(record.layers))
        assert tuple(record.rgb[GREEN_SPOT]) == (0, 255, 0)
        assert tuple(record.rgb[RED_SPOT]) == (255, 0, 0)

    def test_a_colour_change_reaches_the_drawn_pixels(self, widget):
        key = _add(widget)
        record = _record(widget, key)
        record.layers[0].color = "magenta"
        widget.aligned_images.recomposite(key)
        assert tuple(record.rgb[GREEN_SPOT]) == (255, 0, 255)
        # What the overlay draws is the new composite, not a stale copy.
        assert record.overlay._data is record.drawn
        assert tuple(record.overlay._data[GREEN_SPOT][:3]) == (255, 0, 255)

    def test_a_hidden_channel_leaves_the_composite(self, widget):
        key = _add(widget)
        record = _record(widget, key)
        record.layers[1].visible = False
        widget.aligned_images.recomposite(key)
        assert tuple(record.rgb[RED_SPOT]) == (0, 0, 0)
        assert tuple(record.rgb[GREEN_SPOT]) == (0, 255, 0)
        assert record.overlay._data[RED_SPOT][3] == 0  # and so drawn clear

    def test_a_display_edit_never_moves_the_image(self, widget):
        key = _add(widget)
        widget.aligned_images.set_placement(key, 4e-6, -3e-6, 7.0)
        record = _record(widget, key)
        before = (record.placement, record.overlay.centre, record.overlay.rotation)
        record.layers[0].color = "cyan"
        widget.aligned_images.recomposite(key)
        widget.aligned_images.set_signal_only(key, False)
        widget.aligned_images.set_opacity(key, 0.3)
        after = (record.placement, record.overlay.centre, record.overlay.rotation)
        assert after == before


class TestSignalOnly:
    def test_is_how_an_image_starts(self, widget):
        record = _record(widget, _add(widget))
        assert record.signal_only
        assert widget.aligned_image_panel.check_signal_only.isChecked()

    def test_draws_dark_clear_and_signal_solid(self, widget):
        record = _record(widget, _add(widget))
        drawn = record.overlay._data
        assert drawn.shape == (SIZE, SIZE, 4)
        assert drawn[DARK][3] == 0
        assert drawn[GREEN_SPOT][3] == 255
        # Colour times coverage is the composite again: nothing is lost over black.
        restored = drawn[..., :3].astype(float) * drawn[..., 3:4] / 255.0
        assert np.abs(restored - record.rgb).max() <= 1.0

    def test_off_draws_the_whole_frame(self, widget):
        key = _add(widget)
        widget.aligned_images.set_signal_only(key, False)
        drawn = _record(widget, key).overlay._data
        assert (drawn[..., 3] == 255).all()
        assert tuple(drawn[DARK][:3]) == (0, 0, 0)


class TestTheDisplayState:
    def test_is_plain_values_a_record_can_write(self, widget):
        key = _add(widget)
        layer = _record(widget, key).layers[0]
        layer.autocontrast = False
        layer.clim = (np.float32(10.0), np.float64(3000.0))
        state = widget.aligned_images.display_state(key)
        text = yaml.safe_dump(state)
        assert "numpy" not in text
        assert state["channels"][0]["clim"] == [10.0, 3000.0]
        assert state["signal_only"] is True

    def test_set_back_it_shows_the_image_the_same_way(self, widget):
        first = _add(widget, "first")
        record = _record(widget, first)
        record.layers[0].color = "yellow"
        record.layers[1].visible = False
        record.layers[1].gamma = 0.5
        widget.aligned_images.recomposite(first)
        widget.aligned_images.set_signal_only(first, False)
        widget.aligned_images.set_opacity(first, 0.35)
        state = widget.aligned_images.display_state(first)

        second = _add(widget, "second")
        widget.aligned_images.set_display_state(second, state)
        again = _record(widget, second)
        assert widget.aligned_images.display_state(second) == state
        assert np.array_equal(again.rgb, record.rgb)
        assert again.overlay.opacity == pytest.approx(0.35)
        assert not again.signal_only

    def test_channels_are_matched_by_name_not_position(self, widget):
        key = _add(widget)
        state = widget.aligned_images.display_state(key)
        state["channels"] = list(reversed(state["channels"]))
        state["channels"][0]["color"] = "blue"  # RFP, now listed first
        widget.aligned_images.set_display_state(key, state)
        assert [l.color for l in _record(widget, key).layers] == ["green", "blue"]

    def test_renamed_channels_fall_back_to_position(self, widget):
        key = _add(widget)
        state = widget.aligned_images.display_state(key)
        for entry, name in zip(state["channels"], ("A", "B")):
            entry["name"] = name
        state["channels"][0]["color"] = "cyan"
        widget.aligned_images.set_display_state(key, state)
        assert _record(widget, key).layers[0].color == "cyan"


class TestThePanel:
    def test_shows_the_selected_images_display(self, widget):
        panel = widget.aligned_image_panel
        first = _add(widget, "first")
        widget.aligned_images.set_opacity(first, 0.25)
        widget.aligned_images.set_signal_only(first, False)
        second = _add(widget, "second")
        assert panel.slider_opacity.value() == 60
        assert panel.check_signal_only.isChecked()
        panel.combo.setCurrentIndex(panel.combo.findData(first))
        assert panel.slider_opacity.value() == 25
        assert not panel.check_signal_only.isChecked()
        panel.combo.setCurrentIndex(panel.combo.findData(second))
        assert panel.check_signal_only.isChecked()

    def test_its_controls_reach_the_selected_image(self, widget):
        panel = widget.aligned_image_panel
        key = _add(widget)
        panel.check_signal_only.setChecked(False)
        panel.slider_opacity.setValue(80)
        record = _record(widget, key)
        assert not record.signal_only
        assert record.overlay.opacity == pytest.approx(0.8)


class TestEditsAreAnnouncedOnceTheySettle:
    def test_a_slider_drag_is_one_announcement(self, widget):
        key = _add(widget)
        panel = widget.aligned_image_panel
        heard = []
        widget.image_display_changed.connect(heard.append)
        for value in (50, 45, 40, 35):
            panel.slider_opacity.setValue(value)
        panel.check_signal_only.setChecked(False)
        assert heard == []  # not per step
        assert _wait_until(lambda: heard == [key])
        _wait(600)
        assert heard == [key]

    def test_setting_a_display_from_a_record_is_not_announced(self, widget):
        key = _add(widget)
        state = widget.aligned_images.display_state(key)
        state["opacity"] = 0.2
        state["signal_only"] = False
        state["channels"][0]["color"] = "yellow"
        heard = []
        widget.image_display_changed.connect(heard.append)
        widget.set_aligned_image_display(key, state)
        _wait(600)
        assert heard == []
        # And the panel shows what was set.
        assert widget.aligned_image_panel.slider_opacity.value() == 20
        assert not widget.aligned_image_panel.check_signal_only.isChecked()

    def test_a_removed_image_is_not_announced(self, widget):
        key = _add(widget)
        heard = []
        widget.image_display_changed.connect(heard.append)
        widget.aligned_image_panel.slider_opacity.setValue(10)
        widget.remove_aligned_image(key)
        _wait(600)
        assert heard == []


class TestTheChannelControls:
    def test_edit_the_selected_images_layers(self, widget):
        key = _add(widget)
        heard = []
        widget.image_display_changed.connect(heard.append)
        widget.aligned_image_panel.btn_channels.click()
        panel = widget._channels_panel
        assert panel.isVisible()
        assert panel._layers is _record(widget, key).layers
        panel.colormap.setCurrentText("cyan")
        record = _record(widget, key)
        assert record.layers[0].color == "cyan"
        assert _wait_until(lambda: tuple(record.rgb[GREEN_SPOT]) == (0, 255, 255))
        assert _wait_until(lambda: heard == [key])

    def test_edits_that_queue_up_are_blended_once(self, widget, monkeypatch):
        key = _add(widget)
        widget.aligned_image_panel.btn_channels.click()
        blends = []
        real = widget.aligned_images.recomposite
        monkeypatch.setattr(
            widget.aligned_images,
            "recomposite",
            lambda k: blends.append(k) or real(k),
        )
        panel = widget._channels_panel
        for value in (90, 80, 70, 60):
            panel.opacity.setValue(value)  # the selected channel's opacity
        assert _wait_until(lambda: blends == [key])
        _wait(50)
        assert blends == [key]
        assert _record(widget, key).layers[0].opacity == pytest.approx(0.6)

    def test_the_button_closes_them_again(self, widget):
        _add(widget)
        widget.aligned_image_panel.btn_channels.click()
        widget.aligned_image_panel.btn_channels.click()
        assert not widget._channels_panel.isVisible()

    def test_follow_the_selection(self, widget):
        panel = widget.aligned_image_panel
        first = _add(widget, "first")
        second = _add(widget, "second")
        panel.btn_channels.click()
        assert widget._channels_panel._layers is _record(widget, second).layers
        panel.combo.setCurrentIndex(panel.combo.findData(first))
        assert widget._channels_panel._layers is _record(widget, first).layers
        assert widget._channels_panel.isVisible()

    def test_go_with_the_last_image(self, widget):
        key = _add(widget)
        widget.aligned_image_panel.btn_channels.click()
        widget.remove_aligned_image(key)
        assert not widget._channels_panel.isVisible()

    def test_move_to_the_image_left_when_theirs_is_removed(self, widget):
        first = _add(widget, "first")
        second = _add(widget, "second")
        widget.aligned_image_panel.btn_channels.click()
        widget.remove_aligned_image(second)
        assert widget._channels_panel.isVisible()
        assert widget._channels_panel._layers is _record(widget, first).layers
        assert widget._channels_key == first

    def test_go_with_the_widget(self, widget):
        """A top-level tool window: it would float over whatever came next once
        the tab is left (FIB-962) unless the widget hides it."""
        _add(widget)
        widget.show()
        widget.aligned_image_panel.btn_channels.click()
        assert widget._channels_panel.isVisible()
        widget.hide()
        assert not widget._channels_panel.isVisible()
