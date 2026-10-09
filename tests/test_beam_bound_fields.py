"""A recipe field bound to a beam parameter shows it as the beam does: the unit,
scale, step and decimals it leaves out are the beam parameter's display hint."""

from fibsem.devices.beam import Beam
from fibsem.structures import (
    FibsemMillingSettings,
    ImageSettings,
    get_fields_with_metadata,
)


def test_the_milling_field_of_view_is_shown_as_the_beams():
    hfw = get_fields_with_metadata(FibsemMillingSettings)["hfw"]

    assert (hfw["unit"], hfw["scale"]) == ("m", 1e6)
    assert (hfw["step"], hfw["decimals"]) == (
        Beam.hfw.display.step,
        Beam.hfw.display.decimals,
    )
    # what the field says itself still wins
    assert (hfw["label"], hfw["minimum"], hfw["hidden"]) == (
        "Field of View",
        20.0,
        True,
    )


def test_the_milling_current_takes_the_beams_unit_and_keeps_its_label():
    current = get_fields_with_metadata(FibsemMillingSettings)["milling_current"]

    assert current["unit"] == "A"
    assert current["label"] == "Milling Current"


def test_the_beams_advanced_flag_stays_with_the_beam_panel():
    # Beam.voltage is advanced in the beam panel; the recipe field says for itself
    assert Beam.voltage.display.advanced
    assert get_fields_with_metadata(ImageSettings)["hfw"]["advanced"] is False


def test_a_field_bound_to_something_the_beam_does_not_declare_is_unchanged():
    application_file = get_fields_with_metadata(FibsemMillingSettings)[
        "application_file"
    ]

    assert application_file["unit"] is None
    assert application_file["scale"] is None


def test_image_settings_show_dwell_time_in_microseconds():
    dwell = get_fields_with_metadata(ImageSettings)["dwell_time"]

    assert (dwell["label"], dwell["unit"], dwell["scale"]) == ("Dwell Time", "s", 1e6)
