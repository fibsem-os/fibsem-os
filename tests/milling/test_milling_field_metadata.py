"""Milling fields in the device Parameter's vocabulary: SI limits and a Display."""

import pytest

from fibsem.milling.properties import distance_metadata
from fibsem.milling.strategy.coincidence import CoincidenceMillingStrategyConfig
from fibsem.milling.strategy.overtilt import OvertiltTrenchMillingConfig
from fibsem.structures import RangeLimit, get_fields_with_metadata


def test_a_distance_is_metres_shown_in_microns():
    metadata = distance_metadata("Width", "The width.", limits=RangeLimit(1e-6, 1e-4))
    assert metadata["unit"] == "m"
    assert metadata["limits"] == RangeLimit(1e-6, 1e-4)
    assert metadata["display"].scale == 1e6
    assert metadata["display"].label == "Width"


def test_hidden_and_advanced_come_from_the_display():
    """The strategy lists read the merged metadata, where a Display's flags land."""
    config = CoincidenceMillingStrategyConfig()
    assert "bbox" in config._hidden_attributes
    assert "secret_parameter" in OvertiltTrenchMillingConfig().advanced_attributes


def test_whole_number_fields_are_typed_as_ints():
    """Typed from the annotation, so the strategy form shows a whole-number box
    rather than a float one that read 1800 back as 1800.0."""
    metadata = get_fields_with_metadata(CoincidenceMillingStrategyConfig)
    assert metadata["timeout"]["type"] is int
    assert (metadata["timeout"]["minimum"], metadata["timeout"]["maximum"]) == (
        30,
        9000,
    )


def test_an_area_is_shown_in_square_microns():
    area = get_fields_with_metadata(OvertiltTrenchMillingConfig)["area_parameter"]
    assert area["scale"] == pytest.approx(1e12)
    assert area["display_unit"] == "µm²"
