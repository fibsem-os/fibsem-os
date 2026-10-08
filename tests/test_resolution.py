"""Resolution: a (width, height) tuple with the "WxH" spelling."""

import pytest

from fibsem.structures import Resolution


def test_it_is_the_plain_tuple():
    resolution = Resolution(1536, 1024)
    assert resolution == (1536, 1024)
    assert tuple(resolution) == (1536, 1024)
    assert (1536, 1024) in [resolution]


def test_it_spells_and_reads_wxh():
    assert str(Resolution(1536, 1024)) == "1536x1024"
    assert Resolution.parse("1536x1024") == (1536, 1024)
    assert Resolution.parse("1536 X 1024") == (1536, 1024)


@pytest.mark.parametrize("text", ["1536", "1536x", "axb", "1x2x3", None])
def test_it_refuses_what_is_not_wxh(text):
    with pytest.raises(ValueError, match="WxH"):
        Resolution.parse(text)


def test_its_aspect_is_width_over_height():
    assert Resolution(1536, 1024).aspect == 1.5
