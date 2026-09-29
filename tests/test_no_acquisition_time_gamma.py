"""Acquisition does not modify the pixels it returns.

`ImageSettings` used to carry an `autogamma` flag that applied gamma correction to the
array after acquisition. It went (FIB-505): unlike `autocontrast`, which configures the
detector *before* the image exists, gamma is a display correction applied afterwards,
and baking it into stored data is destructive and unrecoverable. The canvas applies it
at display time instead, where it is adjustable.

Nothing shipped ever enabled it -- all seven configuration files set it false -- so this
removes a feature that was off everywhere rather than changing what anyone was getting.
"""

import inspect
import json
import os
from pathlib import Path

import numpy as np
import pytest

from fibsem import acquire
from fibsem.structures import FibsemImageMetadata, ImageSettings

CONFIG_DIR = Path(__file__).resolve().parents[1] / "fibsem" / "config"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "metadata"


def test_image_settings_has_no_gamma_flag() -> None:
    """The request describes the acquisition. Gamma is not part of one."""
    settings = ImageSettings()

    assert not hasattr(settings, "autogamma")
    assert "autogamma" not in settings.to_dict()


def test_autocontrast_stays() -> None:
    """It is a genuine instrument operation, and is deliberately unaffected."""
    settings = ImageSettings()

    assert hasattr(settings, "autocontrast")
    assert "autocontrast" in settings.to_dict()


@pytest.mark.parametrize("value", [True, False])
def test_a_stored_configuration_carrying_the_old_flag_still_loads(value: bool) -> None:
    """`autogamma` was a `[USER]` field, so a site's own config may still set it.

    It is ignored rather than raising -- `from_dict` reads named keys, so an unknown one
    passes through harmlessly. A site that had it on stops getting gamma baked into its
    data, which is the point of the change.
    """
    settings = ImageSettings.from_dict(
        {
            "hfw": 50e-6,
            "dwell_time": 1e-6,
            "beam_type": "ELECTRON",
            "autocontrast": True,
            "autogamma": value,
        }
    )

    assert settings.hfw == 50e-6
    assert settings.autocontrast is True
    assert not hasattr(settings, "autogamma")


def test_images_written_before_v7_still_load() -> None:
    """The fixtures are real instrument output and all carry the key."""
    for fixture in FIXTURE_DIR.glob("*.json"):
        raw = json.loads(fixture.read_text(encoding="utf-8"))
        assert "autogamma" in raw["image"], f"{fixture.name} should predate v7"

        metadata = FibsemImageMetadata.from_dict(raw)

        assert metadata.image_settings.hfw is not None
        assert "autogamma" not in metadata.to_dict()["image"]


def test_no_shipped_configuration_still_declares_it() -> None:
    """Leaving the key in would advertise a setting that does nothing."""
    declaring = [
        path.name for path in CONFIG_DIR.glob("*.yaml") if "autogamma" in path.read_text(encoding="utf-8")
    ]

    assert declaring == []


def test_acquisition_does_not_transform_the_array() -> None:
    """The read that matters: no post-acquisition processing step remains.

    Asserted against the source rather than by acquiring, because the point is the
    absence of a call -- an acquisition test passes whether or not gamma ran, unless it
    happens to pick data the correction would have changed.
    """
    source = inspect.getsource(acquire.acquire_image)

    assert "auto_gamma" not in source
    assert "autogamma" not in source


def test_gamma_remains_available_to_call_deliberately() -> None:
    """Removed from the acquisition path, not from the library.

    Someone who wants gamma can still apply it; it is no longer applied on their behalf
    and written to disk.
    """
    from fibsem.autofunctions.gamma import auto_gamma
    from fibsem.structures import FibsemImage

    image = FibsemImage(data=np.full((8, 8), 200, dtype=np.uint8))
    corrected = auto_gamma(image, method="autogamma")

    assert corrected.data.shape == image.data.shape
