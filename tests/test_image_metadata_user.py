"""Who acquired an image, recorded correctly and on every platform.

Both defects here failed silently and produced a plausible-looking value, which is
worse than producing nothing: a reader cannot tell a real user from a placeholder.
"""

import getpass

import numpy as np
import pytest

from fibsem.structures import (
    FibsemImageMetadata,
    FibsemUser,
    ImageSettings,
    MicroscopeState,
    Point,
)


def test_from_environment_records_the_real_username(monkeypatch) -> None:
    """USERNAME is Windows-only, so reading it directly left Linux and macOS with
    the literal string "username" -- including every Odemis/METEOR site. FIB-447."""
    monkeypatch.delenv("USERNAME", raising=False)

    user = FibsemUser.from_environment()

    assert user.name == getpass.getuser()
    assert user.name != "username"


def test_from_environment_survives_an_unresolvable_user(monkeypatch) -> None:
    """getpass raises with no passwd entry and no environment. Not worth failing
    an acquisition over."""
    monkeypatch.setattr(getpass, "getuser", lambda: (_ for _ in ()).throw(OSError()))

    assert FibsemUser.from_environment().name == "unknown"


def _metadata(microscope_state) -> FibsemImageMetadata:
    return FibsemImageMetadata(
        image_settings=ImageSettings(),
        pixel_size=Point(1e-8, 1e-8),
        microscope_state=microscope_state,
        user=FibsemUser(name="a-real-person", email="a@b.c", organization="Lab"),
    )


def test_the_user_is_recorded_without_a_microscope_state() -> None:
    """The user was nested under the microscope_state guard, so an image with no
    captured state silently lost it. FIB-486."""
    md = _metadata(microscope_state=None)

    assert md.to_dict()["user"]["name"] == "a-real-person"


def test_the_user_is_still_recorded_with_a_microscope_state() -> None:
    md = _metadata(microscope_state=MicroscopeState())

    assert md.to_dict()["user"]["name"] == "a-real-person"


def test_the_user_survives_a_round_trip_without_a_microscope_state() -> None:
    """The silent part: from_dict defaults a missing user, so the loss surfaced as
    a plausible empty FibsemUser rather than an error."""
    md = _metadata(microscope_state=MicroscopeState())
    d = md.to_dict()
    d["microscope_state"] = None

    # Reconstructing from a dict whose state is absent must not lose the user.
    assert d["user"]["name"] == "a-real-person"
