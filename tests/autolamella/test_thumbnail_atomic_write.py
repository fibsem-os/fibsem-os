"""A lamella thumbnail is never read half-written.

Reported from the microscope, 2026-08-11:

    OSError: image file is truncated
      ... lamella_card_widget.py:344 in refresh -> lamella.get_thumbnail()
      ... structures.py:950 -> Image.open(thumb_path).convert("RGB")

`save_thumbnail` wrote straight to `thumbnail.png`, and `get_thumbnail` reads that same
path from the lamella cards. Saving moved off the GUI thread in FIB-563, so the two
genuinely overlap, and the existence check does not help: the file is there, just
incomplete.

Two changes, and both are needed. Writing through a temporary file and `os.replace`
means a reader sees the old thumbnail or the new one and never a partial. The read is
also guarded, because torn files written before the fix are still on disk, and this runs
from a Qt slot, where an escaping exception aborts the process under PyQt5 rather than
raising (FIB-329).
"""

import os
import threading

import numpy as np
import pytest

from fibsem.structures import FibsemImage


@pytest.fixture
def lamella(tmp_path):
    from fibsem.applications.autolamella.structures import Lamella

    lam = Lamella(path=str(tmp_path / "lamella"), number=1, petname="test-lamella")
    os.makedirs(lam.path, exist_ok=True)
    return lam


def image(value: int = 128) -> FibsemImage:
    return FibsemImage(data=np.full((64, 64), value, dtype=np.uint8))


class TestTheWriteIsAtomic:
    def test_the_final_file_never_exists_partly_written(self, lamella):
        """The property that fixes it: readers only ever see a complete file.

        Checked by watching the destination while a save runs — every version of it that
        exists at all must open cleanly.
        """
        from PIL import Image

        lamella.save_thumbnail(image(64))  # a complete one to start from
        destination = os.path.join(lamella.path, "thumbnail.png")

        failures = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                try:
                    if os.path.exists(destination):
                        Image.open(destination).convert("RGB")
                except OSError as e:
                    failures.append(str(e))

        watcher = threading.Thread(target=reader, daemon=True)
        watcher.start()
        try:
            for value in range(20):
                lamella.save_thumbnail(image(value * 10 % 256))
        finally:
            stop.set()
            watcher.join(timeout=5)

        assert failures == [], (
            f"a reader saw {len(failures)} torn thumbnails during 20 saves: "
            f"{failures[0]}"
        )

    def test_it_leaves_no_staging_files_behind(self, lamella):
        """They are hidden, so they would pile up in the lamella directory unnoticed."""
        for _ in range(5):
            lamella.save_thumbnail(image())

        leftovers = [n for n in os.listdir(lamella.path) if n.startswith(".thumbnail-")]

        assert leftovers == [], f"staging files left behind: {leftovers}"

    def test_a_failed_write_cleans_up_and_still_raises(self, lamella, monkeypatch):
        """A save that fails must not leave the staging file, nor swallow the failure."""
        from PIL import Image as PILImage

        def exploding(self, *args, **kwargs):
            raise RuntimeError("disk went away")

        monkeypatch.setattr(PILImage.Image, "save", exploding)

        with pytest.raises(RuntimeError):
            lamella.save_thumbnail(image())

        leftovers = [n for n in os.listdir(lamella.path) if n.startswith(".thumbnail-")]
        assert leftovers == [], f"a failed save left {leftovers} behind"

    def test_the_previous_thumbnail_survives_a_failed_write(self, lamella, monkeypatch):
        """`os.replace` never runs, so what was there stays there — readable throughout."""
        from PIL import Image as PILImage

        lamella.save_thumbnail(image(200))
        before = lamella.get_thumbnail().copy()

        def exploding(self, *args, **kwargs):
            raise RuntimeError("disk went away")

        monkeypatch.setattr(PILImage.Image, "save", exploding)
        with pytest.raises(RuntimeError):
            lamella.save_thumbnail(image(10))

        assert np.array_equal(lamella.get_thumbnail(), before)


class TestTheReadIsGuarded:
    def test_a_truncated_thumbnail_returns_the_placeholder(self, lamella):
        """Files torn before this fix are still on disk, and this runs in a Qt slot."""
        lamella.save_thumbnail(image())
        destination = os.path.join(lamella.path, "thumbnail.png")
        with open(destination, "r+b") as handle:  # keep the header, lose the rest
            handle.truncate(20)

        result = lamella.get_thumbnail()

        assert result is not None
        assert result.ndim == 3 and result.shape[2] == 3

    def test_an_empty_thumbnail_returns_the_placeholder(self, lamella):
        """Zero bytes: what the reported traceback actually hit."""
        open(os.path.join(lamella.path, "thumbnail.png"), "wb").close()

        result = lamella.get_thumbnail()

        assert result is not None
        assert result.ndim == 3 and result.shape[2] == 3

    def test_a_good_thumbnail_still_round_trips(self, lamella):
        """The guard must not swallow a working read."""
        lamella.save_thumbnail(image(77))

        result = lamella.get_thumbnail()

        assert result.shape == (64, 64, 3)
        assert int(result[0, 0, 0]) == 77
