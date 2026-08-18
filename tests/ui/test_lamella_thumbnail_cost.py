"""A lamella card does not re-decode a full-resolution frame to draw a 66px picture.

`thumbnail.png` was the acquired frame written out verbatim -- 1536x1024 RGB, 823 KB on
real data -- and `LamellaCardWidget.refresh` re-opened, re-decoded and re-scaled it every
time it ran, with no cache anywhere. 26 ms per card. `_rebuild_lamella_list` clears and
re-adds the whole strip whenever a lamella is added or removed, so a 100-lamella
experiment paid 2.8 s of frozen GUI to add one position (FIB-681).

Two halves, and both are needed. Writing at display size fixes what lands on disk from
now on; caching the scaled image fixes the experiments already saved with a full frame
under that name, which no amount of resizing-on-write can reach.

The properties worth pinning are not the timings -- they are the ones whose failure is
silent:

* **The write only ever shrinks.** A caller handing over an already-small image gets it
  back unchanged; upscaling would invent detail and break `get_thumbnail`'s round trip.
* **A rewritten thumbnail is picked up.** The cache is keyed by the file's stamp, so a
  workflow replacing a thumbnail mid-run must invalidate it. Serving the stale one is
  invisible until someone notices a card showing the wrong picture.
* **The cache is bounded.** Every thumbnail a workflow rewrites lands under a new key, so
  an unbounded dict grows for the length of a run.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.applications.autolamella.structures import Lamella
from fibsem.applications.autolamella.ui import lamella_card_widget as lcw
from fibsem.applications.autolamella.ui.lamella_card_widget import LamellaCardWidget
from fibsem.config import MODE_COMPACT, MODE_COZY
from fibsem.structures import FibsemImage

_app = QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(autouse=True)
def _clear_cache():
    """The cache is module-level, so tests would otherwise see each other's entries."""
    lcw._THUMBNAIL_CACHE.clear()
    yield
    lcw._THUMBNAIL_CACHE.clear()


@pytest.fixture
def lamella(tmp_path):
    lam = Lamella(path=str(tmp_path / "lamella"), number=1, petname="test-lamella")
    os.makedirs(lam.path, exist_ok=True)
    return lam


def frame(w: int, h: int, value: int = 128) -> FibsemImage:
    return FibsemImage(data=np.full((h, w), value, dtype=np.uint8))


def stored_size(lamella: Lamella):
    from PIL import Image

    with Image.open(os.path.join(lamella.path, "thumbnail.png")) as handle:
        return handle.size


class TestTheWriteIsDisplaySized:
    def test_an_acquired_frame_is_stored_at_display_size(self, lamella):
        """The real case: a 1536x1024 frame must not land on disk at 1536x1024."""
        lamella.save_thumbnail(frame(1536, 1024))

        assert stored_size(lamella) == (512, 341)

    @pytest.mark.parametrize(
        "source",
        [
            (1536, 1024),   # the acquired frame
            (1024, 1536),   # portrait
            (3072, 3072),   # a stitched square overview
            (1024, 3072),   # a tall stitched strip
            (409, 384),     # an odd small frame, below the bound
            (1001, 667),    # a ratio no integer pair hits cleanly
        ],
        ids=lambda s: f"{s[0]}x{s[1]}",
    )
    def test_the_stored_aspect_is_the_closest_integers_allow(self, lamella, source):
        """The card scales to fill and crops the overflow, so a stored picture whose
        ratio has drifted is cropped somewhere the frame was not.

        Exactness is not available: 1536x1024 at a 512 bound wants 341.33 px of
        height, and a PNG has whole pixels. What must hold is that the remaining
        error is *rounding* and not squashing -- no neighbouring integer height is a
        better fit for the source ratio than the one chosen.
        """
        lamella.save_thumbnail(frame(*source))

        width, height = stored_size(lamella)
        wanted = source[0] / source[1]

        # The long edge is pinned -- either to the bound, or to the source when that is
        # already smaller -- so the short edge is the one carrying the rounding, and the
        # one whose neighbours are worth comparing. Varying the pinned edge instead just
        # proposes sizes the bound forbids: a 1024x3072 strip stored 171x512 is a better
        # fit at 171x513, which is over the bound and not on offer.
        candidates = (
            [(width, h) for h in (height - 1, height, height + 1) if h > 0]
            if height <= width
            else [(w, height) for w in (width - 1, width, width + 1) if w > 0]
        )
        closest = min(candidates, key=lambda wh: abs(wanted - wh[0] / wh[1]))
        assert closest == (width, height), (
            f"{source[0]}x{source[1]} stored as {width}x{height} "
            f"(ratio {width / height:.4f}); {closest[0]}x{closest[1]} is closer to {wanted:.4f}"
        )
        # and a wholesale squash, which rounding can never produce
        assert abs(width / height - wanted) / wanted < 0.005

    def test_a_small_frame_is_never_upscaled(self, lamella):
        """`get_thumbnail`'s round trip returns what was put in, at its own size."""
        lamella.save_thumbnail(frame(64, 64, value=77))

        assert stored_size(lamella) == (64, 64)
        assert lamella.get_thumbnail().shape == (64, 64, 3)

    def test_the_file_is_dramatically_smaller(self, lamella):
        """823 KB per lamella on disk was the other half of the cost."""
        # Noise rather than a flat fill: a solid colour compresses to almost nothing at
        # either size, which would make this pass without the resize.
        rng = np.random.default_rng(0)
        data = rng.integers(0, 255, size=(1024, 1536), dtype=np.uint8)
        lamella.save_thumbnail(FibsemImage(data=data))

        size_kb = os.path.getsize(os.path.join(lamella.path, "thumbnail.png")) / 1024

        assert size_kb < 300, f"thumbnail is still {size_kb:.0f} KB"


class TestTheScaledImageIsCached:
    def test_a_second_card_does_not_reread_the_file(self, lamella, monkeypatch):
        """The rebuild case: every card in the strip shares one decode."""
        lamella.save_thumbnail(frame(1536, 1024))

        reads = []
        original = Lamella.get_thumbnail
        monkeypatch.setattr(
            Lamella,
            "get_thumbnail",
            lambda self: (reads.append(self.path), original(self))[1],
        )

        first = LamellaCardWidget(lamella, mode=MODE_COZY)
        second = LamellaCardWidget(lamella, mode=MODE_COZY)
        second.refresh()

        assert len(reads) == 1, f"decoded {len(reads)} times for one thumbnail"
        assert not first._thumb_label.pixmap().isNull()
        assert not second._thumb_label.pixmap().isNull()

    def test_a_rewritten_thumbnail_is_picked_up(self, lamella):
        """A workflow replaces this mid-run; a stale card is a silent wrong picture."""
        lamella.save_thumbnail(frame(1536, 1024, value=10))
        card = LamellaCardWidget(lamella, mode=MODE_COZY)
        before = card._thumb_label.pixmap().toImage().pixelColor(2, 2).red()

        lamella.save_thumbnail(frame(1536, 1024, value=240))
        card.refresh()
        after = card._thumb_label.pixmap().toImage().pixelColor(2, 2).red()

        assert before != after, "the card kept drawing the pre-rewrite thumbnail"
        assert after > before

    def test_each_label_size_is_cached_separately(self, lamella):
        """Cozy and standard scale the same file to different sizes."""
        lamella.save_thumbnail(frame(1536, 1024))

        card = LamellaCardWidget(lamella, mode=MODE_COZY)
        cozy_width = card._thumb_label.pixmap().width()
        card.set_mode("standard")
        standard_width = card._thumb_label.pixmap().width()

        assert cozy_width != standard_width
        assert len(lcw._THUMBNAIL_CACHE) == 2

    def test_a_compact_card_reads_nothing(self, lamella, monkeypatch):
        """No thumbnail in that arrangement, so decoding one is pure waste."""
        lamella.save_thumbnail(frame(1536, 1024))
        monkeypatch.setattr(
            Lamella, "get_thumbnail", lambda self: pytest.fail("decoded for a hidden label")
        )

        LamellaCardWidget(lamella, mode=MODE_COMPACT)

    def test_a_missing_thumbnail_does_not_poison_the_cache(self, lamella):
        """The placeholder is cached against "no file"; saving one must supersede it."""
        card = LamellaCardWidget(lamella, mode=MODE_COZY)  # nothing on disk yet
        placeholder = card._thumb_label.pixmap().toImage().pixelColor(2, 2).red()

        lamella.save_thumbnail(frame(1536, 1024, value=240))
        card.refresh()

        assert card._thumb_label.pixmap().toImage().pixelColor(2, 2).red() != placeholder


class TestTheCacheIsSafe:
    # A cached QImage outliving the ndarray it was built from was the other risk here,
    # since Qt documents `QImage(buffer, ...)` as not copying. There is no test for it
    # because PyQt5 makes it unreachable: sip copies the Python buffer, so the image
    # always owns its pixels -- checked by pointer, by mutating the source in place,
    # and via `sip.voidptr`, which copies too. Any test would pass no matter what this
    # module did, which is worse than none. See `_arr_to_qimage`, and revisit if this
    # ever moves to PySide.

    def test_the_cache_is_bounded(self, lamella, monkeypatch):
        """Every rewrite lands under a new key, so a run would grow this forever."""
        monkeypatch.setattr(lcw, "_THUMBNAIL_CACHE_MAX", 4)

        for value in range(12):
            lamella.save_thumbnail(frame(64, 64, value=value * 20))
            lcw._thumbnail_image(lamella, 66, 44)

        assert len(lcw._THUMBNAIL_CACHE) <= 4

    def test_the_bound_evicts_the_least_recently_used(self, lamella, tmp_path):
        """Evicting the entry being drawn every frame would defeat the cache."""
        lamella.save_thumbnail(frame(64, 64))
        others = []
        for i in range(3):
            other = Lamella(path=str(tmp_path / f"other-{i}"), number=i + 2, petname=f"o{i}")
            os.makedirs(other.path, exist_ok=True)
            other.save_thumbnail(frame(64, 64, value=i * 30))
            others.append(other)

        lcw._THUMBNAIL_CACHE.clear()
        lcw._THUMBNAIL_CACHE_MAX_original = lcw._THUMBNAIL_CACHE_MAX
        try:
            lcw._THUMBNAIL_CACHE_MAX = 3
            lcw._thumbnail_image(lamella, 66, 44)  # the one we keep touching
            for other in others:
                lcw._thumbnail_image(lamella, 66, 44)  # ...touched again
                lcw._thumbnail_image(other, 66, 44)
        finally:
            lcw._THUMBNAIL_CACHE_MAX = lcw._THUMBNAIL_CACHE_MAX_original

        kept = [key[0] for key in lcw._THUMBNAIL_CACHE]
        assert os.path.join(lamella.path, "thumbnail.png") in kept
