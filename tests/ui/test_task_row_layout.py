"""Task rows wrap rather than running off the edge of the review panel.

Needs Qt, so it skips in CI like everything else under tests/ui — the layout
behaviour it guards can't be observed without a real layout.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("napari")

from PyQt5.QtWidgets import QApplication, QGridLayout

from fibsem.applications.autolamella.ui.lamella_task_image_widget import (
    _IMAGES_PER_LINE,
    LamellaTaskImageWidget,
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _lines_and_columns(widget, count):
    row = widget._build_task_row_with_placeholders(
        "Task", [f"/tmp/image-{i}.tif" for i in range(count)]
    )
    grid = row.findChildren(QGridLayout)[0]
    occupied = [
        grid.getItemPosition(i)[:2] for i in range(grid.count())
    ]  # (row, column)
    return occupied


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5])
def test_no_line_holds_more_than_the_wrap_width(app, count):
    """The panel scrolls vertically only, so a tile past the width is unreachable.

    Before fluorescence images a task had at most the SEM/FIB pair and always fitted;
    a task can now also carry a z-stack, which is what made this reachable.
    """
    widget = LamellaTaskImageWidget()
    occupied = _lines_and_columns(widget, count)

    per_line = {}
    for row, column in occupied:
        per_line.setdefault(row, []).append(column)

    assert len(occupied) == count, "every image must get a tile"
    assert all(len(cols) <= _IMAGES_PER_LINE for cols in per_line.values())
    assert max(c for _, c in occupied) < _IMAGES_PER_LINE


def test_the_common_pair_still_fits_on_one_line(app):
    """The SEM/FIB pair is the overwhelmingly common case and must not start
    wrapping — that would be a visible regression for every existing experiment."""
    assert len({row for row, _ in _lines_and_columns(LamellaTaskImageWidget(), 2)}) == 1


def test_a_task_with_three_images_wraps_onto_two_lines(app):
    """SEM + FIB + fluorescence: the case that motivated wrapping."""
    assert len({row for row, _ in _lines_and_columns(LamellaTaskImageWidget(), 3)}) == 2


# tile sizing


def _render(tmp_path, resolution):
    """Save an image at the given resolution and return its rendered tile size.

    Built with full metadata rather than via generate_blank_image: that one carries
    no microscope_state, which from_dict requires, so its metadata does not survive
    the tif round-trip and the loader has no pixel size to read.
    """
    import numpy as np

    from fibsem.structures import (
        FibsemImage,
        FibsemImageMetadata,
        ImageSettings,
        MicroscopeState,
        Point,
    )
    from fibsem.applications.autolamella.ui.lamella_task_image_widget import _load_and_resize

    width, height = resolution
    metadata = FibsemImageMetadata(
        image_settings=ImageSettings(resolution=resolution),
        pixel_size=Point(1e-7, 1e-7),
        microscope_state=MicroscopeState(),
    )
    image = FibsemImage(data=np.zeros((height, width), dtype=np.uint8), metadata=metadata)
    path = image.save(str(tmp_path / f"ref_{width}x{height}.tif"))

    arr, _ = _load_and_resize(path)
    h, w = arr.shape[:2]
    return w, h


@pytest.mark.parametrize("resolution", [(1536, 1024), (1024, 1024), (2048, 2048), (512, 1024)])
def test_every_tile_fits_the_placeholder_box(app, tmp_path, resolution):
    """A tile must never exceed the placeholder it replaces.

    Beam images are 3:2 and are width-limited, so filling the width was fine. A
    fluorescence stack is square: scaled to the full width it is half again as tall
    as its neighbours and taller than its own placeholder, so the row jumps when it
    loads and everything below shifts.
    """
    from fibsem.applications.autolamella.ui.lamella_task_image_widget import (
        _PLACEHOLDER_HEIGHT,
        _TARGET_WIDTH,
    )

    w, h = _render(tmp_path, resolution)

    assert w <= _TARGET_WIDTH
    assert h <= _PLACEHOLDER_HEIGHT


def test_a_three_by_two_image_still_fills_the_width(app, tmp_path):
    """The common beam-image case must be untouched — it was already width-limited,
    and shrinking it would be a visible regression on every existing experiment."""
    from fibsem.applications.autolamella.ui.lamella_task_image_widget import _TARGET_WIDTH

    assert _render(tmp_path, (1536, 1024))[0] == _TARGET_WIDTH


def test_a_square_image_is_limited_by_height_not_width(app, tmp_path):
    """A square stack fits the box by its height, leaving it narrower than a beam
    image rather than taller."""
    from fibsem.applications.autolamella.ui.lamella_task_image_widget import (
        _PLACEHOLDER_HEIGHT,
        _TARGET_WIDTH,
    )

    w, h = _render(tmp_path, (2048, 2048))

    assert h == _PLACEHOLDER_HEIGHT
    assert w < _TARGET_WIDTH
