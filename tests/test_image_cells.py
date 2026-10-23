"""Images cut into named cells (fibsem.imaging.cells)."""

import json

import numpy as np
import pytest

from fibsem.imaging.cells import (
    GRID,
    TILES,
    cells_picture,
    column_name,
    from_crop,
    image_cells,
    named,
    tile_layout,
)


def test_columns_are_lettered_like_a_spreadsheet():
    assert [column_name(i) for i in (0, 1, 25, 26, 27)] == ["A", "B", "Z", "AA", "AB"]


def _covers(cells, width, height):
    covered = np.zeros((height, width), int)
    for cell in cells:
        covered[cell.y : cell.y + cell.height, cell.x : cell.x + cell.width] += 1
    return (covered == 1).all()


def test_a_plain_image_is_a_grid_of_cells_that_fit_a_preview():
    kind, cells = image_cells(3072, 2048)
    assert kind == GRID
    assert [c.name for c in cells[:5]] == ["A1", "B1", "C1", "D1", "A2"]
    assert cells[-1].name == "D3"
    assert _covers(cells, 3072, 2048)
    # A small image still has cells to name.
    assert len(image_cells(128, 128)[1]) == 4


def test_a_tiled_overview_s_cells_are_its_tiles_with_the_overlap_split():
    # Three 1000 px tiles pasted 900 px apart.
    kind, cells = image_cells(2800, 1000, (3, 1, 0.1))
    assert kind == TILES
    assert [c.box for c in cells] == [
        [0, 0, 950, 1000],
        [950, 0, 900, 1000],
        [1850, 0, 950, 1000],
    ]


def test_the_layout_is_read_from_the_run_s_record_beside_the_mosaic(tmp_path):
    def record(stem, settings, rows, cols):
        (tmp_path / stem).mkdir()
        tiles = [{"row": r, "col": c} for r in range(rows) for c in range(cols)]
        (tmp_path / stem / "overview-metadata.json").write_text(
            json.dumps({"settings": settings, "tiles": tiles})
        )

    record("beam", {"overlap": 0.2}, 2, 3)
    record("fm", {"overview": {"overlap": 0.1}}, 4, 4)
    record("one", {"overlap": 0.1}, 1, 1)
    assert tile_layout(str(tmp_path / "beam.tif")) == (3, 2, 0.2)
    assert tile_layout(str(tmp_path / "fm.ome.tiff")) == (4, 4, 0.1)
    assert tile_layout(str(tmp_path / "one.tif")) is None  # one tile: a grid
    assert tile_layout(str(tmp_path / "plain.tif")) is None


def test_cells_are_found_by_name_and_refused_with_the_range():
    _, cells = image_cells(3072, 2048)
    assert [c.name for c in named(cells, ["b2", "A1"])] == ["B2", "A1"]
    with pytest.raises(ValueError, match="A1 to D3"):
        named(cells, ["E1"])


def test_a_point_on_a_crop_s_picture_maps_back_to_the_image(tmp_path):
    from PIL import Image

    path = str(tmp_path / "wide.png")
    Image.new("RGB", (3072, 2048)).save(path)
    # B2 is 768 px wide: its picture is the cell at native resolution.
    assert from_crop(path, "B2") == (1.0, 768, 683, [768, 683, 768, 682])
    # A crop wider than its picture is narrowed to 768.
    scale, x0, y0, _ = from_crop(path, [100, 50, 1536, 400])
    assert (scale, x0, y0) == (2.0, 100, 50)
    with pytest.raises(ValueError):
        from_crop(path, "Q9")
    with pytest.raises(ValueError):
        from_crop(path, [1, 2, 3])


def test_the_cells_are_drawn_on_the_preview_at_its_size():
    data = np.full((256, 384), 100, np.uint8)
    _, cells = image_cells(1536, 1024)
    drawn = cells_picture(data, cells, 0.25)
    assert drawn.shape == (256, 384, 3)
    # The edge between columns A and B, at 768 source px.
    assert drawn[100:150, 191:194].max() > 200
