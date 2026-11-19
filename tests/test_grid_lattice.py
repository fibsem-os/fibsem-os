"""The grid-bar lattice (`fibsem.analysis.lattice`) on rendered grids of known
pitch, angle and bar width, and on an image with no lattice in it."""

import math
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from scipy.ndimage import gaussian_filter

from fibsem.analysis import lattice
from fibsem.analysis.images import Planes
from fibsem.analysis.tools import ImageTools
from fibsem.server.app_routes import build_analysis_router


def _grid(
    shape=(1200, 1600),
    a=(120.0, 0.0),
    b=(0.0, 120.0),
    bar=40.0,
    origin=(37.0, 21.0),
    dark=True,
    seed=0,
):
    """A grid rendered with steps *a*, *b* (px), bars *bar* px wide centred on
    whole lattice coordinates from *origin*: film and bars, broken squares,
    blobs for cells and ice, blur and noise, as 8 bits."""
    rng = np.random.default_rng(seed)
    h, w = shape
    basis = np.column_stack([a, b])
    ys, xs = np.mgrid[0:h, 0:w]
    uv = np.linalg.inv(basis) @ np.vstack(
        [xs.ravel() - origin[0], ys.ravel() - origin[1]]
    )
    # The spacing between each direction's bars, perpendicular to them.
    inverse = np.linalg.inv(basis)
    spacing = [1 / np.hypot(*inverse[0]), 1 / np.hypot(*inverse[1])]
    centred = np.abs(uv - np.rint(uv))
    on_bar = (centred[0] < bar / spacing[0] / 2) | (centred[1] < bar / spacing[1] / 2)
    film, bars = (180.0, 80.0) if dark else (80.0, 180.0)
    image = np.where(on_bar, bars, film).reshape(h, w)
    # A fifth of the squares broken: their film gone, darker than the bars.
    cells = np.floor(uv).astype(int)
    broken = rng.random((1000, 1000)) < 0.2
    hole = broken[cells[0] % 1000, cells[1] % 1000] & ~on_bar
    image[hole.reshape(h, w)] = 50.0
    for _ in range(60):  # cells and ice
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(6, 25)
        image[(xs - cx) ** 2 + (ys - cy) ** 2 < r**2] = rng.choice([30.0, 230.0])
    image = gaussian_filter(image, 1.2) + rng.normal(0, 25, shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def _planes(data, pixel_size=1e-6):
    return Planes(data=data[None, None], names=["SEM"], pixel_size=pixel_size)


def _true_centre(i, j, a, b, origin):
    return np.array(origin) + np.column_stack([a, b]) @ np.array([i + 0.5, j + 0.5])


def _nearest_true(px, a, b, origin):
    """The distance from *px* to the nearest true square centre."""
    basis = np.column_stack([a, b])
    k = np.floor(np.linalg.inv(basis) @ (np.array(px) - origin))
    return min(
        np.hypot(*(np.array(px) - _true_centre(k[0] + di, k[1] + dj, a, b, origin)))
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
    )


@pytest.mark.parametrize("dark", [True, False], ids=["dark bars", "bright bars"])
def test_a_turned_grid_is_found_with_its_pitch_angle_and_bars(dark):
    turn = math.radians(10)
    a = (120 * math.cos(turn), 120 * math.sin(turn))
    b = (-120 * math.sin(turn), 120 * math.cos(turn))
    origin = (37.0, 21.0)
    answer = lattice.find_grid_squares(_planes(_grid(a=a, b=b, dark=dark)))

    found = answer["lattice"]
    assert found is not None, answer["reason"]
    assert found["a"]["pitch_px"] == pytest.approx(120, rel=0.01)
    assert found["b"]["pitch_px"] == pytest.approx(120, rel=0.01)
    assert found["angle_deg"] == pytest.approx(10, abs=0.5)
    assert found["angle_between_deg"] == pytest.approx(90, abs=0.5)
    assert found["a"]["bar_width_px"] == pytest.approx(40, abs=4)
    assert found["b"]["bar_width_px"] == pytest.approx(40, abs=4)
    assert found["a"]["bar_width_m"] == pytest.approx(40e-6, abs=4e-6)
    assert found["bars"] == ("darker" if dark else "brighter")
    assert answer["quality"]["score"] > 0.5
    assert len(answer["squares"]) == lattice.DEFAULT_LIMIT
    assert answer["squares_in_view"] > 100
    for square in answer["squares"]:
        assert _nearest_true(square["px"], a, b, origin) < 4


def test_a_point_is_checked_against_the_bars():
    a, b, origin = (120.0, 0.0), (0.0, 120.0), (37.0, 21.0)
    centre = _true_centre(5, 4, a, b, origin)
    on_bar = centre + [60.0, 0.0]  # the bar's centre line
    near_bar = centre + [30.0, 0.0]  # 10 px from its edge
    answer = lattice.find_grid_squares(
        _planes(_grid(a=a, b=b, origin=origin)),
        points=[centre.tolist(), on_bar.tolist(), near_bar.tolist()],
    )

    middle, bar, near = answer["points"]
    assert middle["square"] is not None and not middle["on_bar"]
    assert middle["bar_distance_px"] == pytest.approx(40, abs=4)
    assert middle["bar_distance_m"] == pytest.approx(40e-6, abs=4e-6)
    assert np.hypot(*middle["offset_px"]) < 4
    assert bar["on_bar"] and bar["square"] is None
    assert bar["bar_distance_px"] == pytest.approx(-20, abs=4)
    assert near["square"] == middle["square"]
    assert near["bar_distance_px"] == pytest.approx(10, abs=4)
    # The squares listed are the nearest the first point, it first.
    assert answer["squares"][0]["id"] == middle["square"]


def test_a_tilted_view_s_two_directions_are_fitted_apart():
    # Foreshortened along one axis and sheared, as a tilted FIB view is.
    a, b, origin = (190.0, 30.0), (-40.0, 115.0), (11.0, 52.0)
    answer = lattice.find_grid_squares(
        _planes(_grid(a=a, b=b, origin=origin, bar=30.0, dark=False, seed=3))
    )

    found = answer["lattice"]
    assert found is not None, answer["reason"]
    assert found["a"]["step_px"] == pytest.approx(list(a), abs=2.5)
    assert found["b"]["step_px"] == pytest.approx(list(b), abs=2.5)
    assert found["angle_between_deg"] == pytest.approx(
        math.degrees(math.acos(np.dot(a, b) / (np.hypot(*a) * np.hypot(*b)))), abs=1
    )
    assert found["a"]["bar_width_px"] == pytest.approx(30, abs=4)
    assert found["b"]["bar_width_px"] == pytest.approx(30, abs=4)
    for square in answer["squares"]:
        assert _nearest_true(square["px"], a, b, origin) < 5


def test_a_mosaic_s_seams_are_flagged_and_each_tile_placed():
    # Two rows of three tiles, each placed up to 20 px off the others.
    a, b = (120.0, 0.0), (0.0, 120.0)
    offsets = [[(0, 0), (18, 14), (-12, 10)], [(10, -15), (0, 12), (-16, -8)]]
    origins = [[(37.0 + dx, 21.0 + dy) for dx, dy in row] for row in offsets]
    mosaic = np.zeros((1200, 1600), np.uint8)
    for r, row in enumerate(origins):
        for c, origin in enumerate(row):
            tile = _grid(a=a, b=b, origin=origin, seed=3 * r + c)
            ys, xs = slice(600 * r, 600 * (r + 1)), slice(534 * c, 534 * (c + 1))
            mosaic[ys, xs] = tile[ys, xs]
    answer = lattice.find_grid_squares(_planes(mosaic), limit=60)

    assert answer["lattice"] is not None, answer["reason"]
    assert any("seams" in w for w in answer["warnings"])
    checked = 0
    for square in answer["squares"]:
        x, y = square["px"]
        if min(abs(x - 534), abs(x - 1068), abs(y - 600)) < 90:
            continue  # a square near a seam is fitted on both tiles
        origin = origins[int(y // 600)][min(int(x // 534), 2)]
        assert _nearest_true(square["px"], a, b, origin) < 6
        checked += 1
    assert checked > 20


def test_no_lattice_is_said_so_not_guessed():
    rng = np.random.default_rng(4)
    blobs = gaussian_filter(rng.normal(0, 1, (1000, 1400)), 15)
    image = np.clip(128 + blobs * 2000 + rng.normal(0, 20, blobs.shape), 0, 255)
    answer = lattice.find_grid_squares(
        _planes(image.astype(np.uint8)), points=[[100, 100]]
    )

    assert answer["lattice"] is None
    assert answer["reason"]
    assert answer["points"] == [{"px": [100.0, 100.0], "square": None}]
    assert "squares" not in answer


def test_the_route_answers_and_saves_the_overlay(tmp_path):
    a, b, origin = (120.0, 0.0), (0.0, 120.0), (37.0, 21.0)
    (tmp_path / "grid").mkdir()
    Image.fromarray(_grid(a=a, b=b, origin=origin)).save(tmp_path / "grid" / "o.png")
    app = FastAPI()
    app.include_router(build_analysis_router(ImageTools(lambda: str(tmp_path))))
    client = TestClient(app)

    answer = client.post(
        "/experiment/analysis/grid_squares",
        json={"path": "grid/o.png", "points": [[400, 300]], "limit": 3},
    )
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body["measurement"] == "grid"
    assert Path(body["path"]).parts[-2:] == ("drawn", "o-grid.png")
    assert (tmp_path / "grid" / "drawn" / "o-grid.png").is_file()
    assert body["plot"]["image_b64_jpeg"]
    result = body["result"]
    assert result["lattice"]["a"]["pitch_m"] is None  # a picture has no scale
    assert len(result["squares"]) == 3
    # A picture carries no stage geometry: no stage position, still an answer.
    assert result["squares"][0]["stage_position"] is None
    assert result["points"][0]["stage_position"] is None

    refused = client.post(
        "/experiment/analysis/grid_squares", json={"path": "grid/o.png", "limit": 999}
    )
    assert refused.status_code in (400, 422)

    many = client.post(
        "/experiment/analysis/grid_squares",
        json={"path": "grid/o.png", "limit": 150, "box": [0, 0, 800, 600]},
    )
    assert many.status_code == 200, many.text
    assert len(many.json()["result"]["squares"]) == 130  # every square in view


def test_a_rejected_fit_still_answers_its_best_candidate(tmp_path):
    # Bars too faint to accept, but there: the candidate still finds them.
    rng = np.random.default_rng(0)
    ys, xs = np.mgrid[0:1000, 0:1400]
    on_bar = ((xs % 120) < 40) | ((ys % 120) < 40)
    image = 128 + np.where(on_bar, -4, 4) + rng.normal(0, 60, xs.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)
    planes = _planes(image)
    fit = lattice.fit_lattice(planes)
    answer = lattice.grid_answer(fit, planes.pixel_size, points=[[80, 80]])

    assert answer["lattice"] is None
    assert answer["reason"].startswith("No lattice accepted")
    assert answer["points"] == [{"px": [80.0, 80.0], "square": None}]
    candidate = answer["candidate"]
    assert candidate["a"]["pitch_px"] == pytest.approx(120, rel=0.02)
    assert candidate["b"]["pitch_px"] == pytest.approx(120, rel=0.02)
    assert max(candidate["quality"]["explained"]) < lattice.MIN_EXPLAINED
    assert candidate["points"][0]["bar_distance_px"] is not None
    assert candidate["squares"]
    assert lattice.grid_figure(planes, fit, answer).ndim == 3


def test_a_flat_image_has_no_candidate():
    planes = _planes(np.full((400, 600), 90, np.uint8))
    fit = lattice.fit_lattice(planes)
    answer = lattice.grid_answer(fit, planes.pixel_size)

    assert answer["lattice"] is None and answer["candidate"] is None
    assert "flat" in answer["reason"]
    assert lattice.grid_figure(planes, fit, answer).ndim == 3


def test_a_box_fits_one_part_and_extends_the_lattice_beyond():
    # A grid on the left, no grid on the right: fitted in a box on the left.
    a, b, origin = (120.0, 0.0), (0.0, 120.0), (37.0, 21.0)
    rng = np.random.default_rng(5)
    image = _grid(a=a, b=b, origin=origin)
    image[:, 700:] = np.clip(128 + rng.normal(0, 30, (1200, 900)), 0, 255)
    answer = lattice.find_grid_squares(
        _planes(image),
        points=[[1300.0, 700.0]],
        limit=200,
        box=[0, 0, 700, 1200],
    )

    assert answer["lattice"] is not None, answer["reason"]
    assert answer["box"] == [0, 0, 700, 1200]
    assert answer["lattice"]["a"]["pitch_px"] == pytest.approx(120, rel=0.01)
    inside = [s for s in answer["squares"] if s["px"][0] < 640]
    outside = [s for s in answer["squares"] if s["px"][0] > 760]
    assert inside and all(not s["extrapolated"] for s in inside)
    assert outside and all(s["extrapolated"] and s["match"] is None for s in outside)
    for square in inside + outside:
        assert _nearest_true(square["px"], a, b, origin) < 4
    assert answer["points"][0]["extrapolated"]
    assert answer["points"][0]["bar_distance_px"] is not None


def test_a_pitch_hint_is_reported_against_the_fit():
    a, b, origin = (120.0, 0.0), (0.0, 120.0), (37.0, 21.0)
    planes = _planes(_grid(a=a, b=b, origin=origin))  # 1 um pixels

    agreed = lattice.find_grid_squares(planes, pitch_um=125)
    assert agreed["hint"]["agrees"] and not agreed["hint"]["changed_choice"]
    assert agreed["hint"]["longer_step_um"] == pytest.approx(120, rel=0.01)

    wrong = lattice.find_grid_squares(planes, pitch_um=300)
    assert not wrong["hint"]["agrees"]
    assert "disagrees" in wrong["hint"]["note"]
    # A hint prefers; it does not force a lattice the image doesn't have.
    assert wrong["lattice"]["a"]["pitch_px"] == pytest.approx(120, rel=0.01)

    unscaled = Planes(data=planes.data, names=["SEM"], pixel_size=None)
    assert (
        "no pixel size"
        in lattice.find_grid_squares(unscaled, pitch_um=125)["hint"]["note"]
    )
