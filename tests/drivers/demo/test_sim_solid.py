"""The simulated sample: objects that lie under one another, seen along each
beam."""

import os

import numpy as np
import pytest
from scipy import ndimage as ndi

import fibsem.config as cfg
from fibsem import utils
from fibsem.drivers.demo.sim import solid as sim_solid
from fibsem.drivers.demo.sim.scene import SampleScene
from fibsem.structures.settings import (
    BeamType,
    FibsemRectangleSettings,
    ImageSettings,
)

TFS_SHUTTLE_CONFIG = os.path.join(
    cfg.SHIPPED_CONFIG_PATH, "microscope-configuration.yaml"
)
MILLING_TILT = np.deg2rad(12.0)
PLAIN = {
    "cell_type": "none",
    "contamination_density": 0.0,
    "ice_density": 0.0,
    "frost_density": 0.0,
    "sphere_density": 0.0,
    "crack_density": 0.0,
    "rip_fraction": 0.0,
}


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(
        manufacturer="Demo", config_path=TFS_SHUTTLE_CONFIG
    )
    microscope.system.sim["sample"] = {"enabled": True, "coincidence_offset": 0.0}
    microscope._connection.setup_sample_scene()
    pose = microscope.get_stage_position()
    pose.t = MILLING_TILT
    microscope.move_stage_absolute(pose)
    yield microscope
    microscope.disconnect()


def _use(microscope, **kwargs) -> SampleScene:
    kwargs = {"coincidence_offset": 0.0, **kwargs}
    scene = SampleScene(noise_sigma=0.0, noise_fraction=0.0, **kwargs)
    scene.anchor(microscope.get_stage_position())
    microscope._connection.sample_scene = scene
    return scene


def _image(microscope, beam_type, hfw=150e-6) -> np.ndarray:
    settings = ImageSettings(
        resolution=(512, 342), hfw=hfw, dwell_time=1e-9, save=False, beam_type=beam_type
    )
    return microscope.acquire_image(image_settings=settings).data.astype(np.float32)


def _mill(microscope, *rectangles) -> None:
    for width, height, y in rectangles:
        microscope.draw_pattern(
            FibsemRectangleSettings(
                width=width, height=height, depth=10e-6, centre_x=0.0, centre_y=y
            )
        )
    microscope.run_milling()


def _stack(scene, x, y) -> dict:
    """The (bottom, top) of each piece at one world point (um)."""
    xa = x + np.arange(-2, 3) * 0.2
    ya = y + np.arange(-2, 3) * 0.2
    col = sim_solid.columns(scene, xa, ya, (0.2, 0.2))
    return {
        k: (float(np.broadcast_to(col[k][0], (5, 5))[2, 2]), float(col[k][1][2, 2]))
        for k in sim_solid.SLOTS
    }


def test_a_cell_lies_under_the_ice_with_its_nucleus_inside():
    """What the height field cannot hold: at a cell's nucleus the stack is
    film, cell, nucleus inside the cell, and ice over the top."""
    scene = SampleScene(rip_fraction=0.0)
    nucleus = next(
        f for f in scene.features if f.kind == "cell" and f.part == "nucleus"
    )
    stack = _stack(scene, nucleus.x * 1e6, nucleus.y * 1e6)
    cell, ice = stack["cell"][1], stack["ice"][1]
    lo, hi = stack["nucleus"]
    film = stack["film"][1]
    assert film == sim_solid.FILM_TOP
    assert film < lo < hi < cell < ice


def test_a_sphere_rests_on_the_ice_with_room_under_its_sides():
    scene = SampleScene(cell_type="none", sphere_density=2.0, rip_fraction=0.0)
    ball = max((f for f in scene.features if f.part == "sphere"), key=lambda f: f.sigma)
    r = ball.sigma * 1e6
    centre = _stack(scene, ball.x * 1e6, ball.y * 1e6)
    side = _stack(scene, ball.x * 1e6 + 0.8 * r, ball.y * 1e6)
    assert centre["particle"][1] > centre["ice"][1]
    # out toward its side the ball has lifted off the ice: a gap under it
    assert side["particle"][0] > side["ice"][1]


def _shift(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The translation (rows, columns) taking a onto b, by phase correlation."""
    fa, fb = np.fft.fft2(a - a.mean()), np.fft.fft2(b - b.mean())
    r = np.fft.ifft2(fa.conj() * fb / (np.abs(fa * fb) + 1e-9)).real
    peak = np.array(np.unravel_index(np.argmax(r), r.shape), dtype=float)
    size = np.array(r.shape)
    return np.where(peak > size / 2, peak - size, peak)


@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
def test_the_grid_lands_where_the_projection_puts_it(
    microscope, beam_type, monkeypatch
):
    """Positions come from the stage projection: with the ice left flat over
    the bars (it stands up over them otherwise, which a glancing beam sees as
    a shift), the bars are on the pixels the projection maps them to."""
    from fibsem.geometry.projection import BeamStageProjection

    monkeypatch.setattr(sim_solid, "BAR_RISE", 0.0)
    scene = _use(microscope, **PLAIN)
    image = _image(microscope, beam_type, hfw=400e-6)
    projection = BeamStageProjection.from_microscope(microscope, beam_type=beam_type)
    position = microscope._connection.scene_stage_position()
    height, width = image.shape
    pixel = 400e-6 / width

    def world(col, row):
        dx, dy = (col - width / 2) * pixel, (row - height / 2) * pixel
        return np.array(scene.view_to_world(beam_type, position, projection, dx, dy))

    # the pixel-to-world map is affine: three points give it
    o, cx, cy = world(0, 0), world(1, 0), world(0, 1)
    cols, rows = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    wx = o[0] + cols * (cx - o)[0] + rows * (cy - o)[0]
    wy = o[1] + cols * (cx - o)[1] + rows * (cy - o)[1]
    bars = scene.film_masks(wx, wy)[0].astype(np.float32)
    assert 0.05 < bars.mean() < 0.95
    # the offset of the bars that best matches the image
    r = 6
    match = {
        (dy, dx): abs(
            np.corrcoef(
                image[r:-r, r:-r].ravel(),
                np.roll(bars, (dy, dx), (0, 1))[r:-r, r:-r].ravel(),
            )[0, 1]
        )
        for dy in range(-r, r + 1)
        for dx in range(-r, r + 1)
    }
    assert max(np.abs(max(match, key=match.get))) <= 1


def test_a_trench_is_cut_along_the_milling_beam(microscope):
    """A rectangle milled in the FIB view is a square-edged dark rectangle in
    that view, where it was drawn."""
    _use(microscope, **PLAIN)
    _mill(microscope, (30e-6, 6e-6, 0.0))
    image = _image(microscope, BeamType.ION, hfw=60e-6)
    pixel = 60e-6 / 512
    labels, n = ndi.label(image < 40)
    assert n > 0
    sizes = ndi.sum(np.ones_like(labels), labels, range(1, n + 1))
    rows, cols = np.nonzero(labels == 1 + int(np.argmax(sizes)))
    assert (np.ptp(cols) + 1) * pixel == pytest.approx(30e-6, rel=0.08)
    assert (np.ptp(rows) + 1) * pixel == pytest.approx(6e-6, rel=0.15)
    assert len(rows) / ((np.ptp(rows) + 1) * (np.ptp(cols) + 1)) > 0.9
    centre = ((cols.min() + cols.max()) / 2 - 256, (rows.min() + rows.max()) / 2 - 171)
    assert abs(centre[0]) <= 3 and abs(centre[1]) <= 3


@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
def test_a_shallow_trench_shows_its_floor(microscope, beam_type):
    """A pattern that stops in the ice leaves a floor both beams see, darker
    the deeper it is; only a cut through the sample is black."""
    grey = []
    for depth in (0.3e-6, 1.0e-6, 10e-6):
        _use(microscope, **PLAIN)
        microscope.draw_pattern(
            FibsemRectangleSettings(
                width=12e-6, height=6e-6, depth=depth, centre_x=0.0, centre_y=0.0
            )
        )
        microscope.run_milling()
        image = _image(microscope, beam_type, hfw=40e-6)
        dark = ndi.binary_erosion(image < 0.8 * np.median(image), iterations=8)
        assert dark.sum() > 500
        grey.append(float(np.median(image[dark])))
    shallow, deeper, through = grey
    assert shallow > deeper + 4 > through + 30, grey
    assert through < 40, grey


def test_a_lamella_between_two_trenches_stays(microscope):
    _use(microscope, **PLAIN)
    _mill(microscope, (20e-6, 6e-6, 3.5e-6), (20e-6, 6e-6, -3.5e-6))
    image = _image(microscope, BeamType.ION, hfw=40e-6)
    profile = image[:, 236:276].mean(axis=1)
    pixel = 40e-6 / 512
    band = profile[171 - int(0.3e-6 / pixel) : 171 + int(0.3e-6 / pixel) + 1]
    assert band.min() > 60, band
    assert profile[171 - int(3.5e-6 / pixel)] < 40
    assert profile[171 + int(3.5e-6 / pixel)] < 40


def test_a_face_polished_in_many_passes_is_as_open_as_one_cut_at_once(
    microscope, monkeypatch
):
    """Thin strips milled one after another make one trench: the face under
    the last is no darker than under a single trench of the same reach."""
    opening = []
    for cuts in (
        [(12e-6, 11.5e-6, 6.75e-6)],
        [(12e-6, 4e-6, 10.5e-6)]
        + [(12e-6, 2e-6, y * 1e-6) for y in (8.0, 6.5, 5.0, 3.5, 2.0)],
    ):
        _use(microscope, **PLAIN)
        for cut in cuts:
            _mill(microscope, cut)
        _, hit = _trace(microscope, monkeypatch, BeamType.ELECTRON, 40e-6)
        face = hit["kind"][:, 236:276] == 2
        assert face.sum() > 500
        opening.append(float(np.percentile(hit["opening"][:, 236:276][face], 10)))
    print("OPEN", opening)
    assert opening[0] > 0.3
    assert opening[1] > 0.8 * opening[0], opening


def test_a_cut_face_shows_the_cell_and_its_nucleus(microscope, monkeypatch):
    """A trench through a cell leaves a face made of ice, the cell and the
    nucleus inside it, each its own tone."""
    scene = _use(microscope, rip_fraction=0.0, sphere_density=0.0)
    nucleus = min(
        (f for f in scene.features if f.kind == "cell" and f.part == "nucleus"),
        key=lambda f: np.hypot(f.x, f.y),
    )
    for f in scene.features:
        f.x -= nucleus.x
        f.y -= nucleus.y
    _mill(microscope, (14e-6, 8e-6, 5e-6))
    hits = []
    trace = sim_solid.trace
    monkeypatch.setattr(
        sim_solid, "trace", lambda *a: hits.append(trace(*a)) or hits[-1]
    )
    _image(microscope, BeamType.ELECTRON, hfw=40e-6)
    face = hits[-1]["material"][hits[-1]["kind"] == 2]
    for material in (sim_solid.ICE, sim_solid.CELL, sim_solid.NUCLEUS):
        assert (face == material).sum() > 20, material
    tones = {sim_solid.INSIDE[m][0] for m in (1, 2, 3)}
    assert len(tones) == 3
    # what is in the cell is faint against it: a cryo face shows little
    cell = np.array(sim_solid.INSIDE[sim_solid.CELL])
    for part in (sim_solid.NUCLEUS, sim_solid.DROPLET, sim_solid.VESICLE):
        off = np.abs(np.array(sim_solid.INSIDE[part]) / cell - 1)
        assert (off > 0.03).all() and (off < 0.3).all(), part


def _trace(microscope, monkeypatch, beam_type, hfw):
    """An image, and what each of its pixels met."""
    hits = []
    trace = sim_solid.trace
    monkeypatch.setattr(
        sim_solid, "trace", lambda *a: hits.append(trace(*a)) or hits[-1]
    )
    image = _image(microscope, beam_type, hfw=hfw)
    monkeypatch.setattr(sim_solid, "trace", trace)
    return image, hits[-1]


def test_rpe_cells_stand_apart_as_domes():
    """The rpe cells: big domes a few micrometres tall, clear of one another,
    each with a nucleus."""
    scene = SampleScene(cell_type="rpe", rip_fraction=0.0)
    bodies = [f for f in scene.features if f.kind == "cell" and f.part == "body"]
    nuclei = [f for f in scene.features if f.part == "nucleus"]
    assert len(bodies) == len(nuclei) > 50
    xy = np.array([(f.x, f.y) for f in bodies])
    radius = np.array([f.sigma for f in bodies]) * sim_solid.RPE_RADIUS
    assert 11e-6 <= radius.min() and radius.max() <= 21e-6
    gap = np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1))
    np.fill_diagonal(gap, np.inf)
    assert (gap > radius[:, None] + radius[None]).all()
    body = bodies[0]
    stack = _stack(scene, body.x * 1e6, body.y * 1e6)
    assert stack["ice"][1] == pytest.approx(body.height * 1e6, abs=1.5)
    assert stack["cell"][1] < stack["ice"][1]


def test_contamination_is_a_clump_of_grains_on_the_ice():
    scene = SampleScene(cell_type="none", rip_fraction=0.0)
    speck = next(
        f for f in scene.features if f.kind == "contamination" and f.part == "body"
    )
    grains = sim_solid._clump(speck, int(scene.seed))
    assert len(grains) >= 2
    assert grains == sim_solid._clump(speck, int(scene.seed))
    x, y, r, _ = grains[0]
    stack = _stack(scene, x, y)
    assert stack["particle"][1] > stack["ice"][1]


def test_an_ice_plate_is_a_slab_standing_on_the_ice():
    scene = SampleScene(cell_type="none", rip_fraction=0.0, ice_density=2.0)
    plates = [f for f in scene.features if f.kind == "ice"]
    lo, hi = sim_solid.PLATE
    tall = []
    for plate in plates[:12]:
        x, y, r = plate.x * 1e6, plate.y * 1e6, plate.sigma * 1e6
        on = _stack(scene, x, y)["ice"][1]
        # the ice beside it: clear of the plate, though a bar or another
        # plate may lie there, and a broken hole has no ice at all
        beside = [
            _stack(scene, x + 2.5 * r * c, y + 2.5 * r * s)["ice"][1]
            for c, s in ((1, 0), (-1, 0), (0, 1), (0, -1))
        ]
        tall.append(on - float(np.median(beside)))
    usual = [t for t in tall if 0.5 * lo < t < hi + 1.5]
    assert len(usual) >= 0.6 * len(tall), tall
    assert max(usual) - min(usual) > 0.2, "each plate its own thickness"


@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
def test_ice_plates_show_in_both_beams(microscope, beam_type):
    """A plate is brighter than the ice it lies on, so it shows at a field of
    view that takes in a grid square."""

    def bright(density):
        _use(microscope, **{**PLAIN, "ice_density": density})
        image = _image(microscope, beam_type)
        return float((image > 1.1 * np.median(image)).mean())

    none, some = bright(0.0), bright(6.0)
    assert some > none + 0.04


def test_a_milled_edge_has_a_bright_rim(microscope, monkeypatch):
    """The top next to a trench glows in the SEM; away from it, it doesn't."""
    _use(microscope, **PLAIN)
    before = _image(microscope, BeamType.ELECTRON, hfw=60e-6)
    _mill(microscope, (20e-6, 6e-6, 0.0))
    after, hit = _trace(microscope, monkeypatch, BeamType.ELECTRON, 60e-6)
    rim = sim_solid._rim(hit, 60.0 / 512)
    flip = slice(None, None, -1) if hit["flip"] else slice(None)
    near = (rim[flip] > 0.8) & (hit["kind"][flip] == 1)
    far = (rim[flip] == 0) & (hit["kind"][flip] == 1)
    assert near.sum() > 100 and far.sum() > 1000
    assert (after - before)[near].mean() > 40
    assert abs((after - before)[far].mean()) < 3


def test_curtaining_streaks_a_cut_face_unless_turned_off(microscope, monkeypatch):
    """A cut face carries streaks along the milling beam (a tone that varies
    from column to column); with curtaining off the face is even."""
    spread = {}
    for curtaining in (1.0, 0.0):
        _use(microscope, curtaining=curtaining, **PLAIN)
        _mill(microscope, (20e-6, 8e-6, 0.0))
        image, hit = _trace(microscope, monkeypatch, BeamType.ELECTRON, 40e-6)
        flip = slice(None, None, -1) if hit["flip"] else slice(None)
        face = hit["kind"][flip] == 2
        row = np.argmax(face.sum(axis=1))
        spread[curtaining] = image[row][face[row]].std()
    assert spread[1.0] > 1.4 * max(spread[0.0], 0.5)


def test_a_gis_coat_is_a_layer_on_top_and_a_bright_band_on_a_cut(
    microscope, monkeypatch
):
    scene = _use(microscope, **PLAIN)
    bare = _stack(scene, 0.0, 0.0)
    assert bare["coat"][1] == sim_solid.NONE
    scene.deposit(25)  # 1 um
    coated = _stack(scene, 0.0, 0.0)
    assert coated["coat"][0] == pytest.approx(bare["ice"][1])
    assert coated["coat"][1] - coated["coat"][0] == pytest.approx(1.0, abs=0.05)
    _mill(microscope, (20e-6, 8e-6, 0.0))
    image, hit = _trace(microscope, monkeypatch, BeamType.ELECTRON, 40e-6)
    flip = slice(None, None, -1) if hit["flip"] else slice(None)
    face = hit["kind"][flip] == 2
    material = hit["material"][flip]
    coat = face & (material == sim_solid.COAT)
    ice = face & (material == sim_solid.ICE)
    assert coat.sum() > 50 and ice.sum() > 50
    assert image[coat].mean() > 1.5 * image[ice].mean()


def test_a_spot_burn_is_a_small_pit_with_a_bright_ring(microscope, monkeypatch):
    from fibsem.geometry.projection import BeamStageProjection

    scene = _use(microscope, **PLAIN)
    before = _image(microscope, BeamType.ELECTRON, hfw=20e-6)
    scene.burn(
        [(0.0, 0.0)],
        BeamType.ION,
        microscope._connection.scene_stage_position(),
        BeamStageProjection.from_microscope(microscope, beam_type=BeamType.ION),
    )
    after, hit = _trace(microscope, monkeypatch, BeamType.ELECTRON, 20e-6)
    assert (hit["kind"] == 2).sum() > 20  # the pit's floor
    change = after - before
    assert change.max() > 40  # the ring
    rows, cols = np.nonzero(change > 40)
    assert np.ptp(rows) < 100 and np.ptp(cols) < 100  # under 4 um across


def test_damage_is_shapes_on_the_lamella_face(microscope):
    """A damaged lamella is one region of irregular pieces, cut through the
    lamella where they lie and nowhere else."""
    scene = _use(
        microscope,
        damage_chance=1.0,
        damage_below=(0.5e-6, 0.5e-6),
        **PLAIN,
    )
    _mill(microscope, (12e-6, 4e-6, 2.4e-6), (12e-6, 4e-6, -2.4e-6))
    assert scene.damage_started is None
    _mill(microscope, (12e-6, 4e-6, 2.15e-6), (12e-6, 4e-6, -2.15e-6))
    assert scene.damage_started is not None
    region = scene.milled[-1]
    assert region.face and all(len(piece) >= 3 for piece in region.face)
    xa = np.arange(-8.0, 8.0, 0.1)
    ya = np.arange(-30.0, 30.0, 0.1)
    col = sim_solid.columns(scene, xa, ya, (0.1, 0.1))
    cut = sim_solid._removed(scene, *np.meshgrid(xa, ya), col)[-1]
    gone = cut["to"] > cut["from"]
    assert 0 < gone.sum() < 0.5 * gone.size


@pytest.mark.parametrize("current, breaks", [(60e-12, False), (1e-9, True)])
def test_a_high_current_damages_a_lamella_a_low_one_leaves(microscope, current, breaks):
    """`damage_below` is the thickness at risk at `damage_current`; a pass at
    a higher current puts a thicker lamella at risk, and says so."""
    scene = _use(
        microscope,
        damage_chance=1.0,
        damage_below=(0.3e-6, 0.3e-6),
        damage_current=60e-12,
        **PLAIN,
    )
    microscope.set_beam_current(current, BeamType.ION)
    _mill(microscope, (12e-6, 4e-6, 2.4e-6), (12e-6, 4e-6, -2.4e-6))
    record = scene.mill_passes[-1]
    assert 0.6e-6 < record["gap"] < 0.9e-6
    assert record["damage_below"] == pytest.approx(0.3e-6 * (current / 60e-12) ** 0.5)
    if not breaks:
        assert record["damage_chance"] == 0.0 and scene.damage_started is None
        return
    # thicker than the usual limit: at risk, but at less than the full chance
    limit, gap = record["damage_below"], record["gap"]
    chance = (limit - gap) / (limit - 0.3e-6)
    assert 0.2 < chance < 0.8
    assert record["damage_chance"] == pytest.approx(chance)
    if scene.damage_started is not None:
        assert scene.damage_started["current"] == pytest.approx(current)
        assert scene.damage_started["below"] > scene.damage_started["gap"]


def test_a_thicker_lamella_is_less_likely_to_break_at_a_high_current():
    """Over the usual limit the chance falls with thickness, to nothing at
    the limit the current sets; under the usual limit it is the full chance."""
    chances = []
    for gap in (0.2e-6, 0.5e-6, 0.9e-6, 1.3e-6):
        scene = SampleScene(
            damage_chance=0.4, damage_below=(0.3e-6, 0.3e-6), damage_current=60e-12
        )
        record = {"gap": gap, "current": 1e-9, "pass": 1, "band": (0, 1e-5, 0, gap)}
        scene._damage(record, np.random.default_rng(0))
        chances.append(record["damage_chance"])
    assert chances[0] == pytest.approx(0.4)
    assert chances[0] > chances[1] > chances[2] > chances[3] == 0.0


def test_the_current_matters_to_damage_only_where_a_sample_is_configured():
    assert SampleScene().damage_current == 0.0
    assert SampleScene.from_config({}).damage_current == pytest.approx(60e-12)


def _columns(scene, x, y, half=20.0, step=0.1) -> dict:
    xa = x + np.arange(-half, half, step)
    ya = y + np.arange(-half, half, step)
    return sim_solid.columns(scene, xa, ya, (step, step))


def _body(scene):
    bodies = [f for f in scene.features if f.kind == "cell" and f.part == "body"]
    return min(bodies, key=lambda f: np.hypot(f.x, f.y))


def test_a_holey_film_has_ice_across_its_holes_and_none_where_one_broke():
    scene = SampleScene(film="holey", **{**PLAIN, "broken_hole_fraction": 0.05})
    col = _columns(scene, 60.0, 60.0, half=40.0, step=0.2)
    film = col["film"][1] > sim_solid.NONE / 2
    ice = col["ice"][1] > sim_solid.NONE / 2
    bar = col["bar"][1] > sim_solid.NONE / 2
    hole = ice & ~film & ~bar
    assert 0.01 < hole.mean() < 0.5  # a lattice of holes, iced over
    assert (~ice & ~film).any()  # and a few broken through
    assert col["hole"][hole].mean() > 5 * col["hole"][film].mean()


@pytest.mark.slow
def test_every_cell_type_stands_up_under_its_own_ice():
    for kind in ("yeast", "mammalian", "bacteria", "neuron", "rpe"):
        scene = SampleScene(**{**PLAIN, "cell_type": kind})
        f = _body(scene)
        col = _columns(scene, f.x * 1e6, f.y * 1e6, half=3.0)
        ice, cell = col["ice"][1], col["cell"][1]
        inside = cell > sim_solid.NONE / 2
        assert inside[30, 30], kind
        # the cell is under the ice, and the ice stands higher for it
        assert np.all(cell[inside] < ice[inside]), kind
        assert ice[30, 30] > 0.4, kind


def test_no_two_cells_are_alike_inside():
    scene = SampleScene(**{**PLAIN, "cell_type": "rpe"})
    bodies = [f for f in scene.features if f.kind == "cell" and f.part == "body"]
    nuclei = [f for f in scene.features if f.part == "nucleus"]
    lobes = [sim_solid._lobes(f, scene.seed) for f in nuclei[:40]]
    assert len({len(n) for n in lobes}) > 1  # some lobed, some not
    assert len({round(n[0][2], 3) for n in lobes}) > 30  # each its own size
    counts = [len(sim_solid._inclusions(scene, f)) for f in bodies[:40]]
    assert max(counts) > 10 * max(min(counts), 1)  # packed, or almost empty

    # what is in a cell lies inside it, clear of its top
    f = _body(scene)
    col = _columns(scene, f.x * 1e6, f.y * 1e6, half=8.0)
    lo, hi = col["organelle"]
    there = hi > sim_solid.NONE / 2
    assert there.any()
    assert np.all(hi[there] < col["cell"][1][there])
    assert np.all(lo[there] > sim_solid.FILM_TOP)
    assert set(np.unique(col["organelle_is"][there])) <= {
        sim_solid.DROPLET,
        sim_solid.VESICLE,
    }
    # and a view too coarse to show it does without
    coarse = _columns(scene, f.x * 1e6, f.y * 1e6, half=8.0, step=1.0)
    assert not (coarse["organelle"][1] > sim_solid.NONE / 2).any()


def test_a_milled_frame_walked_in_part_matches_one_walked_whole(
    microscope, monkeypatch
):
    _use(microscope, **{**PLAIN, "cell_type": "rpe"})
    _mill(microscope, (14e-6, 6e-6, 5e-6), (14e-6, 6e-6, -5e-6), (3e-6, 3e-6, 20e-6))
    calls = []
    seen = sim_solid._seen

    def record(col, removed, z, shift, height):
        out = seen(col, removed, z, shift, height)
        calls.append((col, removed, z, shift, height, out))
        return out

    monkeypatch.setattr(sim_solid, "_seen", record)
    for beam_type in (BeamType.ION, BeamType.ELECTRON):
        _image(microscope, beam_type, hfw=80e-6)
        col, removed, z, shift, height, (hit_q, hit_m) = calls[-1]
        assert removed and "walk" in col
        width = col["surface"].shape[1]
        whole = sim_solid._walk(
            col, removed, z, shift, slice(0, height), slice(0, width)
        )
        assert np.array_equal(hit_q, whole[0]) and np.array_equal(hit_m, whole[1])
        # and the milling did change what is seen
        assert not np.array_equal(hit_q, col["walk"][1][0])


def test_the_surface_from_above_says_where_cells_and_cuts_are(microscope):
    scene = _use(microscope, **{**PLAIN, "cell_type": "rpe"})
    f = _body(scene)
    x = f.x * 1e6 + np.arange(-30, 30, 0.5)
    y = f.y * 1e6 + np.arange(-30, 30, 0.5)
    z, mat = sim_solid.surface(scene, x, y)
    assert mat[60, 60] == sim_solid.CELL and z[60, 60] > 2.0
    assert mat[0, 0] in (sim_solid.ICE, sim_solid.CELL)
    assert (mat == sim_solid.ICE).any()

    _mill(microscope, (10e-6, 6e-6, 0.0))
    px, py = np.mean(scene.milled[0].points, axis=0) * 1e6
    z, _ = sim_solid.surface(
        scene, px + np.arange(-2, 3) * 0.5, py + np.arange(-2, 3) * 0.5
    )
    assert z[2, 2] < -1.0  # the cut went down through the ice


def test_the_fiducial_cross_is_the_brightest_thing_in_view(microscope):
    _use(microscope, fiducial=True, **PLAIN)
    for beam_type in (BeamType.ELECTRON, BeamType.ION):
        image = _image(microscope, beam_type, hfw=80e-6)
        h, w = image.shape
        centre = image[h // 2 - 60 : h // 2 + 60, w // 2 - 60 : w // 2 + 60]
        assert np.percentile(centre, 99) > np.median(image) + 60


def test_a_thick_lamella_cracks_wider():
    """Damage on a lamella still 1 um thick is coarser than on a thin one, so
    it shows at the field of view rough milling is watched at."""
    scene = SampleScene(seed=5, **PLAIN)
    ends = [(-6.0, 0.0), (6.0, 0.0)]

    def area(thickness):
        rng = np.random.default_rng(2)
        pieces = sim_solid.damage(scene, "crack", 3, ends, -0.2, rng, thickness)
        return sum(
            0.5
            * abs(
                np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
            )
            for p in pieces
        )

    assert area(1.0) > 2 * area(0.2)
    assert area(0.3) == pytest.approx(area(0.2))


def test_the_record_says_what_was_milled_and_goes_to_json(microscope):
    """After a run the scene's record lists each pass and region as plain
    values, and says of any point whether it was milled away: before any FM
    image, and for a pattern that stopped short."""
    import json

    scene = _use(microscope, **PLAIN)
    _mill(microscope, (12e-6, 6e-6, 0.0))
    x, y = np.mean(scene.milled[0].points, axis=0)
    record = scene.record(
        {"in the cut": (x, y, 1e-6), "beside it": (x + 40e-6, y, 1e-6)}
    )
    assert json.loads(json.dumps(record)) == record
    assert len(record["passes"]) == 1 and len(record["regions"]) == 1
    assert record["regions"][0]["depth"] == pytest.approx(10e-6)
    assert not record["regions"][0]["damage"]
    assert record["points"]["in the cut"]["milled"] is True
    assert record["points"]["beside it"]["milled"] is False

    shallow = _use(microscope, **PLAIN)
    microscope.draw_pattern(
        FibsemRectangleSettings(
            width=12e-6, height=6e-6, depth=0.3e-6, centre_x=0.0, centre_y=0.0
        )
    )
    microscope.run_milling()
    x, y = np.mean(shallow.milled[0].points, axis=0)
    points = shallow.record({"under the floor": (x, y, 0.5e-6)})["points"]
    assert points["under the floor"]["milled"] is False


@pytest.mark.slow
def test_the_record_lists_each_target_and_whether_it_is_gone(microscope):
    scene = _use(microscope, cell_type="rpe", seed=4)
    targets = scene.record()["targets"]
    assert len(targets) == len(scene.fm_targets()) > 0
    assert not any(t["milled"] for t in targets[:20])


def test_a_gis_coat_closes_a_broken_hole_and_keeps_the_relief_under_it():
    """The coat lies over what is there: it grows in over a broken hole and
    closes a small one, and a bar still stands up under it."""
    scene = SampleScene(film="holey", **{**PLAIN, "broken_hole_fraction": 0.05})
    bare = _columns(scene, 60.0, 60.0, half=40.0, step=0.2)
    broken = bare["surface"] < sim_solid.NONE / 2
    assert broken.any()
    scene.deposit(120)  # 4.8 um
    coated = _columns(scene, 60.0, 60.0, half=40.0, step=0.2)
    lo, hi = coated["coat"]
    over = broken & (hi > sim_solid.NONE / 2)
    assert over.mean() > 0.9 * broken.mean()
    # a layer over the hole, not a plug down through it
    assert 0.5 < (hi - lo)[over].mean() < 4.9
    relief = np.ptp(bare["surface"][~broken])
    assert np.ptp(hi[~broken]) > 0.6 * relief


WAFFLE = {**PLAIN, "preparation": "waffle"}


def _on_and_off_a_bar(scene) -> tuple:
    """A world point (um) in the middle of a grid bar, and one in the middle
    of a square."""
    line = np.arange(-150.0, 150.0, 1.0)
    bars = scene.film_masks(line[None, :] * 1e-6, line[:, None] * 1e-6)[0]
    far = ndi.distance_transform_edt(bars), ndi.distance_transform_edt(~bars)
    (by, bx), (sy, sx) = (np.unravel_index(np.argmax(d), d.shape) for d in far)
    return (line[bx], line[by]), (line[sx], line[sy])


def _face(microscope, orientation: str) -> None:
    """Turn the stage to one of its named orientations."""
    to = microscope.get_orientation(orientation)
    pose = microscope.get_stage_position()
    pose.r, pose.t = to.r, to.t
    microscope.move_stage_absolute(pose)


def test_a_waffle_is_a_slab_of_ice_with_the_bars_standing_in_it():
    scene = SampleScene(**WAFFLE)
    bar, square = _on_and_off_a_bar(scene)
    in_square, on_bar = _stack(scene, *square), _stack(scene, *bar)
    film = sim_solid.film_top(scene)
    assert film == -25.0
    # in a square: ice from the film at the bottom up to the plane in focus
    assert in_square["film"][1] == film
    assert in_square["ice"][0] == film
    assert -1.5 < in_square["ice"][1] < 0.5
    assert in_square["bar"][1] < sim_solid.NONE / 2
    # a bar stands on the film, nearly as tall as the slab, under a skin of ice
    lo, hi = on_bar["bar"]
    assert lo == film and hi - lo > 23.0
    assert on_bar["ice"][0] == pytest.approx(hi)
    assert 0.2 < on_bar["ice"][1] - hi < 1.5
    assert not [f for f in scene.features if f.kind == "cell"]


def test_a_thicker_waffle_buries_its_bars():
    flush = SampleScene(**WAFFLE)
    thick = SampleScene(slab_thickness=40e-6, **WAFFLE)
    bar, _ = _on_and_off_a_bar(thick)
    stack = _stack(thick, *bar)
    assert stack["ice"][1] - stack["bar"][1] > 14.0
    x = bar[0] + np.arange(-2, 3) * 0.2
    y = bar[1] + np.arange(-2, 3) * 0.2
    shows = [
        float(sim_solid.columns(s, x, y, (0.2, 0.2))["bar_weight"][2, 2])
        for s in (flush, thick)
    ]
    assert shows[0] > 0.8 and shows[1] < 0.1


def test_a_view_looks_down_through_a_waffle_and_no_further_through_a_plunged_grid():
    assert sim_solid.z_min(SampleScene(**PLAIN)) == sim_solid.Z_MIN
    assert sim_solid.z_min(SampleScene(**WAFFLE)) == -28.0


def test_a_waffle_must_be_a_thickness_the_beam_can_mill():
    with pytest.raises(ValueError, match="slab_thickness"):
        SampleScene(slab_thickness=80e-6, **WAFFLE)
    with pytest.raises(ValueError, match="preparation"):
        SampleScene(preparation="sandwich")
    scene = SampleScene.from_config(
        {"preparation": "waffle", "slab_thickness": 30e-6, "hat_scratches": 0.5}
    )
    assert sim_solid.film_top(scene) == -30.0


def test_the_hats_polish_leaves_lines_in_a_waffles_top():
    """Fine parallel grooves, which a perfect polish (0) does not leave."""
    _, square = _on_and_off_a_bar(SampleScene(**WAFFLE))
    x = square[0] + np.arange(-20, 20, 0.1)
    y = square[1] + np.arange(-20, 20, 0.1)
    rough = []
    for strength in (1.0, 0.0):
        scene = SampleScene(hat_scratches=strength, **WAFFLE)
        top = sim_solid.columns(scene, x, y, (0.1, 0.1))["ice"][1]
        rough.append(float((top - ndi.gaussian_filter(top, 20)).std()))
    assert rough[0] > 0.01 and rough[1] < 0.2 * rough[0], rough


def test_a_trench_goes_through_a_waffle_only_if_it_is_set_as_deep(microscope):
    """The slab is 25 um thick: a pattern set deeper, milled square on to
    the grid, cuts a hole through it; one set shallower stops at a floor in
    the ice."""
    _face(microscope, "FIB")
    bottom = []
    for depth in (30e-6, 5e-6):
        scene = _use(microscope, **WAFFLE)
        microscope.draw_pattern(
            FibsemRectangleSettings(
                width=20e-6, height=20e-6, depth=depth, centre_x=0.0, centre_y=0.0
            )
        )
        microscope.run_milling()
        px, py = np.mean(scene.milled[0].points, axis=0) * 1e6
        near = np.arange(-2, 3) * 0.5
        z, mat = sim_solid.surface(scene, px + near, py + near)
        bottom.append((float(z[2, 2]), int(mat[2, 2])))
    through, floor = bottom
    assert through[1] == sim_solid.VOID
    assert floor[1] == sim_solid.ICE and floor[0] == pytest.approx(-5.0, abs=1.0)


def test_a_trench_wall_in_a_waffle_is_a_dark_face_not_a_bright_top(microscope):
    """Looking into a trench at the milling angle, the far wall is the side
    of the slab: darker than the top round it, and darker further down."""
    _face(microscope, "FIB")
    _use(microscope, **WAFFLE)
    microscope.draw_pattern(
        FibsemRectangleSettings(
            width=30e-6, height=30e-6, depth=30e-6, centre_x=0.0, centre_y=0.0
        )
    )
    microscope.run_milling()
    _face(microscope, "MILLING")
    image = _image(microscope, BeamType.ION, hfw=80e-6)
    top = float(np.median(image[:20]))
    inside = ndi.binary_erosion(image < 0.6 * top, iterations=6)
    assert inside.sum() > 2000
    rows = np.nonzero(inside.any(axis=1))[0]
    column = image[rows.min() : rows.max() + 1, 246:266].mean(axis=1)
    assert column.max() < 0.6 * top
    assert column[:5].mean() != pytest.approx(column[-5:].mean(), abs=5.0)
