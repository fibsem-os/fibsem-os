"""The solid renderer: the sample as objects that lie under one another."""

import os

import numpy as np
import pytest
from scipy import ndimage as ndi

import fibsem.config as cfg
from fibsem import utils
from fibsem.drivers.demo import sim_solid
from fibsem.drivers.demo.sim_scene import SampleScene
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


def test_the_renderer_comes_from_the_config():
    assert SampleScene.from_config({"renderer": "solid"}).renderer == "solid"


def test_a_cell_lies_under_the_ice_with_its_nucleus_inside():
    """What the height field cannot hold: at a cell's nucleus the stack is
    film, cell, nucleus inside the cell, and ice over the top."""
    scene = SampleScene(renderer="solid", rip_fraction=0.0)
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
    scene = SampleScene(
        renderer="solid", cell_type="none", sphere_density=2.0, rip_fraction=0.0
    )
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
def test_the_grid_lands_where_the_relief_renderer_puts_it(
    microscope, beam_type, monkeypatch
):
    """Positions come from the same projection: with the ice left flat over
    the bars (it stands up over them otherwise, which a glancing beam sees as
    a shift), the bars are on the same pixels whichever renderer draws them."""
    monkeypatch.setattr(sim_solid, "BAR_RISE", 0.0)
    _use(microscope, renderer="relief", **PLAIN)
    relief = _image(microscope, beam_type, hfw=400e-6)
    _use(microscope, renderer="solid", **PLAIN)
    solid = _image(microscope, beam_type, hfw=400e-6)
    assert np.abs(_shift(relief, solid)).max() <= 1


def test_a_trench_is_cut_along_the_milling_beam(microscope):
    """A rectangle milled in the FIB view is a square-edged dark rectangle in
    that view, where it was drawn."""
    _use(microscope, renderer="solid", **PLAIN)
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


def test_a_lamella_between_two_trenches_stays(microscope):
    _use(microscope, renderer="solid", **PLAIN)
    _mill(microscope, (20e-6, 6e-6, 3.5e-6), (20e-6, 6e-6, -3.5e-6))
    image = _image(microscope, BeamType.ION, hfw=40e-6)
    profile = image[:, 236:276].mean(axis=1)
    pixel = 40e-6 / 512
    band = profile[171 - int(0.3e-6 / pixel) : 171 + int(0.3e-6 / pixel) + 1]
    assert band.min() > 60, band
    assert profile[171 - int(3.5e-6 / pixel)] < 40
    assert profile[171 + int(3.5e-6 / pixel)] < 40


def test_a_cut_face_shows_the_cell_and_its_nucleus(microscope, monkeypatch):
    """A trench through a cell leaves a face made of ice, the cell and the
    nucleus inside it, each its own tone."""
    scene = _use(microscope, renderer="solid", rip_fraction=0.0, sphere_density=0.0)
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
    scene = SampleScene(cell_type="rpe", renderer="solid", rip_fraction=0.0)
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
    scene = SampleScene(cell_type="none", renderer="solid", rip_fraction=0.0)
    speck = next(
        f for f in scene.features if f.kind == "contamination" and f.part == "body"
    )
    grains = sim_solid._clump(speck, int(scene.seed))
    assert len(grains) >= 2
    assert grains == sim_solid._clump(speck, int(scene.seed))
    x, y, r, _ = grains[0]
    stack = _stack(scene, x, y)
    assert stack["particle"][1] > stack["ice"][1]


def test_a_milled_edge_has_a_bright_rim(microscope, monkeypatch):
    """The top next to a trench glows in the SEM; away from it, it doesn't."""
    _use(microscope, renderer="solid", **PLAIN)
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
        _use(microscope, renderer="solid", curtaining=curtaining, **PLAIN)
        _mill(microscope, (20e-6, 8e-6, 0.0))
        image, hit = _trace(microscope, monkeypatch, BeamType.ELECTRON, 40e-6)
        flip = slice(None, None, -1) if hit["flip"] else slice(None)
        face = hit["kind"][flip] == 2
        row = np.argmax(face.sum(axis=1))
        spread[curtaining] = image[row][face[row]].std()
    assert spread[1.0] > 1.8 * max(spread[0.0], 0.5)


def test_a_gis_coat_is_a_layer_on_top_and_a_bright_band_on_a_cut(
    microscope, monkeypatch
):
    scene = _use(microscope, renderer="solid", **PLAIN)
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

    scene = _use(microscope, renderer="solid", **PLAIN)
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
        renderer="solid",
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


def _columns(scene, x, y, half=20.0, step=0.1) -> dict:
    xa = x + np.arange(-half, half, step)
    ya = y + np.arange(-half, half, step)
    return sim_solid.columns(scene, xa, ya, (step, step))


def _body(scene):
    bodies = [f for f in scene.features if f.kind == "cell" and f.part == "body"]
    return min(bodies, key=lambda f: np.hypot(f.x, f.y))


def test_a_holey_film_has_ice_across_its_holes_and_none_where_one_broke():
    scene = SampleScene(
        renderer="solid", film="holey", **{**PLAIN, "broken_hole_fraction": 0.05}
    )
    col = _columns(scene, 60.0, 60.0, half=40.0, step=0.2)
    film = col["film"][1] > sim_solid.NONE / 2
    ice = col["ice"][1] > sim_solid.NONE / 2
    bar = col["bar"][1] > sim_solid.NONE / 2
    hole = ice & ~film & ~bar
    assert 0.01 < hole.mean() < 0.5  # a lattice of holes, iced over
    assert (~ice & ~film).any()  # and a few broken through
    assert col["hole"][hole].mean() > 5 * col["hole"][film].mean()


def test_every_cell_type_stands_up_under_its_own_ice():
    for kind in ("yeast", "mammalian", "bacteria", "neuron", "rpe"):
        scene = SampleScene(renderer="solid", **{**PLAIN, "cell_type": kind})
        f = _body(scene)
        col = _columns(scene, f.x * 1e6, f.y * 1e6, half=3.0)
        ice, cell = col["ice"][1], col["cell"][1]
        inside = cell > sim_solid.NONE / 2
        assert inside[30, 30], kind
        # the cell is under the ice, and the ice stands higher for it
        assert np.all(cell[inside] < ice[inside]), kind
        assert ice[30, 30] > 0.4, kind


def test_no_two_cells_are_alike_inside():
    scene = SampleScene(renderer="solid", **{**PLAIN, "cell_type": "rpe"})
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
    _use(microscope, renderer="solid", **{**PLAIN, "cell_type": "rpe"})
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
