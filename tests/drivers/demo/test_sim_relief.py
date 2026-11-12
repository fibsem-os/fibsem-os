"""The relief renderer: the scene as a height field, seen along each beam."""

import os

import numpy as np
import pytest
from scipy import ndimage as ndi

import fibsem.config as cfg
from fibsem import utils
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


def _shift(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The translation (rows, columns) taking a onto b, by phase correlation."""
    fa, fb = np.fft.fft2(a - a.mean()), np.fft.fft2(b - b.mean())
    r = np.fft.ifft2(fa.conj() * fb / (np.abs(fa * fb) + 1e-9)).real
    peak = np.array(np.unravel_index(np.argmax(r), r.shape), dtype=float)
    size = np.array(r.shape)
    return np.where(peak > size / 2, peak - size, peak)


def test_the_renderer_comes_from_the_config():
    assert SampleScene.from_config({"renderer": "relief"}).renderer == "relief"
    assert SampleScene().renderer == "flat"


@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
def test_the_grid_lands_where_the_flat_renderer_puts_it(microscope, beam_type):
    """Positions come from the same projection: the bars (at the film's height)
    are on the same pixels whichever renderer draws them. The flat renderer
    has no heights, so the bars are kept level with the squares here."""
    _use(microscope, **PLAIN)
    flat = _image(microscope, beam_type, hfw=400e-6)
    _use(microscope, renderer="relief", bar_height=0.0, **PLAIN)
    relief = _image(microscope, beam_type, hfw=400e-6)
    assert np.abs(_shift(flat, relief)).max() <= 1


def test_a_trench_is_cut_along_the_milling_beam(microscope):
    """A rectangle milled in the FIB view is a square-edged dark rectangle in
    that view, where it was drawn."""
    _use(microscope, renderer="relief", **PLAIN)
    microscope.draw_pattern(
        FibsemRectangleSettings(
            width=30e-6, height=6e-6, depth=1e-6, centre_x=0.0, centre_y=0.0
        )
    )
    microscope.run_milling()
    image = _image(microscope, BeamType.ION, hfw=60e-6)
    pixel = 60e-6 / 512
    labels, n = ndi.label(image < 40)
    assert n > 0
    sizes = ndi.sum(np.ones_like(labels), labels, range(1, n + 1))
    rows, cols = np.nonzero(labels == 1 + int(np.argmax(sizes)))
    width = (cols.max() - cols.min() + 1) * pixel
    height = (rows.max() - rows.min() + 1) * pixel
    assert width == pytest.approx(30e-6, rel=0.08)
    assert height == pytest.approx(6e-6, rel=0.15)
    # square: the dark region fills its bounding box
    assert len(rows) / ((np.ptp(rows) + 1) * (np.ptp(cols) + 1)) > 0.9
    centre = ((cols.min() + cols.max()) / 2 - 256, (rows.min() + rows.max()) / 2 - 171)
    assert abs(centre[0]) <= 3 and abs(centre[1]) <= 3


def test_a_lamella_between_two_trenches_stays(microscope):
    """Two patterns with a gap between them leave the gap standing: in the
    FIB view a bright band between two dark ones."""
    _use(microscope, renderer="relief", **PLAIN)
    gap = 1e-6
    for y in (gap / 2 + 3e-6, -(gap / 2 + 3e-6)):
        microscope.draw_pattern(
            FibsemRectangleSettings(
                width=20e-6, height=6e-6, depth=1e-6, centre_x=0.0, centre_y=y
            )
        )
    microscope.run_milling()
    image = _image(microscope, BeamType.ION, hfw=40e-6)
    profile = image[:, 236:276].mean(axis=1)
    pixel = 40e-6 / 512
    band = profile[171 - int(0.3e-6 / pixel) : 171 + int(0.3e-6 / pixel) + 1]
    above = profile[171 - int(3.5e-6 / pixel)]
    below = profile[171 + int(3.5e-6 / pixel)]
    assert band.min() > 60, band
    assert above < 40 and below < 40, (above, below)


@pytest.mark.parametrize("cell_type", ["rpe", "neuron"])
def test_rpe_and_neuron_cells_grow_and_stand_up(cell_type):
    """RPE grows as packed islands, neurons as a soma with processes; both
    stand above the film in the relief renderer."""
    from fibsem.drivers.demo import sim_relief

    scene = SampleScene(cell_type=cell_type, renderer="relief")
    parts = {f.part for f in scene.features if f.kind == "cell"}
    assert "body" in parts
    if cell_type == "neuron":
        assert "neurite" in parts
    body = next(f for f in scene.features if f.kind == "cell" and f.part == "body")
    xa = body.x * 1e6 + np.arange(-20, 20, 0.5)
    ya = body.y * 1e6 + np.arange(-20, 20, 0.5)
    x, y = np.meshgrid(xa, ya)
    h, _, _, _, _, _ = sim_relief.surface(scene, x, y, (0.5, 0.5))
    assert h[40, 40] - np.median(h) > 0.5


def test_ice_plates_frost_and_spheres():
    """Ice plates are hexagons, frost is bright grains, and the spheres are
    dark round balls standing on the film."""
    from fibsem.drivers.demo import sim_relief

    scene = SampleScene(cell_type="none", renderer="relief", sphere_density=2.0)
    parts = {f.part for f in scene.features if f.kind == "contamination"}
    assert {"frost", "sphere"} <= parts
    ball = max((f for f in scene.features if f.part == "sphere"), key=lambda f: f.sigma)
    xa = ball.x * 1e6 + np.arange(-5, 5, 0.1)
    ya = ball.y * 1e6 + np.arange(-5, 5, 0.1)
    x, y = np.meshgrid(xa, ya)
    h, mat, _, _, _, _ = sim_relief.surface(scene, x, y, (0.1, 0.1))
    assert mat[50, 50] == sim_relief.SPHERE
    assert h[50, 50] - np.median(h) > 0.5 * ball.sigma * 1e6
    assert sim_relief.MATERIAL[sim_relief.SPHERE][0] < 0.5


def test_a_crack_is_a_dark_line_in_the_film():
    """A hairline crack darkens a thin line of the film without opening it."""
    scene = SampleScene(cell_type="none", film="continuous", crack_density=1.0)
    arm = max(scene.cracks, key=len)
    x, y = arm[len(arm) // 2, :2]
    along = scene.crack_depth(np.array([x]), np.array([y]))
    away = scene.crack_depth(np.array([x + 5e-6]), np.array([y + 5e-6]))
    assert along[0] > 0.5 and away[0] == 0.0
    assert (
        not SampleScene(crack_density=0.0)
        .crack_depth(np.array([x]), np.array([y]))
        .any()
    )


def test_a_coat_takes_the_charging_away():
    """An uncoated grid charges in the SEM, more with a longer dwell; a
    sputter coat takes most of it away, and a GIS coat some."""
    scene = SampleScene()
    assert scene.charge_strength(None) == 0.0
    short, long = scene.charge_strength(0.2e-6), scene.charge_strength(3e-6)
    assert 0 < short < long
    scene.sputter(15)
    assert scene.charge_strength(3e-6) < 0.5 * long
    gis = SampleScene()
    gis.deposit(25)
    assert gis.gis_thickness == pytest.approx(1e-6)
    assert gis.charge_strength(3e-6) < long


def test_the_demo_coater_and_gis_coat_the_sample(monkeypatch):
    """Coating through the coating service coats the simulated sample."""
    monkeypatch.setenv("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(
            cfg.SHIPPED_CONFIG_PATH, "sim-arctis-configuration.yaml"
        ),
    )
    try:
        microscope.system.sim["sample"] = {"enabled": True}
        microscope._connection.setup_sample_scene()
        scene = microscope._connection.sample_scene
        microscope.coating.sputter(10)
        microscope.coating.deposit(20)
        assert scene.sputter_thickness > 0
        assert scene.gis_thickness == pytest.approx(0.8e-6)
    finally:
        microscope.disconnect()


def _lamella(microscope, gap, height=6e-6, depth=1e-6):
    for y in (gap / 2 + height / 2, -(gap / 2 + height / 2)):
        microscope.draw_pattern(
            FibsemRectangleSettings(
                width=20e-6, height=height, depth=depth, centre_x=0.0, centre_y=y
            )
        )
    microscope.run_milling()


def test_a_pass_records_the_true_gap(microscope):
    """Milling drifts and overcuts a little: each pass records the gap it
    planned and the one it left, and they differ."""
    scene = _use(microscope, renderer="relief", damage_chance=0.0, **PLAIN)
    _lamella(microscope, 1e-6)
    record = scene.mill_passes[-1]
    assert record["planned_gap"] == pytest.approx(1e-6, rel=1e-6)
    assert record["gap"] != pytest.approx(1e-6, abs=1e-9)
    assert abs(record["gap"] - 1e-6) < 0.3e-6
    assert record["gap_at_depth"] > record["gap"]


def test_a_shallow_trench_leaves_a_floor(microscope):
    """A pattern cuts only as deep as it is set: short of the bottom, the
    FIB sees a floor, not the dark through-cut."""
    _use(microscope, renderer="relief", cell_type="rpe", film="continuous")
    microscope.draw_pattern(
        FibsemRectangleSettings(
            width=20e-6, height=8e-6, depth=0.2e-6, centre_x=0.0, centre_y=0.0
        )
    )
    microscope.run_milling()
    image = _image(microscope, BeamType.ION, hfw=40e-6)
    middle = image[171 - 20 : 171 + 20, 256 - 40 : 256 + 40]
    assert np.median(middle) > 40


def test_a_thin_lamella_can_be_damaged(microscope):
    """Below the seeded thickness, a pass can start damage; the pass and the
    kind are recorded, and it grows on the passes after."""
    scene = _use(
        microscope,
        renderer="relief",
        damage_chance=1.0,
        damage_below=(0.5e-6, 0.5e-6),
        **PLAIN,
    )
    _lamella(microscope, 1e-6)
    assert scene.damage_started is None
    _lamella(microscope, 0.3e-6, height=0.35e-6)
    started = scene.damage_started
    assert started["pass"] == 1
    assert started["kind"] in ("ragged", "crack", "broken")
    _lamella(microscope, 0.25e-6, height=0.375e-6)
    assert scene.mill_passes[-1]["damage"]["severity"] == 2


def test_centriole_pairs_are_sharp_at_their_height():
    """A centriole pair is recorded with its midpoint and height, renders
    brightest in the plane at that height, and the camera's hot pixels sit
    on the same pixels, as bright, in every frame."""
    from fibsem.drivers.demo.sim_scene import fm_channel_weights
    from fibsem.geometry.projection import FMStageProjection

    with_none = SampleScene(cell_type="rpe", fm_no_target_chance=1.0, seed=5)
    assert with_none.fm_targets() == []
    scene = SampleScene(cell_type="rpe", fm_no_target_chance=0.0, seed=5)
    target = min(scene.fm_targets(), key=lambda t: np.hypot(*t["midpoint"][:2]))
    (x0, y0, z0), (x1, y1, z1) = target["pair"]
    assert 0.7e-6 < np.hypot(x1 - x0, y1 - y0) < 1.3e-6
    mx, my, mz = target["midpoint"]
    for g in scene.features:
        g.x, g.y = g.x - mx, g.y - my
    for spot in scene.fm_spots:
        spot["pair"] = [(x - mx, y - my, z) for x, y, z in spot["pair"]]
        x, y, z, sigma = spot["pool"]
        spot["pool"] = (x - mx, y - my, z, sigma)

    microscope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(
            cfg.SHIPPED_CONFIG_PATH, "sim-arctis-configuration.yaml"
        ),
    )
    try:
        microscope.system.sim["sample"] = {"enabled": True}
        microscope._connection.setup_sample_scene()
        pose = microscope._connection.scene_stage_position_at_fm()
        scene.anchor(pose)
        live = FMStageProjection.from_microscope(microscope)
        projection = FMStageProjection(
            geometry=live.geometry, pixel_size=0.1e-6, shape=(201, 201)
        )
        weights = fm_channel_weights("525", 488)

        def frame(z, seed):
            return scene.render_fm(
                pose,
                (201, 201),
                projection,
                weights=weights,
                defocus=z,
                rng=np.random.default_rng(seed),
            )

        sharp, far = frame(mz, 1), frame(mz + 2.5e-6, 2)
    finally:
        microscope.disconnect()
    centre = (slice(90, 111), slice(90, 111))
    assert sharp[centre].max() > 1.3 * far[centre].max()
    hot = (sharp == far) & (sharp > 4000)
    assert hot.sum() >= int(scene.fm_hot_pixels * 201 * 201) - 1


def test_milling_cuts_away_a_centriole_outside_the_lamella():
    """An FM point is gone once a pattern cut along the beam passes through
    it, and kept while it lies in the slab between two trenches."""
    from fibsem.drivers.demo.sim_relief import BeamCut
    from fibsem.drivers.demo.sim_scene import MilledRegion

    scene = SampleScene(cell_type="rpe", fm_no_target_chance=1.0, seed=5)

    def trench(w0, w1):
        # in the milling view a point (x, y, h) um sits at (x, y + h)
        polygon = np.array([[-5.0, w0], [5.0, w0], [5.0, w1], [-5.0, w1]])
        corners = polygon * 1e-6
        return MilledRegion(
            points=corners, beam_cut=BeamCut(polygon=polygon, turn=1.0, fs=1.0, k=1.0)
        )

    point = (0.0, 0.0, 1e-6)  # sits at w = 1 um in the milling view
    scene.milled = [trench(2.0, 8.0), trench(-8.0, 0.5)]
    assert not scene.milled_through(*point)
    scene.milled.append(trench(0.5, 1.5))
    assert scene.milled_through(*point)


@pytest.mark.parametrize(
    "setting, channel",
    [
        ({"fm_red": "cilia"}, ("610", 561)),
        ({"fm_green": "lipid_droplets"}, ("525", 488)),
        ({"fm_far_red": "mitochondria"}, ("670", 640)),
        ({"fm_far_red": "membrane"}, ("670", 640)),
    ],
)
def test_each_channel_can_carry_a_label(setting, channel):
    """Cilia (red), lipid droplets (green), mitochondria or membrane (far
    red) show in their channel when set, and the channel is dark without."""
    from fibsem.drivers.demo.sim_scene import fm_channel_weights
    from fibsem.geometry.projection import FMStageProjection

    microscope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(
            cfg.SHIPPED_CONFIG_PATH, "sim-arctis-configuration.yaml"
        ),
    )
    try:
        microscope.system.sim["sample"] = {"enabled": True}
        microscope._connection.setup_sample_scene()
        pose = microscope._connection.scene_stage_position_at_fm()
        live = FMStageProjection.from_microscope(microscope)
        projection = FMStageProjection(
            geometry=live.geometry, pixel_size=0.5e-6, shape=(201, 201)
        )

        def frame(**kw):
            base = dict(cell_type="rpe", fm_no_target_chance=0.0, seed=5)
            scene = SampleScene(**{**base, **kw})
            # a target cell, with a cilium, under the view
            spot = next(t for t in scene.fm_spots if t["cilium"])
            mx, my, _ = spot["midpoint"]
            for g in scene.features:
                g.x, g.y = g.x - mx, g.y - my
            for t in scene.fm_spots:
                if t["cilium"]:
                    t["cilium"] = [(x - mx, y - my, z) for x, y, z in t["cilium"]]
                    t["cilium_path"] = [
                        (x - mx, y - my, z) for x, y, z in t["cilium_path"]
                    ]
            scene.anchor(pose)
            image = scene.render_fm(
                pose,
                (201, 201),
                projection,
                weights=fm_channel_weights(*channel),
                defocus=1e-6,
                rng=np.random.default_rng(0),
            )
            return image.astype(np.float32)

        plain = frame(fm_red="none", fm_green="none")
        labelled = frame(**{"fm_red": "none", "fm_green": "none", **setting})
    finally:
        microscope.disconnect()
    assert (labelled - plain).max() > 300


def test_reflection_shows_bars_bright_ice_grey_and_cells_dark():
    """In reflected light the bar tops are brightest, the ice a mid grey in
    fringes, a cell darker than the ice round it, and a cut-through trench
    nothing."""
    from fibsem.drivers.demo import sim_relief
    from fibsem.drivers.demo.sim_relief import BeamCut
    from fibsem.drivers.demo.sim_scene import MilledRegion

    scene = SampleScene(renderer="relief", cell_type="rpe", seed=5)
    body = min(
        (f for f in scene.features if f.kind == "cell" and f.part == "body"),
        key=lambda f: np.hypot(f.x, f.y),
    )
    cx, cy = body.x * 1e6, body.y * 1e6
    xs = np.arange(cx - 60, cx + 60, 0.5)
    ys = np.arange(cy - 60, cy + 60, 0.5)
    x, y = np.meshgrid(xs, ys)
    r = sim_relief.reflectance(scene, x, y, (0.5, 0.5))
    _, mat, *_ = sim_relief.surface(scene, x, y, (0.5, 0.5))
    # the bar tops; their rounded edges tilt the light away
    top = scene.bar_ridge(x / 1e6, y / 1e6) > 0.9 * scene.bar_height
    assert np.median(r[(mat == sim_relief.METAL) & top]) > 0.9
    ice = r[mat == sim_relief.ICE]
    cell = r[mat == sim_relief.CELL]
    assert 0.2 < np.median(ice) < 0.8 and ice.std() > 0.02
    assert np.median(cell) < np.median(ice)

    polygon = np.array(
        [[cx - 5, cy - 3], [cx + 5, cy - 3], [cx + 5, cy + 3], [cx - 5, cy + 3]]
    )
    scene.milled = [
        MilledRegion(
            points=polygon * 1e-6,
            beam_cut=BeamCut(polygon=polygon, turn=1.0, fs=1.0, k=0.0),
        )
    ]
    sim_relief._cache(scene).clear()
    r = sim_relief.reflectance(scene, x, y, (0.5, 0.5))
    inside = (np.abs(x - cx) < 4) & (np.abs(y - cy) < 2)
    assert r[inside].max() == 0.0


def test_a_burn_aimed_on_a_cell_lands_where_the_beam_meets_it():
    """Aimed at a point on the film plane, a grazing beam meets a cell
    standing above it further back along the beam: the hit is on the cell's
    surface, where the plane point and the raised point show alike."""
    from fibsem.drivers.demo import sim_relief
    from fibsem.drivers.demo.sim_relief import ViewFrame

    scene = SampleScene(renderer="relief", cell_type="mammalian", seed=5)
    body = min(
        (f for f in scene.features if f.kind == "cell" and f.part == "body"),
        key=lambda f: np.hypot(f.x, f.y),
    )
    fs, k = np.sin(np.deg2rad(15)), np.cos(np.deg2rad(15))
    frame = ViewFrame(anchor=(0.0, 0.0), theta=0.0, turn=1.0, fs=fs, k=k)
    shift = scene._along_beam_to_surface((body.x, body.y), frame)
    um = sim_relief.UM
    x, y = body.x * um, (body.y + shift) * um
    gx, gy = np.meshgrid(x + np.arange(-24, 25) * 0.25, y + np.arange(-24, 25) * 0.25)
    h = float(sim_relief.surface(scene, gx, gy, (0.25, 0.25))[0][24, 24])
    assert h > 0.5  # on the cell, not the film
    assert abs(fs * shift * um + k * h) < 0.1


def test_the_bars_stand_above_the_squares():
    """A bar is a ridge and a square a shallow well: the surface over a bar
    stands about bar_height above the middle of the square beside it, and
    with bar_height 0 the two are level."""
    from fibsem.drivers.demo import sim_relief

    def rise(**kwargs):
        scene = SampleScene(
            renderer="relief", grid_rotation=0.0, centre_mark=False, **PLAIN, **kwargs
        )
        grid = scene.grids[0]
        pitch = scene.grid_pitch * 1e6
        # the grid's centre is a square's centre; the next bar is half a pitch over
        ys = grid.y * 1e6 + np.arange(-2, 2, 0.5)
        xs = grid.x * 1e6 + np.array([0.0, pitch / 2])
        x, y = np.meshgrid(xs, ys)
        h = sim_relief.surface(scene, x, y, (0.5, 0.5))[0]
        return float(np.mean(h[:, 1] - h[:, 0])), scene.bar_height * 1e6

    rise_um, height_um = rise()
    assert rise_um == pytest.approx(height_um, abs=1.5)
    assert abs(rise(bar_height=0.0)[0]) < 1.5
