"""Scene realism: contamination, beam shift and scan rotation (FIB-874).

Each layer exists because it exercises something in the measurement or the
app: contamination is the aperiodic content that breaks a mesh-pitch
alias, a beam offset is the only way the sim can produce a real lateral
dx, and scan rotation turns the content the way a real raster does.
"""

import dataclasses
import os

import numpy as np
import pytest
from scipy import ndimage as ndi

import fibsem.config as cfg
from fibsem import utils
from fibsem.alignment.coincidence import (
    REFUSAL_LATERAL_OFFSET,
    check_coincidence,
    ensure_coincident,
)
from fibsem.microscopes.sim_scene import SampleScene
from fibsem.projection import BeamStageProjection
from fibsem.structures import BeamType, ImageSettings

TFS_SHUTTLE_CONFIG = os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")
MILLING_TILT = np.deg2rad(12.0)


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(
        manufacturer="Demo", config_path=TFS_SHUTTLE_CONFIG
    )
    microscope.system.sim["sample"] = {
        "enabled": True,
        "coincidence_offset": 8e-6,
        "fiducial": True,
    }
    microscope._setup_sample_scene()
    pose = microscope.get_stage_position()
    pose.t = MILLING_TILT
    microscope.move_stage_absolute(pose)
    yield microscope
    microscope.disconnect()


def _scene(microscope, **kwargs) -> SampleScene:
    """A deterministic scene: no noise, and no ice or rips unless asked -
    the fiducial finders here look for the darkest/brightest pixels."""
    kwargs = {"ice_density": 0.0, "rip_fraction": 0.0, "fiducial": True, **kwargs}
    scene = SampleScene(noise_sigma=0.0, noise_fraction=0.0, **kwargs)
    scene.anchor(microscope.get_stage_position())
    microscope._sample_scene = scene
    return scene


def _settings(beam_type, hfw=150e-6):
    return ImageSettings(
        resolution=(1536, 1024),
        hfw=hfw,
        dwell_time=1e-9,
        save=False,
        beam_type=beam_type,
    )


def _fiducial(image):
    """Centre of the fiducial: the centroid of its pixels. In the FIB view
    the foreshortened cross is a long dark ridge, and the smoothed minimum
    wanders along it; the centroid tracks the crossing to a pixel."""
    data = image.data.astype(np.float32)
    if image.metadata.image_settings.beam_type is BeamType.ION:
        mask = data < np.percentile(data, 0.5)
    else:
        mask = data > np.percentile(data, 99.5)
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def test_contamination_adds_fine_aperiodic_structure(microscope):
    def speckle(density):
        _scene(
            microscope,
            coincidence_offset=0.0,
            cell_type="none",
            contamination_density=density,
        )
        image = microscope.acquire_image(image_settings=_settings(BeamType.ELECTRON))
        data = image.data.astype(np.float32)
        # speck-scale blobs: a band-pass at their size, counted as components
        dog = ndi.gaussian_filter(data, 2) - ndi.gaussian_filter(data, 10)
        return int(ndi.label(dog > 15)[1])

    clean, dirty = speckle(0.0), speckle(30.0)
    assert dirty > 3 * clean and dirty - clean > 15, (clean, dirty)
    scene = microscope._sample_scene
    assert sum(f.kind == "contamination" for f in scene.features) == int(
        30.0 * (scene.extent / 100e-6) ** 2
    )


def test_beam_shift_moves_the_content_the_way_the_alignment_expects(microscope):
    """The oracle is the alignment code itself: displace the stage, align by
    beam shift against the earlier reference, and the image must be back."""
    from fibsem.alignment import AlignmentSubsystem, beam_shift_alignment_v2

    _scene(
        microscope,
        coincidence_offset=0.0,
        cell_type="none",
        grid_intensity=0.0,
        contamination_density=0.0,
    )
    settings = _settings(BeamType.ION)
    reference = microscope.acquire_image(image_settings=settings)
    before = _fiducial(reference)

    microscope.stable_move(dx=3e-6, dy=-2e-6, beam_type=BeamType.ION)
    displaced = _fiducial(microscope.acquire_image(image_settings=settings))
    assert np.hypot(displaced[0] - before[0], displaced[1] - before[1]) > 10

    beam_shift_alignment_v2(
        microscope, reference, subsystem=AlignmentSubsystem.BEAM_SHIFT
    )

    after = _fiducial(microscope.acquire_image(image_settings=settings))
    assert np.hypot(after[0] - before[0], after[1] - before[1]) < 3
    shift = microscope.get_beam_shift(BeamType.ION)
    assert np.hypot(shift.x, shift.y) > 1e-6  # it really was the beam that moved


def test_a_small_beam_offset_is_measured_as_dx_and_tolerated(microscope):
    _scene(
        microscope,
        coincidence_offset=8e-6,
        beam_offset={"ion": (1.3e-6, 0.0)},
        current_offset_scale=0.0,
    )

    result = ensure_coincident(microscope, tolerance=1e-6)

    assert result.converged, result.reason
    # dx is the shift that brings FIB onto SEM: minus the displacement
    assert result.measurements[0].dx == pytest.approx(-1.3e-6, abs=0.4e-6)
    assert abs(result.measurements[0].dz) == pytest.approx(8e-6, abs=1e-6)


def test_a_large_beam_offset_is_refused_as_lateral(microscope):
    _scene(
        microscope,
        coincidence_offset=8e-6,
        beam_offset={"ion": (8e-6, 0.0)},
        current_offset_scale=0.0,
    )

    m = check_coincidence(microscope)

    assert not m.is_reliable
    assert m.refusal_reason == REFUSAL_LATERAL_OFFSET
    assert m.dx == pytest.approx(-8e-6, abs=0.6e-6)


def test_changing_beam_current_moves_the_beam_reproducibly(microscope):
    """Each ion current carries its own seeded offset: switching current
    shifts the FIB view, switching back restores it, and the beam-shift
    alignment can undo the difference - the milling-current alignment."""
    from fibsem.alignment import AlignmentSubsystem, beam_shift_alignment_v2

    _scene(
        microscope,
        coincidence_offset=0.0,
        cell_type="none",
        grid_intensity=0.0,
        contamination_density=0.0,
        current_offset_scale=2e-6,
    )
    settings = _settings(BeamType.ION)
    currents = microscope.get_available_values("current", BeamType.ION)
    low, high = currents[1], currents[-2]

    microscope.set("current", low, BeamType.ION)
    reference = microscope.acquire_image(image_settings=settings)
    at_low = _fiducial(reference)
    microscope.set("current", high, BeamType.ION)
    at_high = _fiducial(microscope.acquire_image(image_settings=settings))
    assert np.hypot(at_high[0] - at_low[0], at_high[1] - at_low[1]) > 3
    microscope.set("current", low, BeamType.ION)
    again = _fiducial(microscope.acquire_image(image_settings=settings))
    assert np.hypot(again[0] - at_low[0], again[1] - at_low[1]) < 1.5

    microscope.set("current", high, BeamType.ION)
    beam_shift_alignment_v2(
        microscope, reference, subsystem=AlignmentSubsystem.BEAM_SHIFT
    )
    aligned = _fiducial(microscope.acquire_image(image_settings=settings))
    assert np.hypot(aligned[0] - at_low[0], aligned[1] - at_low[1]) < 2


def test_scan_rotation_turns_the_content(microscope):
    """A quarter-turn of the scan puts a feature that sat to the right of
    centre above or below it, at the same distance."""
    scene = _scene(
        microscope,
        coincidence_offset=0.0,
        cell_type="none",
        grid_intensity=0.0,
        contamination_density=0.0,
    )
    microscope.stable_move(dx=10e-6, dy=0.0, beam_type=BeamType.ELECTRON)
    pose = microscope.get_stage_position()
    projection = BeamStageProjection.from_microscope(microscope, BeamType.ELECTRON)
    resolution = (1536, 1024)
    hfw = 150e-6
    px = hfw / resolution[0]

    def fiducial_px(scan_rotation):
        rotated = dataclasses.replace(projection, scan_rotation=scan_rotation)
        data = scene.render(
            BeamType.ELECTRON,
            pose,
            hfw,
            resolution,
            rotated,
            rng=np.random.default_rng(0),
        )
        y, x = np.unravel_index(
            np.argmax(ndi.gaussian_filter(data.astype(np.float32), 3)), data.shape
        )
        return x - resolution[0] / 2, y - resolution[1] / 2

    x0, y0 = fiducial_px(0.0)
    assert abs(abs(x0) - 10e-6 / px) < 3 and abs(y0) < 3
    x90, y90 = fiducial_px(np.pi / 2)
    assert abs(x90) < 3 and abs(abs(y90) - 10e-6 / px) < 3
    x180, y180 = fiducial_px(np.pi)
    assert abs(x180 + x0) < 3 and abs(y180) < 3


@pytest.mark.parametrize(
    "cell_type, parts",
    [
        ("mammalian", {"body", "nucleus", "organelle"}),
        ("yeast", {"body", "nucleus", "bud"}),
        ("bacteria", {"body"}),
        ("mixed", {"body", "nucleus", "organelle", "bud"}),
    ],
)
def test_cell_types_generate_their_parts(cell_type, parts):
    scene = SampleScene(cell_type=cell_type, contamination_density=0.0)
    cells = [f for f in scene.features if f.kind == "cell"]
    assert {f.part for f in cells} == parts
    assert all(f.cell_id >= 0 for f in cells)


def test_mammalian_is_the_default_and_unknown_types_are_rejected():
    assert SampleScene().cell_type == "mammalian"
    with pytest.raises(ValueError):
        SampleScene(cell_type="tardigrade")
    assert SampleScene.from_config({"cell_type": "bacteria"}).cell_type == "bacteria"


def test_mammalian_cells_are_spread_out_with_the_nucleus_inside_the_body():
    scene = SampleScene(cell_type="mammalian", contamination_density=0.0)
    cells = [f for f in scene.features if f.kind == "cell"]
    bodies = {f.cell_id: f for f in cells if f.part == "body"}
    nuclei = {f.cell_id: f for f in cells if f.part == "nucleus"}
    assert set(bodies) == set(nuclei)
    for cell_id, body in bodies.items():
        nucleus = nuclei[cell_id]
        assert np.hypot(nucleus.x - body.x, nucleus.y - body.y) < 0.8 * body.sigma
        assert nucleus.sigma < 0.5 * body.sigma
    centres = np.array([(b.x, b.y) for b in bodies.values()])
    for i, c in enumerate(centres):
        others = np.delete(centres, i, axis=0)
        assert np.hypot(*(others - c).T).min() > 1.5 * scene.mammalian_radius[1]


def test_the_fm_dyes_the_nucleus_and_the_cytoplasm_differently(microscope):
    """In the FM the DNA dye (365 excitation) lights the nucleus only; the
    cytoplasmic dye (450) lights the whole body - so the DAPI image is a
    subset of the GFP image, brightest where GFP is bright too."""
    from fibsem.microscopes.sim_scene import fm_channel_weights
    from fibsem.projection import FMStageProjection

    scene = _scene(
        microscope,
        coincidence_offset=0.0,
        cell_type="mammalian",
        contamination_density=0.0,
        grid_intensity=0.0,
    )
    res = (512, 512)
    projection = FMStageProjection(
        geometry=microscope.hardware_geometry(),
        pixel_size=150e-6 / res[0],
        shape=(res[1], res[0]),
    )
    pose = microscope.get_stage_position()
    gfp = scene.render_fm(
        pose, res, projection, weights=fm_channel_weights("Fluorescence", 450)
    ).astype(np.float32)
    dapi = scene.render_fm(
        pose, res, projection, weights=fm_channel_weights("Fluorescence", 365)
    ).astype(np.float32)

    dapi_smooth = ndi.gaussian_filter(dapi, 2)
    gfp_smooth = ndi.gaussian_filter(gfp, 2)
    background = np.median(dapi_smooth)
    nuclei = dapi_smooth > background + 0.3 * (dapi_smooth.max() - background)
    assert 0 < nuclei.mean() < 0.05  # nuclei are small
    # and they sit in cytoplasm: the GFP there is well above the bare film
    # (large adherent cells cover most of the field, so the film is the
    # dim tail of the GFP image, not its median)
    film = np.percentile(gfp_smooth, 2)
    assert gfp_smooth[nuclei].mean() > 2 * film


def _world_grid(hfw=300e-6, shape=(1024, 1024)):
    px = hfw / shape[1]
    xs = (np.arange(shape[1]) - shape[1] / 2)[None, :] * px
    ys = (np.arange(shape[0]) - shape[0] / 2)[:, None] * px
    return xs, ys, px


def test_the_film_is_continuous_by_default():
    assert SampleScene().film == "continuous"


@pytest.mark.xfail(
    reason="the fine-pass correlator aliases on the 4 um hole lattice: at the "
    "fiducial the two bands disagree by a hole pitch, and an 80 um error "
    "converges falsely through a lattice alias. Handling a periodic film "
    "(notching the lattice frequency, or masking the holes) is measurement "
    "work still to do - FIB-868.",
    strict=False,
)
def test_the_fine_pass_measures_on_holey_film(microscope):
    _scene(microscope, coincidence_offset=8e-6, film="holey")
    m = check_coincidence(microscope)
    assert m.is_reliable, m.refusal_reason
    assert abs(m.dz) == pytest.approx(8e-6, abs=1e-6)


def test_holey_film_is_a_lattice_of_holes_on_the_film_only():
    scene = SampleScene(film="holey", hole_diameter=2e-6, hole_pitch=4e-6, seed=1)
    xs, ys, px = _world_grid()
    bars, holes, rips, rim, beyond = scene.film_masks(xs, ys)
    assert not (holes & bars).any()
    # R2/2: a 2 um hole every 4 um covers pi / 16 of the film
    fill = holes[~bars & ~rips].mean()
    assert fill == pytest.approx(np.pi / 16, rel=0.15)
    # and it is a lattice: hole centres repeat at the pitch along the grid axes
    n_holes = ndi.label(holes)[1]
    film_area = (~bars & ~rips).sum() * px**2
    assert n_holes == pytest.approx(film_area / scene.hole_pitch**2, rel=0.15)
    assert not SampleScene(film="continuous").film_masks(xs, ys)[1].any()


def test_some_squares_are_ripped_and_rips_are_square_sized():
    scene = SampleScene(rip_fraction=0.1, seed=2)
    xs, ys, px = _world_grid(hfw=1200e-6, shape=(1024, 1024))
    bars, holes, rips, rim, beyond = scene.film_masks(xs, ys)
    n, areas = ndi.label(rips)[1], None
    squares_in_view = (1200e-6**2) / scene.grid_pitch**2
    assert 0.03 * squares_in_view < n < 0.25 * squares_in_view
    labels, n = ndi.label(rips)
    areas = ndi.sum(rips, labels, range(1, n + 1)) * px**2
    # a rip is a good part of its square, not a speck
    assert np.median(areas) > 0.1 * scene.grid_pitch**2
    assert not SampleScene(rip_fraction=0.0).film_masks(xs, ys)[2].any()


def test_rips_differ_in_size_orientation_and_position():
    """Every square torn, and no two tears alike: areas spread widely, the
    long axes point every way, and the tears sit off their squares' centres."""
    scene = SampleScene(rip_fraction=1.0, seed=3, grid_rotation=0.0)
    xs, ys, px = _world_grid(hfw=1200e-6, shape=(1024, 1024))
    bars, holes, rips, rim, beyond = scene.film_masks(xs, ys)
    labels, n = ndi.label(rips)
    assert n > 30
    areas = ndi.sum(rips, labels, range(1, n + 1)) * px**2
    assert np.std(areas) / np.mean(areas) > 0.3
    angles, offsets = [], []
    for k in range(1, n + 1):
        rows, cols = np.nonzero(labels == k)
        if rows.size < 200:
            continue
        cov = np.cov(np.vstack([cols, rows]))
        w, v = np.linalg.eigh(cov)
        angles.append(np.arctan2(v[1, -1], v[0, -1]) % np.pi)
        # the tear's centre against its square's centre, in pitches
        cx_, cy_ = cols.mean() * px + xs.min(), rows.mean() * px + ys.min()
        half = scene.grid_pitch / 2
        offsets.append(
            np.hypot(
                ((cx_ + half) % scene.grid_pitch) - half,
                ((cy_ + half) % scene.grid_pitch) - half,
            )
            / scene.grid_pitch
        )
    assert np.std(angles) > 0.4
    assert np.median(offsets) > 0.08


def test_a_rip_has_a_bright_curled_edge_in_the_sem(microscope):
    scene = _scene(
        microscope,
        coincidence_offset=0.0,
        cell_type="none",
        contamination_density=0.0,
        fiducial=False,
        grid_intensity=0.0,
        rip_fraction=1.0,
    )
    frame = microscope.acquire_image(_settings(BeamType.ELECTRON, hfw=150e-6)).data
    film = np.median(frame)
    dark = frame < film - 40
    assert dark.mean() > 0.02, "no rip in view"
    edge = ndi.binary_dilation(dark, iterations=6) & ~dark
    assert np.percentile(frame[edge], 90) > film + 25
    assert scene.rip_fraction == 1.0


def test_ice_plates_are_generated_and_render_bright(microscope):
    scene = _scene(
        microscope,
        coincidence_offset=0.0,
        cell_type="none",
        contamination_density=0.0,
        grid_intensity=0.0,  # so the plates are the brightest thing in view
        ice_density=2.0,
    )
    ice = [f for f in scene.features if f.kind == "ice"]
    assert len(ice) == int(2.0 * (scene.extent / 100e-6) ** 2)
    assert all(f.sharpness >= 4 and f.wobble > 0 for f in ice)
    image = microscope.acquire_image(
        image_settings=_settings(BeamType.ELECTRON, hfw=300e-6)
    )
    data = image.data.astype(np.float32)
    # flat plates: large bright connected regions with sharp edges, well
    # above the film (an absolute step, not a percentile - the plates are
    # a good fraction of the field)
    bright = ndi.gaussian_filter(data, 1) > np.median(data) + 40
    labels, n = ndi.label(bright)
    px = 300e-6 / 1536
    areas = ndi.sum(bright, labels, range(1, n + 1)) * px**2
    assert (areas > np.pi * (5e-6) ** 2).sum() >= 3


def test_beyond_the_grid_is_rim_then_holder():
    scene = SampleScene(grid_radius=1.4e-3, grid_rim_width=150e-6)
    xs = np.array([[0.0, 1.45e-3, 1.7e-3]])
    ys = np.array([[0.0]])
    bars, holes, rips, rim, beyond = scene.film_masks(xs, ys)
    # the rim is the ring round the film; past it, the holder
    assert rim.tolist() == [[False, True, False]]
    assert beyond.tolist() == [[False, False, True]]


def test_milled_patterns_persist_in_the_world(microscope):
    """A rectangle milled at the milling pose is a trench from then on: it
    sits where the pattern was drawn in the FIB view, it moves with the
    stage, and the SEM sees it too (FIB-877)."""
    from fibsem.structures import FibsemRectangleSettings

    _scene(
        microscope,
        coincidence_offset=0.0,
        cell_type="none",
        contamination_density=0.0,
        grid_intensity=0.0,
        ice_density=0.0,
    )
    settings = _settings(BeamType.ION)
    before = microscope.acquire_image(image_settings=settings)

    # a trench 30 x 6 um, 20 um above and 10 um right of the FIB centre
    # (microscope image coordinates: metres from the centre, y up)
    microscope.draw_rectangle(
        FibsemRectangleSettings(
            width=30e-6, height=6e-6, depth=1e-6, centre_x=10e-6, centre_y=20e-6
        )
    )
    microscope.run_milling(milling_current=1e-9, milling_voltage=30e3)
    after = microscope.acquire_image(image_settings=settings)
    assert len(microscope._sample_scene.milled) == 1

    def trench_pixels(image):
        """The trench: the largest dark connected region (the fiducial's
        thin arms are the only other dark thing in this scene)."""
        labels, n = ndi.label(image.data < 100)
        assert n > 0, "no trench rendered"
        sizes = ndi.sum(np.ones_like(labels), labels, range(1, n + 1))
        ys, xs = np.nonzero(labels == 1 + int(np.argmax(sizes)))
        return xs, ys

    assert (after.data < 100).sum() > (before.data < 100).sum() + 100
    xs, ys = trench_pixels(after)
    centre = (xs.mean(), ys.mean())
    # microscope image coordinates are centred with y up
    height, width = after.data.shape
    px = after.metadata.pixel_size.x
    expected = (width / 2 + 10e-6 / px, height / 2 - 20e-6 / px)
    assert np.hypot(centre[0] - expected[0], centre[1] - expected[1]) < 4
    assert (xs.max() - xs.min()) * px == pytest.approx(30e-6, rel=0.15)

    # it moves with the stage: 25 um to the right in the FIB view
    microscope.stable_move(dx=25e-6, dy=0.0, beam_type=BeamType.ION)
    moved = microscope.acquire_image(image_settings=settings)
    xs2, ys2 = trench_pixels(moved)
    assert abs(xs2.mean() - centre[0]) == pytest.approx(25e-6 / px, abs=4)
    assert ys2.mean() == pytest.approx(centre[1], abs=4)

    # and the SEM sees a dark trench too (a wider field: the FIB's 20 um
    # "up" is ~77 um along the foreshortened surface at this pose)
    sem = microscope.acquire_image(
        image_settings=_settings(BeamType.ELECTRON, hfw=400e-6)
    )
    labels, n = ndi.label(sem.data < 40)
    assert n >= 1
    sizes = ndi.sum(np.ones_like(labels), labels, range(1, n + 1))
    sem_px = sem.metadata.pixel_size.x
    assert sizes.max() * sem_px**2 > 0.5 * 30e-6 * 6e-6  # a trench-sized dark patch


def test_the_fiducial_can_be_switched_off():
    # off by default - no real grid has one - and on when asked
    assert not any(f.kind == "fiducial" for f in SampleScene().features)
    scene = SampleScene(fiducial=True)
    assert any(f.kind == "fiducial" for f in scene.features)
    assert SampleScene.from_config({"fiducial": True}).fiducial is True


def test_a_rotated_pattern_mills_the_footprint_it_was_drawn_with(microscope):
    """A rectangle drawn at 45 deg in the FIB view at the milling angle must
    come back as that same rotated rectangle in the FIB view - the
    foreshortening is undone on the corners, not on a height alone."""
    from fibsem.structures import FibsemRectangleSettings

    _scene(
        microscope,
        coincidence_offset=0.0,
        cell_type="none",
        contamination_density=0.0,
        grid_intensity=0.0,
        fiducial=False,
    )
    settings = _settings(BeamType.ION)
    microscope.draw_rectangle(
        FibsemRectangleSettings(
            width=24e-6,
            height=3e-6,
            depth=1e-6,
            centre_x=-5e-6,
            centre_y=8e-6,
            rotation=np.deg2rad(45),
        )
    )
    microscope.run_milling(milling_current=1e-9, milling_voltage=30e3)
    image = microscope.acquire_image(image_settings=settings)
    trench = image.data < 100

    # the pattern's own footprint in image pixels (centred, y up)
    height, width = image.data.shape
    px = image.metadata.pixel_size.x
    yy, xx = np.mgrid[0:height, 0:width]
    dx = (xx - width / 2) * px - (-5e-6)
    dy = -(yy - height / 2) * px - 8e-6
    c, s_ = np.cos(np.deg2rad(45)), np.sin(np.deg2rad(45))
    u, v = dx * c + dy * s_, -dx * s_ + dy * c
    expected = (np.abs(u) <= 12e-6) & (np.abs(v) <= 1.5e-6)
    iou = (trench & expected).sum() / (trench | expected).sum()
    assert iou > 0.8, f"trench footprint IoU {iou:.2f} against the drawn pattern"


def _fib_and_sem(microscope, hfw=150e-6):
    fib = microscope.acquire_image(image_settings=_settings(BeamType.ION, hfw=hfw)).data
    sem = microscope.acquire_image(
        image_settings=_settings(BeamType.ELECTRON, hfw=hfw)
    ).data
    return fib.astype(np.float32), sem.astype(np.float32)


def test_holes_and_trenches_are_dark_in_both_beams(microscope):
    """No material, no signal: a hole in the film and a milled trench read
    dark in the FIB as well as the SEM - the FIB is not an inverted SEM."""
    from fibsem.structures import FibsemRectangleSettings

    scene = _scene(
        microscope,
        coincidence_offset=0.0,
        cell_type="none",
        contamination_density=0.0,
        fiducial=False,
        film="holey",
    )
    microscope.draw_rectangle(
        FibsemRectangleSettings(
            width=20e-6, height=5e-6, depth=1e-6, centre_x=0.0, centre_y=-25e-6
        )
    )
    microscope.run_milling(milling_current=1e-9, milling_voltage=30e3)
    fib, sem = _fib_and_sem(microscope, hfw=100e-6)
    for name, image in (("FIB", fib), ("SEM", sem)):
        film = np.median(image)
        # the darkest few percent are holes and the trench, well below the film
        assert np.percentile(image, 2) < film - 30, f"{name}: no dark holes/trench"
        assert (image < film - 30).mean() > 0.05, f"{name}: too little dark structure"


def test_the_fib_sees_cells_as_outlined_not_inverted(microscope):
    """In the FIB a cell body is slightly darker than the film with a bright
    rim (the edge effect); in the SEM it is brighter than the film."""
    scene = _scene(
        microscope,
        coincidence_offset=0.0,
        cell_type="yeast",
        contamination_density=0.0,
        fiducial=False,
        grid_intensity=0.0,
    )
    fib, sem = _fib_and_sem(microscope, hfw=100e-6)
    film_fib, film_sem = np.median(fib), np.median(sem)
    # the SEM: cells brighter than the film
    assert (sem > film_sem + 25).mean() > 0.02
    # the FIB (its own pixels - the two views are foreshortened differently,
    # so a mask cannot be carried across): bodies darker than the film...
    dark = fib < film_fib - 15
    assert dark.mean() > 0.005
    interior = ndi.binary_erosion(dark, iterations=3)
    # ...with a bright outline just outside them: the body's darkening
    # tapers over several pixels before the edge effect's rim, so look a
    # little way out from the dark core
    rim = ndi.binary_dilation(dark, iterations=12) & ~ndi.binary_dilation(
        dark, iterations=6
    )
    assert np.percentile(fib[rim], 90) > film_fib + 8
    assert np.percentile(fib[rim], 90) > fib[interior].mean() + 20


def test_the_film_is_brighter_at_grazing_incidence():
    """SE yield rises with tilt: the same bare film reads brighter in a
    view that sees it at a steeper angle."""
    from fibsem.microscopes.sim_scene import BEAM_LAYERS

    t = BEAM_LAYERS[BeamType.ION]
    grazing = t["film"] + t["film_tilt_gain"] * (1 - 0.26)
    face_on = t["film"] + t["film_tilt_gain"] * (1 - 0.95)
    assert grazing > face_on + 30
