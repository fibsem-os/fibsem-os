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
    microscope.system.sim["sample"] = {"enabled": True, "coincidence_offset": 8e-6}
    microscope._setup_sample_scene()
    pose = microscope.get_stage_position()
    pose.t = MILLING_TILT
    microscope.move_stage_absolute(pose)
    yield microscope
    microscope.disconnect()


def _scene(microscope, **kwargs) -> SampleScene:
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
    assert np.hypot(after[0] - before[0], after[1] - before[1]) < 2
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
    film = np.percentile(gfp_smooth, 10)
    assert gfp_smooth[nuclei].mean() > 3 * film
