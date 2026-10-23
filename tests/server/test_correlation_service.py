"""Correlating saved images through the App Server: the correlation service's
queries read inside the open experiment, work at Watch, and save a run where the
correlation tab finds it."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # TestClient transport

from fastapi.testclient import TestClient  # noqa: E402

from fibsem.correlation.history import (  # noqa: E402
    LamellaCorrelation,
    correlation_dir,
)
from fibsem.fm.structures import (  # noqa: E402
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.structures import BeamType, ImageSettings  # noqa: E402

TOKEN = "test-token"
CORRELATION = "/services/correlation"

# A known FM->FIB transform: FIB = S * R @ (x, y, z * Z_SCALE) + T. The fit
# recovers it exactly from noise-free pairs that span the stack's depth.
FM_PX, FM_PZ = 200e-9, 400e-9
Z_SCALE = FM_PZ / FM_PX
ANGLE = np.radians(25.0)
R = np.array(
    [
        [1, 0, 0],
        [0, np.cos(ANGLE), -np.sin(ANGLE)],
        [0, np.sin(ANGLE), np.cos(ANGLE)],
    ]
)
S, T = 1.4, np.array([60.0, 40.0, 0.0])
FM_FIDUCIALS = [
    (10, 12, 2),
    (80, 15, 6),
    (20, 85, 9),
    (75, 70, 3),
    (50, 40, 8),
    (30, 55, 1),
]
POI = (45, 50, 5)


def _to_fib(point):
    x, y, z = point
    return S * R @ np.array([x, y, z * Z_SCALE]) + T


@pytest.fixture
def experiment(tmp_path):
    folder = tmp_path / "experiment"
    (folder / "01-lamella").mkdir(parents=True)
    return folder


@pytest.fixture
def client(microscope, experiment):
    context = SimpleNamespace(experiment_path=str(experiment))
    app = build_server(
        microscope,
        app_context=context,
        auth=AuthConfig.generate(level="watch", token=TOKEN),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers.update({"Authorization": f"Bearer {TOKEN}"})
        yield client


@pytest.fixture
def images(microscope, experiment):
    lamella = experiment / "01-lamella"
    fib = microscope.acquire_image(
        ImageSettings(beam_type=BeamType.ION, resolution=[384, 256], hfw=150e-6)
    )
    fib_path = str(lamella / "ref_fib.tif")
    fib.save(fib_path)
    metadata = FluorescenceImageMetadata(
        acquisition_datetime="2026-01-01T00:00:00",
        pixel_size_x=FM_PX,
        pixel_size_y=FM_PX,
        pixel_size_z=FM_PZ,
        resolution=(100, 100),
        channels=[
            FluorescenceChannelMetadata(
                name="reflection",
                excitation_wavelength=635.0,
                power=0.01,
                exposure_time=0.002,
                gain=None,
                offset=0.0,
            )
        ],
    )
    fm = FluorescenceImage(
        data=np.zeros((1, 10, 100, 100), dtype=np.uint16), metadata=metadata
    )
    fm_path = fm.save(str(lamella / "stack.ome.tiff"))
    return SimpleNamespace(lamella=str(lamella), fib=fib_path, fm=fm_path)


def _correlate(client, images, **extra):
    kwargs = {
        "fib_image_path": images.fib,
        "fm_image_path": images.fm,
        "fib_points": [_to_fib(p)[:2].tolist() for p in FM_FIDUCIALS],
        "fm_points": [list(map(float, p)) for p in FM_FIDUCIALS],
        "poi_points": [list(map(float, POI))],
        **extra,
    }
    return client.post(f"{CORRELATION}/queries/correlate", json={"kwargs": kwargs})


def test_correlate_at_watch_maps_the_target_and_saves_a_run(client, images):
    resp = _correlate(client, images, lamella_path=images.lamella)
    assert resp.status_code == 200, resp.text
    result = resp.json()["result"]

    assert result["pairs"] == 6
    assert result["rms_error_px"] < 0.01
    expected = _to_fib(POI)[:2]
    assert np.allclose(result["poi"][0]["image_px"], expected, atol=0.05)
    assert os.path.isfile(result["overlay_path"])
    runs = LamellaCorrelation.discover(correlation_dir(images.lamella)).runs
    assert [r.path for r in runs] == [result["run_path"]]


def test_without_a_lamella_nothing_is_saved(client, images):
    resp = _correlate(client, images)
    assert resp.status_code == 200, resp.text
    assert "run_path" not in resp.json()["result"]
    assert not os.path.exists(correlation_dir(images.lamella))


def test_predict_maps_more_points_through_a_saved_run(client, images):
    run_path = _correlate(client, images, lamella_path=images.lamella).json()["result"][
        "run_path"
    ]
    point = (60.0, 30.0, 4.0)

    resp = client.post(
        f"{CORRELATION}/queries/predict",
        json={"kwargs": {"run_path": run_path, "fm_points": [list(point)]}},
    )

    assert resp.status_code == 200, resp.text
    mapped = resp.json()["result"]["points"][0]["image_px"]
    assert np.allclose(mapped, _to_fib(point)[:2], atol=0.05)


def test_mismatched_pairs_are_refused(client, images):
    resp = client.post(
        f"{CORRELATION}/queries/correlate",
        json={
            "kwargs": {
                "fib_image_path": images.fib,
                "fm_image_path": images.fm,
                "fib_points": [[1.0, 2.0]] * 4,
                "fm_points": [[1.0, 2.0, 3.0]] * 3,
                "poi_points": [[1.0, 2.0, 3.0]],
            }
        },
    )
    assert resp.status_code == 422
    assert "pairs" in resp.text


@pytest.mark.parametrize(
    "points, said",
    [
        (
            dict(fib_points=[[1.0, 2.0]] * 3, fm_points=[[1.0, 2.0, 3.0]] * 3),
            "at least",
        ),
        (dict(poi_points=[]), "point of interest"),
    ],
)
def test_points_the_gate_refuses_are_refused_in_the_querys_terms(
    client, images, points, said
):
    resp = _correlate(client, images, **points)
    assert resp.status_code == 422
    assert said in resp.text


def test_a_beam_image_is_refused_as_the_fm_stack(client, images):
    """It would fit, unseeded and never judged (round 11 F6)."""
    resp = _correlate(client, images, fm_image_path=images.fib)
    assert resp.status_code == 422, resp.text
    assert "ref_fib.tif is a FIB image, not an FM stack" in resp.text


def test_paths_outside_the_experiment_are_refused(client, images, tmp_path):
    resp = _correlate(client, images, lamella_path=str(tmp_path))
    assert resp.status_code == 403, resp.text


def test_refine_point_answers_a_proposal(client, images):
    resp = client.post(
        f"{CORRELATION}/queries/refine_point",
        json={
            "kwargs": {
                "image_path": images.fm,
                "x": 50.0,
                "y": 50.0,
                "z": 5.0,
                "beam": "fm",
            }
        },
    )
    assert resp.status_code == 200, resp.text
    result = resp.json()["result"]
    assert result["method"] == "Gaussian"
    assert result["status"] in ("ok", "unchanged", "error")
    assert result["initial"] == [50.0, 50.0, 5.0]


def test_served_with_no_microscope(images, experiment):
    """An analysis service: the App Server serves it with the microscope's
    routes off and no microscope at all."""
    app = build_server(
        None,
        app_context=SimpleNamespace(experiment_path=str(experiment)),
        auth=AuthConfig.generate(level="watch", token=TOKEN),
        serve_microscope=False,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers.update({"Authorization": f"Bearer {TOKEN}"})
        assert client.get("/capabilities").json()["routers"]["services"] is True
        resp = _correlate(client, images)
        assert resp.status_code == 200, resp.text
        assert client.get("/microscope/services/correlation").status_code == 404


def test_the_agent_s_correlation_tools_reach_their_queries(client, images):
    """Each tool's arguments go to its query as the route takes them, wrapped in
    kwargs, as the sidecar sends them (sim session 2026-10-19 F4: they went flat
    and every call was refused)."""
    from fibsem.ai.agent.catalog import CATALOG, route_call

    tools = {spec.name: spec for spec in CATALOG}

    def call(tool, **arguments):
        method, path, body = route_call(tools[tool], arguments)
        resp = client.request(method, path, json=body)
        assert resp.status_code == 200, resp.text
        return resp.json()["result"]

    correlated = call(
        "correlation_correlate",
        fib_image_path=images.fib,
        fm_image_path=images.fm,
        fib_points=[_to_fib(p)[:2].tolist() for p in FM_FIDUCIALS],
        fm_points=[list(map(float, p)) for p in FM_FIDUCIALS],
        poi_points=[list(map(float, POI))],
        lamella_path=images.lamella,
    )
    assert correlated["pairs"] == 6
    predicted = call(
        "correlation_predict",
        run_path=correlated["run_path"],
        fm_points=[list(map(float, POI))],
    )
    assert np.allclose(predicted["points"][0]["image_px"], _to_fib(POI)[:2], atol=0.05)
    refined = call(
        "correlation_refine_point", image_path=images.fm, x=50.0, y=50.0, beam="fm"
    )
    assert refined["initial"][:2] == [50.0, 50.0]
    corrected = call(
        "correlation_correlate",
        fib_image_path=images.fib,
        fm_image_path=images.fm,
        fib_points=[_to_fib(p)[:2].tolist() for p in FM_FIDUCIALS],
        fm_points=[list(map(float, p)) for p in FM_FIDUCIALS],
        poi_points=[list(map(float, POI))],
        fm_surface_point=[50.0, 50.0, SURFACE_Z],
        ri={"n2": 1.4},
        ri_factor=1.3,
    )
    assert corrected["ri_correction"]["factor"] == pytest.approx(1.3)
    burns = call("correlation_find_spot_burns", lamella_path=images.lamella)
    assert burns["fractions"] == []


SURFACE_Z = 2.0


def _expected_pre(factor):
    x, y, z = POI
    return _to_fib((x, y, SURFACE_Z + (z - SURFACE_Z) * factor))[:2]


def test_an_fm_surface_point_corrects_the_targets_depth_before_the_fit(client, images):
    resp = _correlate(
        client, images, fm_surface_point=[50.0, 50.0, SURFACE_Z], ri_factor=1.3
    )
    assert resp.status_code == 200, resp.text
    result = resp.json()["result"]
    assert result["ri_correction"]["mode"] == "pre"
    assert result["ri_correction"]["factor"] == pytest.approx(1.3)
    assert np.allclose(result["poi"][0]["image_px"], _expected_pre(1.3), atol=0.05)


def test_without_a_factor_the_lookup_table_gives_it_at_tilt_0(client, images):
    from fibsem.correlation.config import RISettings
    from fibsem.correlation.refractive_index import lookup_zeta

    resp = _correlate(
        client,
        images,
        fm_surface_point=[50.0, 50.0, SURFACE_Z],
        ri={"n2": 1.4},
    )
    assert resp.status_code == 200, resp.text
    applied = resp.json()["result"]["ri_correction"]
    d = RISettings()
    zeta = lookup_zeta(0.0, d.depth_um, d.na, 1.4, d.wavelength_um)
    assert applied["factor"] == pytest.approx(zeta)
    assert applied["params"]["tilt_deg"] == 0.0
    assert applied["params"]["computed"] is True


def test_the_experiment_protocol_s_ri_settings_are_the_starting_point(
    client, images, experiment
):
    from fibsem.correlation.config import RISettings
    from fibsem.correlation.refractive_index import lookup_zeta

    (experiment / "experiment.yaml").write_text("positions: []\n")
    (experiment / "protocol.yaml").write_text("correlation:\n  ri:\n    na: 0.5\n")
    resp = _correlate(
        client,
        images,
        lamella_path=images.lamella,
        fm_surface_point=[50.0, 50.0, SURFACE_Z],
    )
    assert resp.status_code == 200, resp.text
    d = RISettings()
    zeta = lookup_zeta(0.0, d.depth_um, 0.5, d.n2, d.wavelength_um)
    assert resp.json()["result"]["ri_correction"]["factor"] == pytest.approx(zeta)


def test_a_fib_surface_point_moves_the_target_after_the_fit(client, images):
    plain = _correlate(client, images).json()["result"]["poi"][0]
    resp = _correlate(client, images, fib_surface_point=[190.0, 60.0], ri_factor=1.3)
    assert resp.status_code == 200, resp.text
    applied = resp.json()["result"]["ri_correction"]
    assert applied["mode"] == "post"
    assert np.allclose(applied["poi_uncorrected"]["image_px"], plain["image_px"])
    moved = resp.json()["result"]["poi"][0]["image_px"]
    assert not np.allclose(moved, plain["image_px"])


@pytest.mark.parametrize(
    "extra, said",
    [
        (
            dict(fm_surface_point=[1.0, 2.0, 3.0], fib_surface_point=[1.0, 2.0]),
            "one surface point",
        ),
        (dict(ri_factor=1.3), "surface point"),
        (
            dict(fm_surface_point=[1.0, 2.0, 3.0], ri={"index": 1.3}),
            "not index",
        ),
    ],
)
def test_ri_arguments_that_cannot_apply_are_refused(client, images, extra, said):
    resp = _correlate(client, images, **extra)
    assert resp.status_code == 422, resp.text
    assert said in resp.text


SPOT_BURNS = """\
positions:
- petname: 01-lamella
  path: {path}
  task_config:
    SPOT_BURN_FIDUCIAL:
      task_type: SPOT_BURN_FIDUCIAL
      coordinates:
      - {{x: 0.25, y: 0.5}}
      - {{x: 0.75, y: 0.5}}
      reference_imaging:
        field_of_view1: {hfw}
"""


def _spot_burns(client, images, **extra):
    return client.post(
        f"{CORRELATION}/queries/spot_burns",
        json={"kwargs": {"lamella_path": images.lamella, **extra}},
    )


def test_spot_burns_are_found_and_placed_on_a_fib_image_of_their_width(
    client, images, experiment
):
    (experiment / "experiment.yaml").write_text(
        SPOT_BURNS.format(path=images.lamella, hfw=150e-6)
    )
    resp = _spot_burns(client, images, fib_image_path=images.fib)
    assert resp.status_code == 200, resp.text
    found = resp.json()["result"]
    assert found["fractions"] == [[0.25, 0.5], [0.75, 0.5]]
    assert found["fib_px"] == [[96.0, 128.0], [288.0, 128.0]]
    assert found["reason"] == ""


def test_spot_burns_placed_at_another_width_are_not_placed(client, images, experiment):
    (experiment / "experiment.yaml").write_text(
        SPOT_BURNS.format(path=images.lamella, hfw=100e-6)
    )
    found = _spot_burns(client, images, fib_image_path=images.fib).json()["result"]
    assert found["fib_px"] is None
    assert "100 µm" in found["reason"]


def test_no_spot_burns_says_why(client, images):
    found = _spot_burns(client, images).json()["result"]
    assert found["fractions"] == []
    assert "experiment.yaml" in found["reason"]
