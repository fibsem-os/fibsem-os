"""One Coincidence Polishing pass on the Demo microscope.

The FIB image is a synthetic band and the fluorescence frames carry a target
spot and a control spot, so the pass's measurements can be checked against
what was drawn: the band's edges and width, the target refitted where it is,
and its signal against the control and against an earlier pass.
"""

import json
import math
import os

import numpy as np
import pytest
import yaml

import fibsem.config as fconfig
from fibsem import utils
from fibsem.fm.structures import ChannelSettings, EmissionFilter
from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.patterning.patterns import RectanglePattern
from fibsem.milling.strategy import get_strategy_names
from fibsem.milling.strategy.coincidence_polishing import (
    CoincidencePolishingConfig,
    CoincidencePolishingStrategy,
)
from fibsem.milling.tasks import FibsemMillingTaskConfig, run_milling_task
from fibsem.structures.image import FibsemImage
from fibsem.structures.settings import BeamType, FibsemMillingSettings

SIM_ARCTIS_CONFIG_PATH = os.path.join(
    fconfig.SHIPPED_CONFIG_PATH, "sim-arctis-configuration.yaml"
)

BAND_NM = 480.0
FIB_PIXEL = 15e-6 / 1536  # 9.8 nm
TARGET = (120.0, 90.0)
CONTROL = (60.0, 40.0)


@pytest.fixture
def fm_microscope(tmp_path):
    """A Demo microscope with a fluorescence module and the sample scene off."""
    with open(SIM_ARCTIS_CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    config.setdefault("sim", {}).setdefault("sample", {})["enabled"] = False
    path = tmp_path / "sim-arctis-configuration.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=str(path))
    assert microscope.fm is not None
    yield microscope
    microscope.disconnect()


def _band_image() -> FibsemImage:
    """A horizontal bright band BAND_NM thick across the image's centre."""
    width, height = 1536, 1024
    image = FibsemImage.generate_blank_image(
        resolution=(width, height), hfw=width * FIB_PIXEL
    )
    ys = np.arange(height)[:, None]
    half = BAND_NM * 1e-9 / FIB_PIXEL / 2
    centre = height / 2
    cover = np.clip(
        np.minimum(ys + 0.5, centre + half) - np.maximum(ys - 0.5, centre - half), 0, 1
    )
    data = np.broadcast_to(40 + cover * 160, (height, width)).copy()
    data = np.where(np.abs(ys - centre) < 150, data, 120)
    data += np.random.default_rng(0).normal(0, 2.0, data.shape)
    image.data = np.clip(data, 0, 255).astype(np.uint8)
    return image


@pytest.fixture
def scene(fm_microscope, monkeypatch):
    """The FIB answers the band; each FM frame has the target, brightest at the
    objective's starting position, and a steady control. ``scene["target"]``
    scales the target, as milling would fade it."""
    state = {"target": 1.0}
    real_acquire = fm_microscope.acquire_image

    def acquire_image(image_settings=None, beam_type=None):
        if beam_type is BeamType.ION:
            return _band_image()
        return real_acquire(image_settings=image_settings, beam_type=beam_type)

    fm = fm_microscope.fm
    z0 = fm.objective.position
    real_fm_acquire = fm.acquire_image

    def fm_acquire(channel_settings=None):
        image = real_fm_acquire(channel_settings)
        shape = np.asarray(image.data).shape[-2:]
        yy, xx = np.mgrid[: shape[0], : shape[1]]
        dz = (fm.objective.position - z0) / 0.6e-6
        focus = math.exp(-(dz**2))

        def spot(x, y, amplitude):
            return amplitude * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 2.0**2))

        frame = 100 + spot(*TARGET, 2000 * state["target"] * focus)
        frame = frame + spot(*CONTROL, 1000 * focus)
        image.data = frame.astype(np.uint16).reshape(np.asarray(image.data).shape)
        return image

    monkeypatch.setattr(fm_microscope, "acquire_image", acquire_image)
    monkeypatch.setattr(fm, "acquire_image", fm_acquire)
    return state


def _task(path, **config) -> FibsemMillingTaskConfig:
    strategy = CoincidencePolishingStrategy(config=CoincidencePolishingConfig(**config))
    task = FibsemMillingTaskConfig(
        name="Polish",
        field_of_view=50e-6,
        stages=[
            FibsemMillingStage(
                name="Polish Top 01",
                milling=FibsemMillingSettings(milling_current=100e-12, hfw=50e-6),
                pattern=RectanglePattern(width=9e-6, height=0.5e-6, depth=0.5e-6),
                strategy=strategy,
            )
        ],
    )
    task.alignment.enabled = False
    task.acquisition.acquire_fib = False
    task.acquisition.acquire_sem = False
    task.acquisition.imaging.path = str(path)
    return task


def _gfp() -> ChannelSettings:
    return ChannelSettings(
        name="gfp",
        excitation_wavelength=470,
        emission_filter=EmissionFilter("Fluorescence", multi_band=True),
        power=0.1,
        exposure_time=0.1,
    )


def _pass(microscope, path, **config):
    """Run one pass; the stage's strategy record, as the milling event has it."""
    records = []

    def keep(kind, payload):
        if kind == "milling":
            records.append(payload)

    microscope.session.record_signal.connect(keep)
    try:
        run_milling_task(microscope, _task(path, **config))
    finally:
        microscope.session.record_signal.disconnect(keep)
    (record,) = records
    assert record["status"] == "completed", record
    return record["stages"][0]["strategy"]


def test_the_config_round_trips_on_a_stage():
    """An agent sends the stage as a dict: the strategy and its pass settings
    come back from it."""
    task = _task(
        "/tmp",
        face="bottom",
        targets=[[1.5, 2.5]],
        control=[3.0, 4.0],
        target_channel=_gfp(),
        objective_position=6e-3,
    )
    stage = FibsemMillingStage.from_dict(task.stages[0].to_dict())
    config = stage.strategy
    assert isinstance(config, CoincidencePolishingConfig)
    assert config.face == "bottom"
    assert config.targets == [[1.5, 2.5]]
    assert config.control == [3.0, 4.0]
    assert config.target_channel.name == "gfp"
    assert config.objective_position == 6e-3
    assert config.reflection_channel.name == "reflection"


def test_it_is_found_by_name_but_kept_out_of_the_protocol_selectors():
    assert "Coincidence Polishing" not in get_strategy_names()
    assert CoincidencePolishingStrategy.selectable is False


def test_a_pass_mills_images_and_measures(fm_microscope, scene, tmp_path):
    said = _pass(
        fm_microscope,
        tmp_path,
        face="top",
        target_channel=_gfp(),
        targets=[[TARGET[0] + 2, TARGET[1] - 1]],  # a little off, as drift leaves it
        control=list(CONTROL),
    )
    facts = said["facts"]
    assert said["name"] == "Coincidence Polishing"
    assert facts["Face"] == "top"

    # The band: its width, and its edges either side of the image centre, in
    # the image coordinates a pattern is placed in (+y up).
    assert facts["Band FWHM (nm)"] == pytest.approx(BAND_NM, abs=2 * FIB_PIXEL * 1e9)
    top, bottom = facts["Top edge y (µm)"], facts["Bottom edge y (µm)"]
    assert top == pytest.approx(BAND_NM / 2e3, abs=0.02)
    assert bottom == pytest.approx(-BAND_NM / 2e3, abs=0.02)

    # The target refitted onto the spot, and read against the control.
    assert facts["Target 1 fit"] == "ok"
    assert facts["Target 1 moved (px)"] == pytest.approx(math.hypot(2, 1), abs=0.5)
    assert facts["Target 1 ratio to control"] == pytest.approx(2.0, rel=0.15)
    assert "Target 1 change from baseline (%)" not in facts

    # Every file the agent may open itself is there.
    for key in ("FIB image", "Reflection stack", "Target stack", "Pass data"):
        assert os.path.isfile(facts[key]), key
    assert os.path.isfile(said["figure"])
    with open(facts["Pass data"]) as f:
        data = json.load(f)
    assert data["face"] == "top"
    (target,) = data["targets"]
    assert target["point"] == pytest.approx(list(TARGET), abs=0.5)
    assert data["band"]["thickness_nm"]["median"] == facts["Band FWHM (nm)"]


def test_the_next_pass_reads_its_change_from_the_baseline(
    fm_microscope, scene, tmp_path
):
    settings = dict(
        target_channel=_gfp(),
        targets=[list(TARGET)],
        control=list(CONTROL),
        acquire_reflection_stack=False,
    )
    first = _pass(fm_microscope, tmp_path / "1", **settings)
    scene["target"] = 0.5  # the pass milled into it
    second = _pass(
        fm_microscope,
        tmp_path / "2",
        face="bottom",
        baseline=first["facts"]["Pass data"],
        **settings,
    )
    assert second["facts"]["Face"] == "bottom"
    assert second["facts"]["Target 1 change from baseline (%)"] == pytest.approx(
        -50, abs=8
    )


def test_a_failed_fit_is_reported_not_raised(fm_microscope, scene, tmp_path):
    """A target where there is no spot: the pass still answers, saying so."""
    said = _pass(
        fm_microscope,
        tmp_path,
        target_channel=_gfp(),
        targets=[[5.0, 5.0]],
        fit_cutout_px=3,
        acquire_reflection_stack=False,
        acquire_fib_image=False,
    )
    assert said["facts"]["Target 1 fit"] in ("ok", "failed")
    assert "Target 1 signal" in said["facts"]


def test_each_look_can_be_switched_off(fm_microscope, scene, tmp_path):
    said = _pass(
        fm_microscope,
        tmp_path,
        acquire_fib_image=False,
        acquire_reflection_stack=False,
    )
    facts = said["facts"]
    assert facts["Face"] == "top"
    assert not any(
        key in facts for key in ("FIB image", "Reflection stack", "Target stack")
    )
    assert os.path.isfile(facts["Pass data"])
    assert "figure" not in said


def test_the_objective_is_put_back_before_the_stacks(fm_microscope, scene, tmp_path):
    objective = fm_microscope.fm.objective
    home = objective.position
    objective.move_absolute(home - 90e-9)  # drifted while milling
    said = _pass(
        fm_microscope,
        tmp_path,
        objective_position=home,
        acquire_fib_image=False,
    )
    assert said["facts"]["Objective moved (nm)"] == pytest.approx(90, abs=1)
    assert objective.position == pytest.approx(home)


def test_a_face_that_is_neither_top_nor_bottom_is_refused(fm_microscope, tmp_path):
    strategy = CoincidencePolishingStrategy(CoincidencePolishingConfig(face="left"))
    stage = _task(tmp_path).stages[0]
    with pytest.raises(ValueError, match="top or bottom"):
        strategy.run(fm_microscope, stage)


def test_the_agent_path_runs_a_pass_through_the_milling_service(
    fm_microscope, scene, tmp_path
):
    """milling_run_stage: the stage as a dict, through the milling service, with
    the FM stacks taken inside its run. The answer carries the pass report."""
    stage = _task(
        tmp_path,
        target_channel=_gfp(),
        targets=[list(TARGET)],
        control=list(CONTROL),
    ).stages[0]
    answer = fm_microscope.milling.run_stage(
        FibsemMillingStage.from_dict(stage.to_dict()), path=str(tmp_path)
    )
    assert answer["status"] == "completed", answer
    said = answer["stages"][0]["strategy"]
    assert said["facts"]["Target 1 fit"] == "ok"
    assert os.path.isfile(said["facts"]["Target stack"])
    assert said["facts"]["Pass data"].startswith(str(tmp_path))
