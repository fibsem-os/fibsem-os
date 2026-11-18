"""The simulator writes its sample's record after every milling pass when
FIBSEM_SIM_RECORD names a file: a live eval's truth about a run."""

import json
import os

import numpy as np

import fibsem.config as cfg
from fibsem import utils
from fibsem.drivers.demo.simulator import SIM_RECORD_ENV
from fibsem.structures.settings import FibsemRectangleSettings

CONFIG = os.path.join(cfg.SHIPPED_CONFIG_PATH, "microscope-configuration.yaml")


def test_each_milling_pass_rewrites_the_record_file(tmp_path, monkeypatch):
    path = tmp_path / "sim-record.json"
    monkeypatch.setenv(SIM_RECORD_ENV, str(path))
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=CONFIG)
    try:
        microscope.system.sim["sample"] = {"enabled": True}
        microscope._connection.setup_sample_scene()
        pose = microscope.get_stage_position()
        pose.t = np.deg2rad(12.0)
        microscope.move_stage_absolute(pose)
        for n in (1, 2):
            microscope.draw_pattern(
                FibsemRectangleSettings(
                    width=10e-6,
                    height=3e-6,
                    depth=1e-6,
                    centre_x=0.0,
                    centre_y=5e-6 * n,
                )
            )
            microscope.run_milling()
            record = json.loads(path.read_text())
            assert len(record["passes"]) == n
        assert not (tmp_path / "sim-record.json.part").exists()
    finally:
        microscope.disconnect()


def test_no_file_without_the_setting(tmp_path, monkeypatch):
    monkeypatch.delenv(SIM_RECORD_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=CONFIG)
    try:
        microscope.system.sim["sample"] = {"enabled": True}
        microscope._connection.setup_sample_scene()
        microscope.draw_pattern(
            FibsemRectangleSettings(
                width=10e-6, height=3e-6, depth=1e-6, centre_x=0.0, centre_y=0.0
            )
        )
        microscope.run_milling()
        assert list(tmp_path.iterdir()) == []
    finally:
        microscope.disconnect()
