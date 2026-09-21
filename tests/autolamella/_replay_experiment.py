"""An experiment directory recorded by the real code, for the replay tests.

The replay reads records that the microscope, task and milling code write as a
side effect of logging, so the only test that can catch one of them changing
shape is one that runs them. This runs a real task, a real milling task and a
real spot burn on the Demo microscope, with the app's own logging set up the
way an experiment sets it up, and leaves the directory behind.
"""

import logging
import os
import time
from pathlib import Path

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.structures import Lamella
from fibsem.applications.autolamella.workflows.tasks.select_position import (
    SelectMillingPositionTask,
    SelectMillingPositionTaskConfig,
)
from fibsem.fm.structures import FluorescenceImage
from fibsem.imaging.spot import SpotBurnSettings
from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.tasks import FibsemMillingTask, FibsemMillingTaskConfig
from fibsem.structures import BeamType, Point

LAMELLA = "01-test"
MILLING_TASK = "Rough Milling"
MILLING_STAGES = ("Rough Mill 01", "Rough Mill 02")
SPOTS = (Point(0.25, 0.5), Point(0.75, 0.5))
FM_STACK = f"{LAMELLA}-zstack.ome.tiff"
FM_PLANES, FM_CHANNELS = 3, 2


def record_demo_experiment(root: Path, monkeypatch) -> Path:
    """Run a task, a mill and a spot burn on Demo, logging into *root*, and save
    an FM z-stack the way an acquisition does (the Demo instrument has no FM)."""
    root.mkdir(parents=True, exist_ok=True)
    microscope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml"),
    )
    # After the session: setup_session configures logging of its own, and an
    # experiment's logfile is set up once the microscope is connected.
    handlers, level = logging.getLogger().handlers[:], logging.getLogger().level
    utils.configure_logging(path=str(root))
    try:
        lamella = Lamella(path=root / LAMELLA, number=1, petname="test")
        lamella.path.mkdir(parents=True, exist_ok=True)
        lamella.milling_pose = microscope.get_microscope_state()
        SelectMillingPositionTask(
            microscope=microscope,
            config=SelectMillingPositionTaskConfig(use_autofocus=False),
            lamella=lamella,
        ).run()

        milling = FibsemMillingTaskConfig.from_stages(
            stages=[FibsemMillingStage(name=name) for name in MILLING_STAGES],
            name=MILLING_TASK,
        )
        milling.alignment.enabled = False
        FibsemMillingTask(microscope, milling).run()

        FluorescenceImage.generate_blank_image(
            resolution=(64, 48), zlevels=FM_PLANES, n_channels=FM_CHANNELS, random=True
        ).save(str(lamella.path / FM_STACK))

        with monkeypatch.context() as m:
            m.setattr(time, "sleep", lambda *_: None)  # the exposure countdown
            microscope.run_spot_burn(
                settings=SpotBurnSettings(
                    coordinates=list(SPOTS), exposure_time=1.0, milling_current=1e-10
                ),
                beam_type=BeamType.ION,
            )
    finally:
        microscope.disconnect()
        for handler in logging.getLogger().handlers:
            handler.close()
        logging.basicConfig(handlers=handlers, level=level, force=True)
    return root
