"""A Demo microscope that is a compustage, or not, by its configuration.

A real backend learns its stage type at connect, and the Demo learns it from
`sim.is_compustage`, its stand-in for that probe, which its stage device keeps. A test
that wants one stage type or the other builds it this way.
"""

import os
import tempfile
from typing import Optional, Tuple

import fibsem.config as cfg
from fibsem import utils


def demo_session(
    compustage: bool,
    config: str = "microscope-configuration.yaml",
    sim: Optional[dict] = None,
) -> Tuple["FibsemMicroscope", "MicroscopeSettings"]:  # noqa: F821
    """`utils.setup_session` for the Demo, from *config* with `sim.is_compustage`
    set and any other `sim:` keys in *sim*. The file is written to a directory that
    outlives the call, because the microscope keeps its path to write calibrations
    back to."""
    data = utils.load_yaml(os.path.join(cfg.CONFIG_PATH, config))
    data["sim"] = {
        **(data.get("sim") or {}),
        **(sim or {}),
        "is_compustage": compustage,
    }
    path = os.path.join(tempfile.mkdtemp(prefix="demo-stage-"), config)
    utils.save_yaml(path, data)
    return utils.setup_session(config_path=path, manufacturer="Demo")
