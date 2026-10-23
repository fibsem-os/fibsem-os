"""An AutoLamella protocol from before task protocols, as a task protocol.

The old file named a method (``options.method``: on-grid, trench or waffle), who
supervised each stage (``options.supervise``, or ``supervision`` by stage name) and
each stage's milling under ``milling``. Its milling came in three shapes, newest
first: a list of stage dicts; ``{stages: [...]}`` with each stage's milling and
pattern settings flat in one block; or one flat block. The oldest kept rough and
polishing milling together under ``lamella``, the last
``options.num_polishing_stages`` of them polishing.

The method's stages become the tasks a task protocol runs: Trench Milling,
Undercut, Setup Lamella Position, Mill Fiducial, Rough Milling (with the
microexpansion stages, and the notch as its stress relief) and Polishing. The
lift-out methods have no tasks to become, and are refused.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

# The milling keys the old file used.
FIDUCIAL = "fiducial"
MICROEXPANSION = "microexpansion"
MILL_POLISHING = "mill_polishing"
MILL_ROUGH = "mill_rough"
NOTCH = "notch"
TRENCH = "trench"
UNDERCUT = "undercut"
STRESS_RELIEF = "stress-relief"

# The pattern a milling key meant when its block named none (no ``type``).
PATTERN_BY_KEY = {
    "cut": "Rectangle",
    "fiducial": "Fiducial",
    "flatten": "Rectangle",
    "undercut": "Undercut",
    "horseshoe": "Horseshoe",
    "lamella": "Trench",
    "polish_lamella": "Trench",
    "thin_lamella": "Trench",
    "sever": "Rectangle",
    "sharpen": "Rectangle",
    "needle": "Rectangle",
    "copper_release": "Horseshoe",
    "serial_trench": "Horseshoe",
    "serial_undercut": "Rectangle",
    "serial_sever": "Rectangle",
    "lamella_sever": "Rectangle",
    "lamella_polish": "Trench",
    "trench": "Trench",
    "notch": "WaffleNotch",
    "microexpansion": "MicroExpansion",
    "clover": "Clover",
    "autolamella": "Trench",
    "MillUndercut": "Rectangle",
    "rectangle": "Rectangle",
    "MillRough": "Trench",
    "MillRegularCut": "Trench",
    "MillPolishing": "Trench",
    "mill_rough": "Trench",
    "mill_polishing": "Trench",
}

# What a flat stage block held for its milling, rather than its pattern.
MILLING_KEYS = (
    "dwell_time",
    "hfw",
    "milling_current",
    "milling_voltage",
    "patterning_mode",
    "preset",
    "rate",
    "spacing",
    "spot_size",
    "application_file",
    "milling_channel",
)

ON_GRID = "AutoLamella-OnGrid"
TRENCH_METHOD = "AutoLamella-Trench"
WAFFLE = "AutoLamella-Waffle"
METHODS = {
    ON_GRID: ("autolamella-on-grid", "on-grid", "autoliftout"),
    TRENCH_METHOD: ("autolamella-trench", "trench"),
    WAFFLE: ("autolamella-waffle", "waffle"),
}
LIFTOUT_METHODS = (
    "autolamella-liftout",
    "liftout",
    "autolamella-serial-liftout",
    "serial-liftout",
)

# A stage's supervision, by its name in ``supervision`` and its key in
# ``options.supervise``.
SETUP_LAMELLA = ("SetupLamella", "setup_lamella")
ROUGH = ("MillRough", "mill_rough")
POLISHING = ("MillPolishing", "mill_polishing")
TRENCH_STAGE = ("MillTrench", "trench")
UNDERCUT_STAGE = ("MillUndercut", "undercut")

ROUGH_TASK = "Rough Milling"
POLISHING_TASK = "Polishing"
TRENCH_TASK = "Trench Milling"
FIDUCIAL_TASK = "Mill Fiducial"
UNDERCUT_TASK = "Undercut"
SETUP_TASK = "Setup Lamella Position"


def is_pre_task(data: Any) -> bool:
    """Whether ``data`` is a protocol from before task protocols."""
    return (
        isinstance(data, dict)
        and isinstance(data.get("milling"), dict)
        and "task_config" not in data
        and "tasks" not in data
    )


def upgrade(data: Dict[str, Any]) -> Dict[str, Any]:
    """The task protocol ``data``, a pre-task protocol, becomes. Raises
    ValueError for a method with no tasks, or for milling its method needs and
    the file does not have."""
    data = copy.deepcopy(data)
    options = data.get("options") or {}
    name = data.get("name") or options.get("name") or "Default-AutoLamella-Protocol"
    method = _method(data.get("method") or options.get("method") or ON_GRID)
    milling = _milling(data["milling"], options)

    def stages(key: str) -> List[Any]:
        from fibsem.milling import FibsemMillingStage

        if not milling.get(key):
            raise ValueError(
                f"The protocol has no {key!r} milling, which {method} uses."
            )
        return [FibsemMillingStage.from_dict(stage) for stage in milling[key]]

    def task(key: str, stage_list: List[Any], title: str) -> Dict[str, Any]:
        from fibsem.milling.tasks import FibsemMillingTaskConfig

        return {
            key: FibsemMillingTaskConfig.from_stages(stage_list, name=title).to_dict()
        }

    def attention(stage: tuple) -> str:
        supervised = _supervised(data, options, stage)
        return "supervised" if supervised else "automated"

    task_config: Dict[str, Dict[str, Any]] = {}
    plan: List[Dict[str, Any]] = []
    if method in (ON_GRID, WAFFLE):
        rough = stages(MILL_ROUGH)
        if options.get("use_microexpansion", True):
            rough += stages(MICROEXPANSION)
        rough_milling = task(MILL_ROUGH, rough, "Rough Milling")
        if options.get("use_notch", False):
            rough_milling.update(task(STRESS_RELIEF, stages(NOTCH), "Notch"))
        task_config[ROUGH_TASK] = _task("MILL_ROUGH", ROUGH_TASK, rough_milling)
        task_config[POLISHING_TASK] = _task(
            "MILL_POLISHING",
            POLISHING_TASK,
            task(MILL_POLISHING, stages(MILL_POLISHING), "Polishing"),
        )
        task_config[FIDUCIAL_TASK] = _task(
            "MILL_FIDUCIAL",
            FIDUCIAL_TASK,
            task(FIDUCIAL, stages(FIDUCIAL), "Fiducial"),
        )
        task_config[SETUP_TASK] = _task(
            "SELECT_MILLING_POSITION",
            SETUP_TASK,
            {},
            milling_angle=_milling_angle(options),
        )
        plan = [
            _entry(SETUP_TASK, attention(SETUP_LAMELLA)),
            _entry(FIDUCIAL_TASK, attention(SETUP_LAMELLA)),
            _entry(ROUGH_TASK, attention(ROUGH), [FIDUCIAL_TASK]),
            _entry(POLISHING_TASK, attention(POLISHING), [ROUGH_TASK]),
        ]
    if method in (TRENCH_METHOD, WAFFLE):
        task_config[TRENCH_TASK] = _task(
            "MILL_TRENCH",
            TRENCH_TASK,
            task(TRENCH, stages(TRENCH), "Trench"),
            orientation="FIB",
        )
        plan.insert(0, _entry(TRENCH_TASK, attention(TRENCH_STAGE)))
    if method == WAFFLE:
        task_config[UNDERCUT_TASK] = _task(
            "MILL_UNDERCUT",
            UNDERCUT_TASK,
            task(UNDERCUT, stages(UNDERCUT), "Undercut"),
            orientation="SEM",
        )
        plan.insert(1, _entry(UNDERCUT_TASK, attention(UNDERCUT_STAGE), [TRENCH_TASK]))

    description = f"auto-converted protocol from {name} - {method}"
    return {
        "name": name,
        "description": description,
        "task_config": task_config,
        "workflow_config": {"name": name, "description": description, "tasks": plan},
        "options": {"turn_beams_off": bool(options.get("turn_beams_off", False))},
    }


def _method(name: str) -> str:
    key = str(name).lower()
    for method, aliases in METHODS.items():
        if key == method.lower() or key in aliases:
            return method
    if key in LIFTOUT_METHODS:
        raise ValueError(
            f"A {name} protocol has no task protocol to become: lift-out is not "
            "a task workflow."
        )
    raise ValueError(f"Unknown method {name!r} in the protocol.")


def _supervised(data: dict, options: dict, stage: tuple) -> bool:
    by_name, key = stage
    if isinstance(data.get("supervision"), dict):
        return bool(data["supervision"].get(by_name, True))
    return bool((options.get("supervise") or {}).get(key, True))


def _milling_angle(options: dict) -> float:
    for key in ("milling_angle", "milling_tilt_angle", "lamella_tilt_angle"):
        if key in options:
            return options[key]
    return 18


def _milling(milling: dict, options: dict) -> Dict[str, list]:
    """Each key's milling as a list of stage dicts."""
    milling = dict(milling)
    if "lamella" in milling:
        lamella = milling.pop("lamella")["stages"]
        polishing = options.get("num_polishing_stages", 1)
        milling.setdefault(MILL_ROUGH, {"stages": lamella[:-polishing]})
        milling.setdefault(MILL_POLISHING, {"stages": lamella[-polishing:]})
    converted = {}
    for key, value in milling.items():
        if isinstance(value, list):
            converted[key] = value
        elif isinstance(value, dict):
            point = value.get("point")
            blocks = value["stages"] if "stages" in value else [value]
            converted[key] = [_stage(block, key, point) for block in blocks]
    return converted


def _stage(block: dict, key: str, point: Optional[dict]) -> Dict[str, Any]:
    """A flat stage block as a stage dict. With no ``type``, the pattern the
    milling key always meant."""
    pattern_name = block.get("type")
    if pattern_name is None:
        pattern_name = PATTERN_BY_KEY.get(key)
        if pattern_name is None:
            raise ValueError(f"The {key!r} milling names no pattern ('type').")
    skip = (*MILLING_KEYS, "strategy", "name", "type")
    pattern = {k: v for k, v in block.items() if k not in skip}
    pattern["name"] = pattern_name
    if pattern_name in ("Trench", "Horseshoe"):
        pattern = _trench(pattern)
    if point is not None:
        pattern["point"] = point
    return {
        "name": block.get("name", key),
        "milling": {k: block[k] for k in MILLING_KEYS if k in block},
        "strategy": block.get("strategy") or {},
        "pattern": pattern,
    }


def _trench(pattern: dict) -> dict:
    """A trench's lamella size, offset and size ratio as its trench heights and
    spacing."""
    ratio = pattern.pop("size_ratio", 1.0)
    height = pattern.pop("trench_height")
    pattern["width"] = pattern.pop("lamella_width")
    pattern["spacing"] = pattern.pop("lamella_height") + 2 * pattern.pop("offset", 0)
    pattern["upper_trench_height"] = height / max(ratio, 1.0)
    pattern["lower_trench_height"] = height * min(ratio, 1.0)
    return pattern


def _task(
    task_type: str, task_name: str, milling: Dict[str, Any], **fields: Any
) -> Dict[str, Any]:
    return {
        "task_type": task_type,
        "task_name": task_name,
        "milling": milling,
        **fields,
    }


def _entry(
    name: str, attention: str, requires: Optional[List[str]] = None
) -> Dict[str, Any]:
    return {"name": name, "attention": attention, "requires": list(requires or [])}
