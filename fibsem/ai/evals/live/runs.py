"""A live case's part in a run (``fibsem.ai.evals.runner``): setting the
run up as the operator would before the agent starts, and recording what
the run did once it ends, to score."""

import sys
from pathlib import Path
from typing import Any, Dict, List

from fibsem.ai.evals.cases import Case
from fibsem.ai.evals.live.client import LiveClient
from fibsem.image_format import read_metadata_dict

# The run's record, beside its result: what the experiment holds at the end.
RECORD = "record.json"
# Which case's task a scripted agent runs, and the case's folder, for its
# ``sim_run`` set-up (a scripted agent reads nothing else of the case)
TASK_ENV = "FIBSEM_EVAL_TASK"
CASE_ENV = "FIBSEM_EVAL_CASE"


def scripted_command() -> List[str]:
    """The scripted agent, run in fibsem-agent's place (``run --scripted``)."""
    return [sys.executable, "-m", "fibsem.ai.evals.live.scripted"]


def set_up(case: Case, found_at: Path, home: Path) -> None:
    """The case's ``sim_run`` set-up, as the operator: the agent's level,
    then each tool call in ``setup`` (a grid loaded, the stage moved)."""
    sim_run = case.data["sim_run"]
    with LiveClient.as_operator(found_at, home) as operator:
        operator.set_level(sim_run.get("level", "watch"))
        for step in sim_run.get("setup") or []:
            operator.call(step["tool"], **(step.get("arguments") or {}))


def record(found_at: Path, home: Path, experiment: Path) -> Dict[str, Any]:
    """What the run left in its experiment: each item's name, id and the
    stage position of each of its poses, and each SEM or FIB image saved."""
    items = []
    with LiveClient.as_operator(found_at, home) as operator:
        listed = operator.call("item_list") or {}
        for row in listed.get("items") or []:
            name = row.get("item_name")
            detail = operator.call("item_get", item_name=name) or {}
            items.append(
                {
                    "name": name,
                    "id": row.get("item_id"),
                    "poses": detail.get("poses") or {},
                }
            )
    return {
        "items": items,
        "images": saved_images(experiment),
        "overviews": overviews(experiment),
    }


def overviews(experiment: Path) -> List[str]:
    """The stitched overviews saved in *experiment* (not their tiles)."""
    return [
        path.relative_to(experiment).as_posix()
        for path in sorted(Path(experiment).rglob("*.tif"))
        if "overview" in path.name and not path.name.startswith("tile_")
    ]


# Folders of images that are steps of something else: an overview's tiles,
# an autofocus sweep's
STEPS = ("overview-", "autofocus")


def saved_images(experiment: Path) -> List[Dict[str, Any]]:
    """Each SEM or FIB image saved in *experiment* on its own: its path, beam,
    field, pixel size, working distance and stage position."""
    import tifffile

    found = []
    for path in sorted(Path(experiment).rglob("*_[ei]b.tif")):
        rel = path.relative_to(experiment)
        if any(part.startswith(STEPS) for part in rel.parts[:-1]):
            continue
        try:
            with tifffile.TiffFile(path) as tif:
                metadata = read_metadata_dict(tif) or {}
        except Exception:  # noqa: BLE001 - an unreadable file is left out
            continue
        image = metadata.get("image") or {}
        state = metadata.get("microscope_state") or {}
        beam = image.get("beam_type")
        settings = (
            state.get("electron_beam" if beam == "ELECTRON" else "ion_beam") or {}
        )
        found.append(
            {
                "path": rel.as_posix(),
                "beam": beam,
                "hfw": image.get("hfw"),
                "pixel_size": (metadata.get("pixel_size") or {}).get("x"),
                "working_distance": settings.get("working_distance"),
                "stage_position": state.get("stage_position"),
                "time": state.get("timestamp"),
            }
        )
    return found
