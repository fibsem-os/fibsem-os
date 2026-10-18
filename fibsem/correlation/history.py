"""A lamella's correlation history — the on-disk run folders.

Each correlation that records a result writes a run to
``<lamella>/Correlation/<timestamp>/correlation.json`` (FIB-264). This names and
writes those folders (:func:`new_run_dir`, :func:`save_run`) and reconstructs the
history from them so a new correlation can seed from the previous one (FIB-299).
Nothing else is persisted — the history *is* the folders.

It also finds what a folder on disk holds for a correlation: the experiment and
lamella above a run (:func:`experiment_and_lamella_dirs`), the lamella's
spot-burn pattern (:func:`find_spot_burns`), and the images and saved files a
project folder opens with (:func:`discover_project`). Pure functions over the
file system, no Qt: the correlation tab, its launcher, the protocol editor and
the correlation service read the same folders the same way.
"""

from __future__ import annotations

import datetime
import glob
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from fibsem.constants import DATETIME_FILE
from fibsem.correlation.structures import (
    CorrelationResult,
    CorrelationState,
    load_correlation_file,
)
from fibsem.structures import Point

CORRELATION_DIRNAME = "Correlation"
CORRELATION_FILENAME = "correlation.json"
EXPERIMENT_FILENAME = "experiment.yaml"

# The project file before it was consolidated (FIB-264): still *read* for
# back-compat with projects saved by earlier versions; nothing writes them now.
LEGACY_DATA_FILENAME = "correlation_data.json"
LEGACY_RESULT_FILENAME = "correlation_result.json"

# Preference order within a run folder: the consolidated file, then either legacy.
_RUN_FILES = (CORRELATION_FILENAME, LEGACY_RESULT_FILENAME, LEGACY_DATA_FILENAME)


def correlation_dir(lamella_path: str) -> str:
    """Where a lamella's correlation runs live: ``<lamella>/Correlation``."""
    return os.path.join(lamella_path, CORRELATION_DIRNAME)


def new_run_dir(lamella_path: str, now: Optional[datetime.datetime] = None) -> str:
    """A new run folder for a lamella, named for ``now`` but NOT created.

    It is created only when the run records something (:func:`save_run`):
    creating it on open made every open a run, and an abandoned session became
    the next open's starting coordinates (FIB-320).
    """
    now = now or datetime.datetime.now()
    return os.path.join(correlation_dir(lamella_path), now.strftime(DATETIME_FILE))


def save_run(run_dir: str, state: CorrelationState) -> str:
    """Write ``state`` as the run's ``correlation.json``, creating the folder.

    Returns the file written.
    """
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, CORRELATION_FILENAME)
    state.save(path)
    return path


def _run_file(folder: str) -> Optional[str]:
    for name in _RUN_FILES:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return None


def load_run(run_dir: str) -> CorrelationState:
    """The correlation saved in a run folder, whichever file it was written as."""
    path = _run_file(run_dir)
    if path is None:
        raise FileNotFoundError(f"no correlation is saved in {run_dir}")
    return load_correlation_file(path)


@dataclass
class CorrelationRun:
    """One saved correlation, loaded from its timestamped folder."""

    path: str  # the run folder
    name: str  # folder basename (the timestamp)
    state: CorrelationState


@dataclass
class LamellaCorrelation:
    """The runs found under a lamella's ``Correlation`` directory, oldest first."""

    runs: List[CorrelationRun] = field(default_factory=list)

    @staticmethod
    def discover(correlation_dir: str) -> "LamellaCorrelation":
        """Load every run folder under ``correlation_dir`` (``<lamella>/Correlation``).

        Folders are ordered by name — the timestamp format is lexicographically
        chronological. Unreadable folders are logged and skipped so one bad run
        doesn't hide the rest.
        """
        runs: List[CorrelationRun] = []
        if not os.path.isdir(correlation_dir):
            return LamellaCorrelation()
        for folder in sorted(glob.glob(os.path.join(correlation_dir, "*"))):
            if not os.path.isdir(folder):
                continue
            path = _run_file(folder)
            if path is None:
                continue
            try:
                state = load_correlation_file(path)
            except Exception:
                logging.warning("Skipping unreadable correlation run: %s", path)
                continue
            runs.append(
                CorrelationRun(path=folder, name=os.path.basename(folder), state=state)
            )
        return LamellaCorrelation(runs=runs)

    def latest(self) -> Optional[CorrelationRun]:
        """The most recent run, or None if there are none."""
        return self.runs[-1] if self.runs else None


def correlation_record(
    result: CorrelationResult, lamella: Any, run_folder: Optional[str], root: Any
) -> Dict[str, Any]:
    """An accepted correlation, for the experiment's record (FIB-1068): what it
    gave, how well it fits, and the run folder that holds the rest. Whoever
    moves an item's point of interest to a run's answer records this."""
    from fibsem.correlation.verdict import verdict_tier

    data = result.input_data
    pixel_size = data.fib_image_pixel_size if data is not None else None
    folder = run_folder
    if run_folder and root:
        try:
            folder = os.path.relpath(run_folder, str(root))
        except ValueError:  # another drive
            pass
    ri = result.refractive_index_correction_mode
    return {
        "item": {"id": lamella.id, "name": lamella.name},
        "poi": result.poi[0].px_m.to_dict(),
        "rms_px": result.rms_error,
        "rms_nm": result.rms_error * pixel_size * 1e9 if pixel_size else None,
        "fiducials": len(result.delta_2d),
        "refractive_index": (
            {"mode": ri, "factor": result.refractive_index_correction_factor}
            if ri
            else None
        ),
        "verdict": verdict_tier(result.diagnostics),
        "seeded": result.seed is not None,
        "folder": folder,
    }


# ── image names, as runs record them ─────────────────────────────────────


_IMAGE_SUFFIXES = (".ome.tiff", ".ome.tif", ".tiff", ".tif")


def recorded_image_basename(name: Optional[str]) -> Optional[str]:
    """A recorded image name as a bare file name.

    Runs written on the instrument PC can record a full Windows path, which
    ``os.path.basename`` on another platform leaves whole, so both separators
    are split on here (FIB-1237).
    """
    if not name:
        return None
    return str(name).replace("\\", "/").rstrip("/").split("/")[-1] or None


def image_stem(name: Optional[str]) -> Optional[str]:
    """What identifies an image file across the forms a run has recorded it in.

    The bare name without its TIFF suffix, and without the ``_ib`` tag: older
    runs recorded the acquisition stem (``ref_Spot Burn Fiducial_final_res_02``)
    where newer ones record the file (``..._final_res_02_ib.tif``). An ``_eb``
    image keeps its tag, so it never matches the ion image of the same stage.
    """
    base = recorded_image_basename(name)
    if base is None:
        return None
    for suffix in _IMAGE_SUFFIXES:
        if base.lower().endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base[:-3] if base.endswith("_ib") else base


# ── what a folder holds ──────────────────────────────────────────────────


def experiment_and_lamella_dirs(
    project_dir: str,
) -> Tuple[Optional[str], Optional[str]]:
    """The experiment folder above a run or lamella folder, and the lamella folder.

    Walks up from ``project_dir`` (at most six levels) to the folder holding
    ``experiment.yaml``; the lamella folder is the one directly under it.
    ``(None, None)`` when there is no experiment above.
    """
    path = os.path.abspath(project_dir)
    lamella_dir = None
    probe = path
    for _ in range(6):
        if os.path.isfile(os.path.join(probe, EXPERIMENT_FILENAME)):
            return probe, lamella_dir
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        # the lamella folder is the one directly under the experiment folder
        lamella_dir = probe
        probe = parent
    return None, None


def _discover_correlation_files(directory: str) -> Dict[str, Optional[str]]:
    """Locate FIB/FM images and saved correlation JSON in a project dir.

    Conventions (matching the widget's auto-save + typical exports):
      - FM image   : ``*.ome.tif`` / ``*.ome.tiff``
      - FIB image  : ``*_ib.tif`` / ``*_ib.tiff`` (else the first non-OME TIFF)
      - correlation: ``correlation.json`` (FIB-264 consolidated file)
      - data       : ``correlation_data.json``   (legacy, still read)
      - result     : ``correlation_result.json`` (legacy, still read)
    """

    def _first(patterns: List[str]) -> Optional[str]:
        for pat in patterns:
            hits = sorted(glob.glob(os.path.join(directory, pat)))
            if hits:
                return hits[0]
        return None

    fib = _first(["*_ib.tif", "*_ib.tiff"])
    if fib is None:
        tifs = sorted(
            glob.glob(os.path.join(directory, "*.tif"))
            + glob.glob(os.path.join(directory, "*.tiff"))
        )
        fib = next((t for t in tifs if not t.endswith((".ome.tif", ".ome.tiff"))), None)

    def _existing(name: str) -> Optional[str]:
        p = os.path.join(directory, name)
        return p if os.path.exists(p) else None

    return {
        "fib": fib,
        "fm": _first(["*.ome.tif", "*.ome.tiff"]),
        "correlation": _existing(CORRELATION_FILENAME),
        "data": _existing(LEGACY_DATA_FILENAME),
        "result": _existing(LEGACY_RESULT_FILENAME),
    }


def find_spot_burns(project_dir: str) -> Tuple[List[Point], str]:
    """The spot-burn pattern for the lamella a project folder belongs to.

    Walks up from ``project_dir`` to the experiment file, matches the lamella by
    its folder name, and reads the spot-burn task's coordinates. Returns
    ``([], reason)`` when any of that is missing, so the caller can say why
    there is no pattern. The standalone launcher's stand-in for what the
    protocol editor passes to the correlation tab's ``add_lamella_setup``.
    """
    import yaml

    experiment_dir, lamella_dir = experiment_and_lamella_dirs(project_dir)
    if experiment_dir is None:
        return [], "no experiment.yaml above the project folder"
    experiment_file = os.path.join(experiment_dir, EXPERIMENT_FILENAME)
    lamella_name = os.path.basename(lamella_dir) if lamella_dir else None
    try:
        with open(experiment_file) as fh:
            experiment = yaml.safe_load(fh)
    except Exception as exc:
        return [], f"could not read {experiment_file}: {exc}"
    positions = experiment.get("positions", []) if isinstance(experiment, dict) else []
    for position in positions:
        name = position.get("petname") or position.get("name")
        folder = (
            str(position.get("path", "")).replace("\\", "/").rstrip("/").split("/")[-1]
        )
        if lamella_name not in (name, folder):
            continue
        for cfg in (position.get("task_config") or {}).values():
            if (
                not isinstance(cfg, dict)
                or cfg.get("task_type") != "SPOT_BURN_FIDUCIAL"
            ):
                continue
            coords = [Point(x=c["x"], y=c["y"]) for c in cfg.get("coordinates", [])]
            if coords:
                return coords, ""
            return [], f"{lamella_name}: spot-burn task has no coordinates"
        return [], f"{lamella_name}: no spot-burn task in the experiment"
    return [], f"{lamella_name!r} is not a lamella in {experiment_file}"


def _images_beside_a_run(lamella_dir: str) -> Dict[str, Optional[str]]:
    """The FIB and FM images for a run folder, from the lamella folder above it.

    A lamella holds a reference image per task and stage, so the plain
    discovery order would take whichever sorts first. Correlation fiducials are
    picked on a post-burn reference, so prefer the spot-burn task's final one.

    This is a guess, used only for a run that records no image names (before
    FIB-1019): picking the wrong reference shifts every FIB coordinate. The
    caller logs what it chose.
    """
    ib = sorted(glob.glob(os.path.join(lamella_dir, "*_ib.tif")))
    burn = [
        p
        for p in ib
        if "spot burn" in os.path.basename(p).lower()
        and "final" in os.path.basename(p).lower()
    ]
    fm = sorted(
        glob.glob(os.path.join(lamella_dir, "*.ome.tif"))
        + glob.glob(os.path.join(lamella_dir, "*.ome.tiff"))
    )
    return {
        "fib": (burn or ib or [None])[-1],
        "fm": (fm or [None])[0],
    }


def _recorded_image_names(found: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """The FIB and FM image names the run's own file records, or None for each.

    Read from the file the project's loader will load, in the same order. An
    unreadable file records nothing here; the load reports it.
    """
    for key in ("correlation", "data", "result"):
        if not found.get(key):
            continue
        try:
            state = load_correlation_file(found[key])
        except Exception:
            continue
        sources = [state.input_data]
        if state.result is not None and state.result.input_data is not None:
            sources.append(state.result.input_data)
        return {
            "fib": next(
                (
                    d.stored_fib_image_filename
                    for d in sources
                    if d.stored_fib_image_filename
                ),
                None,
            ),
            "fm": next(
                (
                    d.stored_fm_image_filename
                    for d in sources
                    if d.stored_fm_image_filename
                ),
                None,
            ),
        }
    return {"fib": None, "fm": None}


# The forms a recorded name has taken: the file itself, or (older FIB records)
# the acquisition stem that ``save`` appended its suffix to.
_RECORDED_IMAGE_SUFFIXES = {
    "fib": ("", "_ib.tif", "_ib.tiff", ".tif", ".tiff"),
    "fm": ("", ".ome.tiff", ".ome.tif"),
}


def _find_recorded_image(name: str, kind: str, folders: List[str]) -> Optional[str]:
    """The file a recorded image name refers to, in the first of ``folders``
    that has it, or None. A full path recorded on another machine is reduced to
    its file name."""
    base = recorded_image_basename(name)
    if base is None:
        return None
    for folder in folders:
        for suffix in _RECORDED_IMAGE_SUFFIXES[kind]:
            path = os.path.join(folder, base + suffix)
            if os.path.isfile(path):
                return path
    return None


def discover_project(directory: str) -> Dict[str, Optional[str]]:
    """The files a correlation project directory opens with.

    ``{"fib", "fm", "correlation", "data", "result"}``, each a path or None (see
    :func:`_discover_correlation_files` for the names looked for). The images are
    the ones the run records it was picked on (FIB-1019), looked for in the run
    folder and then the lamella folder above it. A recorded image that cannot be
    found is left as None rather than replaced by a guess: the fiducials are
    pixels in that image (FIB-1237). Only a run that records no names falls back
    to guessing.
    """
    found = _discover_correlation_files(directory)
    logging.info("Quickstart loading correlation project: %s", directory)

    _, lamella_dir = experiment_and_lamella_dirs(directory)
    above_run = (
        lamella_dir
        if lamella_dir and os.path.abspath(lamella_dir) != os.path.abspath(directory)
        else None
    )
    recorded = _recorded_image_names(found)
    for key in ("fib", "fm"):
        if not recorded[key]:
            continue
        folders = [directory] + ([above_run] if above_run else [])
        found[key] = _find_recorded_image(recorded[key], key, folders)
        if found[key]:
            logging.info(
                "  %s image as recorded by the run: %s",
                key.upper(),
                os.path.basename(found[key]),
            )
        else:
            logging.warning(
                "  the run records %s image %s, which is not in %s; not loading "
                "another in its place",
                key.upper(),
                recorded_image_basename(recorded[key]),
                " or ".join(folders),
            )

    # A run folder holds correlation.json; the images live in the lamella
    # folder above it. The launcher already walks up there for the spot-burn
    # pattern and the transform priors, so do the same for the images rather
    # than opening with points and empty canvases (FIB-1018).
    unrecorded = [k for k in ("fib", "fm") if not recorded[k] and not found[k]]
    if unrecorded and above_run:
        above = _images_beside_a_run(above_run)
        for key in unrecorded:
            if above[key]:
                found[key] = above[key]
                logging.info(
                    "  %s image guessed from the lamella folder: %s",
                    key.upper(),
                    os.path.basename(above[key]),
                )

    return found
