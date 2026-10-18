"""Discovery of the final reference images a task run produced.

Recorded outputs are preferred; the filename convention stays as the fallback for
experiments written before outputs existed, and for runs that failed before
reaching post_task and so left no history entry at all.
"""

import os
from pathlib import Path
from typing import List

from fibsem.applications.autolamella.structures import Lamella
from fibsem.applications.autolamella.task_outputs import (
    final_reference_images,
    fluorescence_images,
    images_by_task,
    latest_final_reference_images,
)
from fibsem.workflows.status import TaskState

CONVENTIONAL = [
    "ref_MillRough_final_res_01_eb.tif",
    "ref_MillRough_final_res_01_ib.tif",
    "ref_MillRough_final_res_02_eb.tif",
    "ref_MillRough_final_res_02_ib.tif",
]


def _lamella(tmp_path: Path) -> Lamella:
    lamella = Lamella(path=tmp_path / "lam", number=0, petname="test")
    os.makedirs(lamella.path, exist_ok=True)
    return lamella


def _write(lamella: Lamella, names: List[str]) -> None:
    for name in names:
        Path(lamella.path, name).write_bytes(b"")


def test_falls_back_to_the_filename_convention_when_nothing_recorded(tmp_path):
    """Old experiments have no outputs; the glob is still the route to their images."""
    lamella = _lamella(tmp_path)
    _write(lamella, CONVENTIONAL)
    task = TaskState(name="MillRough")

    found = final_reference_images(lamella, task)

    assert [os.path.basename(p) for p in found] == CONVENTIONAL


def test_prefers_recorded_outputs_over_the_glob(tmp_path):
    """What the run recorded wins, even when convention-named files also exist.

    The recorded names deliberately do not match the glob, so a result containing
    them can only have come from the record.
    """
    lamella = _lamella(tmp_path)
    _write(lamella, CONVENTIONAL + ["custom_eb.tif", "custom_ib.tif"])
    task = TaskState(
        name="MillRough",
        outputs={"final_sem": ["custom_eb.tif"], "final_fib": ["custom_ib.tif"]},
    )

    found = final_reference_images(lamella, task)

    assert [os.path.basename(p) for p in found] == ["custom_eb.tif", "custom_ib.tif"]


def test_both_routes_agree_on_order(tmp_path):
    """Recorded and globbed discovery must order identically.

    Callers slice the last N off the result to get the highest-resolution pair, so a
    difference in ordering would silently change which images the review panel shows.
    Records arrive grouped by beam; the glob arrives interleaved.
    """
    lamella = _lamella(tmp_path)
    _write(lamella, CONVENTIONAL)

    globbed = final_reference_images(lamella, TaskState(name="MillRough"))
    recorded = final_reference_images(
        lamella,
        TaskState(
            name="MillRough",
            outputs={
                "final_sem": [n for n in CONVENTIONAL if n.endswith("_eb.tif")],
                "final_fib": [n for n in CONVENTIONAL if n.endswith("_ib.tif")],
            },
        ),
    )

    assert recorded == globbed


def test_recorded_relative_paths_resolve_under_the_lamella_directory(tmp_path):
    """Records are stored relative; discovery returns absolute paths."""
    lamella = _lamella(tmp_path)
    _write(lamella, ["ref_MillRough_final_res_01_eb.tif"])
    task = TaskState(
        name="MillRough", outputs={"final_sem": ["ref_MillRough_final_res_01_eb.tif"]}
    )

    found = final_reference_images(lamella, task)

    assert found == [str(Path(lamella.path, "ref_MillRough_final_res_01_eb.tif"))]
    assert os.path.isfile(found[0])


def test_a_run_that_produced_nothing_finds_nothing(tmp_path):
    """No record and no files on disk is an empty result, not an error."""
    lamella = _lamella(tmp_path)

    assert final_reference_images(lamella, TaskState(name="MillRough")) == []


def test_only_final_outputs_are_offered(tmp_path):
    """start_* and one-off acquisitions are recorded but deliberately not shown.

    The review panel has always shown only the final reference set; recording more
    roles must not change what it displays.
    """
    lamella = _lamella(tmp_path)
    _write(lamella, ["start_eb.tif", "other_eb.tif"])
    task = TaskState(
        name="MillRough",
        outputs={"start_sem": ["start_eb.tif"], "other_sem": ["other_eb.tif"]},
    )

    assert final_reference_images(lamella, task) == []


def test_a_set_acquired_twice_in_one_run_does_not_crowd_out_the_other_beam(tmp_path):
    """MillCoincidentTask acquires the final set twice in one run, before and after
    milling. Both calls use the default filename, so the second overwrites the same
    files -- but both record. Left duplicated, the last-two slice callers take would
    return the same FIB image twice and no SEM image at all.
    """
    lamella = _lamella(tmp_path)
    _write(lamella, CONVENTIONAL)
    sem = [n for n in CONVENTIONAL if n.endswith("_eb.tif")]
    fib = [n for n in CONVENTIONAL if n.endswith("_ib.tif")]

    task = TaskState(
        name="MillRough", outputs={"final_sem": sem * 2, "final_fib": fib * 2}
    )
    found = final_reference_images(lamella, task)

    assert [os.path.basename(p) for p in found] == CONVENTIONAL
    assert [os.path.basename(p) for p in found[-2:]] == [
        "ref_MillRough_final_res_02_eb.tif",
        "ref_MillRough_final_res_02_ib.tif",
    ]


def test_a_record_whose_images_are_gone_falls_through_to_the_convention(tmp_path):
    """Unlike the glob, a record can name files that no longer exist. Returning them
    would build a task row of placeholders that never fill, where the old behaviour
    skipped the row entirely.
    """
    lamella = _lamella(tmp_path)
    _write(lamella, CONVENTIONAL)
    task = TaskState(name="MillRough", outputs={"final_sem": ["deleted_eb.tif"]})

    found = final_reference_images(lamella, task)

    assert [os.path.basename(p) for p in found] == CONVENTIONAL


def test_a_record_of_only_deleted_images_with_nothing_on_disk_finds_nothing(tmp_path):
    """...and with no convention-named files either, the row is skipped as before."""
    lamella = _lamella(tmp_path)
    task = TaskState(name="MillRough", outputs={"final_sem": ["deleted_eb.tif"]})

    assert final_reference_images(lamella, task) == []


# fluorescence_images


def test_fluorescence_images_returns_recorded_stacks(tmp_path):
    lamella = _lamella(tmp_path)
    _write(lamella, ["01-lam-zstack-14-26-25.ome.tiff"])
    task = TaskState(
        name="Acquire Fluorescence Image",
        outputs={"fluorescence": ["01-lam-zstack-14-26-25.ome.tiff"]},
    )

    found = fluorescence_images(lamella, task)

    assert found == [str(Path(lamella.path, "01-lam-zstack-14-26-25.ome.tiff"))]


def test_fluorescence_has_no_filename_fallback(tmp_path):
    """FM output never followed a discoverable convention — it carries the lamella
    name and a time-of-day stamp, but no task name. An experiment written before
    runs recorded their outputs has none to find, and guessing a pattern would
    attribute files to the wrong run.
    """
    lamella = _lamella(tmp_path)
    _write(lamella, ["01-lam-zstack-14-26-25.ome.tiff"])  # on disk, but unrecorded

    assert fluorescence_images(lamella, TaskState(name="Acquire")) == []


def test_fluorescence_applies_the_same_guards_as_reference_images(tmp_path):
    """MillCoincidentTask records under `fluorescence` too, so duplicates and
    deleted files are live cases here, not hypothetical."""
    lamella = _lamella(tmp_path)
    _write(lamella, ["stack.ome.tiff"])
    task = TaskState(
        name="Mill Coincident",
        outputs={"fluorescence": ["stack.ome.tiff", "stack.ome.tiff", "gone.ome.tiff"]},
    )

    assert fluorescence_images(lamella, task) == [
        str(Path(lamella.path, "stack.ome.tiff"))
    ]


def test_reference_and_fluorescence_are_separate(tmp_path):
    """Each reads only its own roles — a task with both must not cross-contaminate."""
    lamella = _lamella(tmp_path)
    _write(lamella, CONVENTIONAL + ["stack.ome.tiff"])
    task = TaskState(
        name="MillRough",
        outputs={
            "final_sem": [CONVENTIONAL[0]],
            "fluorescence": ["stack.ome.tiff"],
        },
    )

    assert final_reference_images(lamella, task) == [
        str(Path(lamella.path, CONVENTIONAL[0]))
    ]
    assert fluorescence_images(lamella, task) == [
        str(Path(lamella.path, "stack.ome.tiff"))
    ]


# accumulation across repeated runs of the same task


def test_repeated_runs_accumulate_distinct_fluorescence_stacks(tmp_path):
    """Each FM acquisition writes its own timestamped file, so re-running a task
    adds an image rather than replacing one. Showing only the last run would
    silently hide every earlier acquisition."""
    lamella = _lamella(tmp_path)
    _write(lamella, ["lam-zstack-10-00-00.ome.tiff", "lam-zstack-11-00-00.ome.tiff"])
    first = TaskState(
        name="Acquire", outputs={"fluorescence": ["lam-zstack-10-00-00.ome.tiff"]}
    )
    second = TaskState(
        name="Acquire", outputs={"fluorescence": ["lam-zstack-11-00-00.ome.tiff"]}
    )

    found = fluorescence_images(lamella, first, second)

    assert [os.path.basename(p) for p in found] == [
        "lam-zstack-10-00-00.ome.tiff",
        "lam-zstack-11-00-00.ome.tiff",
    ]


def test_repeated_runs_collapse_when_the_files_are_overwritten(tmp_path):
    """Reference images are written to the same filename every run, so repeated runs
    contribute the same paths and the union is one set — no branching on task type
    needed to get that.

    This pins the assumption the accumulation rests on. If reference filenames ever
    become unique per run, this test fails and the review panel would start showing
    every run's beam images: that should be an explicit decision, not a silent
    consequence of a naming change elsewhere.
    """
    lamella = _lamella(tmp_path)
    _write(lamella, CONVENTIONAL)
    outputs = {"final_sem": CONVENTIONAL[:1], "final_fib": CONVENTIONAL[1:2]}
    first = TaskState(name="MillRough", outputs=dict(outputs))
    second = TaskState(name="MillRough", outputs=dict(outputs))

    assert final_reference_images(lamella, first, second) == final_reference_images(
        lamella, first
    )


def test_the_glob_fallback_yields_one_set_however_many_runs(tmp_path):
    """Old experiments have no records, so the fallback globs the same files for
    every run of a task. The result must be one set, not the same files repeated
    per run — repeats would crowd out real images when callers slice off the last N.

    (Globbing each *distinct* name once is an efficiency detail on top of this; the
    deduplication is what makes the result correct.)
    """
    lamella = _lamella(tmp_path)
    _write(lamella, CONVENTIONAL)
    runs = [TaskState(name="MillRough") for _ in range(3)]

    found = final_reference_images(lamella, *runs)

    assert [os.path.basename(p) for p in found] == CONVENTIONAL


# --- each run in its own folder (FIB-1275) ---------------------------------------


def _run_in_folder(lamella: Lamella, task: str, stamp: str, **outputs) -> None:
    """A run that saved into ``<task>/<stamp>/`` and recorded it, as task run
    folders do; ``outputs`` maps a role to file names in that folder."""
    folder = Path(lamella.path, task, stamp)
    folder.mkdir(parents=True, exist_ok=True)
    recorded = {}
    for role, names in outputs.items():
        for name in names:
            (folder / name).write_bytes(b"")
        recorded[role] = [os.path.join(task, stamp, name) for name in names]
    lamella.task_history.append(TaskState(name=task, outputs=recorded))


FINALS = {
    "final_sem": ["ref_Mill_final_res_01_eb.tif"],
    "final_fib": ["ref_Mill_final_res_01_ib.tif"],
}


def test_the_task_s_images_are_its_latest_run_s(tmp_path):
    """Both runs' images are kept; the task's images now are the later run's."""
    lamella = _lamella(tmp_path)
    _run_in_folder(lamella, "Mill", "20261014-093012", **FINALS)
    _run_in_folder(lamella, "Mill", "20261014-101544", **FINALS)

    found = latest_final_reference_images(lamella, "Mill")

    assert [os.path.relpath(p, lamella.path) for p in found] == [
        os.path.join("Mill", "20261014-101544", "ref_Mill_final_res_01_eb.tif"),
        os.path.join("Mill", "20261014-101544", "ref_Mill_final_res_01_ib.tif"),
    ]
    # every run's, for the History tab, which shows each run
    assert len(final_reference_images(lamella, *lamella.task_history)) == 4


def test_a_later_run_that_saved_no_final_images_leaves_the_earlier_one_s(tmp_path):
    lamella = _lamella(tmp_path)
    _run_in_folder(lamella, "Mill", "20261014-093012", **FINALS)
    _run_in_folder(
        lamella, "Mill", "20261014-101544", start_fib=["ref_Mill_start_ib.tif"]
    )

    found = latest_final_reference_images(lamella, "Mill")

    assert all("20261014-093012" in p for p in found) and len(found) == 2


def test_a_task_whose_runs_record_no_final_images_is_not_found_by_name(tmp_path):
    """Its runs record what they make, so a file of the conventional name in the
    lamella folder is not theirs: an earlier version's, or another's."""
    lamella = _lamella(tmp_path)
    _write(lamella, ["ref_Mill_final_res_01_ib.tif"])
    _run_in_folder(
        lamella, "Mill", "20261014-093012", start_fib=["ref_Mill_start_ib.tif"]
    )

    assert latest_final_reference_images(lamella, "Mill") == []


def test_a_task_from_before_outputs_is_found_by_name(tmp_path):
    lamella = _lamella(tmp_path)
    _write(lamella, CONVENTIONAL)
    lamella.task_history.append(TaskState(name="MillRough"))

    found = latest_final_reference_images(lamella, "MillRough")

    assert [os.path.basename(p) for p in found] == CONVENTIONAL


def test_the_animation_shows_a_task_run_twice_once_with_its_latest_images(tmp_path):
    lamella = _lamella(tmp_path)
    _run_in_folder(lamella, "Mill", "20261014-093012", **FINALS)
    _run_in_folder(
        lamella,
        "Setup",
        "20261014-095000",
        **{
            "final_fib": ["ref_Setup_final_res_01_ib.tif"],
        },
    )
    _run_in_folder(lamella, "Mill", "20261014-101544", **FINALS)

    steps = images_by_task(lamella)

    assert [name for name, _, _ in steps] == ["Setup", "Mill"]
    _, mill_finals, _ = steps[1]
    assert all("20261014-101544" in p for p in mill_finals) and len(mill_finals) == 2
