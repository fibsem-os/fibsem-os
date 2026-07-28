"""Discovery of the final reference images a task run produced.

Recorded outputs are preferred; the filename convention stays as the fallback for
experiments written before outputs existed, and for runs that failed before
reaching post_task and so left no history entry at all.
"""

import os
from pathlib import Path
from typing import List

from fibsem.applications.autolamella.structures import AutoLamellaTaskState, Lamella
from fibsem.applications.autolamella.task_outputs import final_reference_images

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
    task = AutoLamellaTaskState(name="MillRough")

    found = final_reference_images(lamella, task)

    assert [os.path.basename(p) for p in found] == CONVENTIONAL


def test_prefers_recorded_outputs_over_the_glob(tmp_path):
    """What the run recorded wins, even when convention-named files also exist.

    The recorded names deliberately do not match the glob, so a result containing
    them can only have come from the record.
    """
    lamella = _lamella(tmp_path)
    _write(lamella, CONVENTIONAL + ["custom_eb.tif", "custom_ib.tif"])
    task = AutoLamellaTaskState(
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

    globbed = final_reference_images(lamella, AutoLamellaTaskState(name="MillRough"))
    recorded = final_reference_images(
        lamella,
        AutoLamellaTaskState(
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
    task = AutoLamellaTaskState(
        name="MillRough", outputs={"final_sem": ["ref_MillRough_final_res_01_eb.tif"]}
    )

    found = final_reference_images(lamella, task)

    assert found == [str(Path(lamella.path, "ref_MillRough_final_res_01_eb.tif"))]
    assert os.path.isfile(found[0])


def test_a_run_that_produced_nothing_finds_nothing(tmp_path):
    """No record and no files on disk is an empty result, not an error."""
    lamella = _lamella(tmp_path)

    assert final_reference_images(lamella, AutoLamellaTaskState(name="MillRough")) == []


def test_only_final_outputs_are_offered(tmp_path):
    """start_* and one-off acquisitions are recorded but deliberately not shown.

    The review panel has always shown only the final reference set; recording more
    roles must not change what it displays.
    """
    lamella = _lamella(tmp_path)
    _write(lamella, ["start_eb.tif", "other_eb.tif"])
    task = AutoLamellaTaskState(
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

    task = AutoLamellaTaskState(
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
    task = AutoLamellaTaskState(
        name="MillRough", outputs={"final_sem": ["deleted_eb.tif"]}
    )

    found = final_reference_images(lamella, task)

    assert [os.path.basename(p) for p in found] == CONVENTIONAL


def test_a_record_of_only_deleted_images_with_nothing_on_disk_finds_nothing(tmp_path):
    """...and with no convention-named files either, the row is skipped as before."""
    lamella = _lamella(tmp_path)
    task = AutoLamellaTaskState(
        name="MillRough", outputs={"final_sem": ["deleted_eb.tif"]}
    )

    assert final_reference_images(lamella, task) == []
