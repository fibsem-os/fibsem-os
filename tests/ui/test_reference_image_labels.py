"""How the Lamella editor names the images in its pickers.

The workflow names reference images ``ref_<task>_<stage>_<beam>.tif``. A picker that
shows that string makes the operator parse it; one that shows ``Task · stage`` does
not, and the filename is still one hover away in the tooltip.
"""

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.ui.autolamella_lamella_protocol_editor import (  # noqa: E402
    fm_stack_label,
    reference_image_label,
)

TASKS = ["Setup Lamella Position", "Mill Fiducial", "Rough Milling", "Polishing"]


@pytest.mark.parametrize(
    "filename, label",
    [
        ("ref_Mill Fiducial_start_ib.tif", "Mill Fiducial · start"),
        ("ref_Mill Fiducial_final_res_01_eb.tif", "Mill Fiducial · final res 01"),
        ("ref_Rough Milling_post_tilt_ib.tif", "Rough Milling · post tilt"),
        # older experiments wrote the task with underscores
        ("ref_Setup_Lamella_Position_start_ib.tif", "Setup Lamella Position · start"),
        # no stage at all
        ("ref_Polishing_ib.tif", "Polishing"),
    ],
)
def test_task_and_stage_replace_the_filename(filename, label):
    assert reference_image_label(filename, TASKS) == label


def test_the_longest_task_name_wins():
    """A task whose name is a prefix of another's must not claim its images."""
    tasks = ["Mill", "Mill Fiducial"]
    assert reference_image_label("ref_Mill Fiducial_start_ib.tif", tasks) == (
        "Mill Fiducial · start"
    )
    assert reference_image_label("ref_Mill_start_ib.tif", tasks) == "Mill · start"


def test_images_from_no_task_keep_a_readable_stem():
    assert reference_image_label("ref_alignment_ib.tif", TASKS) == "alignment"
    assert (
        reference_image_label("overview-19-07-45_ib.tif", TASKS) == "overview-19-07-45"
    )


def test_z_stacks_read_as_a_clock_time():
    assert (
        fm_stack_label("01-fancy-mite-zstack-18-36-22.ome.tiff") == "Z-stack 18:36:22"
    )
    assert fm_stack_label("something-else.ome.tiff") == "something-else"
