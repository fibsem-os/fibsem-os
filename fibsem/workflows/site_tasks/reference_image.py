######## REFERENCE IMAGE TASK DEFINITIONS ########

from dataclasses import dataclass
from typing import ClassVar, Literal, Optional, Type

from fibsem import utils
from fibsem.constants import TIME_FILE
from fibsem.structures.params import param
from fibsem.workflows.proposals import STATE
from fibsem.workflows.site_task import SiteTask, SiteTaskConfig


@dataclass
class AcquireReferenceImageConfig(SiteTaskConfig):
    """Configuration for the AcquireReferenceImageTask."""

    task_type: ClassVar[str] = "ACQUIRE_REFERENCE_IMAGE"
    display_name: ClassVar[str] = "Acquire Reference Image"
    orientation: Literal["SEM", "FIB", "MILLING"] = param(
        "MILLING",
        choices=("SEM", "FIB", "MILLING"),
        doc="The orientation to acquire reference images in (SEM, FIB, MILLING)",
    )  # change to pose?
    filename: Optional[str] = param(
        None,
        doc="Custom filename for reference images. If None, auto-generates from last completed task name and timestamp.",
    )


class AcquireReferenceImageTask(SiteTask):
    """Task to acquire reference image with specified settings."""

    config: AcquireReferenceImageConfig
    config_cls: ClassVar[Type[AcquireReferenceImageConfig]] = (
        AcquireReferenceImageConfig
    )
    # One question: is the position right, before the images are taken.
    asks = (STATE,)

    def _run(self) -> None:
        """Run the task to acquire reference image with the specified settings."""

        # move to position
        self._move_to_pose()

        msg = (
            f"Acquire reference image for {self.item.name}. Press Continue when ready."
        )
        # The question on the record: the position as arrived at is the
        # proposal, Continue confirms it, and the position as it stands then is
        # the decision -- so a move made first is the delta.
        self.ask(
            STATE,
            {"stage_position": self.microscope.get_stage_position()},
            message=msg,
            decided=lambda: {"stage_position": self.microscope.get_stage_position()},
        )

        # bookkeeping
        image_settings = self.config.imaging

        self.log_status_message(
            "ACQUIRE_REFERENCE_IMAGE", "Acquiring Reference Image..."
        )

        # acquire reference images — use config filename if provided, else use last completed task name
        if self.config.filename is not None:
            base = self.config.filename
        else:
            task_name = "Setup"
            if self.item.last_completed_task is not None:
                task_name = self.item.last_completed_task.name.replace(" ", "-")
            base = f"ReferenceImage-{task_name}"

        filename = f"ref_{base}-{utils.current_timestamp(TIME_FILE)}"
        # phase="final" because this set is what the task exists to produce. The
        # default rule reads the role off the filename and would file a named set
        # under "other", which is right for a task grabbing an extra set mid-run and
        # wrong here -- it hid these images from the History panel entirely.
        self._acquire_set_of_reference_images(
            image_settings=image_settings, filename=filename, phase="final"
        )
