
######## REFERENCE IMAGE TASK DEFINITIONS ########

from dataclasses import dataclass, field
from typing import ClassVar, Literal, Optional, Type

from fibsem import utils
from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig
from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask
from fibsem.applications.autolamella.workflows.ui import ask_user


@dataclass
class AcquireReferenceImageConfig(AutoLamellaTaskConfig):
    """Configuration for the AcquireReferenceImageTask."""
    task_type: ClassVar[str] = "ACQUIRE_REFERENCE_IMAGE"
    display_name: ClassVar[str] = "Acquire Reference Image"
    orientation: Literal["SEM", "FIB", "MILLING"] = field(
        default="MILLING",
        metadata={"help": "The orientation to acquire reference images in (SEM, FIB, MILLING)", "items": ("SEM", "FIB", "MILLING")},
    ) # change to pose?
    filename: Optional[str] = field(
        default=None,
        metadata={"help": "Custom filename for reference images. If None, auto-generates from last completed task name and timestamp."},
    )


class AcquireReferenceImageTask(AutoLamellaTask):
    """Task to acquire reference image with specified settings."""
    config: AcquireReferenceImageConfig
    config_cls: ClassVar[Type[AcquireReferenceImageConfig]] = AcquireReferenceImageConfig

    def _run(self) -> None:
        """Run the task to acquire reference image with the specified settings."""

        # move to position
        self._move_to_milling_pose()

        if self.validate:
            ask_user(self.parent_ui,
                    msg=f"Acquire reference image for {self.lamella.name}. Press continue when ready.",
                    pos="Continue"
                    )

        # bookkeeping
        image_settings = self.config.imaging
        image_settings.path = self.lamella.path

        self.log_status_message("ACQUIRE_REFERENCE_IMAGE", "Acquiring Reference Image...")

        # acquire reference images — use config filename if provided, else use last completed task name
        if self.config.filename is not None:
            base = self.config.filename
        else:
            task_name = "Setup"
            if self.lamella.last_completed_task is not None:
                task_name = self.lamella.last_completed_task.name.replace(" ", "-")
            base = f"ReferenceImage-{task_name}"

        filename = f"ref_{base}-{utils.current_timestamp_v3()}"
        self._acquire_set_of_reference_images(image_settings=image_settings, filename=filename)
