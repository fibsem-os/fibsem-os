######## TASK BASE DEFINITIONS ########


import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
)

import numpy as np

from fibsem import constants, utils
from fibsem.acting import AGENT
from fibsem.applications.autolamella.protocol.constants import (
    FIDUCIAL_KEY,
    MILL_POLISHING_KEY,
    MILL_ROUGH_KEY,
    STRESS_RELIEF_KEY,
    TRENCH_KEY,
    UNDERCUT_KEY,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskConfig,
    Lamella,
)
from fibsem.applications.autolamella.workflows._default_milling_config import (
    DEFAULT_MILLING_CONFIG,
)
from fibsem.applications.autolamella.workflows.core import (
    align_feature_coincident,
    update_status_ui,
)
from fibsem.detection import detection as detection_module
from fibsem.detection.decided import (
    decided_features,
    detection_image_file,
    detection_provenance,
    detection_values,
    record_training_data,
)
from fibsem.detection.detection import (
    DetectedFeatures,
    Feature,
    LamellaBottomEdge,
    LamellaCentre,
    LamellaTopEdge,
)
from fibsem.fm.structures import ChannelSettings
from fibsem.microscope import FibsemMicroscope
from fibsem.milling.patterning.utils import get_pattern_reduced_area
from fibsem.milling.progress import MillingProgress, MillingProgressStatus
from fibsem.structures import (
    DEFAULT_ALIGNMENT_AREA,
    FibsemStagePosition,
    ImageSettings,
    Point,
)
from fibsem.workflows.interaction import (
    INSTRUCTION_TIMEOUT_S,
    SetFluorescenceChannels,
    ShowDetection,
    _abort_requested,
    ask,
)
from fibsem.workflows.proposals import (
    DETECTION,
    DecisionOutcome,
)
from fibsem.workflows.site_task import SiteTask
from fibsem.workflows.status import TaskState

if TYPE_CHECKING:
    from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager

TAutoLamellaTaskConfig = TypeVar(
    "TAutoLamellaTaskConfig", bound="AutoLamellaTaskConfig"
)


class AutoLamellaTask(SiteTask):
    """Base class for AutoLamella tasks: a task on a lamella."""

    config_cls: ClassVar[AutoLamellaTaskConfig]
    config: AutoLamellaTaskConfig

    def __init__(
        self,
        microscope: FibsemMicroscope,
        config: AutoLamellaTaskConfig,
        lamella: Lamella,
        task_manager: Optional["TaskManager"] = None,
    ):
        super().__init__(microscope, config, lamella, task_manager=task_manager)

    @property
    def lamella(self) -> Lamella:
        """The lamella this task runs on: its item."""
        return self.item

    @lamella.setter
    def lamella(self, lamella: Lamella) -> None:
        self.item = lamella

    @property
    def path(self) -> Path:
        """The lamella's folder, which it knows without an experiment."""
        return Path(self.lamella.path)

    def detect(
        self,
        image_settings: ImageSettings,
        checkpoint: str,
        features: Sequence[Feature],
        *,
        position: Optional[FibsemStagePosition] = None,
        message: str = "",
        enabled: bool = True,
    ) -> DetectedFeatures:
        """Take an image, run the model on it, and ask about what it found.

        The detection is a ``detection`` question through ``ask``: recorded
        on the lamella against the image the model ran on, answered on the
        Detection tab (the features shown there first, a marker dragged,
        then Continue) or by an agent, and the decided points are put back
        on the detection with ``feature_m`` recomputed, so the stage moves by
        the corrected value. A rejection (an agent's) fails the task. The
        training data is written here, against the decided points.
        """
        message = message or self.lamella.status_info
        names = ", ".join(f.name for f in features)
        if len(names) > 15:
            names = names[:15] + "..."
        update_status_ui(
            self.task_manager, f"{message}: Detecting Features ({names})..."
        )
        detection = detection_module.take_image_and_detect_features(
            microscope=self.microscope,
            image_settings=image_settings,
            features=features,
            point=position,
            checkpoint=checkpoint,
        )
        if self.responder is not None and not self.unasked_reason(
            DETECTION, enabled=enabled
        ):
            # The Detection tab shows what the model found, as a copy: the
            # operator drags its markers, and the original is what the
            # correction is measured against.
            ask(
                self.responder,
                ShowDetection(detection=deepcopy(detection)),
                abort=lambda: _abort_requested(self.task_manager),
                timeout=INSTRUCTION_TIMEOUT_S,
            )
        decision = self.ask(
            DETECTION,
            detection_values(detection),
            image=detection_image_file(detection, str(self.lamella.path)),
            provenance=detection_provenance(detection),
            message=message,
            enabled=enabled,
        )
        if decision.outcome is DecisionOutcome.Rejected:
            raise RuntimeError(
                f"{self.task_name} was rejected: {decision.reason or 'no reason given'}"
            )
        answer = decided_features(detection, decision.values)
        record_training_data(detection, answer)
        return answer

    def _update_fluorescence_pose(self) -> None:
        """Refresh the lamella's recorded fluorescence pose from the current microscope state.

        The configured objective (focus) position is preserved: get_microscope_state()
        does not capture the objective position, so overwriting the pose here (and
        re-reading the live objective position) previously wiped the user's configured
        focus.
        """
        configured_objective_position = (
            self.lamella.fluorescence_pose.objective_position
            if self.lamella.fluorescence_pose is not None
            else None
        )
        self.lamella.fluorescence_pose = self.microscope.get_microscope_state()
        self.lamella.fluorescence_pose.objective_position = (
            configured_objective_position
        )

    def set_fluorescence_channels_ui(
        self, channel_settings: List[ChannelSettings]
    ) -> None:
        """Set the fluorescence channel settings in the fluorescence widget."""
        if self.responder is None:
            return

        ask(
            self.responder,
            SetFluorescenceChannels(channels=deepcopy(channel_settings)),
            abort=lambda: _abort_requested(self.task_manager),
            timeout=INSTRUCTION_TIMEOUT_S,
        )
