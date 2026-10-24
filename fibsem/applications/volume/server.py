"""The volume app over the App Server: the /app core and the Items and Poses
parts, with each position edit made by the experiment's own method (its
refusals, one save).

Per-position task configs are refused with why: positions share the protocol's.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fibsem.applications.volume.experiment_setup import (
    create_experiment,
    load_for_adoption,
)
from fibsem.applications.volume.poses import (
    POSES,
    move_milling_pose,
    pose_stage_position,
    set_pose_now,
)
from fibsem.applications.volume.protocol import PROTOCOL_FOLDER, VolumeProtocol
from fibsem.server.app_context import AppContext, find_item
from fibsem.server.files import jpeg_preview
from fibsem.server.responses import (
    ExperimentSummary,
    ItemDetail,
    Refusal,
    SummaryRow,
    no_experiment,
    refused,
)
from fibsem.structures.settings import FibsemRectangle, FibsemStagePosition, Point
from fibsem.util.records import content_hash
from fibsem.workflows.item import item_pose_positions
from fibsem.workflows.outputs import item_output_named, saved_file

__all__ = ["VolumeAppContext", "serve"]

# What apply_item_patch edits, each by its experiment method.
POSITION_PATCH_FIELDS = ("iterations", "poi", "alignment_area", "description")


@dataclass
class PositionRow(SummaryRow):
    iterations: int = 0
    slices_cut: int = 0


@dataclass
class PositionSummary(ExperimentSummary):
    items: List[PositionRow] = field(default_factory=list)


@dataclass
class PositionDetail(ItemDetail):
    iterations: Optional[int] = None
    slices_cut: Optional[int] = None


def _not_served(what: str) -> Refusal:
    return refused("invalid", f"Volume does not {what} yet.")


class VolumeAppContext(AppContext):
    """The App Server's view of a volume session."""

    app_name = "Volume"

    # --- experiment record ------------------------------------------------------

    answer_types = {
        ExperimentSummary: PositionSummary,
        ItemDetail: PositionDetail,
    }

    def summary_row(self, experiment, item) -> PositionRow:
        return PositionRow(
            **vars(super().summary_row(experiment, item)),
            iterations=item.iterations,
            slices_cut=experiment.slices_cut(item),
        )

    # --- experiment and protocol files -----------------------------------------

    protocol_type = VolumeProtocol

    def _protocol_folder(self) -> Optional[str]:
        return PROTOCOL_FOLDER

    def _new_experiment(self, directory, name, protocol, metadata):
        return create_experiment(directory, name, protocol, metadata)

    def _load_for_adoption(self, path):
        return load_for_adoption(path)

    # --- Items ------------------------------------------------------------------

    def detail_of(self, experiment, item) -> PositionDetail:
        detail = super().detail_of(experiment, item)
        detail.iterations = item.iterations
        detail.slices_cut = experiment.slices_cut(item)
        return detail

    def slice_depth(self, item, iteration: Optional[int]) -> Optional[float]:
        """The iteration's cut times the protocol's slice thickness."""
        from fibsem.applications.volume.tasks.milling import volume_milling

        protocol = getattr(self._experiment, "task_protocol", None)
        if iteration is None or protocol is None:
            return None
        config = volume_milling(protocol)
        return iteration * config.slice_thickness if config is not None else None

    def item_version(self, item) -> str:
        return self._version(item)

    @staticmethod
    def _version(position) -> str:
        """The position document's version: its patchable fields and poses."""
        return content_hash(
            {
                "name": position.name,
                "iterations": position.iterations,
                "poi": position.poi.to_dict(),
                "alignment_area": position.alignment_area.to_dict(),
                "poses": item_pose_positions(position),
            }
        )

    def output_image(
        self, item_name: str, filename: str, max_width: int = 768
    ) -> Dict[str, Any]:
        experiment = self._experiment
        position, refusal = find_item(experiment, item_name)
        if refusal:
            return refusal
        path = item_output_named(experiment, position, filename) or saved_file(
            experiment, experiment.position_path(position), filename
        )
        if path is None:
            return refused(
                "not_found", f"No output named {filename!r} for {item_name!r}."
            )
        return jpeg_preview(path, filename, max_width)

    def item_task_config(self, item_name: str, task_name: str) -> Dict[str, Any]:
        return _not_served(
            "keep task configs per position; positions share the protocol's"
        )

    def apply_item_task_config_patch(
        self, item_name, task_name, patch, version, timeout=10.0
    ):
        return self.item_task_config(item_name, task_name)

    def apply_protocol_to_item(self, item_name, task_names=None, timeout=10.0):
        return self.item_task_config(item_name, "")

    def add_item(
        self,
        name: Optional[str] = None,
        stage_position: Optional[Dict[str, Any]] = None,
        overview: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add a position, milled at the stage position now or at
        *stage_position*: the microscope state now, with that stage position."""
        experiment, microscope = self._experiment, self._microscope
        if experiment is None:
            return no_experiment()
        refusal = self._no_microscope()
        if refusal:
            return refusal
        if overview is not None:
            return refused(
                "invalid",
                "A volume position is added at a stage position, not on an overview.",
            )
        if stage_position is None:
            refusal = self._not_while_running("add a position where the stage is")
            if refusal:
                return refused(
                    refusal.error_type, f"{refusal.message} Or give its stage_position."
                )
        name = (name or "").strip() or experiment.next_position_name()

        def add():
            state = microscope.get_microscope_state()
            if stage_position is not None:
                state.stage_position = FibsemStagePosition.from_dict(stage_position)
            experiment.add_position(name, state)

        try:
            self._agent_call(add)
        except (ValueError, TypeError, KeyError) as exc:
            return refused("invalid", str(exc))
        return {"applied": True, "item": self.item_detail(name)}

    def remove_item(self, item_name: str) -> Dict[str, Any]:
        """Remove a position; its folder stays. Refused for the one the run is on."""
        experiment = self._experiment
        if experiment is None:
            return no_experiment()
        position, refusal = find_item(experiment, item_name)
        if refusal:
            return refusal
        refusal = self._busy(self._runner.busy_refusal([item_name], "remove it"))
        if refusal:
            return refusal
        why = self._agent_call(lambda: experiment.remove_position(item_name))
        if why:
            return refused("invalid", why)
        return {"applied": True, "item_name": item_name}

    # --- Poses -------------------------------------------------------------------

    def _position_and_microscope(self, item_name: str):
        """The position and microscope a pose route needs, or a refusal."""
        experiment = self._experiment
        if experiment is None:
            return (
                None,
                None,
                {"error_type": "not_found", "error": "No experiment is loaded."},
            )
        refusal = self._no_microscope()
        if refusal:
            return None, None, refusal
        position, refusal = find_item(experiment, item_name)
        if refusal:
            return None, None, refusal
        return position, self._microscope, None

    def move_to_item_pose(self, item_name: str, pose_name: str) -> Dict[str, Any]:
        """Move the stage to one of a position's poses, as the panel's Move to
        does, through the safe absolute move. Runs on the caller's thread,
        which holds control of the microscope."""
        position, microscope, refusal = self._position_and_microscope(item_name)
        if refusal:
            return refusal
        stage_position, why = pose_stage_position(position, pose_name)
        if why:
            return {"error_type": "not_found", "error": why, "pose_names": list(POSES)}
        microscope.safe_absolute_stage_movement(stage_position)
        return {
            "applied": True,
            "item_name": item_name,
            "pose": pose_name,
            "stage_position": microscope.get_stage_position().to_dict(),
        }

    def set_item_pose(self, item_name: str, pose_name: str) -> Dict[str, Any]:
        """Set one of a position's poses from the microscope now, as the panel's
        Update does. The milling pose is kept once a slice is cut. Saved."""
        position, microscope, refusal = self._position_and_microscope(item_name)
        if refusal:
            return refusal
        refusal = self._busy(self._runner.busy_refusal([item_name], "set its pose"))
        if refusal:
            return refusal
        why = self._agent_call(
            lambda: set_pose_now(self._experiment, microscope, item_name, pose_name)
        )
        if why:
            return {"error_type": "invalid", "error": why, "pose_names": list(POSES)}
        return {
            "applied": True,
            "item_name": item_name,
            "pose": pose_name,
            "item": self.item_detail(item_name),
        }

    def move_item(
        self,
        item_name: str,
        stage_position: Optional[Dict[str, Any]] = None,
        overview: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Move a position's milling pose to *stage_position*; the stage doesn't
        move. Refused once a slice is cut. Saved."""
        experiment = self._experiment
        if experiment is None:
            return {"error_type": "not_found", "error": "No experiment is loaded."}
        if overview is not None:
            return _not_served("place positions from an overview")
        if stage_position is None:
            return {
                "error_type": "invalid",
                "error": "Give stage_position: where to move it to.",
            }
        _, refusal = find_item(experiment, item_name)
        if refusal:
            return refusal
        refusal = self._busy(self._runner.busy_refusal([item_name], "move it"))
        if refusal:
            return refusal
        try:
            position = FibsemStagePosition.from_dict(stage_position)
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            return {"error_type": "invalid", "error": f"Not a stage position: {exc}"}
        why = self._agent_call(
            lambda: move_milling_pose(experiment, item_name, position)
        )
        if why:
            return {"error_type": "invalid", "error": why}
        return {"applied": True, "item": self.item_detail(item_name)}

    def set_item_objective_position(self, item_name: str) -> Dict[str, Any]:
        return _not_served("keep an objective position on a position")

    def move_objective_to_item(self, item_name: str) -> Dict[str, Any]:
        return self.set_item_objective_position(item_name)

    def set_objective_position_on_all(self, item_name: str) -> Dict[str, Any]:
        return self.set_item_objective_position(item_name)

    def apply_item_patch(
        self,
        item_name: str,
        patch: Dict[str, Any],
        version: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Edit a position's slices, point of interest, alignment area or
        description, against the version :meth:`item_detail` served: all or
        none, with one save (``VolumeExperiment.patch_position``). A position
        keeps the name it was given."""
        experiment = self._experiment
        if experiment is None:
            return no_experiment()
        position, refusal = find_item(experiment, item_name)
        if refusal:
            return refusal
        unknown = sorted(set(patch) - set(POSITION_PATCH_FIELDS))
        if unknown:
            return {
                "applied": False,
                "invalid_patch": f"A position patch sets {', '.join(POSITION_PATCH_FIELDS)}; "
                f"not {', '.join(unknown)}.",
            }
        refusal = self._busy(self._runner.busy_refusal([item_name], "edit it"))
        if refusal:
            return refusal
        try:
            fields = {
                "iterations": patch.get("iterations"),
                "poi": Point.from_dict(patch["poi"]) if "poi" in patch else None,
                "alignment_area": FibsemRectangle.from_dict(patch["alignment_area"])
                if "alignment_area" in patch
                else None,
                "description": patch.get("description"),
            }
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            return {"applied": False, "invalid_patch": f"Not a position patch: {exc}"}

        def edit():
            if str(version) != self._version(position):
                return {"applied": False, "stale": True}
            why = experiment.patch_position(item_name, **fields)
            if why:
                return {"applied": False, "invalid_patch": why}
            return {
                "applied": True,
                "fields": [f for f in POSITION_PATCH_FIELDS if f in patch],
            }

        result = self._agent_call(edit, timeout=timeout)
        if result.get("applied"):
            result["item"] = self.item_detail(position.name)
        return result


def serve(session) -> bool:
    """Host the App Server over a volume *session*, window or not. Whether it
    is serving (see ``AppSession.start_app_server``)."""
    return session.start_app_server(
        lambda buffer: VolumeAppContext(session, event_buffer=buffer)
    )
