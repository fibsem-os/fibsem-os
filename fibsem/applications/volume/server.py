"""The volume app over the App Server: the /app core and the Items part, with
each position edit made by the experiment's own method (its refusals, one save).

Per-position task configs are refused with why: positions share the protocol's.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fibsem.applications.volume.experiment import VolumeExperiment
from fibsem.applications.volume.protocol import PROTOCOL_FOLDER, VolumeProtocol
from fibsem.server.app_context import (
    AppContext,
    content_hash,
    find_item,
    item_pose_positions,
)
from fibsem.server.files import jpeg_preview
from fibsem.server.responses import ExperimentSummary, ItemDetail, SummaryRow
from fibsem.structures import FibsemRectangle, FibsemStagePosition, Point
from fibsem.util.timestamps import to_iso
from fibsem.workflows.attention import Attention
from fibsem.workflows.outputs import item_output_named, saved_file
from fibsem.workflows.protocol_files import record_recent

__all__ = ["VolumeAppContext", "serve"]

# What apply_item_patch edits, each by its experiment method.
POSITION_PATCH_FIELDS = ("name", "iterations", "poi", "alignment_area")


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


def _not_served(what: str) -> Dict[str, Any]:
    return {
        "available": False,
        "applied": False,
        "error_type": "invalid",
        "error": f"Volume does not {what} yet.",
    }


class VolumeAppContext(AppContext):
    """The App Server's view of a volume session."""

    app_name = "Volume"

    # --- experiment record ------------------------------------------------------

    def protocol(self) -> Dict[str, Any]:
        experiment = self._experiment
        protocol = getattr(experiment, "task_protocol", None) if experiment else None
        if protocol is None:
            return {"available": False, "tasks": []}
        plan = protocol.workflow_config
        tasks = []
        for name in protocol.ordered_task_names:
            config = protocol.task_config[name]
            entry = plan.get(name)
            tasks.append(
                {
                    "name": name,
                    "type": config.task_type,
                    "attention": config.attention.value,
                    "supervise": config.attention is Attention.supervised,
                    "required": entry.required if entry is not None else False,
                    "requires": protocol.requirements(name),
                    "scheduled_at": to_iso(entry.scheduled_at) if entry else None,
                }
            )
        return {"available": True, "name": protocol.name, "tasks": tasks}

    summary_type = PositionSummary
    item_detail_type = PositionDetail

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
        experiment = VolumeExperiment.create(
            Path(directory), name, metadata=metadata, task_protocol=protocol
        )
        record_recent(experiment)
        return experiment

    def _load_for_adoption(self, path):
        try:
            return VolumeExperiment.load(Path(path)), None
        except Exception as exc:  # noqa: BLE001 - a broken file is a data fact
            return None, f"could not load {path}: {exc}"

    # --- Items ------------------------------------------------------------------

    def detail_of(self, experiment, item) -> PositionDetail:
        detail = super().detail_of(experiment, item)
        detail.iterations = item.iterations
        detail.slices_cut = experiment.slices_cut(item)
        return detail

    def slice_depth(self, item, iteration: Optional[int]) -> Optional[float]:
        """The iteration's cut times the protocol's slice thickness."""
        from fibsem.applications.volume.tasks.milling import (
            VolumeFIBMillingTaskConfig,
        )

        experiment = self._experiment
        protocol = getattr(experiment, "task_protocol", None)
        if iteration is None or protocol is None:
            return None
        for config in protocol.task_config.values():
            if isinstance(config, VolumeFIBMillingTaskConfig):
                return iteration * config.slice_thickness
        return None

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
        if experiment is None:
            return {"available": False, "jpeg": None}
        position, refusal = find_item(experiment, item_name)
        if refusal:
            return {"available": False, "jpeg": None, **refusal}
        path = item_output_named(experiment, position, filename) or saved_file(
            experiment, experiment.position_path(position), filename
        )
        if path is None:
            return {
                "available": True,
                "jpeg": None,
                "error": f"No output named {filename!r} for {item_name!r}.",
            }
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
            return {"error_type": "not_found", "error": "No experiment is loaded."}
        refusal = self._no_microscope()
        if refusal:
            return refusal
        if overview is not None:
            return {
                "error_type": "invalid",
                "error": "A volume position is added at a stage position, "
                "not on an overview.",
            }
        if stage_position is None:
            refusal = self._not_while_running("add a position where the stage is")
            if refusal:
                refusal["error"] += " Or give its stage_position."
                return refusal
        name = (name or "").strip() or self._next_name(experiment)

        def add():
            state = microscope.get_microscope_state()
            if stage_position is not None:
                state.stage_position = FibsemStagePosition.from_dict(stage_position)
            experiment.add_position(name, state)

        try:
            self._agent_call(add)
        except (ValueError, TypeError, KeyError) as exc:
            return {"error_type": "invalid", "error": str(exc)}
        return {"applied": True, "item": self.item_detail(name)}

    @staticmethod
    def _next_name(experiment) -> str:
        taken = {p.name for p in experiment.positions}
        n = len(taken) + 1
        while f"position-{n:02d}" in taken:
            n += 1
        return f"position-{n:02d}"

    def remove_item(self, item_name: str) -> Dict[str, Any]:
        """Remove a position; its folder stays. Refused for the one the run is on."""
        experiment = self._experiment
        if experiment is None:
            return {"error_type": "not_found", "error": "No experiment is loaded."}
        position, refusal = find_item(experiment, item_name)
        if refusal:
            return {"error_type": "not_found", **refusal}
        refusal = self._busy(self._runner.busy_refusal([item_name], "remove it"))
        if refusal:
            return refusal
        why = self._agent_call(lambda: experiment.remove_position(item_name))
        if why:
            return {"error_type": "invalid", "error": why}
        return {"applied": True, "item_name": item_name}

    def apply_item_patch(
        self,
        item_name: str,
        patch: Dict[str, Any],
        version: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Edit a position's name, slices, point of interest or alignment area,
        against the version :meth:`item_detail` served. Each field is set by
        its experiment method, in that order; the first refusal stops the
        rest, and the answer says which were applied."""
        experiment = self._experiment
        if experiment is None:
            return {"applied": False, "error": "No experiment is loaded."}
        position, refusal = find_item(experiment, item_name)
        if refusal:
            return {"applied": False, **refusal}
        unknown = sorted(set(patch) - set(POSITION_PATCH_FIELDS))
        if unknown:
            return {
                "applied": False,
                "invalid_patch": f"A position patch sets {', '.join(POSITION_PATCH_FIELDS)}; "
                f"not {', '.join(unknown)}.",
            }
        refusal = self._busy(self._runner.busy_refusal([item_name], "edit it"))
        if refusal:
            return {"applied": False, **refusal}
        try:
            edits = self._position_edits(experiment, item_name, patch)
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            return {"applied": False, "invalid_patch": f"Not a position patch: {exc}"}

        def edit():
            if str(version) != self._version(position):
                return {"applied": False, "stale": True}
            applied: List[str] = []
            for field, call in edits:
                why = call()
                if why:
                    return {
                        "applied": bool(applied),
                        "invalid_patch": why,
                        "path": field,
                        "fields": applied,
                    }
                applied.append(field)
            return {"applied": True, "fields": applied}

        result = self._agent_call(edit, timeout=timeout)
        if result.get("applied"):
            result["item"] = self.item_detail(position.name)
        return result

    @staticmethod
    def _position_edits(
        experiment, name: str, patch: Dict[str, Any]
    ) -> List[tuple[str, Callable[[], str]]]:
        """Each patched field's experiment call, in POSITION_PATCH_FIELDS order;
        the rename last, so the others find the position by its old name."""
        edits = []
        if "iterations" in patch:
            iterations = patch["iterations"]
            if isinstance(iterations, bool) or not isinstance(iterations, int):
                raise TypeError("iterations is a whole number")
            edits.append(
                ("iterations", lambda: experiment.set_iterations(name, iterations))
            )
        if "poi" in patch:
            poi = Point.from_dict(patch["poi"])
            edits.append(("poi", lambda: experiment.set_poi(name, poi)))
        if "alignment_area" in patch:
            area = FibsemRectangle.from_dict(patch["alignment_area"])
            edits.append(
                ("alignment_area", lambda: experiment.set_alignment_area(name, area))
            )
        if "name" in patch:
            new_name = patch["name"]
            if not isinstance(new_name, str):
                raise TypeError("name is text")
            edits.append(("name", lambda: experiment.rename_position(name, new_name)))
        return edits


def serve(session) -> bool:
    """Host the App Server over a volume *session*, window or not. Whether it
    is serving (see ``AppSession.start_app_server``)."""
    return session.start_app_server(
        lambda buffer: VolumeAppContext(session, event_buffer=buffer)
    )
