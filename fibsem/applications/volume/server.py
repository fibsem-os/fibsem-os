"""The volume app over the App Server: the /app core and the Items part, with
each position edit made by the experiment's own method (its refusals, one save).

Not yet served for volume, and refused with why: adding tasks, workflow
options, the experiment and protocol files, and per-position task configs
(positions share the protocol's).
"""

from typing import Any, Callable, Dict, List, Optional

from fibsem.server.app_context import (
    AppContext,
    content_hash,
    find_item,
    item_pose_positions,
)
from fibsem.server.files import jpeg_preview
from fibsem.structures import FibsemRectangle, FibsemStagePosition, Point
from fibsem.util.timestamps import to_iso
from fibsem.workflows.attention import Attention
from fibsem.workflows.outputs import files_by_run, item_output_named, saved_file
from fibsem.workflows.status import TaskStatus

__all__ = ["VolumeAppContext", "serve"]

# What apply_item_patch edits, each by its experiment method.
POSITION_PATCH_FIELDS = ("name", "iterations", "poi", "alignment_area")


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

    def experiment_summary(self) -> Dict[str, Any]:
        experiment = self._experiment
        if experiment is None:
            return {"available": False, "items": []}
        items = [
            {
                "experiment_name": experiment.name,
                "item_name": p.name,
                "item_id": p.id,
                "iterations": p.iterations,
                "slices_cut": experiment.slices_cut(p),
                "failed": any(s.status is TaskStatus.Failed for s in p.task_history),
            }
            for p in experiment.positions
        ]
        return {"available": True, "items": items}

    def task_history(self) -> Dict[str, Any]:
        experiment = self._experiment
        if experiment is None:
            return {"available": False, "items": []}
        items = [
            {
                "item_name": p.name,
                "item_id": p.id,
                "task_name": s.name,
                "task_id": s.task_id,
                "task_type": s.task_type,
                "iteration": s.iteration,
                "task_status": s.status.name,
                "task_status_message": s.status_message,
                "start_timestamp": to_iso(s.start_timestamp),
                "end_timestamp": to_iso(s.end_timestamp),
            }
            for p in experiment.positions
            for s in p.task_history
        ]
        items.sort(key=lambda r: r["start_timestamp"] or "")
        return {"available": True, "items": items}

    # --- not served for volume yet ------------------------------------------------

    def protocols(self) -> Dict[str, Any]:
        return {"folder": None, "protocols": []}

    def create_experiment(self, name, protocol=None, description=None):
        return _not_served("create experiments over the App Server")

    def load_experiment(self, path):
        return _not_served("load experiments over the App Server")

    def load_protocol(self, protocol):
        return _not_served("load protocols")

    def save_protocol_as(self, name, overwrite=False):
        return _not_served("save protocols")

    def add_task(self, task_type, task_name):
        return _not_served("add tasks")

    def set_workflow_options(self, **options):
        return _not_served("set workflow options")

    # --- Items ------------------------------------------------------------------

    def item_detail(self, item_name: str) -> Dict[str, Any]:
        experiment = self._experiment
        if experiment is None:
            return {"available": False, "item_name": item_name}
        position, refusal = find_item(experiment, item_name)
        if refusal:
            return {"available": False, "item_name": item_name, **refusal}
        return {
            "available": True,
            "item_name": position.name,
            "id": position.id,
            "description": position.description,
            # metres, in the milling frame; each slice steps down from it
            "poi": position.poi.to_dict(),
            # fractions of the FIB frame, origin top-left
            "alignment_area": position.alignment_area.to_dict(),
            "iterations": position.iterations,
            "slices_cut": experiment.slices_cut(position),
            "poses": item_pose_positions(position),
            "version": self._version(position),
        }

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

    def task_outputs(self, item_name: str) -> Dict[str, Any]:
        experiment = self._experiment
        if experiment is None:
            return {"available": False, "item_name": item_name}
        position, refusal = find_item(experiment, item_name)
        if refusal:
            return {"available": False, "item_name": item_name, **refusal}
        history = list(position.task_history)
        return {
            "available": True,
            "item_name": item_name,
            "completed_tasks": [
                s.name for s in history if s.status is TaskStatus.Completed
            ],
            "tasks": files_by_run(experiment.position_path(position), history),
        }

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
