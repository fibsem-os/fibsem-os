"""A task config's version and the version-guarded patch every app's agent
edits go through (FIB-864): reads serve the version, writes echo it, and a
mismatch is refused as stale."""

from typing import Any, Dict, List, Optional, Tuple

from fibsem.server.app_context import content_hash
from fibsem.server.config_patch import PatchError, apply_patch
from fibsem.workflows.events import to_plain
from fibsem.workflows.task_config import WORKFLOW_FIELDS

__all__ = [
    "WORKFLOW_FIELD_VERBS",
    "config_version",
    "patch_config",
    "plain_changes",
    "workflow_field_refusal",
]


def config_version(config) -> str:
    """A content hash naming exactly this state of a task config's settings.

    Computed from the serialized document, less the task's name, so the same
    content hashes identically wherever it is held.
    """
    data = config.to_dict()
    for name in WORKFLOW_FIELDS:
        data.pop(name, None)
    return content_hash(to_plain(data))


WORKFLOW_FIELD_VERBS = {
    "attention": "protocol_set_supervision",
    "requires": "protocol_set_task_requirements",
}


def workflow_field_refusal(patch: dict) -> Optional[dict]:
    """Why a task-config *patch* is refused for naming attention or requires,
    as an ``invalid_patch`` answer; None when it names neither. Both are the
    plan entry's, with their own verbs."""
    for path in patch:
        field = str(path).split(".", 1)[0]
        if field in WORKFLOW_FIELD_VERBS:
            return {
                "invalid_patch": f"{field} is not set by a config patch; use "
                f"{WORKFLOW_FIELD_VERBS[field]}.",
                "path": path,
            }
    return None


def plain_changes(changes: List[Tuple[str, Any, Any]]) -> List[Dict[str, Any]]:
    return [
        {"path": p, "old": to_plain(old), "new": to_plain(new)}
        for p, old, new in changes
    ]


def patch_config(config, patch: dict, version: str) -> Dict[str, Any]:
    """Patch *config* when *version* names its state: what changed and its new
    version, or why not, with nothing changed. Saving is the caller's."""
    refused = workflow_field_refusal(patch)
    if refused:
        return {"applied": False, **refused}
    if config_version(config) != version:
        return {"applied": False, "stale": True}
    try:
        changes = apply_patch(config, patch)
    except PatchError as exc:
        return {"applied": False, "invalid_patch": str(exc), "path": exc.path}
    return {
        "applied": True,
        "changes": plain_changes(changes),
        "version": config_version(config),
    }
