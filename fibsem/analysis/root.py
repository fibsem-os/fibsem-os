"""The folder a server's files are read from: every path resolves inside it,
symlinks followed, so ``..``, an absolute path elsewhere and a link out of the
folder are refused alike."""

import os
from typing import Dict, Optional


class FileRefusal(Exception):
    """Why a file is not served: an HTTP status, an error type and a message."""

    def __init__(self, status: int, error_type: str, message: str):
        super().__init__(message)
        self.status = status
        self.error_type = error_type
        self.message = message

    @property
    def detail(self) -> Dict[str, str]:
        return {"error_type": self.error_type, "message": self.message}


def resolve(root: Optional[str], path: str) -> str:
    """*path* (relative to *root*, or absolute) as a real path inside *root*,
    or the :class:`FileRefusal` saying why not. A *root* of None is no open
    experiment."""
    if root is None:
        raise FileRefusal(
            409, "no_experiment", "No experiment is open: load or create one first."
        )
    root = os.path.realpath(root)
    target = os.path.realpath(os.path.join(root, path or ""))
    if os.path.commonpath([root, target]) != root:
        raise FileRefusal(
            403,
            "not_allowed",
            f"Only files inside the open experiment ({root}), not {path}.",
        )
    if not os.path.exists(target):
        raise FileRefusal(
            404,
            "not_found",
            f"No file or folder at {path} in {os.path.basename(root)}. Paths are "
            "inside the experiment; for another experiment's files, pass "
            "experiment.",
        )
    return target
