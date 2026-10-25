"""Which optional extras are installed, read from fibsem's own package metadata.

A command or feature that needs an extra asks ``missing_extra_message`` and prints
what it returns, so the packages named and the install line always match
``pyproject.toml``.
"""

import importlib.metadata
from typing import List, Optional

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement

__all__ = ["missing_extra_message", "missing_packages"]


def _requirements(extra: str) -> List[Requirement]:
    """The extra's requirements that apply on this platform, nested extras expanded."""
    try:
        declared = importlib.metadata.requires("fibsem") or []
    except importlib.metadata.PackageNotFoundError:
        return []
    found: List[Requirement] = []
    seen = set()
    pending = [extra]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        environment = {**default_environment(), "extra": name}
        for line in declared:
            try:
                requirement = Requirement(line)
            except InvalidRequirement:
                continue
            marker = requirement.marker
            if marker is None or "extra" not in str(marker):
                continue
            if not marker.evaluate(environment):
                continue
            if requirement.name == "fibsem":
                pending.extend(requirement.extras)
            else:
                found.append(requirement)
    return found


def missing_packages(extra: str) -> List[str]:
    """The packages of *extra* that are not installed, in declaration order."""
    missing = []
    for requirement in _requirements(extra):
        try:
            importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(requirement.name)
    return missing


def missing_extra_message(extra: str, what: str) -> Optional[str]:
    """Why *what* can't run without *extra*, or None when the extra is installed."""
    missing = missing_packages(extra)
    if not missing:
        return None
    return (
        f"{what} needs the [{extra}] extra ({', '.join(missing)} not installed): "
        f'pip install "fibsem[{extra}]"'
    )
