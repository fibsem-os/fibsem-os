"""Whether a backend's packages are installed, asked without importing them:
listing the models must not load torch."""

import importlib.util
from typing import Iterable, List


def missing(packages: Iterable[str]) -> List[str]:
    """The *packages* (import names) not installed here."""
    return [name for name in packages if importlib.util.find_spec(name) is None]
