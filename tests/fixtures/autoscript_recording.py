"""Reading the Thermo parity recordings so they compare across numpy versions.

numpy 2 writes a scalar as ``np.float64(x)`` in a repr, numpy 1 (Python 3.8) as
``x``, and the two can differ in a float's last digit. A recording is compared with
the reprs dropped and every float, bare or in a message, to 12 significant digits.
"""

import json
import re
from typing import Any

_NP_SCALAR = re.compile(r"np\.float64\(([^()]*)\)")
_FLOAT = re.compile(r"-?\d+\.\d+(?:e[-+]?\d+)?")


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return float(f"{value:.12g}")
    if isinstance(value, str):
        return _FLOAT.sub(lambda m: repr(float(f"{float(m.group()):.12g}")), value)
    if isinstance(value, list):
        return [_round(v) for v in value]
    if isinstance(value, dict):
        return {_round(k): _round(v) for k, v in value.items()}
    return value


def load(text: str) -> Any:
    """The recording in *text*, numpy's scalar reprs dropped and floats rounded."""
    return _round(json.loads(_NP_SCALAR.sub(r"\1", text)))
