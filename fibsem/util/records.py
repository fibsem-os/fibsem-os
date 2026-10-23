"""Dataframe rows as JSON-able records."""

import hashlib
import json
import math
from datetime import datetime
from typing import Any, Dict, List


def json_safe(value: Any) -> Any:
    """Coerce one cell of a dataframe/record into something JSON can carry."""
    if isinstance(value, datetime):
        # pandas' NaT is a datetime too: a time not yet known, as NaN is for a number
        return None if value != value else value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item") and not isinstance(value, str):
        # numpy scalars → native python
        try:
            return json_safe(value.item())
        except (AttributeError, ValueError):
            pass
    if hasattr(value, "isoformat"):
        # pandas.Timestamp and friends
        return value.isoformat()
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value
    return str(value)


def dataframe_records(df) -> List[Dict[str, Any]]:
    """*df*'s rows as dicts of JSON-able values."""
    return [
        {key: json_safe(val) for key, val in record.items()}
        for record in df.to_dict(orient="records")
    ]


def content_hash(data) -> str:
    """A short hash naming exactly this content: a document's version."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
