"""Apply dotted-path patches to typed config objects (FIB-864).

The write half of the config documents: a patch is ``{"a.b.c": value, ...}``
against the live dataclass tree a read served. Deliberately a *patch*, never a
document replace — a replace round-trips a possibly-stale read and silently
reverts concurrent edits, while a patch names its intent, validates per field,
and reads legibly in the event stream.

Rules, all enforced here so every caller (per-item now, protocol-level next)
refuses identically:

- Every path segment must already exist: dataclass field, dict key, or list
  index. A patch edits values; it never creates structure.
- The leaf must be a value, not a section — patching ``milling`` wholesale is
  refused; patch the fields inside it.
- The new value must match the old one's type (bool for bool, number for
  number, string for string, an enum member's name for an enum). ``None``
  fields are refused: their type cannot be inferred.
- Where the enclosing dataclass declares ``minimum``/``maximum`` field
  metadata (the same vocabulary the GUI's forms clamp by), the value must be
  in range — otherwise the server stores a number the form then displays
  differently.
- All-or-nothing: every entry is resolved and validated before anything is
  set, so a failing entry leaves the config untouched.

Qt-free and py3.8-clean; the GUI-thread caller owns marshalling.
"""

from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple

__all__ = ["PatchError", "apply_patch"]


class PatchError(ValueError):
    """One patch entry could not be applied; ``path`` names the entry."""

    def __init__(self, path: str, message: str):
        self.path = path
        super().__init__(message)


def apply_patch(root: Any, patch: Dict[str, Any]) -> List[Tuple[str, Any, Any]]:
    """Validate every entry against the live object, then apply them all.

    Returns ``[(path, old, new), ...]`` in patch order. Raises
    :class:`PatchError` before anything is set if any entry is invalid.
    """
    resolved = [_resolve(root, path, value) for path, value in patch.items()]
    changes: List[Tuple[str, Any, Any]] = []
    for setter, path, old, new in resolved:
        setter(new)
        changes.append((path, old, new))
    return changes


def _resolve(root: Any, path: str, value: Any):
    parts = path.split(".")
    if not all(parts):
        raise PatchError(path, f"malformed path {path!r}")
    obj = root
    for part in parts[:-1]:
        obj = _child(obj, part, path)
    leaf = parts[-1]
    old = _child(obj, leaf, path)
    if isinstance(old, (dict, list)) or is_dataclass(old):
        raise PatchError(
            path,
            f"{path!r} names a section, not a value; patch the fields inside it.",
        )
    new = _coerce(old, value, path)
    _check_bounds(obj, leaf, new, path)

    def setter(coerced: Any, obj=obj, leaf=leaf) -> None:
        if isinstance(obj, dict):
            obj[leaf] = coerced
        elif isinstance(obj, list):
            obj[int(leaf)] = coerced
        else:
            setattr(obj, leaf, coerced)

    return setter, path, old, new


def _child(obj: Any, part: str, path: str) -> Any:
    if isinstance(obj, dict):
        if part not in obj:
            raise PatchError(
                path,
                f"{part!r} not found under {path!r}; "
                f"known keys: {sorted(map(str, obj.keys()))}",
            )
        return obj[part]
    if isinstance(obj, list):
        if not part.lstrip("-").isdigit():
            raise PatchError(path, f"{part!r} is not a list index in {path!r}")
        index = int(part)
        if not (0 <= index < len(obj)):
            raise PatchError(
                path, f"index {index} out of range (0..{len(obj) - 1}) in {path!r}"
            )
        return obj[index]
    if is_dataclass(obj):
        names = {f.name for f in dataclass_fields(obj)}
        if part not in names:
            raise PatchError(
                path,
                f"{part!r} is not a field of {type(obj).__name__} in {path!r}; "
                f"fields: {sorted(names)}",
            )
        return getattr(obj, part)
    raise PatchError(
        path, f"cannot descend into {type(obj).__name__} at {part!r} in {path!r}"
    )


def _coerce(old: Any, value: Any, path: str) -> Any:
    if isinstance(old, Enum):
        members = type(old).__members__
        if isinstance(value, str) and value in members:
            return members[value]
        try:
            return type(old)(value)
        except (KeyError, ValueError):
            raise PatchError(
                path,
                f"{value!r} is not a {type(old).__name__}; one of: {sorted(members)}",
            )
    if isinstance(old, bool):
        if not isinstance(value, bool):
            raise PatchError(path, f"{path!r} is a boolean; got {value!r}")
        return value
    if isinstance(old, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PatchError(path, f"{path!r} is a number; got {value!r}")
        return float(value)
    if isinstance(old, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise PatchError(path, f"{path!r} is an integer; got {value!r}")
        return value
    if isinstance(old, str):
        if not isinstance(value, str):
            raise PatchError(path, f"{path!r} is a string; got {value!r}")
        return value
    if old is None:
        raise PatchError(
            path, f"{path!r} is unset; its type cannot be inferred from None."
        )
    raise PatchError(
        path, f"{path!r} holds a {type(old).__name__}, which patches cannot edit."
    )


def _check_bounds(obj: Any, leaf: str, value: Any, path: str) -> None:
    if not is_dataclass(obj) or not isinstance(value, (int, float)):
        return
    for f in dataclass_fields(obj):
        if f.name != leaf:
            continue
        minimum = f.metadata.get("minimum")
        maximum = f.metadata.get("maximum")
        # The form metadata states bounds in *display* units (a depth of
        # 2e-6 m displays as 2.0 with scale 1e6, and min/max bound the 2.0).
        # Compare in that frame, or every valid SI value would be refused.
        scale = f.metadata.get("scale") or 1.0
        display = value * scale
        # No unit label: metadata's "unit" is the SI unit of the STORED value,
        # while min/max bound the scaled display value — labelling either
        # number with it would mislead (2e-3 m scales to 2000.0, not "2000 m").
        if minimum is not None and display < minimum:
            raise PatchError(
                path,
                f"{value!r} (display value {display!r}) is below the minimum "
                f"{minimum!r} for {path!r}.",
            )
        if maximum is not None and display > maximum:
            raise PatchError(
                path,
                f"{value!r} (display value {display!r}) is above the maximum "
                f"{maximum!r} for {path!r}.",
            )
        return
