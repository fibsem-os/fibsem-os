"""The model card: what a model is, read from the model itself.

A card names the model and its version, the task it does, the backend that runs
it, its classes, and the images it was trained on. It is saved as
``fibsem_model.json`` inside the model, and the manager checks an image against
it before running anything.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

CARD_FILENAME = "fibsem_model.json"


@dataclass
class ClassInfo:
    """One class a model labels: its name and the colour it is drawn in."""

    name: str
    colour: str = ""


@dataclass
class InputSpec:
    """The images a model was trained on. *beams* empty means any beam;
    *pixel_size_range* (metres) None means any pixel size."""

    beams: List[str] = field(default_factory=list)
    channels: int = 1
    pixel_size_range: Optional[Tuple[float, float]] = None


@dataclass
class ModelCard:
    """What a model is: its id and version, its task and backend, its classes by
    id, the images it takes, the packages it needs, and where it came from."""

    id: str
    version: str
    backend: str
    task: str = "segmentation"
    classes: Dict[int, ClassInfo] = field(default_factory=dict)
    input: InputSpec = field(default_factory=InputSpec)
    packages: List[str] = field(default_factory=list)
    gpu: bool = False
    provenance: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # JSON keys are strings; the ids come back as ints in from_dict.
        data["classes"] = {str(k): v for k, v in data["classes"].items()}
        if self.input.pixel_size_range is not None:
            data["input"]["pixel_size_range"] = list(self.input.pixel_size_range)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelCard":
        data = dict(data)
        data["classes"] = {
            int(k): ClassInfo(**v) for k, v in data.get("classes", {}).items()
        }
        spec = dict(data.get("input") or {})
        if spec.get("pixel_size_range") is not None:
            spec["pixel_size_range"] = tuple(spec["pixel_size_range"])
        data["input"] = InputSpec(**spec)
        return cls(**data)


def find_card(
    ref: str, card: Optional[Mapping[str, Any]] = None
) -> Tuple[Optional[ModelCard], str]:
    """The card for the model at *ref*, and where it was found, first found wins:

    1. ``fibsem_model.json`` inside the model folder;
    2. ``<model>.card.json`` beside it, for a model we can't change (a vendor's,
       a third party's, an older checkpoint);
    3. *card*, the ``card:`` block on its registry entry.

    ``(None, "")`` when there is none: the backend's defaults stand in.
    """
    inside = os.path.join(ref, CARD_FILENAME)
    if os.path.isdir(ref) and os.path.isfile(inside):
        return _read(inside), "model"
    beside = os.path.splitext(ref.rstrip("/\\"))[0] + ".card.json"
    if os.path.isfile(beside):
        return _read(beside), "beside"
    if card:
        return ModelCard.from_dict(card), "registry"
    return None, ""


def _read(path: str) -> ModelCard:
    with open(path) as f:
        return ModelCard.from_dict(json.load(f))
