"""Inference as a service: the registered models, run on saved images.

The models are `fibsem.ai.models`': the registry names them, the manager loads
each once, and the app and scripts call the same library. The service runs a
model on a saved image *path*, since a route can't carry an image, and saves
the mask, the overlay and the result beside it.

Being a service is what lets an agent use the models over the App Server, under
``/services/inference`` (`fibsem.services.analysis`), with or without a
microscope. The agent names a registered model, never a
file: loading a checkpoint can run the code inside it.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from fibsem.devices.core import query
from fibsem.imaging.tiling.result import METADATA_FILE
from fibsem.services.core import Service
from fibsem.structures import FibsemImage

#: The folder beside an image its results are saved in.
RESULTS_FOLDER = "inference"


class Inference(Service):
    """The registered models: ``list_models`` and ``model_info`` read them,
    ``segment`` and ``detect`` run one on a saved image."""

    def __init__(self, name: str = "inference", manager: Any = None, **kwargs: Any):
        super().__init__(name, **kwargs)
        self._manager = manager

    @property
    def manager(self) -> Any:
        """The model manager: the one given, else the app's default."""
        if self._manager is None:
            from fibsem.ai.models import default_manager

            self._manager = default_manager()
        return self._manager

    @query
    def list_models(self) -> List[Dict[str, Any]]:
        """Every model the agent may use: its name, task, version, backend,
        classes, the tasks it is the default for, and whether it is ready
        (``ready``, ``missing packages``, ``unavailable``) and loaded. Loads
        nothing."""
        return [
            self.manager.status(entry.name)
            for entry in self.manager.registry.entries()
            if entry.agent
        ]

    @query
    def model_info(self, model: str) -> Dict[str, Any]:
        """The card of the model registered as *model*: its classes and colours,
        the images it takes, the packages it needs and where it came from, with
        where the card was read. Loads nothing."""
        entry = self._agent_entry(model)
        card, source = self.manager.card(entry.name)
        return {**card.to_dict(), "card_source": source}

    @query
    def segment(
        self, image_path: str, model: str = "", path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Segment the saved image at *image_path* with the model registered as
        *model* (``name@version`` pins a version; left out, the default for
        segmentation). Checks the image against the model's card first: the
        beam, and the pixel size it was trained at.

        Saves the mask, an overlay and the result in *path*, by default an
        ``inference`` folder beside the image, named after the image and the
        model. Answers each class's share of the image, area in µm² and region
        count, which model made it (name, version, backend, settings, seconds),
        warnings, and the saved paths. Each class lists its largest regions'
        centres in pixels, for a stage position from the image's metadata.

        A stitched overview (its tiles' folder and record beside it) runs a
        tile at a time, at the tiles' size and pixel size. Moves nothing."""
        loaded = self._segmentation_model(model, "segmentation")
        image = FibsemImage.load(image_path)
        overview = overview_tiles(image_path)
        result = loaded.segment(image, tile=overview)
        saved = _save(result, image, image_path, loaded.name, path)
        answer = {**result.to_dict(), "image_path": image_path, **saved}
        if overview is not None:
            answer["tile_px"] = list(overview)
        return answer

    @query
    def detect(
        self,
        image_path: str,
        features: List[str],
        model: str = "",
        point: Optional[List[float]] = None,
        path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Find *features* (names, such as ``LamellaCentre``) in the saved image
        at *image_path*, in the segmentation of the model registered as *model*
        (left out, the default for detection). *point* ([x, y] pixels) picks
        among several of one feature: the closest wins.

        Answers each feature's position in pixels (x, y; +y down) and in metres
        from the image centre (+y up), the image's pixel size, and which model
        made it. Saves the mask, an overlay and the result as ``segment`` does.
        Moves nothing."""
        from fibsem.ai import detection
        from fibsem.structures import Point

        if not features:
            raise ValueError("name at least one feature to detect")
        found = [detection.feature(name) for name in features]
        loaded = self._segmentation_model(model, "detection")
        image = FibsemImage.load(image_path)
        if point is not None:
            point = Point(float(point[0]), float(point[1]))
        det, result = detection.detect(loaded, image, found, point=point)
        saved = _save(result, image, image_path, loaded.name, path)
        return {
            "features": [f.to_dict() for f in det.features],
            "pixel_size_m": det.pixelsize,
            "model": result.provenance,
            "warnings": list(result.warnings),
            "image_path": image_path,
            **saved,
        }

    def _segmentation_model(self, model: str, task: str) -> Any:
        """The loaded segmentation model registered as *model*, or, left out,
        the default for *task* (a detection runs on a segmentation model)."""
        if not model:
            default = self.manager.registry.default_for(task)
            if default is None:
                raise ValueError(f"no model is the default for {task}; name one")
            model = default.name
        entry = self._agent_entry(model)
        loaded = self.manager.get(model)
        if loaded.card.task != "segmentation":
            raise ValueError(
                f"{entry.name} is a {loaded.card.task} model, not segmentation"
            )
        return loaded

    def _agent_entry(self, model: str) -> Any:
        """The entry for *model*. One kept from the agent (``agent: false``) is
        answered as not registered, so it is never named to it."""
        entry, _ = self.manager.entry(model)
        if not entry.agent:
            from fibsem.ai.models import ModelNotFound

            raise ModelNotFound(f"no model '{entry.name}' is registered for the agent")
        return entry


def _save(
    result: Any, image: FibsemImage, image_path: str, model: str, path: Optional[str]
) -> Dict[str, str]:
    """Save *result* in *path*, by default the ``inference`` folder beside the
    image, named after the image and the model. Answers the saved paths."""
    if path is None:
        path = os.path.join(os.path.dirname(image_path), RESULTS_FOLDER)
    stem = f"{os.path.splitext(os.path.basename(image_path))[0]}-{model}"
    return result.save(path, stem, image=image.data)


def overview_tiles(image_path: str) -> Optional[Tuple[int, int]]:
    """The tile size (height, width in pixels) of the stitched overview at
    *image_path*, read from the record in its tiles' folder beside it, or None
    for an image that is not one. The record was ``tile-layout.json`` before it
    was ``overview-metadata.json``; both are read."""
    folder = os.path.splitext(image_path)[0]
    for name in (METADATA_FILE, "tile-layout.json"):
        record = os.path.join(folder, name)
        if not os.path.isfile(record):
            continue
        try:
            with open(record) as f:
                settings = json.load(f).get("settings") or {}
            width, height = settings["image_settings"]["resolution"]
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return int(height), int(width)
    return None
