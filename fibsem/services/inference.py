"""Inference as a service: the registered models, run on saved images.

The models are `fibsem.ai.models`': the registry names them, the manager loads
each once, and the app and scripts call the same library. The service runs a
model on a saved image *path*, since a route can't carry an image, and saves
the mask, the overlay and the result beside it.

Being a service is what lets an agent use the models over the App Server, under
``/microscope/services/inference``. The agent names a registered model, never a
file: loading a checkpoint can run the code inside it.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fibsem.devices.core import query
from fibsem.services.core import Service
from fibsem.structures import FibsemImage

#: The folder beside an image its results are saved in.
RESULTS_FOLDER = "inference"


class Inference(Service):
    """The registered models: ``list_models`` and ``model_info`` read them,
    ``segment`` runs one on a saved image."""

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
        warnings, and the saved paths. Moves nothing."""
        if not model:
            default = self.manager.registry.default_for("segmentation")
            if default is None:
                raise ValueError("no model is the default for segmentation; name one")
            model = default.name
        entry = self._agent_entry(model)
        image = FibsemImage.load(image_path)
        loaded = self.manager.get(model)
        if loaded.card.task != "segmentation":
            raise ValueError(
                f"{entry.name} is a {loaded.card.task} model, not segmentation"
            )
        result = loaded.segment(image)
        if path is None:
            path = os.path.join(os.path.dirname(image_path), RESULTS_FOLDER)
        stem = f"{os.path.splitext(os.path.basename(image_path))[0]}-{entry.name}"
        saved = result.save(path, stem, image=image.data)
        return {**result.to_dict(), "image_path": image_path, **saved}

    def _agent_entry(self, model: str) -> Any:
        """The entry for *model*. One kept from the agent (``agent: false``) is
        answered as not registered, so it is never named to it."""
        entry, _ = self.manager.entry(model)
        if not entry.agent:
            from fibsem.ai.models import ModelNotFound

            raise ModelNotFound(f"no model '{entry.name}' is registered for the agent")
        return entry
