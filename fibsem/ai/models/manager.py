"""The manager: models loaded once, by name, and run from their cards.

``ModelManager.get`` resolves a name through the registry, finds the model's
card, checks the backend's packages, and loads the model once per name, version
and settings; later calls get the same `Model`. A `Model` checks an image
against its card, runs its backend and builds the result, so a backend is only
load and predict.
"""

import dataclasses
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fibsem.ai.models.card import ModelCard, find_card
from fibsem.ai.models.registry import ModelEntry, ModelRegistry
from fibsem.ai.models.results import SegmentationResult
from fibsem.util import timestamps


class ModelNotFound(LookupError):
    """No model is registered by that name."""


class ModelUnavailable(RuntimeError):
    """The model is registered but can't run here: a missing package, a backend
    this install doesn't have, a version it doesn't hold."""


class InputMismatch(ValueError):
    """The image is not one the model was trained on: the wrong beam, or a pixel
    size outside its range."""


class Model:
    """One loaded model: its entry, card and settings, and the backend running it."""

    def __init__(
        self,
        entry: ModelEntry,
        card: ModelCard,
        card_source: str,
        backend: Any,
        config: Any,
        device: str = "cpu",
    ):
        self.entry = entry
        self.card = card
        self.card_source = card_source
        self.backend = backend
        self.config = config
        self.device = device
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.entry.name

    def check(self, beam: Optional[str], pixel_size: Optional[float]) -> List[str]:
        """Raise InputMismatch for an image the card rules out; answer the
        warnings for one it can't check (an unknown pixel size)."""
        spec = self.card.input
        if spec.beams and beam is not None and beam not in spec.beams:
            raise InputMismatch(
                f"{self.name} takes {' or '.join(spec.beams)} images, not {beam}."
            )
        if spec.pixel_size_range is None:
            return []
        if pixel_size is None:
            return [
                f"the pixel size is unknown; {self.name} was not checked against it"
            ]
        low, high = spec.pixel_size_range
        if not low <= pixel_size <= high:
            raise InputMismatch(
                f"{self.name} was trained at {low * 1e9:.0f}-{high * 1e9:.0f} nm "
                f"pixels; this image is {pixel_size * 1e9:.0f} nm. Take the image "
                "at another field width, or pick a model trained at this scale."
            )
        return []

    def segment(
        self,
        image: Any,
        pixel_size: Optional[float] = None,
        beam: Optional[str] = None,
    ) -> SegmentationResult:
        """The segmentation of *image*: a `FibsemImage`, whose own pixel size and
        beam are used, or an array with *pixel_size* (metres) and *beam*."""
        data = getattr(image, "data", image)
        metadata = getattr(image, "metadata", None)
        if metadata is not None:
            pixel_size = metadata.pixel_size.x
            beam = metadata.image_settings.beam_type.name
        warnings = self.check(beam, pixel_size)
        started = time.perf_counter()
        with self._lock:  # one call at a time per loaded model
            mask = self.backend.predict(np.asarray(data))
        return SegmentationResult(
            mask=np.asarray(mask, dtype=np.uint8),
            classes=self.card.classes,
            pixel_size=pixel_size,
            provenance={
                "model": self.name,
                "version": self.card.version,
                "backend": self.card.backend,
                "device": self.device,
                "config": dataclasses.asdict(self.config),
                "seconds": round(time.perf_counter() - started, 3),
                "beam": beam,
                "created_at": timestamps.now_iso(),
            },
            warnings=warnings,
        )


class ModelManager:
    """Models by name, loaded once and kept until unloaded."""

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry if registry is not None else ModelRegistry.load()
        self._loaded: Dict[Tuple[str, str, str], Model] = {}
        self._lock = threading.Lock()

    def entry(self, name: str) -> Tuple[ModelEntry, str]:
        """The entry for *name* (``name@version``) and the version asked for."""
        try:
            return self.registry.resolve(name)
        except LookupError as error:
            raise ModelNotFound(str(error)) from None

    def card(self, name: str) -> Tuple[ModelCard, str]:
        """The card for *name*, and where it came from: ``model``, ``beside``,
        ``registry``, or ``backend`` for the backend's defaults."""
        entry, _ = self.entry(name)
        return _card_for(entry)

    def get(self, name: str, **config: Any) -> Model:
        """The model registered as *name*, loaded once per version and settings.
        *config* overrides the entry's settings for this model."""
        entry, version = self.entry(name)
        card, source = _card_for(entry)
        if version and version != card.version:
            raise ModelUnavailable(
                f"{entry.name} {version} is asked for; {entry.ref} holds {card.version}."
            )
        settings = {**entry.config, **config}
        key = (entry.name, card.version, repr(sorted(settings.items())))
        with self._lock:
            model = self._loaded.get(key)
            if model is None:
                model = self._loaded[key] = _load(entry, card, source, settings)
        return model

    def status(self, name: str) -> Dict[str, Any]:
        """What the model is and whether it runs here, without loading it."""
        entry, _ = self.entry(name)
        answer: Dict[str, Any] = {"name": entry.name, "agent": entry.agent}
        try:
            card, source = _card_for(entry)
        except ModelUnavailable as error:
            return {**answer, "status": "unavailable", "reason": str(error)}
        missing = _backend_class(card).requirements(card)
        loaded = [model for key, model in self._loaded.items() if key[0] == entry.name]
        answer.update(
            task=card.task,
            version=card.version,
            backend=card.backend,
            card_source=source,
            classes=[info.name for info in card.classes.values()],
            default_for=list(entry.default_for),
            status="missing packages" if missing else "ready",
            loaded=bool(loaded),
        )
        if missing:
            answer["missing"] = missing
        return answer

    def unload(self, name: str) -> None:
        """Unload every loaded copy of *name*."""
        entry, _ = self.entry(name)
        with self._lock:
            for key in [key for key in self._loaded if key[0] == entry.name]:
                self._loaded.pop(key).backend.unload()


def _backend_class(card: ModelCard) -> Any:
    from fibsem.ai.models.backends import BACKENDS

    backend = BACKENDS.get(card.backend)
    if backend is None:
        raise ModelUnavailable(
            f"{card.id} runs on the '{card.backend}' backend, which this install "
            f"doesn't have; it has: {', '.join(sorted(BACKENDS))}."
        )
    return backend


def _card_for(entry: ModelEntry) -> Tuple[ModelCard, str]:
    builtin = entry.builtin_backend
    if builtin is not None:
        from fibsem.ai.models.backends import BACKENDS

        if builtin not in BACKENDS:
            raise ModelUnavailable(f"{entry.name}: no built-in backend '{builtin}'.")
        return BACKENDS[builtin].default_card(), "backend"
    card, source = find_card(entry.ref, entry.card)
    if card is None:
        raise ModelUnavailable(
            f"{entry.name} has no card: none inside {entry.ref}, none beside it, "
            "and no card: block on its registry entry."
        )
    return card, source


def _load(entry: ModelEntry, card: ModelCard, source: str, settings: Dict) -> Model:
    backend_class = _backend_class(card)
    missing = backend_class.requirements(card)
    if missing:
        raise ModelUnavailable(f"{entry.name} needs {', '.join(missing)}.")
    try:
        config = backend_class.Config(**settings)
    except TypeError as error:
        raise ValueError(f"{entry.name}: {error}") from None
    backend = backend_class()
    device = "cpu"
    backend.load(entry.ref, card, config, device)
    return Model(entry, card, source, backend, config, device)


_default: Optional[ModelManager] = None
_default_lock = threading.Lock()


def default_manager() -> ModelManager:
    """The manager the app, the service and `fibsem.ai.models.get` share, over
    the registry in the fibsem config folder."""
    global _default
    with _default_lock:
        if _default is None:
            _default = ModelManager()
        return _default
