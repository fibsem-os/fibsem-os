"""The image a proposal's values sit on, for whoever looks at it: an agent's
proposal list, the agent card's preview."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Union

from fibsem.fm.structures import FluorescenceImage
from fibsem.structures import FibsemImage
from fibsem.workflows.proposals import Proposal

__all__ = ["load_reference_image", "preview_of"]


def load_reference_image(
    experiment: Any, item: Any, proposal: Proposal, key: str = "reference_image"
) -> Optional[Union[FibsemImage, FluorescenceImage]]:
    """The image the proposal's values sit on, from its provenance: a file
    name relative to the item's folder (``Experiment.item_path``). A
    fluorescence result (an OME-TIFF, channels and planes) loads as a
    ``FluorescenceImage``; a beam image as a ``FibsemImage``."""
    path = proposal.provenance.get(key)
    if not path:
        return None
    path = os.path.join(str(experiment.item_path(item)), path)
    if not os.path.exists(path):
        logging.warning(f"Reference image not found: {path}")
        return None
    fluorescence = path.lower().endswith((".ome.tif", ".ome.tiff"))
    try:
        if fluorescence:
            return FluorescenceImage.load(path)
        return FibsemImage.load(path)
    except Exception:
        logging.exception(f"Could not load the reference image: {path}")
        return None


def preview_of(image: Any) -> Optional[Any]:
    """What an agent is shown of a proposal's image: the beam image itself,
    or a fluorescence result's channel composite (max over z, tinted per
    channel), the same one the FM canvas and the thumbnail show."""
    if isinstance(image, FluorescenceImage):
        from fibsem.fm.preview import composite_projection

        try:
            return composite_projection(image)
        except Exception:
            logging.exception("Could not project the fluorescence image")
            return None
    return image
