"""The latest image of each beam: what the window shows, and what the App Server
serves and click-to-move reads (FIB-1451).

Held on the app's session rather than on the image widget, so a reader needs no
widget. Each image is replaced whole, never changed in place, so a reader on
another thread gets one image or the other, not half of each.
"""

from typing import Dict, Optional

from fibsem.structures import BeamType, FibsemImage


class LatestImages:
    """The latest image of each beam, by its metadata's beam type."""

    def __init__(self) -> None:
        self._images: Dict[BeamType, FibsemImage] = {}

    def set(self, image: FibsemImage, beam_type: Optional[BeamType] = None) -> None:
        """Keep *image* as the latest of *beam_type*, or of the beam its
        metadata names. An image with neither is not kept: there is no beam to
        show it on."""
        if beam_type is None:
            beam_type = getattr(getattr(image, "metadata", None), "beam_type", None)
        if isinstance(beam_type, BeamType):
            self._images[beam_type] = image

    def get(self, beam_type: BeamType) -> Optional[FibsemImage]:
        """*beam_type*'s latest image, or None before the first."""
        return self._images.get(beam_type)
