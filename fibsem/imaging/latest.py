"""The latest image of each beam: what the window shows, and what the App Server
serves and click-to-move reads (FIB-1451).

Held on the app's session rather than on the image widget, so a reader needs no
widget. Each image is replaced whole, never changed in place, so a reader on
another thread gets one image or the other, not half of each.
"""

from typing import Any, Callable, Dict, List, Optional

from fibsem.structures import BeamType, FibsemImage


class LatestImages:
    """The latest image of each beam, by its metadata's beam type."""

    def __init__(self) -> None:
        self._images: Dict[BeamType, FibsemImage] = {}
        self._unfollow: List[Callable[[], None]] = []

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

    def follow(self, microscope: Optional[Any]) -> None:
        """Keep what *microscope*'s views are sent: each image shown
        (``imaging.image_shown``) and each live frame, window or none. Stops
        following the one before; None follows nothing."""
        for unfollow in self._unfollow:
            unfollow()
        self._unfollow = []
        if microscope is None:
            return
        imaging = getattr(microscope, "imaging", None)
        signals = [imaging.image_shown] if imaging is not None else []
        signals += [
            beam.live_frame for beam in getattr(microscope, "beams", {}).values()
        ]
        for signal in signals:
            signal.connect(self.set)
            self._unfollow.append(lambda signal=signal: signal.disconnect(self.set))
