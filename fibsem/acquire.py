from __future__ import annotations

from typing import Optional

from fibsem.autofunctions.acb import run_auto_contrast_brightness
from fibsem.microscope import FibsemMicroscope
from fibsem.services.imaging import acquisition_record
from fibsem.structures import (
    BeamType,
    FibsemImage,
    ImageSettings,
)


def acquire_image(microscope: FibsemMicroscope, settings: ImageSettings) -> FibsemImage:
    """Apply the given image settings and acquire a new image: autocontrast when
    asked, acquire, save when asked, record (`microscope.imaging.acquire_image`)."""
    return microscope.imaging.acquire_image(settings)


def acquire_current_image(
    microscope: FibsemMicroscope, beam_type: BeamType, path: Optional[str] = None
) -> FibsemImage:
    """Acquire with the beam as it is set now, and record it.

    For a frame whose point is the beam's current field and resolution -- after
    milling, around a milling strategy -- rather than a protocol's settings,
    which :func:`acquire_image` applies. ``path`` is saved to as given (``.tif``
    added if it has no suffix), without the ``_eb``/``_ib`` suffix
    :func:`acquire_image` adds; None saves nothing.
    """
    image = microscope.acquire_image(image_settings=None, beam_type=beam_type)
    written = image.save(path=path) if path is not None else None
    microscope.record_event("image_acquired", acquisition_record(image, written))
    return image


def take_reference_images(
    microscope: FibsemMicroscope, image_settings: ImageSettings
) -> tuple[FibsemImage, FibsemImage]:
    """An electron and an ion image with the same settings
    (`microscope.imaging.reference_images`)."""
    eb_image, ib_image = microscope.imaging.reference_images(image_settings)
    return eb_image, ib_image


def acquire_channels(
    microscope: FibsemMicroscope,
    image_settings: ImageSettings,
    acquire_sem: bool = True,
    acquire_fib: bool = True,
) -> tuple[Optional[FibsemImage], Optional[FibsemImage]]:
    """An SEM and/or FIB image with the same settings, None for one not asked for
    (`microscope.imaging.reference_images`)."""
    sem_image, fib_image = microscope.imaging.reference_images(
        image_settings, electron=acquire_sem, ion=acquire_fib
    )
    return sem_image, fib_image


def acquire_set_of_channels(
    microscope: FibsemMicroscope,
    image_settings: ImageSettings,
    hfws: tuple[float, ...],
    filename: str = "ref_image",
    acquire_sem: bool = True,
    acquire_fib: bool = True,
) -> list[tuple[Optional[FibsemImage], Optional[FibsemImage]]]:
    """SEM and/or FIB images at each field of view in ``hfws``, saved as
    ``<filename>_res_01``, ``_res_02`` ... (`microscope.imaging.reference_image_sets`)."""
    sets = microscope.imaging.reference_image_sets(
        image_settings, hfws, filename=filename, electron=acquire_sem, ion=acquire_fib
    )
    return [(sem, fib) for sem, fib in sets]
