from __future__ import annotations

import copy
import os
from typing import Optional

from fibsem.autofunctions.acb import run_auto_contrast_brightness
from fibsem.microscope import FibsemMicroscope
from fibsem.services.imaging import acquisition_record
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemRectangle,
    ImageSettings,
    ReferenceImages,
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


def take_set_of_reference_images(
    microscope: FibsemMicroscope,
    image_settings: ImageSettings,
    hfws: tuple[float, float],
    filename: str = "ref_image",
) -> ReferenceImages:
    """Reference images of both beams at a low and a high field of view
    (``hfws``), saved as ``<filename>_low_res`` and ``<filename>_high_res``. The
    settings given are not changed."""
    settings = copy.deepcopy(image_settings)
    settings.save = True
    pairs = []
    for hfw, label in zip(hfws, ("low_res", "high_res")):
        settings.hfw = hfw
        settings.filename = f"{filename}_{label}"
        pairs.append(microscope.imaging.reference_images(settings))
    (low_eb, low_ib), (high_eb, high_ib) = pairs
    return ReferenceImages(low_eb, high_eb, low_ib, high_ib)


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


##### FOCUS STACKING
def acquire_focus_stacked_image(
    microscope: FibsemMicroscope,
    image_settings: ImageSettings,
    n_steps: int = 3,
    auto_focus: bool = True,
) -> FibsemImage:
    """Acquire a focus stacked image by taking multiple images at different vertical positions.

    This function divides the field of view into vertical strips, acquires images for each strip
    (optionally with autofocus), and stacks them together into a single image.

    Args:
        microscope: The microscope connection.
        image_settings: The image settings to use for acquisition.
        n_steps: The number of vertical steps to divide the image into. Default is 3.
        auto_focus: Whether to perform autofocus for each strip. Default is True.

    Returns:
        The focus stacked FibsemImage.

    Example:
        >>> # Acquire a 3-strip focus stacked image with autofocus
        >>> image_settings = ImageSettings(
        ...     resolution=[768, 512],
        ...     beam_type=BeamType.ION,
        ...     hfw=104e-6,
        ... )
        >>> stacked = acquire_focus_stacked_image(microscope, image_settings, n_steps=3)

        >>> # Acquire a 5-strip focus stacked image without autofocus
        >>> stacked = acquire_focus_stacked_image(
        ...     microscope, image_settings, n_steps=5, auto_focus=False
        ... )
    """
    import numpy as np
    from skimage.transform import resize

    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")

    # fraction of the image height for each strip
    strip_height = 1.0 / n_steps

    # A copy: the strips rewrite `save` and `reduced_area`, and the caller's settings
    # were left scanning the last strip with saving turned off.
    image_settings = copy.copy(image_settings)
    images: list[FibsemImage] = []
    image_settings.save = False  # ensure we don't save intermediate strips
    for i in range(n_steps):
        # calculate the reduced area for this strip
        reduced_area = FibsemRectangle(
            left=0, top=i * strip_height, width=1, height=strip_height
        )
        image_settings.reduced_area = reduced_area

        # Perform autofocus if requested
        if auto_focus:
            microscope.auto_focus(
                beam_type=image_settings.beam_type, reduced_area=reduced_area
            )

        image = acquire_image(microscope, image_settings)
        images.append(image)

    # restore full-frame scanning mode
    microscope.set_full_frame_scanning_mode(image_settings.beam_type)

    # stack images vertically
    arr = np.vstack([img.data for img in images])

    # resize if necessary to match the expected resolution
    if arr.shape != image_settings.resolution[::-1]:
        arr = resize(arr, image_settings.resolution[::-1], preserve_range=True).astype(
            np.uint8
        )  # type: ignore

    # Create the stacked FibsemImage using metadata from the middle strip
    # (or the last middle strip if even number of strips)
    middle_index = (n_steps - 1) // 2
    stacked_image = FibsemImage(
        data=arr, metadata=copy.deepcopy(images[middle_index].metadata)
    )

    # update metadata to reflect that reduced_area is now the full image
    stacked_image.metadata.image_settings.reduced_area = None  # type: ignore

    # set filename
    if image_settings.beam_type is BeamType.ELECTRON:
        filename = f"{image_settings.filename}_eb"

    if image_settings.beam_type is BeamType.ION:
        filename = f"{image_settings.filename}_ib"

    stacked_image.save(os.path.join(image_settings.path, f"{filename}"))  # type: ignore

    return stacked_image
