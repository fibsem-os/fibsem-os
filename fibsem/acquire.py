from __future__ import annotations

import copy
import os
from typing import Optional, Tuple

from fibsem.imaging import autogamma
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemRectangle,
    ImageSettings,
    ReferenceImages,
)


def new_image(
    microscope: FibsemMicroscope,
    settings: ImageSettings,
) -> FibsemImage:
    """Apply the given image settings and acquire a new image.

    Args:
        microscope (FibsemMicroscope): The FibsemMicroscope instance used to acquire the image.
        settings (ImageSettings): The image settings used to acquire the image.

    Returns:
        FibsemImage: The acquired image.
    """

    # set filename
    if settings.beam_type is BeamType.ELECTRON:
        filename = f"{settings.filename}_eb"

    if settings.beam_type is BeamType.ION:
        filename = f"{settings.filename}_ib"

    # run autocontrast
    if settings.autocontrast:
        microscope.autocontrast(beam_type=settings.beam_type, 
                                reduced_area=settings.reduced_area)

    # acquire the image
    image = microscope.acquire_image(
        image_settings=settings,
    )

    if settings.autogamma:
        image = autogamma.auto_gamma(image, method="autogamma")

    # save image
    if settings.save:
        filename = os.path.join(settings.path, filename)
        image.save(path=filename)

    return image

def acquire_image(microscope:FibsemMicroscope, settings:ImageSettings) -> FibsemImage:
    """ passthrough for new_image to match internal api"""
    return new_image(microscope, settings)

def last_image(microscope: FibsemMicroscope, beam_type: BeamType) -> FibsemImage:
    """_summary_

    Args:
        microscope (FibsemMicroscope): microscope instance
        beam_type (BeamType): beam type for image

    Returns:
        FibsemImage: last image acquired by the microscope
    """
    return microscope.last_image(beam_type=beam_type)



def take_reference_images(
    microscope: FibsemMicroscope, image_settings: ImageSettings
) -> Tuple[FibsemImage, FibsemImage]:
    """
    Acquires a pair of electron and ion reference images using the specified imaging settings and
    a FibsemMicroscope instance.

    Args:
        microscope (FibsemMicroscope): A FibsemMicroscope instance for imaging.
        image_settings (ImageSettings): An ImageSettings object with the desired imaging parameters.

    Returns:
        A tuple containing a pair of FibsemImage objects, representing the electron and ion reference
        images acquired using the specified microscope and image settings.

    """
    import time

    from fibsem.microscopes.tescan import TescanMicroscope # TODO: handle this better

    tmp_beam_type = image_settings.beam_type
    
    # acquire electron image
    image_settings.beam_type = BeamType.ELECTRON
    eb_image = acquire_image(microscope, image_settings)
    
    # acquire ion image
    image_settings.beam_type = BeamType.ION
    if isinstance(microscope, TescanMicroscope):
        time.sleep(1)
    ib_image = acquire_image(microscope, image_settings)
    image_settings.beam_type = tmp_beam_type  # reset to original beam type

    return eb_image, ib_image
   

def take_set_of_reference_images(
    microscope: FibsemMicroscope,
    image_settings: ImageSettings,
    hfws: Tuple[float, float],
    filename: str = "ref_image",
) -> ReferenceImages:
    """
    Takes a set of reference images at low and high magnification using a FibsemMicroscope.
    The image settings and half-field widths for the low- and high-resolution images are
    specified using an ImageSettings object and a tuple of two floats, respectively.
    The optional filename parameter can be used to customize the image labels.

    Args:
        microscope (FibsemMicroscope): A FibsemMicroscope object to acquire the images from.
        image_settings (ImageSettings): An ImageSettings object with the desired imaging parameters.
        hfws (Tuple[float, float]): A tuple of two floats specifying the horizontal field widths (in microns)
            for the low- and high-resolution images, respectively.
        filename (str, optional): A filename to be included in the image filenames. Defaults to "ref_image".

    Returns:
        A ReferenceImages object containing the low- and high-resolution electron and ion beam images.

    Notes:
        This function sets image_settings.save to True before taking the images.
        The returned ReferenceImages object contains the electron and ion beam images as FibsemImage objects.
    """
    # force save
    image_settings.save = True

    image_settings.hfw = hfws[0]
    image_settings.filename = f"{filename}_low_res"
    low_eb, low_ib = take_reference_images(microscope, image_settings)

    image_settings.hfw = hfws[1]
    image_settings.filename = f"{filename}_high_res"
    high_eb, high_ib = take_reference_images(microscope, image_settings)

    reference_images = ReferenceImages(low_eb, high_eb, low_ib, high_ib)


    # more flexible version
    # reference_images = []
    # for i, hfw in enumerate(hfws):
    #     image_settings.hfw = hfw
    #     image_settings.filename = f"{filename}_res_{i:02d}"
    #     eb_image, ib_image = take_reference_images(microscope, image_settings)
    #     reference_images.append([eb_image, ib_image])

    return reference_images


def acquire_channels(microscope: FibsemMicroscope,
                           image_settings: ImageSettings,
                           acquire_sem: bool = True,
                           acquire_fib: bool = True) -> tuple[Optional[FibsemImage], Optional[FibsemImage]]:
    """Acquire SEM and/or FIB images based on the specified flags.
    Args:
        microscope (FibsemMicroscope): The microscope instance to use for image acquisition.
        image_settings (ImageSettings): The settings to use for image acquisition.
        acquire_sem (bool, optional): Whether to acquire an SEM image. Defaults to True.
        acquire_fib (bool, optional): Whether to acquire a FIB image. Defaults to True.
    Returns:
        tuple[Optional[FibsemImage], Optional[FibsemImage]]: A tuple containing the acquired SEM and FIB images.
            If a particular image type was not acquired, its corresponding value will be None.
    """

    sem_image: Optional[FibsemImage] = None
    fib_image: Optional[FibsemImage] = None
    if acquire_sem:
        image_settings.beam_type = BeamType.ELECTRON
        sem_image = acquire_image(microscope, image_settings)
    if acquire_fib:
        image_settings.beam_type = BeamType.ION
        fib_image = acquire_image(microscope, image_settings)
    return sem_image, fib_image

def acquire_set_of_channels(microscope: FibsemMicroscope, 
                            image_settings: ImageSettings, 
                            hfws: tuple[float, ...],
                            filename: str = "ref_image",
                            acquire_sem: bool = True, 
                            acquire_fib: bool = True) -> list[tuple[Optional[FibsemImage], Optional[FibsemImage]]]:
    """Acquire a set of SEM and/or FIB images at different horizontal field widths (hfws).
    Args:
        microscope (FibsemMicroscope): The microscope instance to use for image acquisition.
        image_settings (ImageSettings): The settings to use for image acquisition.
        hfws (list[float]): A list containing the horizontal field widths for the images.
        acquire_sem (bool, optional): Whether to acquire SEM images. Defaults to True.
        acquire_fib (bool, optional): Whether to acquire FIB images. Defaults to True.
    Returns:
        list[tuple[Optional[FibsemImage], Optional[FibsemImage]]]: A list of tuples containing the acquired SEM and FIB images
            for each horizontal field width. If a particular image type was not acquired, its corresponding value will be None.
    """
    images = []

    image_settings = copy.deepcopy(image_settings)
    image_settings.save = True  # ensure saving


    # extend suffexes if more hfws are provided than suffixes
    suffixes = []
    while len(suffixes) < len(hfws):
        suffixes.append(f"res_{len(suffixes)+1:02d}")

    for hfw, suffix in zip(hfws, suffixes):
        image_settings.hfw = hfw
        image_settings.filename = f"{filename}_{suffix}"
        # image_settings.filename = f"{filename}_{int(hfw*1e6)}um"
        sem_image, fib_image = acquire_channels(microscope, image_settings, acquire_sem, acquire_fib)
        images.append((sem_image, fib_image))
    return images

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

    images: list[FibsemImage] = []
    for i in range(n_steps):
        # calculate the reduced area for this strip
        reduced_area = FibsemRectangle(
            left=0, top=i * strip_height,
            width=1, height=strip_height
        )
        image_settings.reduced_area = reduced_area

        # Perform autofocus if requested
        if auto_focus:
            microscope.auto_focus(
                beam_type=image_settings.beam_type,
                reduced_area=reduced_area
            )

        image = acquire_image(microscope, image_settings)
        images.append(image)

    # stack images vertically
    arr = np.vstack([img.data for img in images])

    # resize if necessary to match the expected resolution
    if arr.shape != image_settings.resolution[::-1]:
        arr = resize(arr, image_settings.resolution[::-1], preserve_range=True).astype(np.uint8) # type: ignore

    # Create the stacked FibsemImage using metadata from the middle strip
    # (or the last middle strip if even number of strips)
    middle_index = (n_steps - 1) // 2
    stacked_image = FibsemImage(data=arr, metadata=copy.deepcopy(images[middle_index].metadata))

    # update metadata to reflect that reduced_area is now the full image
    stacked_image.metadata.image_settings.reduced_area = None # type: ignore

    return stacked_image
