import numpy as np
from skimage.filters.rank import gradient
from skimage import img_as_float
from skimage.filters import laplace
from skimage.morphology import disk
import scipy.ndimage

from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.fm.structures import ChannelSettings, ZParameters, FluorescenceImage
from typing import List, Tuple, Dict


def acquire_channels(
    microscope: FluorescenceMicroscope, settings: List[ChannelSettings]
) -> List[FluorescenceImage]:
    """Acquire images for multiple channels."""
    
    if not isinstance(settings, list):
        settings = [settings]  # Ensure settings is a list

    images: List[FluorescenceImage] = []
    for channel in settings:
        microscope.set_channel(channel)
        image = microscope.acquire_image(channel)
        images.append(image)
    return images # TODO: migrate to 5D FluorescenceImage structure

def acquire_z_stack(
    microscope: FluorescenceMicroscope,
    channel_settings: ChannelSettings,
    zparams: ZParameters,
) -> FluorescenceImage:
    """Acquire a Z-stack of images for a given channel."""

    z_init = microscope.objective.position  # initial z position of the objective
    z_positions = zparams.generate_positions(z_init=z_init)
    images: List[FluorescenceImage] = []

    if not isinstance(channel_settings, list):
        channel_settings = [channel_settings]

    # TODO: support multi-channel Z-stacks
    for ch in channel_settings:

        ch_images: List[FluorescenceImage] = []
        for z in z_positions:
            # Move objective to the specified z position
            microscope.objective.move_absolute(z)
            # Acquire image at the current z position
            image = microscope.acquire_image(channel_settings=ch)
            ch_images.append(image)

        # stack the images along the z-axis
        zstack = FluorescenceImage.create_z_stack(ch_images)

        # TODO: properly handle metadata + image structure

        images.append(zstack)  # TODO: migrate to 5D FluorescenceImage structure

    # restore objective to initial position
    microscope.objective.move_absolute(z_init)

    return images


########## CALIBRATION FUNCTIONS ##########


def get_sharpness(img: np.ndarray, **kwargs) -> float:
    """Calculate sharpness (accutance) of an image.
    (Acutance: https://en.wikipedia.org/wiki/Acutance)
    """
    disk_size = kwargs.get("disk_size", 5)
    print(f"calculating sharpness (accutance) of image {img.shape}: {img.dtype}")
    # normalise and convert to uint8, for faster processing
    img_norm = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    filtered = scipy.ndimage.median_filter(img_norm, 3)
    return np.mean(gradient(filtered, disk(disk_size)))


def get_variance(img: np.ndarray, **kwargs) -> float:
    """Get variance of the Laplacian of an image.
    (Faster than get_sharpness, but more sensitive to noise)"""
    # Convert to float to avoid precision issues
    img_float = img_as_float(img)
    laplacian = laplace(img_float)
    return np.var(laplacian)


DEFAULT_FOCUS_METHOD = "variance"
FOCUS_FN_MAP = {"sharpness": get_sharpness, "variance": get_variance}


def run_auto_focus(
    microscope: FluorescenceMicroscope,
    channel_settings: ChannelSettings,
    method: str = DEFAULT_FOCUS_METHOD,
) -> float:
    """Run autofocus by acquiring images at different z positions and finding the best focus."""

    # TODO:
    # - allow user to specify focus function
    # - allow user to specify z parameters (defaults for coarse/fine focus?)

    # set up z parameters
    zparams = ZParameters(zmin=-10e-6, zmax=10e-6, zstep=2.5e-6)
    z_positions = zparams.generate_positions(microscope.objective.position)

    # get the focus metric function
    focus_fn = FOCUS_FN_MAP.get(method, FOCUS_FN_MAP[DEFAULT_FOCUS_METHOD])

    scores = []
    for pos in z_positions:
        # set objective position
        microscope.objective.move_absolute(pos)

        # acquire image
        img = microscope.acquire_image(channel_settings=channel_settings)

        # calculate sharpness of image
        score = focus_fn(img.data)
        print(f"Z Position: {pos:.2e} microns, Score: {score:.2f}")
        scores.append(score)

    # get best focus position
    idx = np.argmax(scores)
    best_focus = z_positions[idx]

    print(
        f"Best focus found at z position: {best_focus:.2e} microns with score: {scores[idx]:.2f}"
    )
    # move objective to best focus position
    microscope.objective.move_absolute(best_focus)

    return best_focus
