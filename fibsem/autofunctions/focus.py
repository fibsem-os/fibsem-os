import logging

import numpy as np
import skimage

from fibsem import acquire, utils
from fibsem import config as cfg
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemRectangle,
    ImageSettings,
    MicroscopeSettings,
)


def auto_focus_beam(
    microscope: FibsemMicroscope,
    settings: MicroscopeSettings,
    beam_type: BeamType,
    metric_fn=None,
    focus_image_settings: ImageSettings = None,
    step_size: float = 0.05e-3,
    num_steps: int = 5,
    kwargs: dict = {},
    verbose: bool = False,
) -> None:
    """Image-based autofocus by sweeping working distance and maximising a focus metric.

    If no metric_fn is provided, delegates to the hardware autofocus routine
    (microscope.auto_focus).

    Args:
        microscope: FibsemMicroscope instance.
        settings: MicroscopeSettings (used for default image path).
        beam_type: Beam to focus (ELECTRON or ION).
        metric_fn: Callable(FibsemImage, **kwargs) -> float. If None, uses hardware autofocus.
        focus_image_settings: ImageSettings for focus sweep images. Defaults to a 768x512 ROI.
        step_size: Working-distance step size in metres (default 0.05 mm).
        num_steps: Number of steps either side of current WD (default 5).
        kwargs: Extra keyword arguments forwarded to metric_fn.
        verbose: Log step-by-step details if True.
    """
    if metric_fn is None:
        microscope.auto_focus(beam_type=beam_type)
        return

    if focus_image_settings is None:
        focus_image_settings = ImageSettings(
            resolution=[768, 512],
            dwell_time=200e-9,
            hfw=100e-6,
            beam_type=beam_type,
            save=True,
            path=settings.image.path,
            autocontrast=True,
            autogamma=False,
            filename=f"{utils.current_timestamp()}_",
            reduced_area=FibsemRectangle(0.3, 0.3, 0.4, 0.4),
        )

    current_wd = microscope.get("working_distance", beam_type)

    if verbose:
        logging.info(f"{metric_fn.__name__} based auto-focus routine")
        logging.info(f"doc: {metric_fn.__doc__}")
        logging.info(f"initial working distance: {current_wd:.2e}")

    min_wd = current_wd - (num_steps * step_size / 2)
    max_wd = current_wd + (num_steps * step_size / 2)
    wds = np.linspace(min_wd, max_wd, num_steps + 1)

    metrics = []
    for i, wd in enumerate(wds):
        logging.info(f"image {i}: {wd:.2e}")
        microscope.set("working_distance", wd, beam_type)

        focus_image_settings.filename = f"{utils.current_timestamp()}_sharpness_{i}"
        img = acquire.new_image(microscope, focus_image_settings)

        metric = metric_fn(img, **kwargs)
        metrics.append(metric)

    idx = np.argmax(metrics)

    if verbose:
        pairs = list(zip(wds, metrics))
        logging.info([f"{wd:.2e}: {metric:.4f}" for wd, metric in pairs])
        logging.info(f"{idx}, {wds[idx]:.2e}, {metrics[idx]:.4f}")

    microscope.set(
        key="working_distance",
        value=wds[idx],
        beam_type=beam_type,
    )


def _sharpness(img: FibsemImage, **kwargs) -> float:
    """Gradient-based acutance metric (Acutance: https://en.wikipedia.org/wiki/Acutance)."""
    from skimage.filters.rank import gradient
    from skimage.morphology import disk
    disk_size = kwargs.get("disk_size", 5)
    logging.info(f"calculating sharpness (accutance) of image {img}: {disk_size}")
    return np.mean(gradient(skimage.filters.median(np.copy(img.data)), disk(disk_size)))


def _dog(img: FibsemImage, **kwargs) -> float:
    """Difference-of-Gaussians focus metric."""
    low = kwargs.get("low", 3)
    high = kwargs.get("high", 9)
    from skimage.filters import difference_of_gaussians
    return np.mean(difference_of_gaussians(np.copy(img.data), low, high))
