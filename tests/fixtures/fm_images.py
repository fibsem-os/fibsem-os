"""Small fluorescence images for tests: a channel and a stack around it."""

from typing import Optional, Sequence

import numpy as np

from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)


def gfp(**fields) -> FluorescenceChannelMetadata:
    """A GFP channel; *fields* override or add to its settings."""
    settings = dict(
        name="GFP",
        excitation_wavelength=488.0,
        power=0.5,
        exposure_time=0.1,
        gain=1.0,
        offset=0.0,
    )
    return FluorescenceChannelMetadata(**{**settings, **fields})


def fm_image(
    data: np.ndarray,
    channels: Optional[Sequence[FluorescenceChannelMetadata]] = None,
    pixel_size: float = 1e-7,
    acquired: str = "2026-08-10T00:00:00",
    filepath: Optional[str] = None,
    **metadata,
) -> FluorescenceImage:
    """A (C, Z, Y, X) stack with square pixels, one GFP channel unless told
    otherwise; *metadata* sets any other field (pixel_size_z, stage_position...)."""
    return FluorescenceImage(
        data=data,
        metadata=FluorescenceImageMetadata(
            acquisition_datetime=acquired,
            pixel_size_x=pixel_size,
            pixel_size_y=pixel_size,
            channels=list(channels) if channels is not None else [gfp()],
            **metadata,
        ),
        filepath=filepath,
    )


def noise(size: int, seed: int, height: Optional[int] = None) -> np.ndarray:
    """A one-channel, one-slice uint16 frame of seeded noise."""
    rng = np.random.default_rng(seed)
    return (rng.random((1, 1, height or size, size)) * 4000).astype(np.uint16)
