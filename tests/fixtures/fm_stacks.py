"""Small fluorescence stacks for tests."""

import numpy as np

from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)


def planes_stack(path, planes=3):
    """A one-channel stack whose plane z holds a bright block in column z."""
    data = np.zeros((1, planes, 96, 96), dtype=np.uint16)
    for z in range(planes):
        data[0, z, 40:56, z * 32 + 8 : z * 32 + 24] = 4000
    channel = FluorescenceChannelMetadata(
        name="GFP",
        excitation_wavelength=488.0,
        power=0.5,
        exposure_time=0.1,
        gain=1.0,
        offset=0.0,
        color="gray",
    )
    metadata = FluorescenceImageMetadata(
        acquisition_datetime="2026-01-01T00:00:00",
        pixel_size_x=1e-7,
        pixel_size_y=1e-7,
        resolution=(96, 96),
        channels=[channel],
        z_positions=[i * 3e-7 for i in range(planes)],
    )
    FluorescenceImage(data=data, metadata=metadata).save(path)
