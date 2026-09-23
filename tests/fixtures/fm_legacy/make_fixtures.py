"""Write the legacy fluorescence fixtures with the `save` FibsemOS had before FIB-279.

Run against an export of the last commit before the fix, not against this checkout --
the point is bytes written by the old code:

    git archive 179bbeff7 fibsem | tar -x -C /tmp/fibsem-179bbeff7
    PYTHONPATH=/tmp/fibsem-179bbeff7 python tests/fixtures/fm_legacy/make_fixtures.py

Every plane holds `plane_value(c, z)` plus a small ramp, so a test can rebuild the
expected array without reading anything but the fixture.
"""

import os

import numpy as np

from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SHAPES = [(1, 4), (4, 1), (2, 3), (3, 2), (2, 2), (3, 3), (4, 4)]
HEIGHT, WIDTH = 6, 7


def expected(nc: int, nz: int) -> np.ndarray:
    data = np.zeros((nc, nz, HEIGHT, WIDTH), dtype=np.uint16)
    ramp = np.arange(HEIGHT * WIDTH).reshape(HEIGHT, WIDTH) % 3
    for c in range(nc):
        for z in range(nz):
            data[c, z] = (10 * c + z + 1) * 97 + ramp
    return data


def _metadata(nc: int, nz: int) -> FluorescenceImageMetadata:
    return FluorescenceImageMetadata(
        acquisition_date="2026-09-23T10:00:00",
        pixel_size_x=2e-7,
        pixel_size_y=2e-7,
        resolution=(WIDTH, HEIGHT),
        channels=[
            FluorescenceChannelMetadata(
                name=f"ch{c}",
                excitation_wavelength=488.0,
                power=0.5,
                exposure_time=0.1,
                gain=1.0,
                offset=0.0,
            )
            for c in range(nc)
        ],
        z_positions=[z * 5e-7 for z in range(nz)] if nz > 1 else None,
    )


def main() -> None:
    for nc, nz in SHAPES:
        image = FluorescenceImage(data=expected(nc, nz), metadata=_metadata(nc, nz))
        image.save(os.path.join(HERE, f"c{nc}z{nz}.ome.tiff"))
    # A 2D image handed to save, as an overview tile is.
    tile = FluorescenceImage(data=expected(1, 1)[0, 0], metadata=_metadata(1, 1))
    tile.save(os.path.join(HERE, "tile.ome.tiff"))


if __name__ == "__main__":
    main()
