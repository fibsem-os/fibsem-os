"""Bench check of beam shift alignment on a connected microscope.

Takes a reduced-area image, shifts the beam by (10, 5) um, aligns back to the
first image with multi_step_alignment_v2, and plots the three images with a
centre crosshair. Use it to check alignment under a given scan rotation.

    python scripts/beam_shift_alignment_test.py --scan_rotation 180 --beam_type ION
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from fibsem import acquire, alignment, utils
from fibsem.structures.settings import BeamType, FibsemRectangle


def run_beam_shift_alignment_test(
    scan_rotation: int = 180, hfw: float = 150e-6, beam_type: BeamType = BeamType.ION
):
    microscope, settings = utils.setup_session()

    print(
        f"Running test alignment with scan rotation: {scan_rotation} degrees, HFW: {hfw}, Beam Type: {beam_type.name}"
    )

    microscope.set_scan_rotation(np.radians(scan_rotation), beam_type=beam_type)
    microscope.reset_beam_shifts()

    settings.image.hfw = hfw
    settings.image.beam_type = beam_type
    settings.image.reduced_area = FibsemRectangle(0.25, 0.25, 0.5, 0.5)
    image1 = acquire.acquire_image(microscope=microscope, settings=settings.image)

    microscope.beam_shift(dx=10e-6, dy=5e-6, beam_type=beam_type)
    image2 = acquire.acquire_image(microscope=microscope, settings=settings.image)

    alignment.multi_step_alignment_v2(
        microscope, ref_image=image1, use_autocontrast=True
    )
    image3 = acquire.acquire_image(microscope=microscope, settings=settings.image)

    fig, ax = plt.subplots(1, 3, figsize=(15, 7))
    fig.suptitle(
        f"Test Alignment: {beam_type.name} - Scan Rotation: {scan_rotation}deg"
    )
    titles = ["Previous Image", "Shifted Image", "Aligned Image"]
    for a, image, title in zip(ax, [image1, image2, image3], titles):
        h, w = image.data.shape[:2]
        a.imshow(image.data, cmap="gray")
        a.set_title(title)
        a.axis("off")
        a.plot([w // 2, w // 2], [0, h], color="yellow", lw=1)
        a.plot([0, w], [h // 2, h // 2], color="yellow", lw=1)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Check beam shift alignment on the connected microscope."
    )
    parser.add_argument(
        "--scan_rotation",
        type=int,
        default=180,
        help="Scan rotation in degrees (default: 180)",
    )
    parser.add_argument(
        "--hfw",
        type=float,
        default=150e-6,
        help="Horizontal field width in metres (default: 150e-6)",
    )
    parser.add_argument(
        "--beam_type",
        type=str,
        choices=["ION", "ELECTRON"],
        default="ION",
        help="Beam to align (default: ION)",
    )
    args = parser.parse_args()
    run_beam_shift_alignment_test(
        scan_rotation=args.scan_rotation,
        hfw=args.hfw,
        beam_type=BeamType[args.beam_type],
    )


if __name__ == "__main__":
    main()
