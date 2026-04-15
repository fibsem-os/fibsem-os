"""Standalone test script for plot_multi_step_alignment_v2.

Run directly:
    python fibsem/ui/widgets/tests/test_alignment_plot_v2.py
"""

import numpy as np

from fibsem import acquire, utils
from fibsem.alignment import (
    AlignmentResult,
    crosscorrelation_cv2,
    plot_multi_step_alignment_v2,
    shift_from_crosscorrelation_v2,
)
from fibsem.structures import Point


def make_shifted_image(ref_image, offset_x: int, offset_y: int):
    """Return a copy of ref_image with data shifted by (offset_x, offset_y) pixels."""
    import copy

    new_image = copy.deepcopy(ref_image)
    new_image.data[:] = np.roll(
        np.roll(ref_image.data, offset_y, axis=0), offset_x, axis=1
    )
    return new_image


def main():
    microscope, settings = utils.setup_session(debug=False)

    # acquire a reference with a recognisable pattern
    ref = acquire.acquire_image(microscope, settings.image)
    # ref.data[:] = 0
    h, w = ref.data.shape
    cx, cy = w // 2, h // 2
    # ref.data[cy - 40:cy + 40, cx - 40:cx + 40] = 200   # bright square at centre

    offsets = [(10, 5), (-8, 12), (3, -6)]
    results = []

    for i, (ox, oy) in enumerate(offsets):
        new = make_shifted_image(ref, ox, oy)
        shift_x, shift_y, score = crosscorrelation_cv2(ref.data, new.data)
        dx, dy, _ = shift_from_crosscorrelation_v2(ref, new)
        r = AlignmentResult(
            shift=Point(dx, dy),
            shift_px=Point(shift_x, shift_y),
            score=score,
            image=new,
        )
        results.append(r)
        print(
            f"Step {i + 1}: offset=({ox},{oy})px  "
            f"detected=({shift_x:.2f},{shift_y:.2f})px  "
            f"dx={r.shift.x * 1e9:.1f}nm dy={r.shift.y * 1e9:.1f}nm  score={score:.3f}"
        )

    fig = plot_multi_step_alignment_v2(
        ref, results, title="Test Alignment v2", save=False
    )
    fig.savefig("/tmp/test_alignment_plot_v2.png", dpi=100)
    print("Saved → /tmp/test_alignment_plot_v2.png")


if __name__ == "__main__":
    main()
