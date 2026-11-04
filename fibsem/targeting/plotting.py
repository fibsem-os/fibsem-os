"""Debug/diagnostic matplotlib visualisations for the detection pipeline."""

from __future__ import annotations

import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from fibsem.structures import FibsemImage


def _plot_detection_pipeline(
    image: FibsemImage,
    blobbs: np.ndarray,
    mask: np.ndarray,
    winner_image: np.ndarray,
    list_of_millpositions: list[tuple[int, int]],
    positions: list[tuple[int, int]],
    pattern_mask: Optional[np.ndarray] = None,
):
    from fibsem.segmentation.utils import decode_segmap_v2  # torch at module scope

    fig, ax = plt.subplots(1, 4, figsize=(30, 12))
    ax[0].imshow(image.data, cmap="gray")
    ax[0].imshow(decode_segmap_v2(mask), alpha=0.5)
    ax[0].set_title("Original Mask")

    ax[1].set_title(f"{len(list_of_millpositions)} Detected Positions")
    ax[1].imshow(blobbs)
    ax[1].imshow(decode_segmap_v2(mask), alpha=0.5)
    y_coords = [pos[0] for pos in list_of_millpositions]
    x_coords = [pos[1] for pos in list_of_millpositions]
    ax[1].scatter(x_coords, y_coords, c="white", marker="+")
    # add text for each position
    for i, pos in enumerate(list_of_millpositions):
        ax[1].text(pos[1] + 10, pos[0] + 10, str(i), color="white", fontsize=12)

    if pattern_mask is None:
        pattern_mask = np.zeros_like(winner_image)
    ax[2].set_title("Winner Grid Mask")
    ax[2].imshow(winner_image)
    ax[2].imshow(pattern_mask, alpha=0.5, cmap="gray")
    # draw a rectangle 50 px from edges of image
    ax[2].add_patch(
        plt.Rectangle(
            (50, 50),
            image.data.shape[1] - 100,
            image.data.shape[0] - 100,
            fill=False,
            linestyle="dashed",
            edgecolor="white",
            linewidth=2,
        )
    )

    ax[3].set_title(f"{len(positions)} Final Positions")
    ax[3].imshow(decode_segmap_v2(mask), alpha=0.5)
    ax[3].imshow(
        decode_segmap_v2(winner_image, colormap=((0, 0, 0), (255, 255, 0))), alpha=0.3
    )
    y_coords = [pos[0] for pos in positions]
    x_coords = [pos[1] for pos in positions]
    ax[3].scatter(x_coords, y_coords, c="white", marker="+", s=50)

    # save figure
    fig.savefig(
        f"milling_patterns_{image.metadata.image_settings.filename}-{time.time()}.png",
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)
