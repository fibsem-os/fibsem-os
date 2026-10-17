"""Low-level grid-mask generation and template-matching utilities."""

from __future__ import annotations

import math

import cv2
import numpy as np


def generate_grid_mask(
    pixelsize: float,
    hole_width: float,
    pitch: float,
    bar_width: float,
    nrows: int = 5,
    ncols: int = 5,
) -> np.ndarray:
    """Generates a grid mask for the given image shape and milling parameters.
    Description
    -----------
    function takes image shape and grid parameters as input and outputs a grid mask.
    Parameters
    ----------
    pixelsize : float
        Pixel size in μm.
    hole_width : float
        Hole width of grid in μm.
    pitch : float
        Pitch of grid in μm.
    bar_width : float
        Bar width of grid in μm.
    Returns
    -------
    np.ndarray of shape (cut_out_height, cut_out_length), dtype int depicting the grid mask.
    cut_out_length : int
        length of the cut out of the reconstructed grid in pixels.
    cut_out_height : int
        height of the cut out of the reconstructed grid in pixels.
    """

    # 1.1 adjust to size of the grid to the horizontal field width (HFW)
    rectangle_size = hole_width / pixelsize
    rectangle_distance = pitch / pixelsize
    rectangle_border = bar_width / pixelsize

    # 1.2 display cut out of the grid
    mheight = int((nrows * rectangle_distance) + rectangle_border)
    mwidth = int((ncols * rectangle_distance) + rectangle_border)

    # 1.3 determine the starting points
    grid = np.zeros([mheight, mwidth], np.uint8)
    y = int(mheight - (rectangle_border))
    x = int(mwidth - (rectangle_border))
    iy = int(y - rectangle_size)
    ix = int(x - rectangle_size)

    # 1.4 construct the rectangles to build the grid
    for row in range(nrows):  # There are 5 rows of rectangles
        for col in range(ncols):  # There are 4 columns of rectangles
            top_left = (
                int(ix - col * rectangle_distance),
                int(iy - row * rectangle_distance),
            )
            bottom_right = (
                int(x - col * rectangle_distance),
                int(y - row * rectangle_distance),
            )
            cv2.rectangle(grid, top_left, bottom_right, (1), 30)

    # 1.5 fill the area between the rectangles
    h, w = grid.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(grid, mask, (0, 0), 1)

    return grid


def generate_grid_mask_v2(
    pixelsize: float,
    hole_width: float,
    pitch: float,
    bar_width: float,
    nrows: int = 5,
    ncols: int = 5,
) -> np.ndarray:
    """Correct reimplementation of :func:`generate_grid_mask`.

    Produces the same output shape as ``generate_grid_mask`` but carves holes
    using numpy broadcasting rather than the cv2.rectangle border-thickness
    hack (which made holes narrower than ``hole_width``).

    Returns
    -------
    np.ndarray, shape (mheight, mwidth), dtype uint8.  1 = bar, 0 = hole.
    """
    hole_px = hole_width / pixelsize
    pitch_px = pitch / pixelsize
    bar_px = bar_width / pixelsize

    mheight = int(nrows * pitch_px + bar_px)
    mwidth = int(ncols * pitch_px + bar_px)

    _MAX_PIXELS = 50_000_000  # 50 MP — anything larger is almost certainly a unit error
    if mheight * mwidth > _MAX_PIXELS:
        raise ValueError(
            f"generate_grid_mask_v2: output would be {mheight}×{mwidth} = {mheight * mwidth:,} px "
            f"({mheight * mwidth / _MAX_PIXELS:.0f}× the 50 MP limit). "
            f"Check that pixelsize ({pixelsize}) is in µm, not metres."
        )

    # Anchor: the bottom/right edge of the first (row/col 0) hole sits one
    # bar-width from the edge — matches the original ix/iy anchor logic.
    y1_ref = mheight - bar_px
    x1_ref = mwidth - bar_px

    in_row = np.zeros(mheight, dtype=bool)
    for r in range(nrows):
        r1 = int(y1_ref - r * pitch_px)
        r0 = int(r1 - hole_px)
        in_row[max(r0, 0) : min(r1, mheight)] = True

    in_col = np.zeros(mwidth, dtype=bool)
    for c in range(ncols):
        c1 = int(x1_ref - c * pitch_px)
        c0 = int(c1 - hole_px)
        in_col[max(c0, 0) : min(c1, mwidth)] = True

    hole_mask = in_row[:, None] & in_col[None, :]
    return (~hole_mask).astype(np.uint8)


def generate_rotation_masks(
    grid_mask: np.ndarray, max_angle: float = 45, stepsize: float = 2.5
) -> list[np.ndarray]:
    """Generate a library of rotated versions of the grid mask for template matching."""
    rows, cols = grid_mask.shape
    rotation_library = []

    center_point = ((cols - 1) / 2.0, (rows - 1) / 2.0)
    for angle in np.arange(0, max_angle, stepsize):
        R = cv2.getRotationMatrix2D(center=center_point, angle=angle, scale=1)
        rotated = cv2.warpAffine(grid_mask, R, (cols, rows))
        rotation_library.append(rotated)

    return rotation_library


def generate_perspective_library(
    rotation_library: list[np.ndarray], working_distance: int, milling_angle: int
):
    """
    Description
    -----------
    function takes rotation library of the grid, cut out dimensions, working distance,
    milling angle as input and outputs the perspective library
    of the grid in the correct perspective.

    Parameters
    ----------
    rotation_library : list of np.ndarray of shape (cut_out_height, cut_out_width), dtype int
        list of rotated versions of the reconstructed grid.
    working_distance : int
        distance from grid to detector in μm.
    milling_angle : int
        angle between orientation of grid and milling direction in °.

    Returns
    -------
    np.ndarray of shape (cut_out_height, cut_out_width), dtype int depicting the fitted grid.

    """
    mheight, mwidth = rotation_library[0].shape

    # 2.4.1 determine the location of the 4 fix points in the correct perspective
    rx, rx2 = 0, int(mwidth)
    ry, ry2 = 0, int(mheight)

    # 2.4.2 determine how the perspective makes the height of the grid appear smaller
    wd = working_distance
    milling_angle = 90 - milling_angle
    compression = (mheight / 2) * (1 - math.cos((milling_angle * 2 * math.pi) / 360))

    # 2.4.3 determine how the perspective makes the front of the grid appear wider
    translation = math.sin((milling_angle * 2 * math.pi) / 360) * (mwidth / 2)
    distortion = ((mwidth * wd) / (wd - translation) - mwidth) / 2

    # 2.4.4 transpose each image from the rotation library into the correct perspective based on
    # the 4 fix points
    pts1 = np.float32([[rx, ry], [rx2, ry], [rx, ry2], [rx2, ry2]])
    pts2 = np.float32(
        [
            [rx + distortion, ry + compression],
            [rx2 - distortion, ry + compression],
            [rx - distortion, ry2 - compression],
            [rx2 + distortion, ry2 - compression],
        ]
    )
    M = cv2.getPerspectiveTransform(pts1, pts2)
    perspective_library = []

    dsize = (mwidth, mheight)
    for rotation in rotation_library:
        dst = cv2.warpPerspective(rotation, M, dsize)
        perspective_library.append(dst)

    return perspective_library, compression, distortion, translation


def match_template(
    pred_mask: np.ndarray,
    perspective_library: list[np.ndarray],
    compression: float,
    distortion: float,
    debug: bool = False,
) -> np.ndarray:
    """Match the template from the perspective library to the predicted mask."""

    # cross-correlation
    score_list = []
    location_list = []
    shortened_library = []

    mheight, mwidth = perspective_library[0].shape

    y0 = int(compression + 50)
    y1 = int(mheight - compression - 50)
    x0 = int(distortion + 400)
    x1 = int(mwidth - distortion - 400)

    # resize the rgb image for speed
    scale = 10
    rgb_resized = pred_mask[::scale, ::scale]
    h, w = pred_mask.shape

    for image in perspective_library:
        shortened = image[y0:y1, x0:x1]
        shortened_resized = shortened[::scale, ::scale]  # resize for speed
        res = cv2.matchTemplate(shortened_resized, rgb_resized, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = tuple([scale * x for x in max_loc])  # scale back to original size

        shortened_library.append(shortened)
        score_list.append(max_val)
        location_list.append(top_left)

    # 2.4.6 determine best match
    winner_idx = score_list.index(max(score_list))

    y = int(location_list[winner_idx][1])
    x = int(location_list[winner_idx][0])
    winner_image = shortened_library[winner_idx][y : y + h, x : x + w]

    if debug:
        _debug_plot_match_template(
            pred_mask=pred_mask,
            shortened=shortened_library[winner_idx],
            top_left=(x, y),
            score=score_list[winner_idx],
            winner_idx=winner_idx,
            score_list=score_list,
        )

    return winner_image


def _debug_plot_match_template(
    pred_mask: np.ndarray,
    shortened: np.ndarray,
    top_left: tuple,
    score: float,
    winner_idx: int,
    score_list: list,
) -> None:
    """Display a debugging figure for :func:`match_template`.

    Shows (left-to-right):
    - The predicted mask (search image)
    - The best-matching template (cropped to pred_mask size), overlaid on the
      shortened template image with a rectangle marking the match location
    - The match score across all rotation candidates
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    h, w = pred_mask.shape
    x, y = top_left

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"match_template debug — best idx={winner_idx}, score={score:.2e}",
        fontsize=12,
    )

    # --- panel 1: predicted mask (search image) ---
    axes[0].imshow(pred_mask, cmap="gray", interpolation="nearest")
    axes[0].set_title("Predicted mask (search image)")
    axes[0].axis("off")

    # --- panel 2: best template with match location highlighted ---
    axes[1].imshow(shortened, cmap="gray", interpolation="nearest")
    rect = mpatches.Rectangle(
        (x, y),
        w,
        h,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )
    axes[1].add_patch(rect)
    axes[1].plot(x + w / 2, y + h / 2, "r+", markersize=12, markeredgewidth=2)
    axes[1].set_title(f"Best template (idx={winner_idx})\nmatch @ x={x}, y={y}")
    axes[1].axis("off")

    # --- panel 3: match scores across rotation candidates ---
    axes[2].bar(range(len(score_list)), score_list, color="steelblue")
    axes[2].axvline(
        winner_idx, color="red", linewidth=2, label=f"best (idx={winner_idx})"
    )
    axes[2].set_xlabel("Rotation index")
    axes[2].set_ylabel("Score (TM_CCOEFF)")
    axes[2].set_title("Match scores across rotation library")
    axes[2].legend()

    plt.tight_layout()
    plt.show()
