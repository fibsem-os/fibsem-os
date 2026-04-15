"""Compare shift_from_crosscorrelation (original) vs shift_from_crosscorrelation_v2 (cv2)
on real FibsemImage data.

Usage:
    python fibsem/ui/widgets/tests/test_alignment_comparison.py \
        --ref path/to/reference.tif \
        --images path/to/alignment_images/

    # or pass images explicitly
    python fibsem/ui/widgets/tests/test_alignment_comparison.py \
        --ref path/to/reference.tif \
        --images img1.tif img2.tif img3.tif

Outputs:
    - Printed comparison table (shift agreement, response scores)
    - /tmp/alignment_comparison.png  — debug plot (v2 layout)
    - /tmp/alignment_comparison_agreement.png — agreement scatter plot
"""

import argparse
import os

import numpy as np

from fibsem.alignment import (
    AlignmentResult,
    crosscorrelation_cv2,
    plot_multi_step_alignment_v2,
    shift_from_crosscorrelation,
    shift_from_crosscorrelation_v2,
)
from fibsem.structures import FibsemImage, Point


def load_images(ref_path: str, image_paths: list[str]) -> tuple:
    ref = FibsemImage.load(ref_path)
    images = [FibsemImage.load(p) for p in image_paths]
    return ref, images


def run_comparison(ref: FibsemImage, images: list[FibsemImage]) -> list[dict]:
    rows = []
    for i, img in enumerate(images):
        # original method
        dx_v1, dy_v1, xcorr = shift_from_crosscorrelation(
            ref, img, lowpass=128, highpass=6, sigma=6, use_rect_mask=True
        )

        # cv2 method
        shift_x_px, shift_y_px, score = crosscorrelation_cv2(ref.data, img.data)
        dx_v2, dy_v2, _ = shift_from_crosscorrelation_v2(ref, img)

        pixel_size = ref.metadata.pixel_size.x
        agree_x = abs(dx_v1 - dx_v2)
        agree_y = abs(dy_v1 - dy_v2)
        agree_px = np.sqrt((agree_x / pixel_size) ** 2 + (agree_y / pixel_size) ** 2)

        rows.append({
            "i": i + 1,
            "dx_v1_nm": dx_v1 * 1e9,
            "dy_v1_nm": dy_v1 * 1e9,
            "dx_v2_nm": dx_v2 * 1e9,
            "dy_v2_nm": dy_v2 * 1e9,
            "agree_px": agree_px,
            "score": score,
            "img": img,
            "shift": Point(dx_v2, dy_v2),
            "shift_px": Point(shift_x_px, shift_y_px),
        })
    return rows


def print_table(rows: list[dict], pixel_size: float) -> None:
    header = (
        f"{'Step':>4}  "
        f"{'v1 dx (nm)':>12}  {'v1 dy (nm)':>12}  "
        f"{'v2 dx (nm)':>12}  {'v2 dy (nm)':>12}  "
        f"{'agree (px)':>10}  {'score':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        flag = "  ✓" if r["score"] >= 0.5 else ("  ?" if r["score"] >= 0.25 else "  ✗")
        print(
            f"{r['i']:>4}  "
            f"{r['dx_v1_nm']:>12.1f}  {r['dy_v1_nm']:>12.1f}  "
            f"{r['dx_v2_nm']:>12.1f}  {r['dy_v2_nm']:>12.1f}  "
            f"{r['agree_px']:>10.3f}  {r['score']:>6.3f}{flag}"
        )

    scores = [r["score"] for r in rows]
    agreements = [r["agree_px"] for r in rows]
    print()
    print(f"  mean score:        {np.mean(scores):.3f}  (min {np.min(scores):.3f}, max {np.max(scores):.3f})")
    print(f"  mean agreement:    {np.mean(agreements):.3f} px  (max {np.max(agreements):.3f} px)")


def plot_agreement(rows: list[dict], save_path: str) -> None:
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10, 4))
    axes = fig.subplots(1, 2)

    step_nums = [r["i"] for r in rows]
    scores = [r["score"] for r in rows]
    agreements = [r["agree_px"] for r in rows]

    # agreement per step
    ax = axes[0]
    colours = ["green" if a < 1.0 else ("orange" if a < 2.0 else "red") for a in agreements]
    ax.bar(step_nums, agreements, color=colours)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="1 px")
    ax.set_xlabel("Image")
    ax.set_ylabel("Disagreement (px)")
    ax.set_title("v1 vs v2 shift disagreement")
    ax.set_xticks(step_nums)
    ax.legend(fontsize="small")

    # score vs agreement scatter
    ax = axes[1]
    ax.scatter(scores, agreements, color="steelblue", zorder=3)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Response score (v2)")
    ax.set_ylabel("Disagreement (px)")
    ax.set_title("Score vs disagreement")
    for r in rows:
        ax.annotate(str(r["i"]), (r["score"], r["agree_px"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    print(f"Agreement plot  → {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare alignment methods on real data.")
    parser.add_argument("--ref", required=True, help="Path to reference .tif image")
    parser.add_argument("--images", nargs="+", required=True,
                        help="Paths to alignment .tif images, or a single directory")
    args = parser.parse_args()

    # accept a directory as --images
    image_paths = args.images
    if len(image_paths) == 1 and os.path.isdir(image_paths[0]):
        image_paths = sorted(
            os.path.join(image_paths[0], f)
            for f in os.listdir(image_paths[0])
            if f.endswith(".tif") or f.endswith(".tiff")
        )
        if not image_paths:
            raise ValueError(f"No .tif files found in {args.images[0]}")

    print(f"Reference:  {args.ref}")
    print(f"Images:     {len(image_paths)} files\n")

    ref, images = load_images(args.ref, image_paths)
    rows = run_comparison(ref, images)

    print_table(rows, pixel_size=ref.metadata.pixel_size.x)
    print()

    # v2 debug plot
    results = [
        AlignmentResult(
            shift=r["shift"],
            shift_px=r["shift_px"],
            score=r["score"],
            image=r["img"],
        )
        for r in rows
    ]
    fig = plot_multi_step_alignment_v2(ref, results, title="Alignment comparison", save=False)
    debug_path = "/tmp/alignment_comparison.png"
    fig.savefig(debug_path, dpi=100)
    print(f"Debug plot  → {debug_path}")

    plot_agreement(rows, "/tmp/alignment_comparison_agreement.png")


if __name__ == "__main__":
    main()
