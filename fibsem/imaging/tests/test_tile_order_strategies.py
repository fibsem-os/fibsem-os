"""Visual comparison of all TileOrderStrategy options for a 5×5 grid.

Run directly:
    python fibsem/imaging/tests/test_tile_order_strategies.py
"""

import matplotlib.pyplot as plt

from fibsem.imaging.tiled import compute_tile_grid, order_tiles, plot_tile_positions
from fibsem.structures import ImageSettings, OverviewAcquisitionSettings, TileOrderStrategy


def main():
    base = OverviewAcquisitionSettings(
        image_settings=ImageSettings(resolution=(1024, 1024), hfw=100e-6),
        nrows=5, ncols=5, overlap=0.1,
    )

    strategies = list(TileOrderStrategy)
    fig, axes = plt.subplots(1, len(strategies), figsize=(6 * len(strategies), 6))

    for ax, strategy in zip(axes, strategies):
        settings = OverviewAcquisitionSettings(
            image_settings=base.image_settings,
            nrows=base.nrows, ncols=base.ncols, overlap=base.overlap,
            tile_order=strategy,
        )
        tiles = compute_tile_grid(settings)
        ordered = order_tiles(tiles, strategy)
        plot_tile_positions(ordered, settings, ax=ax)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
