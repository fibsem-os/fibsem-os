"""TiledOverviewWidget — interactive tile grid for planning tiled acquisitions.

Displays a grid of tiles as coloured rectangles over an optional SEM background
image. Click a tile to toggle it enabled/disabled.

Two modes
---------
Grid mode (default)
    Tiles are drawn in grid-unit space. Use set_grid(nrows, ncols).

Physical mode
    Call set_tile_settings(OverviewAcquisitionSettings) to position tiles in
    physical space (µm). Call set_image(array, hfw_um) to add a background SEM
    image. Both can be called independently or together.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle as MplRectangle
from matplotlib.text import Text as MplText
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

_logger = logging.getLogger(__name__)

# Colours
_COLOR_ENABLED = "#4CAF50"
_COLOR_ENABLED_EDGE = "#81C784"
_COLOR_DISABLED = "#37474F"
_COLOR_DISABLED_EDGE = "#546E7A"
_COLOR_DISABLED_ALPHA = 0.45
_COLOR_BG = "#1e2124"
_COLOR_TEXT = "#FFFFFF"
_COLOR_TEXT_DISABLED = "#78909C"

_TILE_PAD = 0.06   # fractional gap in grid-unit mode
_PATCH_ALPHA = 0.35


@dataclass
class TileState:
    row: int
    col: int
    # Physical offsets to tile centre (metres). None in grid-unit mode.
    dx: Optional[float] = None
    dy: Optional[float] = None
    enabled: bool = True


class TiledOverviewWidget(QWidget):
    """Interactive grid widget for enabling/disabling acquisition tiles.

    Signals
    -------
    tiles_changed : list[tuple[int, int]]
        Emitted on every toggle with the list of currently *enabled* (row, col) pairs.
    """

    tiles_changed = pyqtSignal(list)

    def __init__(self, nrows: int = 3, ncols: int = 4, parent=None):
        super().__init__(parent)
        self._nrows = nrows
        self._ncols = ncols
        self._tiles: List[TileState] = [
            TileState(r, c) for r in range(nrows) for c in range(ncols)
        ]

        # Physical-mode state
        self._physical_mode = False
        self._tile_fov_x_um: float = 1.0   # tile width in µm
        self._tile_fov_y_um: float = 1.0   # tile height in µm
        self._dx_step_um: float = 1.0
        self._dy_step_um: float = 1.0

        # Background image state
        self._bg_image: Optional[np.ndarray] = None
        self._image_hfw_um: float = 1.0
        self._image_fov_y_um: float = 1.0
        self._bg_artist: Optional[AxesImage] = None

        self._patches: Dict[Tuple[int, int], MplRectangle] = {}
        self._labels: Dict[Tuple[int, int], MplText] = {}

        self._setup_ui()
        self._full_redraw()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def nrows(self) -> int:
        return self._nrows

    @property
    def ncols(self) -> int:
        return self._ncols

    def set_grid(self, nrows: int, ncols: int) -> None:
        """Resize the grid in grid-unit mode, resetting all tiles to enabled."""
        self._nrows = nrows
        self._ncols = ncols
        self._physical_mode = False
        self._tiles = [TileState(r, c) for r in range(nrows) for c in range(ncols)]
        self._full_redraw()

    def set_tile_settings(self, settings) -> None:
        """Position tiles from an OverviewAcquisitionSettings object.

        Switches to physical-coordinate mode. Preserves enabled/disabled state
        for tiles that exist in the new grid.
        """
        from fibsem.imaging.tiled import compute_tile_grid

        prev_disabled = {(t.row, t.col) for t in self._tiles if not t.enabled}

        self._nrows = settings.nrows
        self._ncols = settings.ncols

        img_w, img_h = settings.image_settings.resolution
        self._tile_fov_x_um = settings.image_settings.hfw * 1e6
        self._tile_fov_y_um = self._tile_fov_x_um * (img_h / img_w) if img_w > 0 else self._tile_fov_x_um
        overlap = settings.overlap
        self._dx_step_um = self._tile_fov_x_um * (1 - overlap)
        self._dy_step_um = self._tile_fov_y_um * (1 - overlap)

        positions = compute_tile_grid(settings)
        self._tiles = [
            TileState(
                row=p.row, col=p.col,
                dx=p.dx, dy=p.dy,
                enabled=(p.row, p.col) not in prev_disabled,
            )
            for p in positions
        ]
        self._physical_mode = True
        self._full_redraw()

    def set_image(self, image: np.ndarray, hfw_um: float) -> None:
        """Display a background SEM image.

        Parameters
        ----------
        image:
            Greyscale (H×W) or RGB (H×W×3) array.
        hfw_um:
            Horizontal field width of the image in micrometres.
        """
        self._bg_image = image
        self._image_hfw_um = hfw_um
        h, w = image.shape[:2]
        self._image_fov_y_um = hfw_um * (h / w) if w > 0 else hfw_um
        self._full_redraw()

    def clear_image(self) -> None:
        """Remove the background image."""
        self._bg_image = None
        self._bg_artist = None
        self._full_redraw()

    def enabled_tiles(self) -> List[Tuple[int, int]]:
        return [(t.row, t.col) for t in self._tiles if t.enabled]

    def disabled_tiles(self) -> List[Tuple[int, int]]:
        return [(t.row, t.col) for t in self._tiles if not t.enabled]

    def set_tile_enabled(self, row: int, col: int, enabled: bool) -> None:
        tile = self._tile_at(row, col)
        if tile is None:
            return
        tile.enabled = enabled
        self._update_patch(tile)
        self._update_status()
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        fig = Figure(facecolor=_COLOR_BG)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self._canvas = FigureCanvasQTAgg(fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._ax = fig.add_subplot(111)
        self._ax.set_facecolor(_COLOR_BG)

        self._status = QLabel()
        self._status.setStyleSheet("color: #AAAAAA; font-size: 11px; padding: 2px 4px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._canvas)
        layout.addWidget(self._status)

        self._canvas.mpl_connect("button_press_event", self._on_click)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _tile_patch_rect(self, tile: TileState) -> Tuple[float, float, float, float]:
        """Return (x_bl, y_bl, width, height) for the patch (bottom-left corner)."""
        if self._physical_mode and tile.dx is not None and tile.dy is not None:
            # Centre of grid shifted to origin
            grid_cx = (self._ncols - 1) * self._dx_step_um / 2
            grid_cy = (self._nrows - 1) * self._dy_step_um / 2

            cx = tile.dx * 1e6 - grid_cx
            # tile.dy is already negated in compute_tile_grid (stage y inversion);
            # adding half grid height re-centres vertically.
            cy = tile.dy * 1e6 + grid_cy

            x = cx - self._tile_fov_x_um / 2
            y = cy - self._tile_fov_y_um / 2
            return x, y, self._tile_fov_x_um, self._tile_fov_y_um
        else:
            pad = _TILE_PAD
            return tile.col + pad, tile.row + pad, 1 - 2 * pad, 1 - 2 * pad

    def _tile_center(self, tile: TileState) -> Tuple[float, float]:
        x, y, w, h = self._tile_patch_rect(tile)
        return x + w / 2, y + h / 2

    def _set_axes_limits(self) -> None:
        ax = self._ax
        ax.set_aspect("equal")
        ax.axis("off")

        if self._physical_mode:
            # Encompass both image and tile grid
            half_iw = self._image_hfw_um / 2 if self._bg_image is not None else 0
            half_ih = self._image_fov_y_um / 2 if self._bg_image is not None else 0

            grid_half_w = ((self._ncols - 1) * self._dx_step_um + self._tile_fov_x_um) / 2
            grid_half_h = ((self._nrows - 1) * self._dy_step_um + self._tile_fov_y_um) / 2

            half_w = max(half_iw, grid_half_w) * 1.05
            half_h = max(half_ih, grid_half_h) * 1.05

            ax.set_xlim(-half_w, half_w)
            ax.set_ylim(-half_h, half_h)
        else:
            ax.set_xlim(-0.1, self._ncols + 0.1)
            ax.set_ylim(self._nrows + 0.1, -0.1)  # row 0 at top

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _full_redraw(self) -> None:
        self._ax.cla()
        self._ax.set_facecolor(_COLOR_BG)
        self._patches.clear()
        self._labels.clear()
        self._bg_artist = None

        self._set_axes_limits()
        self._draw_background()

        for tile in self._tiles:
            self._add_patch(tile)

        self._update_status()
        self._canvas.draw_idle()

    def _draw_background(self) -> None:
        if self._bg_image is None:
            return
        half_w = self._image_hfw_um / 2
        half_h = self._image_fov_y_um / 2
        cmap = "gray" if self._bg_image.ndim == 2 else None
        self._bg_artist = self._ax.imshow(
            self._bg_image,
            extent=[-half_w, half_w, -half_h, half_h],
            cmap=cmap,
            origin="upper",
            aspect="auto",
            zorder=0,
        )

    def _add_patch(self, tile: TileState) -> None:
        x, y, w, h = self._tile_patch_rect(tile)
        color, edge, alpha = self._tile_colors(tile)
        rect = MplRectangle(
            (x, y), w, h,
            linewidth=1.5,
            edgecolor=edge,
            facecolor=color,
            alpha=alpha,
            zorder=2,
        )
        self._ax.add_patch(rect)
        self._patches[(tile.row, tile.col)] = rect

        cx, cy = x + w / 2, y + h / 2
        label = self._ax.text(
            cx, cy, f"{tile.row},{tile.col}",
            ha="center", va="center",
            fontsize=8,
            color=_COLOR_TEXT if tile.enabled else _COLOR_TEXT_DISABLED,
            zorder=3,
        )
        self._labels[(tile.row, tile.col)] = label

    def _update_patch(self, tile: TileState) -> None:
        key = (tile.row, tile.col)
        patch = self._patches.get(key)
        label = self._labels.get(key)
        if patch is None:
            return
        color, edge, alpha = self._tile_colors(tile)
        patch.set_facecolor(color)
        patch.set_edgecolor(edge)
        patch.set_alpha(alpha)
        if label is not None:
            label.set_color(_COLOR_TEXT if tile.enabled else _COLOR_TEXT_DISABLED)

    @staticmethod
    def _tile_colors(tile: TileState) -> Tuple[str, str, float]:
        if tile.enabled:
            return _COLOR_ENABLED, _COLOR_ENABLED_EDGE, _PATCH_ALPHA
        return _COLOR_DISABLED, _COLOR_DISABLED_EDGE, _COLOR_DISABLED_ALPHA

    def _update_status(self) -> None:
        n_on = sum(1 for t in self._tiles if t.enabled)
        total = len(self._tiles)
        mode = "physical" if self._physical_mode else "grid"
        self._status.setText(f"{n_on} / {total} tiles enabled  [{mode} mode]")

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _on_click(self, event) -> None:
        if event.inaxes is not self._ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        tile = self._hit_test(event.xdata, event.ydata)
        if tile is None:
            return

        tile.enabled = not tile.enabled
        self._update_patch(tile)
        self._update_status()
        self._canvas.draw_idle()
        self.tiles_changed.emit(self.enabled_tiles())

    def _hit_test(self, x: float, y: float) -> Optional[TileState]:
        """Return the tile whose patch contains (x, y), or None."""
        for tile in self._tiles:
            tx, ty, tw, th = self._tile_patch_rect(tile)
            if tx <= x <= tx + tw and ty <= y <= ty + th:
                return tile
        return None

    def _tile_at(self, row: int, col: int) -> Optional[TileState]:
        if 0 <= row < self._nrows and 0 <= col < self._ncols:
            idx = row * self._ncols + col
            return self._tiles[idx]
        return None
