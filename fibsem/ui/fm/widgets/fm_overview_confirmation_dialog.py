"""Pre-flight summary for an FM overview acquisition.

House style: a meta line, count chips, a detail block, a duration broken down by where
it goes, and a primary action in the footer. Each fact appears once — the window title
is the heading, the Channels row is the channel count, and the skipped chip is what
"sparse" would have said.

The style itself moved to `fibsem.ui.widgets.preflight` when the third dialog in this
shape appeared — as the second one's docstring said to do — and the *dialog* followed it
there once the beam overview turned out to present its run in exactly this way. What is
left here is only the facts a fluorescence run is described by.
"""

from typing import List, Optional, Sequence, Tuple

from PyQt5.QtWidgets import QWidget

from fibsem import constants
from fibsem.fm.structures import (
    AutoFocusMode,
    AutoFocusSettings,
    ChannelSettings,
    ObjectiveStartPosition,
    OverviewParameters,
    ZParameters,
)
from fibsem.fm.timing import estimate_tileset_acquisition_time
from fibsem.structures import TileOrderStrategy
from fibsem.ui.widgets.preflight import (
    OverviewPreflightDialog,
    PathValue,
    format_bytes,
    format_duration,
    mosaic_pixels,
)

# Fluorescence images are 16-bit. Stated rather than read off a tile, because the dialog
# is shown before anything has been acquired.
FM_BYTES_PER_PIXEL = 2


class FMOverviewConfirmationDialog(OverviewPreflightDialog):
    """Confirm an overview before it runs, with what it will cost."""

    def __init__(
        self,
        parameters: OverviewParameters,
        channel_settings: List[ChannelSettings],
        zparams: Optional[ZParameters] = None,
        tile_fov: Optional[tuple] = None,
        autofocus_settings: Optional[AutoFocusSettings] = None,
        objective_current: Optional[float] = None,
        objective_focus: Optional[float] = None,
        tile_resolution: Optional[tuple] = None,
        save_directory: Optional[str] = None,
        unreachable: Optional[Sequence[Tuple[int, int]]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent, unreachable=unreachable)
        self.parameters = parameters
        self.channel_settings = channel_settings
        self.zparams = zparams if parameters.use_zstack else None
        self.tile_fov = tile_fov
        self.autofocus_settings = autofocus_settings
        # Read by the caller and passed in, so this dialog needs no microscope -- and
        # so the number it reports is the one the settings panel was showing.
        self.objective_current = objective_current
        self.objective_focus = objective_focus
        # Both passed in rather than read: this dialog takes no microscope, which is what
        # lets it be built and tested on its own, and the camera resolution is a hardware
        # read on a shared connection (FIB-517).
        self.tile_resolution = tile_resolution
        self.save_directory = save_directory
        self._init_ui()

    # ── content ──────────────────────────────────────────────────────────

    def _tile_counts(self) -> Tuple[int, int]:
        p = self.parameters
        return p.n_enabled_tiles, p.rows * p.cols

    def _estimate(self) -> dict:
        return estimate_tileset_acquisition_time(
            self.channel_settings,
            (self.parameters.rows, self.parameters.cols),
            self.zparams,
            self.parameters.autofocus_mode,
            tile_mask=self.parameters.tile_mask,
        )

    def _meta_line(self) -> str:
        p = self.parameters
        bits = [f"{p.rows} × {p.cols} grid", f"{p.overlap:.0%} overlap"]
        if self.tile_fov is not None:
            fov_x, fov_y = self.tile_fov
            width = ((p.cols - 1) * (1 - p.overlap) + 1) * fov_x * constants.SI_TO_MICRO
            height = (
                ((p.rows - 1) * (1 - p.overlap) + 1) * fov_y * constants.SI_TO_MICRO
            )
            bits.append(f"{width:.0f} × {height:.0f} µm")
        bits.append(f"{p.tile_order.value} order")
        # No "sparse" tag: the skipped chip states the same thing as a number.
        return " · ".join(bits)

    def _sweep_rows(self) -> List[tuple]:
        """How the sweep is configured, now that it is configurable.

        Reports only the enabled passes, and says so plainly when there are none --
        that combination schedules focusing that then does nothing, and it looks
        correct from either control alone.
        """
        if self.autofocus_settings is None:
            return []

        rows = [("Focus method", self.autofocus_settings.method.value.title())]
        if self.autofocus_settings.channel_name:
            rows.append(("Focus channel", self.autofocus_settings.channel_name))

        enabled = [p for p in self.autofocus_settings.passes if p.enabled]
        if not enabled:
            rows.append(("Sweep", "no passes enabled — nothing will be focused"))
        else:
            # `search_range` is the total span, not a half-width -- positions cover
            # +/- range/2 -- so it is reported as a span rather than as +/-.
            # One pass per line: joined with a separator they wrap mid-pass, and the
            # separator can land at a line end where it reads as a bullet.
            rows.append(
                (
                    "Sweep",
                    "\n".join(
                        f"{p.search_range * constants.SI_TO_MICRO:.0f} µm span, "
                        f"{p.step_size * constants.SI_TO_MICRO:.1f} µm steps "
                        f"({p.n_steps} images)"
                        for p in enabled
                    ),
                )
            )
        return rows

    def _rows(self) -> List[tuple]:
        """Label/value pairs for the detail block."""
        p = self.parameters
        estimate = self._estimate()

        detail = [
            ("Channels", ", ".join(ch.name for ch in self.channel_settings) or "—"),
        ]
        if p.use_zstack and self.zparams is not None:
            detail.append(("Z-stack", f"{self.zparams.num_planes} planes per tile"))
        else:
            detail.append(("Z-stack", "off"))

        # Before the auto-focus row, in the order they take effect: the objective is put
        # here, and then a sweep searches around it.
        focus = p.objective_start is ObjectiveStartPosition.FOCUS
        target = self.objective_focus if focus else self.objective_current
        where = (
            "the saved focus position" if focus else "the current objective position"
        )
        if target is not None:
            where += f", {target * constants.SI_TO_MILLI:.3f} mm"
            # How far it will travel, which is the part worth checking before a run:
            # a start position an unexpected distance away is a setting left over from
            # something else.
            if (
                self.objective_current is not None
                and abs(target - self.objective_current) > 1e-9
            ):
                delta = (target - self.objective_current) * constants.SI_TO_MILLI
                where += f"  ({delta:+.3f} mm from here)"
        detail.append(("Start from", where))

        if p.autofocus_mode is AutoFocusMode.NONE:
            detail.append(("Auto-focus", "off"))
        else:
            mode = p.autofocus_mode.name.replace("_", " ").lower()
            note = ""
            if (
                p.autofocus_mode is AutoFocusMode.EACH_ROW
                and p.tile_order is TileOrderStrategy.SPIRAL
            ):
                # The runner promotes this; say so before it happens rather than
                # leaving the user to find it in the log afterwards.
                note = "  → per tile, for a spiral"
            detail.append(("Auto-focus", f"{mode}{note}"))
            detail.extend(self._sweep_rows())

        detail.append(("Images", f"{estimate['total_images']:,}"))

        # What it will cost on disk. One file per tile plus the stitch beside them, all
        # uncompressed, so the array sizes are the estimate rather than a floor for it.
        # Each tile is already max-projected when it is written, so the z-planes multiply
        # the *time* and not the bytes -- which is why this reads from the channel count
        # and not from `zparams`.
        if self.tile_resolution is not None:
            width, height = self.tile_resolution
            channels = max(1, len(self.channel_settings))
            mosaic_w, mosaic_h = mosaic_pixels(p.rows, p.cols, p.overlap, width, height)
            tile_bytes = channels * width * height * FM_BYTES_PER_PIXEL
            total = (
                p.n_enabled_tiles * tile_bytes
                + channels * mosaic_w * mosaic_h * FM_BYTES_PER_PIXEL
            )
            detail.append(
                (
                    "Disk",
                    f"~{format_bytes(total)}"
                    f"   ({format_bytes(tile_bytes)} per tile"
                    f" · {mosaic_w} × {mosaic_h} px stitched)",
                )
            )

        if self.save_directory:
            detail.append(("Saving to", PathValue(self.save_directory)))
        detail.append(
            (
                "Estimated time",
                f"{format_duration(estimate['total_time'])}"
                f"   ({format_duration(estimate['image_acquisition_time'])} imaging"
                f" · {format_duration(estimate['stage_movement_time'])} moving"
                + (
                    f" · {format_duration(estimate['autofocus_time'])} focusing"
                    if estimate["autofocus_time"]
                    else ""
                )
                + ")",
            )
        )
        return detail
