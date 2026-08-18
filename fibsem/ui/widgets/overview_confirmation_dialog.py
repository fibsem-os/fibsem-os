"""Pre-flight summary for a FIB/SEM overview acquisition.

The beam tab had nothing: pressing Run started driving the stage. That mattered less
when a run was always centred under the stage and always acquired every tile, and it
stopped being true twice — the grid can be dragged somewhere else (FIB-617) and tiles
can be masked off (FIB-618). Both are set on the canvas, silently, and both survive a
tab switch. This is the last place either announces itself.

Same house style as the fluorescence and coincidence dialogs, from the same module, so
the three cannot drift: a meta line, count chips, a detail block, a primary action in
the footer. The fluorescence one is more than a style-mate -- both confirm an overview
and present it identically, so both are `OverviewPreflightDialog` and what is left here
is only the facts a beam run is described by.
"""

from typing import List, Optional, Tuple

from PyQt5.QtWidgets import QWidget

from fibsem import constants
from fibsem.structures import OverviewAcquisitionSettings
from fibsem.ui.widgets.preflight import OverviewPreflightDialog, format_duration


class OverviewConfirmationDialog(OverviewPreflightDialog):
    """Confirm an overview before it runs, with what it will do."""

    def __init__(
        self,
        settings: OverviewAcquisitionSettings,
        view_description: Optional[str] = None,
        offset: Optional[Tuple[float, float]] = None,
        parent: Optional[QWidget] = None,
    ):
        """
        Args:
            settings: what the run will use — already deep-copied by the caller, so the
                numbers shown are the numbers handed to the runner.
            view_description: the beam and the stage orientation, spelled out. Passed in
                rather than derived: the tab knows the pose it has cached, and reading it
                here would be a hardware call on a dialog opening.
            offset: how far the grid's centre sits from the stage, in metres (dx, dy).
                None means it is centred on the stage, which is also what the runner
                falls back to.
        """
        super().__init__(parent)
        self.settings = settings
        self.view_description = view_description
        self.offset = offset
        self._init_ui()

    # ── content ──────────────────────────────────────────────────────────

    def _tile_counts(self) -> Tuple[int, int]:
        return (self.settings.n_enabled_tiles,
                self.settings.nrows * self.settings.ncols)

    def _meta_line(self) -> str:
        s = self.settings
        sym = constants.MICRON_SYMBOL
        return " · ".join([
            f"{s.nrows} × {s.ncols} grid",
            f"{s.overlap:.0%} overlap",
            f"{s.total_fov_x * constants.SI_TO_MICRO:.0f} × "
            f"{s.total_fov_y * constants.SI_TO_MICRO:.0f} {sym}",
            f"{s.tile_order.value} order",
        ])

    def _centre_text(self) -> str:
        """Where the grid sits, in the terms it was placed in.

        Components rather than a distance: the grid is dragged in x and y and checked
        against the canvas in x and y, and "520 µm away" does not say which way.
        """
        if self.offset is None:
            return "the stage position"
        dx, dy = (v * constants.SI_TO_MICRO for v in self.offset)
        sym = constants.MICRON_SYMBOL
        # Under a micron in both is the grid sitting on the stage as far as anything
        # here is concerned, and floating-point dust should not read as a dragged grid.
        if abs(dx) < 1.0 and abs(dy) < 1.0:
            return "the stage position"
        return f"{dx:+.0f}, {dy:+.0f} {sym} from the stage position"

    def _rows(self) -> List[Tuple[str, str]]:
        """Label/value pairs for the detail block."""
        s = self.settings
        image = s.image_settings
        sym = constants.MICRON_SYMBOL
        width, height = image.resolution

        detail: List[Tuple[str, str]] = []
        if self.view_description:
            detail.append(("Acquired in", self.view_description))
        detail.append(("Centred on", self._centre_text()))
        detail.append((
            "Tile",
            f"{width} × {height} px · {image.hfw * constants.SI_TO_MICRO:.0f} {sym} wide",
        ))
        detail.append((
            "Dwell time",
            f"{image.dwell_time * constants.SI_TO_MICRO:.2f} "
            f"{constants.MICROSECOND_SYMBOL}",
        ))
        detail.append(("Auto contrast", "on" if image.autocontrast else "off"))

        # Where it lands, which is the other thing that survives a tab switch unnoticed:
        # the filename names the tile sub-folder, so two runs under one name interleave.
        if image.path:
            detail.append(("Saving to", f"{image.path}/{image.filename}"))

        # "Scan time", not "Estimated time": this is dwell over pixels, and it leaves out
        # the stage entirely. A real run is several times longer -- see
        # `OverviewAcquisitionSettings.scan_time`, which says why that term is not
        # guessed at here.
        detail.append((
            "Scan time",
            f"{format_duration(s.scan_time)}"
            f"   ({format_duration(image.scan_time)} per tile, before stage movement)",
        ))
        return detail
