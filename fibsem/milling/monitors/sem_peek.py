"""SEM peek: every ``interval`` seconds of milled time, pause the mill, take
an SEM image, and resume.

Not a live monitor: it images only while the pattern is paused, so the step
keeps looking at the instrument as usual. Its files: each image
(``sem-NNN.tif``) and a sheet of them all (``sem-peeks.png``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from fibsem.milling.monitors.base import (
    MillingMonitorResult,
    MillingPeriodicMonitor,
    MillingPeriodicMonitorConfig,
)
from fibsem.structures import BeamType, ImageSettings, param


@dataclass
class SemPeekConfig(MillingPeriodicMonitorConfig):
    hfw: float = param(80e-6, unit="m", doc="Field width of each SEM image.")
    dwell_time: float = param(1e-6, unit="s", doc="Dwell time of each SEM image.")
    resolution: Tuple[int, int] = field(default=(1536, 1024))
    autocontrast: bool = param(True, doc="Autocontrast before each image.")


class SemPeekMonitor(MillingPeriodicMonitor[SemPeekConfig]):
    name = "SEM Peek"
    config_class = SemPeekConfig

    def __init__(self, config: Optional[SemPeekConfig] = None) -> None:
        super().__init__(config)
        self.images: List[str] = []
        self._thumbs: List[Tuple[float, Any]] = []
        self._figure: Optional[str] = None

    def on_start(self) -> None:
        if self.folder is not None:
            os.makedirs(self.folder, exist_ok=True)
        super().on_start()

    def on_tick(self) -> None:
        c = self.config
        settings = ImageSettings(
            hfw=c.hfw,
            dwell_time=c.dwell_time,
            resolution=tuple(c.resolution),
            beam_type=BeamType.ELECTRON,
            autocontrast=c.autocontrast,
            save=False,
        )
        milled = self.step.milled_time()
        with self.paused("SEM peek"):
            image = self.step.microscope.imaging.acquire_image(settings)
        reading: dict = {"peek": self.ticks, "milled_time": round(milled, 1)}
        if self.folder is not None:
            path = os.path.join(self.folder, f"sem-{self.ticks:03d}.tif")
            image.save(path)
            self.images.append(path)
            reading["image"] = path
        self._thumbs.append((milled, image.data[::4, ::4]))
        self.report(reading)

    def on_stop(self) -> None:
        super().on_stop()
        if self.folder is not None and self._thumbs:
            self._figure = self._sheet()

    def _sheet(self) -> Optional[str]:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None
        n = len(self._thumbs)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(
            rows, cols, figsize=(3 * cols, 2.2 * rows), squeeze=False
        )
        for ax in axes.flat:
            ax.axis("off")
        for ax, (milled, thumb) in zip(axes.flat, self._thumbs):
            ax.imshow(thumb, cmap="gray")
            ax.set_title(f"{milled:.0f} s milled", fontsize=9)
        fig.tight_layout()
        path = os.path.join(self.folder, "sem-peeks.png")
        try:
            fig.savefig(path, dpi=100)
        except Exception:  # noqa: BLE001 - the figure must not cost the record
            logging.debug("could not save the SEM peek sheet", exc_info=True)
            path = None
        plt.close(fig)
        return path

    def result(self) -> MillingMonitorResult:
        return MillingMonitorResult(
            facts={"Peeks": len(self._thumbs)},
            figure=self._figure,
            files=self.folder,
        )
