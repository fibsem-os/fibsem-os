from __future__ import annotations
import datetime
import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional
from queue import Queue

import numpy as np
import tifffile as tff
from psygnal import Signal

from fibsem import constants
from fibsem.utils import save_json
from fibsem.fm.structures import ChannelSettings, FluorescenceImage, ZParameters
from fibsem.microscope import FibsemMicroscope
from fibsem.milling import (
    setup_milling,
    FibsemMillingStage,
    MillingStrategy,
    MillingStrategyConfig,
)
from fibsem.structures import BeamType, FibsemImage, FibsemRectangle
from fibsem.microscopes.simulator import DemoMicroscope

if TYPE_CHECKING:
    from fibsem.ui.widgets.milling_widget import FibsemMillingWidget2


@dataclass
class CoincidenceMillingStrategyConfig(MillingStrategyConfig):
    """Configuration for the Coincidence Milling Strategy."""

    # channel_settings: ChannelSettings = field(default_factory=lambda: channel_settings)
    # zparams: ZParameters = field(default_factory=lambda: ZParameters(-5e-6, 5e-6, 0.5e-6))
    timeout: int = field(
        default=1800,
        metadata={
            "label": "Timeout",
            "units": "s",
            "minimum": 30,
            "maximum": 9000,
            "tooltip": "Milling will timeout after this duration in seconds, without user intervention",
        },
    )  # seconds, default timeout for milling
    save_fm_images: bool = field(
        default=True,
        metadata={
            "label": "Save FM Images",
            "tooltip": "Save FM images during milling",
        },
    )  # save FM images during milling
    save_rate_limit: int = field(
        default=2,
        metadata={
            "label": "Save Rate Limit",
            "tooltip": "Save one image every n seconds",
        },
    )  # save one image every n seconds
    # acquire_z_stack: bool = False  # acquire a z-stack after milling
    acquire_fib_image: bool = field(
        default=True,
        metadata={
            "label": "Acquire FIB Image",
            "tooltip": "Acquire a FIB image after milling",
        },
    )  # acquire a FIB image after milling
    intensity_drop_fraction: float = field(
        default=0.4,
        metadata={
            "label": "Drop Fraction",
            "minimum": 0.05,
            "maximum": 0.95,
            "tooltip": "Trigger when the rolling mean drops by this fraction below its peak (e.g. 0.4 = 40% drop)",
        },
    )
    rolling_window: int = field(
        default=10,
        metadata={
            "label": "Rolling Window",
            "tooltip": "Number of frames for rolling mean calculation",
        },
    )
    warmup_duration: float = field(
        default=30.0,
        metadata={
            "label": "Warmup Duration",
            "units": "s",
            "tooltip": "Seconds to ignore at start before tracking the peak",
        },
    )
    consecutive_triggers: int = field(
        default=10,
        metadata={
            "label": "Consecutive Triggers",
            "tooltip": "Number of consecutive frames below threshold required to fire the signal",
        },
    )
    supervised: bool = field(
        default=True,
        metadata={
            "label": "Supervised",
            "tooltip": "Supervised: operator stops milling manually. Unsupervised (False): automatically stop when an intensity drop is detected.",
        },
    )
    bbox: Optional[FibsemRectangle] = field(
        default=None,
        metadata={"hidden": True},  # set interactively via the FM ROI, not a form control
    )  # reduced area for intensity monitoring
    # oscillation parameters

    @classmethod
    def from_dict(cls, d: dict) -> "CoincidenceMillingStrategyConfig":
        # asdict() flattens bbox to a plain dict; reconstruct it back into a
        # FibsemRectangle so config.bbox is a usable object after a round-trip.
        d = dict(d)
        bbox = d.get("bbox")
        if bbox is not None and not isinstance(bbox, FibsemRectangle):
            d["bbox"] = FibsemRectangle.from_dict(bbox)
        return cls(**d)


# ASSUMPTIONS:
# BEFORE STARTING:
#   CHANNEL SETTINGS ARE APPLIED TO MICROSCOPE
#   OBJECTIVE IS IN THE CORRECT POSITION
#   MILLING PATTERN IS VALID: Trench / Rectangle
# SUPERVISION:
#   SUPERVISED (default): OPERATOR SUPERVISES THE MILLING PROCESS AND STOPS WHEN NECESSARY
#   UNSUPERVISED (supervised=False): AUTOMATICALLY STOPS ON INTENSITY DROP

# SETUP MILLING
# DRAW PATTERNS
# START MILLING + START FM ACQUISITION
# LOOP:
#   CHECK FOR STOP EVENT
#   CROP FM IMAGE TO BBOX + EMIT
#   CALC MEAN INTENSITY + EMIT
# STOP FM ACQUISITION + STOP MILLING
# FINALIZE MILLING

# WRAP IN TRY/EXCEPT SO THAT SIGNALS ARE DISCONNECTED PROPERLY
# AND MICROSCOPE IS LEFT IN A SAFE STATE (STOP ACQ, RESTORE IMAGING CURRENT)


class CoincidenceMillingStrategy(MillingStrategy[CoincidenceMillingStrategyConfig]):
    """Coincidence Milling Strategy for milling and fluorescence acquisition."""

    name: str = "CoincidenceMilling"
    fullname: str = "Coincidence Fluorescence Milling Strategy"
    config_class = CoincidenceMillingStrategyConfig
    # driven from the coincidence viewer, not offered in the generic selectors
    selectable: bool = False
    cropped_image_signal = Signal(dict)
    intensity_stats_signal = Signal(
        dict
    )  # emits rolling stats every frame for live display

    def __init__(self, config: Optional[CoincidenceMillingStrategyConfig] = None):
        if config is None:
            config = self.config_class()
        self.config = config
        self.microscope: FibsemMicroscope
        self.stage: FibsemMillingStage
        self.intensities: list[float] = []
        self.timestamps: list[float] = []
        self._stats_records: list[dict] = []
        self.path: str
        self.image_path: str
        self.camera_resolution: tuple[int, int]
        self._last_save_time: float = time.time()  # for rate limiting image saves
        # Intensity drop detection state
        self._peak_rolling_mean: float = 0.0
        self._consecutive_trigger_count: int = 0
        self._acquisition_start_time: Optional[float] = None
        self._warmup_complete: bool = False
        self._drop_detected: bool = False  # latched when a drop event fires
        # Key images captured during the run
        self.first_fm_acq: Optional[FluorescenceImage] = None
        self.last_fm_acq: Optional[FluorescenceImage] = None
        self.pre_fib_acq: Optional["FibsemImage"] = None
        self.post_fib_acq: Optional["FibsemImage"] = None

    def _setup_strategy_components(
        self,
        microscope: FibsemMicroscope,
        stage: FibsemMillingStage,
        parent_ui: Optional["FibsemMillingWidget2"] = None,
    ):
        self.microscope = microscope
        self.stage = stage
        self.parent_ui = parent_ui

        if self.microscope is None:
            raise ValueError(
                "Microscope is not set. Please set the microscope before running the strategy."
            )
        if self.microscope.fm is None:
            raise ValueError(
                "Coincidence Milling Strategy requires a Fluorescence Module (FM) to be available on the microscope."
            )

        path = self.stage.imaging.path
        if path is None:
            path = os.getcwd()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        path = os.path.join(path, f"coincidence-images-{timestamp}")
        self.path = path
        self.image_path = os.path.join(self.path, "timelapse")
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.image_path, exist_ok=True)
        logging.info(f"Created directory for coincidence images at {self.path}")

        # cache camera resolution
        self.camera_resolution = self.microscope.fm.camera.resolution

    def _start_milling_and_acquisition(self) -> float:
        # start milling, and estimate time (if supported by microscope)
        # explicitly re-select the FIB view before milling: FM live acquisition
        # may have left a different channel active (see set_channel below in _setup_milling)
        self.microscope.set_channel(self.microscope.milling_channel)
        self.microscope.start_milling()  # asynchronous start
        estimated_time = self.microscope.estimate_milling_time()
        if isinstance(self.microscope, DemoMicroscope):
            estimated_time += 300  # seconds, override for demo purposes
        time.sleep(1)

        # start acquisition after starting milling
        self._acquisition_start_time = time.time()
        self.microscope.fm.start_acquisition()  # type: ignore

        return estimated_time

    def _connect_signals(self):
        # connect the acquisition signal
        self.microscope.fm.acquisition_signal.disconnect(self.on_fm_acquisition_signal)  # type: ignore
        self.microscope.fm.acquisition_signal.connect(self.on_fm_acquisition_signal)  # type: ignore
        if self.config.save_fm_images:
            self.cropped_image_signal.connect(self._save_fm_image)

    def _setup_milling(self):
        # setup milling
        self.microscope.stop_milling()
        setup_milling(self.microscope, self.stage)
        # explicitly select the FIB view before drawing patterns: the FM acquisition
        # path can leave a non-FIB channel active, which breaks pattern drawing/milling
        self.microscope.set_channel(self.microscope.milling_channel)
        self.microscope.draw_patterns(patterns=self.stage.define_patterns())

    def run(
        self,
        microscope: FibsemMicroscope,
        stage: FibsemMillingStage,
        asynch: bool = False,
        parent_ui: Optional["FibsemMillingWidget2"] = None,
    ) -> None:
        """Coincidence Milling Strategy"""
        logging.info(f"Running {self.name} Milling Strategy for {stage.name}")

        self._setup_strategy_components(microscope, stage, parent_ui)

        # acquire pre-task fib image
        if self.config.acquire_fib_image:
            image = microscope.acquire_image(beam_type=BeamType.ION)
            image.save(os.path.join(self.path, "pre-milling-fib-image.tif"))
            self.pre_fib_acq = image
            self.microscope.fib_acquisition_signal.emit(image)

        # connect acquisition signals
        self._connect_signals()

        self._setup_milling()

        # reset detection state before acquisition starts
        self._peak_rolling_mean = 0.0
        self._consecutive_trigger_count = 0
        self._warmup_complete = False
        self._drop_detected = False
        self._stats_records = []

        # start milling and acquisition
        estimated_time = self._start_milling_and_acquisition()

        # monitor milling progress and check for stop event
        self._monitor_milling_progress(estimated_time=estimated_time)

        # stop milling and acquisition, finalize milling
        self._stop_milling_and_acquisition()

        # save intensities and timestamps
        self._save_intensities()

        # acquire a fibsem image
        if self.config.acquire_fib_image:
            image = self.microscope.acquire_image(beam_type=BeamType.ION)
            image.save(os.path.join(self.path, "post-milling-fib-image.tif"))
            self.post_fib_acq = image
            self.microscope.fib_acquisition_signal.emit(image)

        # save summary figure
        self.save_run_summary_figure()

    def _stop_milling_and_acquisition(self):
        if self.microscope is None or self.microscope.fm is None:
            raise ValueError(
                "Microscope or FM is not set. Cannot stop milling and acquisition."
            )

        # stop milling and fm acquisition
        self.microscope.stop_milling()
        self.microscope.fm.stop_acquisition()
        logging.info("Milling finished. FM acquisition stopped.")

        # finalize
        self.microscope.fm.acquisition_signal.disconnect(self.on_fm_acquisition_signal)
        self.cropped_image_signal.disconnect(self._save_fm_image)
        # ensure the FIB view is active so finish_milling (clears patterns, restores
        # imaging current) operates on the correct channel
        self.microscope.set_channel(self.microscope.milling_channel)
        self.microscope.finish_milling(
            imaging_current=self.microscope.system.ion.beam.beam_current,
            imaging_voltage=self.microscope.system.ion.beam.voltage,
        )

        # post image acquisition handled by parent task
        # acquire a z-stack post-milling
        # if self.config.acquire_z_stack:
        #     image = acquire_z_stack(
        #         self.microscope.fm,
        #         channel_settings=self.config.channel_settings,
        #         zparams=self.config.zparams,
        #     )

    @property
    def is_cancelled(self) -> bool:
        """Check if the milling process has been cancelled via the parent UI."""
        if self.parent_ui and hasattr(self.parent_ui, "_milling_stop_event"):
            return self.parent_ui._milling_stop_event.is_set()
        return False

    def _monitor_milling_progress(self, estimated_time: float):
        """Monitor milling progress, emit updates, and check for stop event."""
        remaining_time = estimated_time
        start_time = time.time()
        estimated_end_time = start_time + estimated_time
        timeout_end_time = start_time + self.config.timeout
        max_end_time = min(timeout_end_time, estimated_end_time)
        SLEEP_DURATION = 1  # seconds

        while True:
            # sleep
            time.sleep(SLEEP_DURATION)

            # check for stop event
            if self.is_cancelled:
                logging.info("Milling stop event set. Stopping milling.")
                self.microscope.stop_milling()
                break

            # unsupervised runs: automatically stop on intensity drop
            if not self.config.supervised and self._drop_detected:
                logging.info(
                    "Unsupervised: intensity drop detected. Stopping milling."
                )
                self.microscope.stop_milling()
                break

            # TODO: support pause behaviour... just keep fm acq, pause milling
            # NOTE: the bbox is pushed to self.config.bbox by the viewer via
            # _on_bbox_update as the FM ROI rectangle changes.

            # update milling progress via signal
            remaining_time = max(0.0, max_end_time - time.time())
            self.microscope.milling_progress_signal.emit(
                {
                    "progress": {
                        "state": "update",
                        "start_time": start_time,
                        "milling_state": "UNKNOWN",
                        "estimated_time": estimated_time,
                        "remaining_time": remaining_time,
                    }
                }
            )
            # timeout
            if time.time() >= max_end_time:
                logging.info(
                    f"Max End Time reached: {max_end_time - start_time} seconds have passed. Stopping milling."
                )
                break

            continue

    def _save_intensities(self):
        if self.path is None:
            logging.warning("Path is not set. Cannot save intensities.")
            return

        if self.camera_resolution is None:
            logging.warning("Camera resolution is not set. Cannot save bounding box.")
            return

        if self.config.bbox is None:
            logging.warning("Bounding box is not set. Cannot save bounding box.")
            return

        # write intensities and timestamps as 2d np.arr to .path
        try:
            arr = np.array([self.intensities, self.timestamps])
            intensities_path = os.path.join(self.path, "intensities.npy")
            np.save(file=intensities_path, arr=arr)
            logging.info(f"Saved intensities to {intensities_path}")

            x, y, w, h = self.config.bbox.to_pixel_coordinates(self.camera_resolution)
            bbox_dict = {"x": x, "y": y, "w": w, "h": h}
            save_json(os.path.join(self.path, "bbox.json"), bbox_dict)

        except Exception as e:
            logging.error(f"Error occured while saving intensities... {e}")

        self._save_stats_dataframe()

    def _save_stats_dataframe(self) -> None:
        if not self._stats_records or self.path is None:
            return
        try:
            import pandas as pd

            df = pd.DataFrame(self._stats_records)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
            csv_path = os.path.join(self.path, "intensity_stats.csv")
            df.to_csv(csv_path, index=False)
            logging.info(f"Saved intensity stats DataFrame to {csv_path}")
        except Exception as e:
            logging.error(f"Error saving intensity stats DataFrame: {e}", exc_info=True)

    def on_fm_acquisition_signal(self, image: FluorescenceImage):
        if self.microscope is None or self.microscope.fm is None:
            logging.error(
                "Microscope or FM is not set. Cannot process FM acquisition signal."
            )
            return
        if self.camera_resolution is None:
            logging.error(
                "Camera resolution is not set. Cannot process FM acquisition signal."
            )
            return

        # crop to bounding box if specified
        data = image.data
        if self.config.bbox is not None:
            x, y, w, h = self.config.bbox.to_pixel_coordinates(self.camera_resolution)
            data = image.data[y : y + h, x : x + w]
        self.cropped_image_signal.emit({"arr": data, "image": image})

        if self.first_fm_acq is None:
            self.first_fm_acq = image
        self.last_fm_acq = image

        # calc mean intensity
        mean_intensity = data.mean()
        if np.isnan(mean_intensity):
            return

        # save intensities and timestamps
        self.intensities.append(mean_intensity)
        self.timestamps.append(time.time())

        # --- rolling stats ---
        n = int(self.config.rolling_window)
        rolling_mean = float(np.mean(self.intensities[-n:]))

        # --- warmup guard ---
        elapsed = time.time() - (self._acquisition_start_time or time.time())
        if not self._warmup_complete:
            if elapsed >= self.config.warmup_duration:
                self._warmup_complete = True
            self.intensity_stats_signal.emit(
                {
                    "value": mean_intensity,
                    "rolling_mean": rolling_mean,
                    "peak_rolling_mean": self._peak_rolling_mean,
                    "threshold_value": 0.0,
                    "warmup_complete": False,
                }
            )
            self._stats_records.append(
                {
                    "timestamp": time.time(),
                    "elapsed_time": elapsed,
                    "value": mean_intensity,
                    "rolling_mean": rolling_mean,
                    "peak_rolling_mean": self._peak_rolling_mean,
                    "threshold_value": 0.0,
                    "warmup_complete": False,
                    "drop_detected": False,
                    "drop_fraction": 1.0,
                    "threshold_fraction": 1.0 - self.config.intensity_drop_fraction,
                    "consecutive_count": 0,
                }
            )
            return

        # update peak after warmup
        if rolling_mean > self._peak_rolling_mean:
            self._peak_rolling_mean = rolling_mean

        threshold_value = (
            1.0 - self.config.intensity_drop_fraction
        ) * self._peak_rolling_mean

        # --- drop detection ---
        if rolling_mean < threshold_value:
            self._consecutive_trigger_count += 1
        else:
            self._consecutive_trigger_count = 0

        drop_detected = self._consecutive_trigger_count >= int(
            self.config.consecutive_triggers
        )
        if drop_detected:
            self._consecutive_trigger_count = 0  # reset so it fires once per event
            self._drop_detected = True  # latch for the monitor loop (auto-stop)
            logging.warning(
                f"Intensity drop detected: rolling_mean={rolling_mean:.2f} < "
                f"threshold={threshold_value:.2f} (peak={self._peak_rolling_mean:.2f})"
            )

        drop_fraction = (
            rolling_mean / self._peak_rolling_mean
            if self._peak_rolling_mean > 0
            else 1.0
        )
        stats = {
            "timestamp": time.time(),
            "elapsed_time": elapsed,
            "value": mean_intensity,
            "rolling_mean": rolling_mean,
            "peak_rolling_mean": self._peak_rolling_mean,
            "threshold_value": threshold_value,
            "warmup_complete": True,
            "drop_detected": drop_detected,
            "drop_fraction": drop_fraction,
            "threshold_fraction": 1.0 - self.config.intensity_drop_fraction,
            "consecutive_count": self.config.consecutive_triggers,
        }
        self.intensity_stats_signal.emit(stats)
        self._stats_records.append(stats)

    def save_run_summary_figure(self) -> Optional[str]:
        """Save a 2x2 PNG of pre/post FIB and first/last FM images plus an
        intensity line plot styled to match LinePlotWidget (dark theme).

        Each image panel is resized to ~512 px wide (aspect ratio preserved).
        Returns the saved file path, or None if saving failed.
        """
        import cv2
        import matplotlib.dates as mdates
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        TARGET_W = 256
        _BG = "#262930"
        _FG = "white"

        def _to_display(arr: np.ndarray) -> np.ndarray:
            """Squeeze to 2-D YX, normalise to uint8, resize to TARGET_W wide."""
            while arr.ndim > 2:
                arr = arr[0]
            arr = arr.astype(np.float32)
            mn, mx = arr.min(), arr.max()
            if mx > mn:
                arr = (arr - mn) / (mx - mn) * 255.0
            arr = arr.clip(0, 255).astype(np.uint8)
            h, w = arr.shape
            new_h = max(1, int(h * TARGET_W / w))
            return cv2.resize(arr, (TARGET_W, new_h), interpolation=cv2.INTER_AREA)

        def _style_ax(ax) -> None:
            """Apply LinePlotWidget dark theme to an axes."""
            ax.set_facecolor(_BG)
            ax.tick_params(colors=_FG, which="both", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(_FG)
            ax.xaxis.label.set_color(_FG)
            ax.yaxis.label.set_color(_FG)
            ax.title.set_color(_FG)

        panels = [
            (
                self.pre_fib_acq.data if self.pre_fib_acq is not None else None,
                "Pre-Milling FIB",
            ),
            (
                self.post_fib_acq.data if self.post_fib_acq is not None else None,
                "Post-Milling FIB",
            ),
            (
                self.first_fm_acq.data if self.first_fm_acq is not None else None,
                "First FM Acquisition",
            ),
            (
                self.last_fm_acq.data if self.last_fm_acq is not None else None,
                "Last FM Acquisition",
            ),
        ]

        try:
            fig = Figure(figsize=(11, 14))
            fig.patch.set_facecolor(_BG)
            FigureCanvasAgg(fig)
            gs = fig.add_gridspec(
                3, 2, height_ratios=[1, 1, 0.6], hspace=0.05, wspace=0.05
            )

            # ── 2×2 image panels ──────────────────────────────────────
            for idx, (data, title) in enumerate(panels):
                ax = fig.add_subplot(gs[idx // 2, idx % 2])
                ax.set_facecolor(_BG)
                ax.set_title(title, fontsize=10, color=_FG)
                ax.axis("off")
                if data is not None:
                    try:
                        ax.imshow(_to_display(data), cmap="gray", aspect="equal")
                    except Exception:
                        logging.exception(
                            "save_run_summary_figure: failed to render panel '%s'",
                            title,
                        )
                        ax.text(
                            0.5,
                            0.5,
                            "error",
                            ha="center",
                            va="center",
                            color=_FG,
                            transform=ax.transAxes,
                        )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "N/A",
                        ha="center",
                        va="center",
                        fontsize=12,
                        color="gray",
                        transform=ax.transAxes,
                    )

            # ── Intensity line plot (full width) ──────────────────────
            ax_plot = fig.add_subplot(gs[2, :])
            _style_ax(ax_plot)
            if self.intensities and self.timestamps:
                dt_times = [datetime.datetime.fromtimestamp(t) for t in self.timestamps]

                ax_plot.plot(
                    dt_times,
                    self.intensities,
                    color="#0f7aad",
                    linewidth=1.0,
                    alpha=0.8,
                    label="Mean intensity",
                )

                # rolling mean overlay
                n = int(self.config.rolling_window)
                if len(self.intensities) >= n:
                    rolling = [
                        float(np.mean(self.intensities[max(0, i - n) : i + 1]))
                        for i in range(len(self.intensities))
                    ]
                    ax_plot.plot(
                        dt_times,
                        rolling,
                        color="#00ff66",
                        linewidth=1.5,
                        linestyle=":",
                        alpha=0.9,
                        label=f"Rolling mean (n={n})",
                    )

                ax_plot.xaxis.set_major_formatter(mdates.DateFormatter("%I:%M:%S %p"))
                ax_plot.tick_params(axis="x", rotation=30)
                ax_plot.set_xlabel("Time", fontsize=9, color=_FG)
                ax_plot.set_ylabel("Mean intensity", fontsize=9, color=_FG)
                ax_plot.set_title("FM Intensity over Time", fontsize=10, color=_FG)

                legend = ax_plot.legend(fontsize=8, loc="upper right")
                legend.get_frame().set_facecolor(_BG)
                legend.get_frame().set_edgecolor(_FG)
                for text in legend.get_texts():
                    text.set_color(_FG)
            else:
                ax_plot.text(
                    0.5,
                    0.5,
                    "No intensity data",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="gray",
                    transform=ax_plot.transAxes,
                )
                ax_plot.axis("off")

            out_path = os.path.join(self.path, "run_summary.png")
            fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=_BG)
            logging.info("Saved run summary figure to %s", out_path)
            return out_path

        except Exception:
            logging.exception("save_run_summary_figure: failed to save figure")
            return None

    def _on_bbox_update(self, bbox: Optional[FibsemRectangle]):
        """Handle updates to the bounding box. Pass None to clear (use full image)."""
        logging.info(f"Bounding Box Updated: {bbox}")
        self.config.bbox = bbox

    def _save_fm_image(self, ddict: dict) -> None:
        if self.stage is None:
            return
        try:
            # rate limit saving
            current_time = time.time()
            if self._last_save_time is not None:
                elapsed_time = current_time - self._last_save_time
                if elapsed_time < self.config.save_rate_limit:
                    return
            self._last_save_time = current_time

            # save the full image with metadata
            image: FluorescenceImage = ddict.get("image", None)  # type: ignore
            if image is not None and self.image_path is not None:
                # convert timestamp from isoformat to formatted string
                timestamp = datetime.datetime.fromisoformat(
                    image.metadata.acquisition_date
                ).strftime(constants.DATETIME_FILE_NANO)
                filename = os.path.join(
                    self.image_path, f"fm-coincidence-{timestamp}.ome.tiff"
                )

                image.save(filename)
        except Exception as e:
            logging.error(f"Error saving FM image: {e}", exc_info=True)
