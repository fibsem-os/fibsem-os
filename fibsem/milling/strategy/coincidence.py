from __future__ import annotations

import datetime
import logging
import os
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from psygnal import Signal

from fibsem import acquire, constants
from fibsem.cancellation import OperationCancelledError
from fibsem.fm.structures import FluorescenceImage
from fibsem.microscope import FibsemMicroscope
from fibsem.milling import (
    FibsemMillingStage,
    MillingStrategy,
    MillingStrategyConfig,
    MillingStrategyResult,
    setup_milling,
)
from fibsem.milling.monitors.base import ON_TRIGGER_END, ON_TRIGGER_SUGGEST
from fibsem.milling.monitors.fm_intensity import FmIntensityConfig, FmIntensityMonitor
from fibsem.milling.progress import MillingProgress, MillingProgressStatus
from fibsem.milling.steps import ENDED, STEP_FAILED, STOPPED
from fibsem.services.milling import monitored_by, report_progress
from fibsem.structures import (
    BeamType,
    Display,
    FibsemImage,
    FibsemRectangle,
    RangeLimit,
    param,
)
from fibsem.utils import save_json


@dataclass
class CoincidenceMillingStrategyConfig(MillingStrategyConfig):
    """Configuration for the Coincidence Milling Strategy."""

    # channel_settings: ChannelSettings = field(default_factory=lambda: channel_settings)
    # zparams: ZParameters = field(default_factory=lambda: ZParameters(-5e-6, 5e-6, 0.5e-6))
    timeout: int = param(
        1800,
        unit="s",
        limits=RangeLimit(30, 9000),
        doc="Milling will timeout after this duration in seconds, without user intervention",
        display=Display("Timeout"),
    )  # seconds, default timeout for milling
    save_fm_images: bool = param(
        True, doc="Save FM images during milling", display=Display("Save FM Images")
    )  # save FM images during milling
    save_rate_limit: int = param(
        2, doc="Save one image every n seconds", display=Display("Save Rate Limit")
    )  # save one image every n seconds
    # acquire_z_stack: bool = False  # acquire a z-stack after milling
    acquire_fib_image: bool = param(
        True,
        doc="Acquire a FIB image after milling",
        display=Display("Acquire FIB Image"),
    )  # acquire a FIB image after milling
    intensity_drop_fraction: float = param(
        0.4,
        limits=RangeLimit(0.05, 0.95),
        doc="Trigger when the rolling mean drops by this fraction below its peak (e.g. 0.4 = 40% drop)",
        display=Display("Drop Fraction"),
    )
    rolling_window: int = param(
        10,
        doc="Number of frames for rolling mean calculation",
        display=Display("Rolling Window"),
    )
    warmup_duration: float = param(
        30.0,
        unit="s",
        doc="Seconds to ignore at start before tracking the peak",
        display=Display("Warmup Duration"),
    )
    consecutive_triggers: int = param(
        10,
        doc="Number of consecutive frames below threshold required to fire the signal",
        display=Display("Consecutive Triggers"),
    )
    supervised: bool = param(
        True,
        doc="Supervised: operator stops milling manually. Unsupervised (False): automatically stop when an intensity drop is detected.",
        display=Display("Supervised"),
    )
    bbox: Optional[FibsemRectangle] = param(
        None, display=Display(hidden=True)
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


# Why the monitor loop ended, in words for the record. "timeout" is whichever
# came first: the mill's estimated time or the configured timeout.
_END_REASONS = {
    "stopped": "stopped by the operator",
    "drop": "intensity drop",
    "timeout": "reached the time limit",
}
# Why the monitor ends the step on a drop, and on the time limit.
DROP, TIME_LIMIT = "intensity drop", "reached the time limit"
# The rule's numbers the live display gets during the warmup.
_WARMUP_STATS = (
    "value",
    "rolling_mean",
    "peak_rolling_mean",
    "threshold_value",
    "warmup_complete",
)


class CoincidenceMonitor(FmIntensityMonitor):
    """FM Intensity for the coincidence strategy. Its settings are the
    strategy's, read every frame, so the viewer's live edits (drop fraction,
    FM box, supervised) apply mid-mill: supervised, a drop is suggested;
    unsupervised it ends the step, also when switched after the drop. It ends
    the step at the strategy's ``timeout``. Each frame goes to the strategy."""

    def __init__(self, strategy: "CoincidenceMillingStrategy") -> None:
        super().__init__(FmIntensityConfig(save_interval=0))
        self.strategy = strategy
        self._timer: Optional[threading.Timer] = None

    def _sync(self) -> None:
        sc, c = self.strategy.config, self.config
        c.drop_fraction = float(sc.intensity_drop_fraction)
        c.rolling_window = int(sc.rolling_window)
        c.warmup_duration = float(sc.warmup_duration)
        c.consecutive_triggers = int(sc.consecutive_triggers)
        c.bbox = sc.bbox
        c.on_trigger = ON_TRIGGER_SUGGEST if sc.supervised else ON_TRIGGER_END
        if c.on_trigger == ON_TRIGGER_END and self.drop_at is not None:
            # switched to unsupervised after a drop it only suggested
            self.end(DROP)

    def on_start(self) -> None:
        self._sync()
        super().on_start()
        self._timer = threading.Timer(
            float(self.strategy.config.timeout), lambda: self.end(TIME_LIMIT)
        )
        self._timer.daemon = True
        self._timer.start()

    def on_frame(self, frame: Any) -> None:
        self._sync()
        super().on_frame(frame)

    def on_stats(self, frame: Any, data: Any, stats: dict) -> None:
        self.strategy._on_frame(frame, data, stats)

    def on_stop(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        super().on_stop()


class CoincidenceMillingStrategy(MillingStrategy[CoincidenceMillingStrategyConfig]):
    """Coincidence Milling Strategy for milling and fluorescence acquisition."""

    name: str = "CoincidenceMilling"
    fullname: str = "Coincidence Fluorescence Milling Strategy"
    config_class = CoincidenceMillingStrategyConfig
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
        # The run's FM Intensity monitor, and whether it saw a drop
        self.monitor: Optional[CoincidenceMonitor] = None
        self._drop_detected: bool = False
        # Key images captured during the run
        self.first_fm_acq: Optional[FluorescenceImage] = None
        self.last_fm_acq: Optional[FluorescenceImage] = None
        self.pre_fib_acq: Optional["FibsemImage"] = None
        self.post_fib_acq: Optional["FibsemImage"] = None
        self._stop_event: Optional["threading.Event"] = None
        # why the last run's monitor loop ended: "stopped", "drop", "timeout"
        self.end_reason: Optional[str] = None

    def __deepcopy__(self, memo: dict) -> "CoincidenceMillingStrategy":
        """Copy the configuration and the outcome, not the run.

        After a run the strategy holds the microscope, the stop event and the
        parent widget, none of which can be deep-copied (thread locks) and none of
        which belong to a config being stored on a lamella. What a copy needs is
        what the strategy *is* -- its config -- plus how the last run ended, so a
        record made from the copy still says why the mill stopped.
        """
        copied = type(self)(config=deepcopy(self.config, memo))
        copied.end_reason = self.end_reason
        return copied

    def _setup_strategy_components(
        self,
        microscope: FibsemMicroscope,
        stage: FibsemMillingStage,
    ):
        self.microscope = microscope
        self.stage = stage

        if self.microscope is None:
            raise ValueError(
                "Microscope is not set. Please set the microscope before running the strategy."
            )
        if self.microscope.fm is None:
            raise ValueError(
                "Coincidence Milling Strategy requires a Fluorescence Module (FM) to be available on the microscope."
            )

        path = self.stage.output_dir
        if path is None:
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

    def run(
        self,
        microscope: FibsemMicroscope,
        stage: FibsemMillingStage,
        stop_event: Optional["threading.Event"] = None,
    ) -> MillingStrategyResult:
        """Coincidence Milling Strategy. Says why the mill stopped, how long it
        milled, whether the intensity dropped, and where its summary is.

        The mill is the milling service's, watched by the FM Intensity monitor
        with this strategy's settings (`CoincidenceMonitor`); a stop ends it
        and the strategy still saves what it saw."""
        logging.info(f"Running {self.name} Milling Strategy for {stage.name}")

        self._stop_event = stop_event
        self._setup_strategy_components(microscope, stage)

        # acquire pre-task fib image
        if self.config.acquire_fib_image:
            image = acquire.acquire_current_image(
                microscope,
                BeamType.ION,
                path=os.path.join(self.path, "pre-milling-fib-image.tif"),
            )
            self.pre_fib_acq = image
            if self.microscope.imaging is not None:
                self.microscope.imaging.show(image)

        if self.config.save_fm_images:
            self.cropped_image_signal.connect(self._save_fm_image)

        setup_milling(microscope, milling_stage=stage, stop_event=stop_event)
        microscope.draw_patterns(stage.define_patterns())

        # reset detection state before acquisition starts
        self.end_reason = None
        self.monitor = None
        self.intensities, self.timestamps = [], []
        self._drop_detected = False
        self._stats_records = []

        estimated_time = microscope.estimate_milling_time()
        report_progress(
            microscope,
            MillingProgress(
                status=MillingProgressStatus.STAGE_UPDATE,
                message=f"Coincidence milling: {stage.name}",
                stage_name=stage.name,
                start_time=time.time(),
                estimated_time=estimated_time,
            ),
        )
        milling_started = time.time()
        try:
            with monitored_by(microscope, stage, self._make_monitor):
                microscope.run_milling(stop_event=stop_event)
        except OperationCancelledError:
            self.end_reason = "stopped"
        finally:
            milled_for = time.time() - milling_started
            self.cropped_image_signal.disconnect(self._save_fm_image)
            self.microscope.finish_milling()
        self.end_reason = self.end_reason or self._why_it_ended()
        logging.info("Milling finished. FM acquisition stopped.")

        # save intensities and timestamps
        self._save_intensities()

        # acquire a fibsem image
        if self.config.acquire_fib_image:
            image = acquire.acquire_current_image(
                self.microscope,
                BeamType.ION,
                path=os.path.join(self.path, "post-milling-fib-image.tif"),
            )
            self.post_fib_acq = image
            if self.microscope.imaging is not None:
                self.microscope.imaging.show(image)

        # save summary figure
        summary = self.save_run_summary_figure()
        return MillingStrategyResult(
            end_reason=_END_REASONS.get(self.end_reason, self.end_reason),
            facts={
                "Mill time (s)": round(milled_for, 1),
                "Intensity drop seen": self._drop_detected,
            },
            figure=summary,
        )

    def _make_monitor(self) -> "CoincidenceMonitor":
        self.monitor = CoincidenceMonitor(self)
        return self.monitor

    def _why_it_ended(self) -> Optional[str]:
        """The key of `_END_REASONS` for how the mill step ended."""
        step = self.monitor.step if self.monitor is not None else None
        end = step.end if step is not None else None
        if end is None:
            return None
        if end.kind == STOPPED or (end.kind == ENDED and end.by == "strategy"):
            return "stopped"  # `stop`, or `stop_pattern`
        if end.kind == ENDED and end.by == self.monitor.by:
            return "drop" if end.reason == DROP else "timeout"
        if end.kind == ENDED:
            return f"ended by {end.by or 'the operator'}: {end.reason}"
        if end.kind == STEP_FAILED:
            return f"failed: {end.reason}"
        return "timeout"  # the mill ran its time

    @property
    def is_cancelled(self) -> bool:
        """Whether the caller's stop event (the milling widget's, from the GUI) is set."""
        return self._stop_event is not None and self._stop_event.is_set()

    def _on_frame(self, image: Any, data: np.ndarray, rule: dict) -> None:
        """A frame the monitor took: for the viewer (the crop and the live
        stats) and for this strategy's files."""
        self.cropped_image_signal.emit({"arr": data, "image": image})
        if self.first_fm_acq is None:
            self.first_fm_acq = image
        self.last_fm_acq = image
        self.intensities.append(float(rule["value"]))
        self.timestamps.append(time.time())
        if rule["drop_detected"]:
            self._drop_detected = True
            logging.warning(
                f"Intensity drop detected: rolling_mean={rule['rolling_mean']:.2f} < "
                f"threshold={rule['threshold_value']:.2f} (peak={rule['peak_rolling_mean']:.2f})"
            )
        stats = {
            "timestamp": time.time(),
            "elapsed_time": self.monitor.step.milled_time(),
            **rule,
        }
        self._stats_records.append(stats)
        if not rule["warmup_complete"]:
            # the live display has only the rule's numbers during the warmup
            stats = {k: rule[k] for k in _WARMUP_STATS}
        self.intensity_stats_signal.emit(stats)

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

    def save_run_summary_figure(self) -> Optional[str]:
        """Save a 2x2 PNG of pre/post FIB and first/last FM images plus an
        intensity line plot, in the shared dark plot theme.

        Each image panel is resized to ~512 px wide (aspect ratio preserved).
        Returns the saved file path, or None if saving failed.
        """
        import cv2
        import matplotlib.dates as mdates
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        from fibsem.theme.plots import DARK, plot_theme

        TARGET_W = 256

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
            with plot_theme(DARK) as theme:
                fig = Figure(figsize=(11, 14))
                FigureCanvasAgg(fig)
                gs = fig.add_gridspec(
                    3, 2, height_ratios=[1, 1, 0.6], hspace=0.05, wspace=0.05
                )

                # ── 2×2 image panels ──────────────────────────────────────
                for idx, (data, title) in enumerate(panels):
                    ax = fig.add_subplot(gs[idx // 2, idx % 2])
                    ax.set_title(title, fontsize=10)
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
                                color=theme.text,
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
                            color=theme.text_muted,
                            transform=ax.transAxes,
                        )

                # ── Intensity line plot (full width) ──────────────────────
                ax_plot = fig.add_subplot(gs[2, :])
                if self.intensities and self.timestamps:
                    dt_times = [
                        datetime.datetime.fromtimestamp(t) for t in self.timestamps
                    ]

                    ax_plot.plot(
                        dt_times,
                        self.intensities,
                        color=theme.series[0],
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
                            color=theme.series[1],
                            linewidth=1.5,
                            linestyle=":",
                            alpha=0.9,
                            label=f"Rolling mean (n={n})",
                        )

                    ax_plot.xaxis.set_major_formatter(
                        mdates.DateFormatter("%I:%M:%S %p")
                    )
                    ax_plot.tick_params(axis="x", rotation=30)
                    ax_plot.set_xlabel("Time", fontsize=9)
                    ax_plot.set_ylabel("Mean intensity", fontsize=9)
                    ax_plot.set_title("FM Intensity over Time", fontsize=10)
                    ax_plot.legend(fontsize=8, loc="upper right")
                else:
                    ax_plot.text(
                        0.5,
                        0.5,
                        "No intensity data",
                        ha="center",
                        va="center",
                        fontsize=12,
                        color=theme.text_muted,
                        transform=ax_plot.transAxes,
                    )
                    ax_plot.axis("off")

                out_path = os.path.join(self.path, "run_summary.png")
                fig.savefig(out_path, dpi=100, bbox_inches="tight")
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
                timestamp = image.metadata.acquisition_datetime.strftime(
                    constants.DATETIME_FILE_NANO
                )
                filename = os.path.join(
                    self.image_path, f"fm-coincidence-{timestamp}.ome.tiff"
                )

                image.save(filename)
        except Exception as e:
            logging.error(f"Error saving FM image: {e}", exc_info=True)
