from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from threading import Event as ThreadingEvent
from typing import TYPE_CHECKING, Optional

import numpy as np

from fibsem import acquire, utils
from fibsem.structures import (
    BeamType,
    FibsemImage,
    ImageSettings,
    Point,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from fibsem.microscope import FibsemMicroscope


from fibsem.alignment.methods import (
    crosscorrelation_cv2,
    crosscorrelation_v2,
    shift_from_crosscorrelation,
    shift_from_crosscorrelation_v2,
    shift_from_skimage_phase_correlation,
)
from fibsem.alignment.plotting import (
    _alignment_save_path,
    plot_multi_step_alignment,
)

ALIGNMENT_SUBDIR = "Alignment"


class AlignmentSubsystem(Enum):
    BEAM_SHIFT = "beam-shift"
    STAGE = "stage"
    STAGE_VERTICAL = "stage-vertical"


class AlignmentMethod(Enum):
    CROSS_CORRELATION = "cross-correlation"
    PHASE_CORRELATION = "phase-correlation"
    SKIMAGE_PHASE_CORRELATION = "skimage-phase-correlation"


DEFAULT_ALIGNMENT_METHOD = AlignmentMethod.CROSS_CORRELATION


@dataclass
class AlignmentIteration:
    """Result of a single alignment step."""

    shift: Point  # (x, y) shift applied, in metres
    score: float  # alignment quality score; higher = better
    image: FibsemImage  # new image acquired during this alignment step
    xcorr: Optional[np.ndarray] = None  # cross-correlation map (bandpass method only)
    success: bool = True  # False if score < minimum_response (shift was zeroed)
    method: Optional[AlignmentMethod] = None  # which method produced this result

    @property
    def shift_px(self) -> Point:
        """Shift in pixels, derived from shift (metres) and image pixel size."""
        if self.image is None or self.image.metadata is None:
            return Point(0.0, 0.0)
        px = self.image.metadata.pixel_size.x
        py = self.image.metadata.pixel_size.y
        return Point(self.shift.x / px, self.shift.y / py)

    def to_dict(self) -> dict:
        return {
            "shift": self.shift.to_dict(),
            "score": self.score,
            "shift_px": self.shift_px.to_dict(),
            "success": self.success,
            "method": self.method.value if self.method else None,
        }

    @staticmethod
    def from_dict(d: dict, image: FibsemImage) -> "AlignmentIteration":
        return AlignmentIteration(
            shift=Point.from_dict(d["shift"]),
            score=d["score"],
            image=image,
            success=d.get("success", True),
            method=AlignmentMethod(d["method"]) if d.get("method") else None,
        )


@dataclass
class AlignmentDifferential:
    """Pairwise comparison of shift estimates across alignment methods."""

    results: "list[AlignmentIteration]"  # one per method, in method order
    shifts_px: "dict[str, Point]"  # method name → shift in pixels
    max_disagreement_px: float  # max pairwise |shift_a − shift_b| in pixels
    agreement: bool  # True if max_disagreement_px < threshold
    consensus_shift: Optional[Point] = None  # score-weighted mean shift in metres

    def to_dict(self) -> dict:
        return {
            "shifts_px": {k: v.to_dict() for k, v in self.shifts_px.items()},
            "max_disagreement_px": float(self.max_disagreement_px),
            "agreement": bool(self.agreement),
            "consensus_shift": self.consensus_shift.to_dict()
            if self.consensus_shift
            else None,
        }

    @staticmethod
    def from_dict(d: dict) -> "AlignmentDifferential":
        return AlignmentDifferential(
            results=[],  # images not serialised; reload via AlignmentResult.load()
            shifts_px={k: Point.from_dict(v) for k, v in d["shifts_px"].items()},
            max_disagreement_px=d["max_disagreement_px"],
            agreement=d["agreement"],
            consensus_shift=Point.from_dict(d["consensus_shift"])
            if d.get("consensus_shift")
            else None,
        )


@dataclass
class AlignmentResult:
    """Inputs and per-step results for a single multi-step alignment operation."""

    name: str
    reference_image: FibsemImage
    subsystem: AlignmentSubsystem
    method: AlignmentMethod
    results: list[AlignmentIteration]
    final_image: Optional[FibsemImage] = None
    validation: Optional[AlignmentDifferential] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "subsystem": self.subsystem.value,
            "method": self.method.value,
            "results": [r.to_dict() for r in self.results],
            "validation": self.validation.to_dict() if self.validation else None,
        }

    def save(
        self, base_path: str, plot: bool = True, plot_title: Optional[str] = None
    ) -> str:
        """Save to <base_path>/<name>/. Returns the run directory path."""
        import json

        run_dir = os.path.join(base_path, self.name)
        os.makedirs(run_dir, exist_ok=True)
        self.reference_image.save(path=os.path.join(run_dir, "reference_image"))
        for i, r in enumerate(self.results):
            r.image.save(path=os.path.join(run_dir, f"new_image_{i:02d}"))
        if self.final_image:
            self.final_image.save(path=os.path.join(run_dir, "final_image"))
        fpath = os.path.join(run_dir, "data.json")
        with open(fpath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        if plot:
            try:
                self.plot(title=plot_title, path=run_dir)
            except Exception as e:
                logging.warning(f"Failed to save alignment plot: {e}")
        return run_dir

    def plot(
        self, title: Optional[str] = None, save: bool = True, path: Optional[str] = None
    ) -> "Figure":
        fig = plot_multi_step_alignment(
            self.reference_image,
            self.results,
            save=save,
            title=title,
            final_image=self.final_image,
            path=path,
            validation=self.validation,
        )
        return fig

    @classmethod
    def load(cls, run_dir: str) -> "AlignmentResult":
        """Load a previously saved AlignmentResult from its directory."""
        import json

        with open(os.path.join(run_dir, "data.json")) as f:
            d = json.load(f)

        reference_image = FibsemImage.load(os.path.join(run_dir, "reference_image.tif"))

        results = []
        for i, rd in enumerate(d["results"]):
            image = FibsemImage.load(os.path.join(run_dir, f"new_image_{i:02d}.tif"))
            results.append(AlignmentIteration.from_dict(rd, image))

        final_path = os.path.join(run_dir, "final_image.tif")
        final_image = (
            FibsemImage.load(final_path) if os.path.exists(final_path) else None
        )

        validation = (
            AlignmentDifferential.from_dict(d["validation"])
            if d.get("validation")
            else None
        )

        return cls(
            name=d["name"],
            reference_image=reference_image,
            subsystem=AlignmentSubsystem(d["subsystem"]),
            method=AlignmentMethod(d["method"]),
            results=results,
            final_image=final_image,
            validation=validation,
        )


def _acquire_from_reference_image(
    microscope: FibsemMicroscope,
    ref_image: FibsemImage,
    use_autocontrast: bool = False,
    use_autofocus: bool = False,
) -> FibsemImage:
    """Acquire a new image with the same settings as the reference image."""
    image_settings = ImageSettings.fromFibsemImage(ref_image)
    image_settings.autocontrast = False
    image_settings.save = False

    if use_autocontrast:
        microscope.autocontrast(
            beam_type=image_settings.beam_type, reduced_area=image_settings.reduced_area
        )
    if use_autofocus:
        microscope.auto_focus(
            beam_type=image_settings.beam_type, reduced_area=image_settings.reduced_area
        )

    return acquire.acquire_image(microscope, settings=image_settings)


def _calculate_shift(
    ref_image: FibsemImage,
    new_image: FibsemImage,
    method: AlignmentMethod = DEFAULT_ALIGNMENT_METHOD,
) -> AlignmentIteration:
    """Calculate the shift between two images using the specified alignment method."""
    logging.debug(f"Calculating shift using method: {method.value}...")
    if method is AlignmentMethod.SKIMAGE_PHASE_CORRELATION:
        result = shift_from_skimage_phase_correlation(ref_image, new_image)
        result.method = method
        return result
    elif method is AlignmentMethod.PHASE_CORRELATION:
        dx, dy, score = shift_from_crosscorrelation_v2(ref_image, new_image)
        xcorr = None
    else:
        dx, dy, xcorr, score = shift_from_crosscorrelation(
            ref_image,
            new_image,
            lowpass=50,
            highpass=4,
            sigma=5,
            use_rect_mask=True,
        )
    return AlignmentIteration(
        shift=Point(dx, dy),
        score=score,
        image=new_image,
        xcorr=xcorr,
        method=method,
    )


def compare_alignment_methods(
    ref_image: FibsemImage,
    new_image: FibsemImage,
    agreement_threshold_px: float = 2.0,
) -> AlignmentDifferential:
    """Run all alignment methods on the same image pair and compare their shift estimates.

    Args:
        ref_image: Reference image.
        new_image: Image to align to the reference.
        agreement_threshold_px: Methods are considered in agreement when the
            maximum pairwise shift difference is below this value (in pixels).
            Defaults to 2.0.

    Returns:
        AlignmentDifferential containing per-method results, shifts in pixels,
        the maximum pairwise disagreement, and an agreement flag.
    """
    pixel_size = new_image.metadata.pixel_size.x if new_image.metadata else 1.0

    methods = list(AlignmentMethod)
    results = [_calculate_shift(ref_image, new_image, method) for method in methods]

    shifts_px = {
        method.value: Point(r.shift.x / pixel_size, r.shift.y / pixel_size)
        for method, r in zip(methods, results)
    }

    points = list(shifts_px.values())
    max_disagreement = 0.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            diff = np.hypot(points[i].x - points[j].x, points[i].y - points[j].y)
            max_disagreement = max(max_disagreement, diff)

    logging.debug(
        {
            "msg": "compare_alignment_methods",
            "shifts_px": {k: (v.x, v.y) for k, v in shifts_px.items()},
            "max_disagreement_px": max_disagreement,
            "agreement": max_disagreement < agreement_threshold_px,
        }
    )

    valid = [r for r in results if r.success]
    if not valid:
        valid = results  # fall back to all results if every method failed
    total_score = sum(r.score for r in valid) or 1.0
    consensus_shift = Point(
        x=sum(r.shift.x * r.score for r in valid) / total_score,
        y=sum(r.shift.y * r.score for r in valid) / total_score,
    )

    return AlignmentDifferential(
        results=results,
        shifts_px=shifts_px,
        max_disagreement_px=max_disagreement,
        agreement=max_disagreement < agreement_threshold_px,
        consensus_shift=consensus_shift,
    )


def _apply_shift(
    microscope: FibsemMicroscope,
    dx: float,
    dy: float,
    beam_type: BeamType,
    subsystem: AlignmentSubsystem = AlignmentSubsystem.BEAM_SHIFT,
):
    """Apply the calculated shift to the microscope subsystem."""
    if subsystem is AlignmentSubsystem.BEAM_SHIFT:
        microscope.beam_shift(-dx, dy, beam_type)
    elif subsystem is AlignmentSubsystem.STAGE:
        microscope.stable_move(
            dx=dx,
            dy=-dy,
            beam_type=beam_type,
        )
    elif subsystem is AlignmentSubsystem.STAGE_VERTICAL:
        if beam_type is BeamType.ELECTRON and hasattr(microscope, "move_coincident_from_sem"):
            microscope.move_coincident_from_sem(dx=dx, dy=-dy)  # type: ignore
            return
        microscope.vertical_move(dy=-dy, dx=dx)


def align_with_reference_image(
    microscope: FibsemMicroscope,
    ref_image: FibsemImage,
    use_autocontrast: bool = False,
    use_autofocus: bool = False,
    subsystem: AlignmentSubsystem = AlignmentSubsystem.BEAM_SHIFT,
    method: AlignmentMethod = DEFAULT_ALIGNMENT_METHOD,
) -> AlignmentIteration:
    """Align to a reference image. Delegates to beam_shift_alignment_v2."""
    return beam_shift_alignment_v2(
        microscope=microscope,
        ref_image=ref_image,
        use_autocontrast=use_autocontrast,
        use_autofocus=use_autofocus,
        subsystem=subsystem,
        method=method,
    )


def beam_shift_alignment_v2(
    microscope: FibsemMicroscope,
    ref_image: FibsemImage,
    use_autocontrast: bool = False,
    use_autofocus: bool = False,
    subsystem: AlignmentSubsystem = AlignmentSubsystem.BEAM_SHIFT,
    method: AlignmentMethod = DEFAULT_ALIGNMENT_METHOD,
):
    """Aligns the images by adjusting the beam shift instead of moving the stage.

    This method uses cross-correlation between the reference image and a new image to calculate the
    optimal beam shift for alignment. This approach offers increased precision, but a lower range
    compared to stage movement.

        Args:
        microscope (FibsemMicroscope): An OpenFIBSEM microscope client.
        ref_image (FibsemImage): The reference image to align to.
        use_autocontrast (bool): Whether to use autocontrast for the new image. Defaults to False.
        use_autofocus (bool): Whether to use autofocus before acquiring the new image. Defaults to False.
        subsystem (AlignmentSubsystem): The subsystem to use for alignment.
            BEAM_SHIFT applies correction via beam shift (default).
            STAGE moves the stage instead. STAGE_VERTICAL uses vertical stage movement.
        method (AlignmentMethod): Cross-correlation method to use. Defaults to DEFAULT_ALIGNMENT_METHOD.

    Raises:
        ValueError: If the reference image does not have a valid beam type.

    """
    if ref_image.metadata is None or ref_image.metadata.beam_type is None:
        raise ValueError("Reference image must have a valid beam type for alignment.")

    new_image = _acquire_from_reference_image(
        microscope=microscope,
        ref_image=ref_image,
        use_autocontrast=use_autocontrast,
        use_autofocus=use_autofocus,
    )

    result = _calculate_shift(ref_image, new_image, method)

    _apply_shift(
        microscope=microscope,
        dx=result.shift.x,
        dy=result.shift.y,
        beam_type=ref_image.metadata.beam_type,
        subsystem=subsystem,
    )

    logging.info(
        f"Beam Shift Alignment: dx: {result.shift.x}, dy: {result.shift.y}, score: {result.score}"
    )

    return result


def multi_step_alignment_v2(
    microscope: FibsemMicroscope,
    ref_image: FibsemImage,
    steps: int = 3,
    use_autocontrast: bool = False,
    use_autofocus: bool = False,
    subsystem: AlignmentSubsystem = AlignmentSubsystem.BEAM_SHIFT,
    stop_event: Optional[ThreadingEvent] = None,
    run_name: str = "AlignmentResult",
    acquire_final_image: bool = True,
    validate: bool = True,
    path: Optional[str] = None,
    method: AlignmentMethod = DEFAULT_ALIGNMENT_METHOD,
) -> AlignmentResult:
    """Runs the beam shift alignment multiple times."""

    alignment_results = []
    aborted = False
    for i in range(steps):
        if stop_event is not None and stop_event.is_set():
            aborted = True
            break
        # only use autocontrast on first step
        use_autocontrast = use_autocontrast if i == 0 else False
        use_autofocus = use_autofocus if i == 0 else False
        result = beam_shift_alignment_v2(
            microscope=microscope,
            ref_image=ref_image,
            use_autocontrast=use_autocontrast,
            use_autofocus=use_autofocus,
            subsystem=subsystem,
            method=method,
        )
        alignment_results.append(result)

    # Cancelled before any step completed: there is nothing meaningful to
    # persist. Skip the final-image acquisition (which would touch the
    # microscope after a stop request) and the save (which would otherwise
    # dump an empty AlignmentResult directory, into the CWD when no path is
    # set), and return an empty result.
    if aborted and not alignment_results:
        return AlignmentResult(
            name=run_name,
            reference_image=ref_image,
            subsystem=subsystem,
            method=method,
            results=[],
        )

    if validate:
        acquire_final_image = True

    final_image = None
    validation = None
    if acquire_final_image:
        final_image = _acquire_from_reference_image(
            microscope=microscope,
            ref_image=ref_image,
            use_autocontrast=False,
            use_autofocus=False,
        )
        if validate:
            validation = compare_alignment_methods(ref_image, final_image)
            if not validation.agreement:
                logging.warning(
                    f"Alignment validation failed: max disagreement "
                    f"{validation.max_disagreement_px:.2f}px across methods."
                )

    ts = utils.current_timestamp_v3(timeonly=True)
    run = AlignmentResult(
        name=f"{run_name}-{ts}",
        reference_image=ref_image,
        subsystem=subsystem,
        method=method,
        results=alignment_results,
        final_image=final_image,
        validation=validation,
    )

    save_path: str = path if path is not None else _alignment_save_path(ref_image)[0]
    run.save(save_path, plot_title=run_name)

    return run


def _eucentric_tilt_alignment(
    microscope: FibsemMicroscope,
    image_settings: ImageSettings,
    target_angle: float,
    step_size: float,
    beam_type: Optional[BeamType] = None,
    show: bool = False,
) -> None:
    """Perform eucentric tilt alignment by moving the stage in steps towards the target angle,
    acquiring images at each step, and performing alignment.
    Args:
        microscope (FibsemMicroscope): The microscope to use for alignment.
        image_settings (ImageSettings): The image settings to use for image acquisition.
        target_angle (float): The target tilt angle in degrees.
        step_size (float): The step size in degrees.
        beam_type (Optional[BeamType]): The beam type to use for image acquisition. If None, both beams are used.
        show (bool): Whether to show the images at each step. Defaults to False.
    Returns:
        None
    """
    import matplotlib.pyplot as plt

    from fibsem.structures import FibsemStagePosition

    stage_position = microscope.get_stage_position()
    current_angle = np.degrees(stage_position.t)

    n_steps = int(abs(int(current_angle) - target_angle) // step_size)

    logging.info(
        f"Current Tilt: {current_angle}, Target Tilt:  {target_angle}, Step Size: {step_size},  Num Steps: {n_steps}"
    )
    steps = np.linspace(current_angle, target_angle, num=n_steps)

    image_settings.hfw = 150e-6
    image_settings.save = False
    if beam_type is not None:
        image_settings.beam_type = beam_type
        reference_image = acquire.acquire_image(microscope, image_settings)
    else:
        ref_sem_image, ref_fib_image = acquire.acquire_channels(
            microscope, image_settings
        )

    fib_images = []
    sem_images = []

    for i, angle in enumerate(steps[1:]):
        microscope.move_stage_absolute(FibsemStagePosition(t=np.radians(angle)))

        if beam_type is not None:
            beam_shift_alignment_v2(
                microscope, reference_image, subsystem=AlignmentSubsystem.STAGE
            )
        else:
            beam_shift_alignment_v2(
                microscope, ref_sem_image, subsystem=AlignmentSubsystem.STAGE
            )
            beam_shift_alignment_v2(
                microscope, ref_fib_image, subsystem=AlignmentSubsystem.STAGE_VERTICAL
            )

        sem_image, fib_image = acquire.acquire_channels(microscope, image_settings)

        if show:
            fig, ax = plt.subplots(1, 2, figsize=(10, 7))
            ax[0].imshow(sem_image.data, cmap="gray")
            ax[0].plot(
                sem_image.data.shape[1] // 2, sem_image.data.shape[0] // 2, "y+", ms=50
            )
            ax[1].imshow(fib_image.data, cmap="gray")
            ax[1].plot(
                fib_image.data.shape[1] // 2, fib_image.data.shape[0] // 2, "y+", ms=50
            )
            plt.show()

        sem_images.append(sem_image)
        fib_images.append(fib_image)
        if beam_type is None:
            ref_sem_image = sem_image
            ref_fib_image = fib_image
        elif beam_type is BeamType.ELECTRON:
            reference_image = sem_image
        elif beam_type is BeamType.ION:
            reference_image = fib_image

    # TODO: have a metric to measure if it failed? how??
    final_position = microscope.get_stage_position()
    diff = stage_position - final_position
    logging.info(f"Start Position: {stage_position.pretty}")
    logging.info(f"Final Position: {final_position.pretty}")
    logging.info(f"Difference: {diff.pretty}")

    return sem_images, fib_images
