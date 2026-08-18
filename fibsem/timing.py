"""Forward estimates of how long microscope operations take.

The single home for the measured constants behind every duration the app quotes.
Estimates are computed from the configuration about to run — nothing here reads
past runs, because the parameters differ between runs and a median over mixed
configurations would be confidently wrong.

Where a value could go either way it rounds **up**. An estimate that runs long is
a mild annoyance; one that runs short means an unattended run is not finished when
the user comes back for it. The constants below are the p90 of their measurement,
not the median, for that reason.

Provenance
----------
Measured against ``AutoLamella-2026-07-29-18-00-DEV-TEST`` (Thermo), by pairing the
``INFO``/``DEBUG`` log entries each operation emits. One instrument, small n: treat
these as starting values to be re-measured, not as settled. Each constant carries
its own sample count and spread below.

See FIB-666.
"""

import logging
from typing import TYPE_CHECKING, Iterable, Optional

from fibsem.structures import ImageSettings, MillingAlignment, ReferenceImageParameters

if TYPE_CHECKING:
    from fibsem.milling.base import FibsemMillingStage

# --- Measured constants ------------------------------------------------------

# Fixed cost of one acquisition on top of the scan itself: beam settle, frame grab,
# processing, save. n=78, median 1.52 s, p90 1.93 s, max 1.98 s. Flat across
# resolution, dwell time, beam and autocontrast, so it is a fixed cost per image
# rather than a factor on the scan time -- at 1536x1024 and 1 us dwell it roughly
# doubles the arithmetic, which is why an estimate built from scan time alone came
# out 30-40x short on imaging-dominated tasks.
IMAGE_OVERHEAD_S = 2.0

# Stage moves. Absolute: n=19, median 7.41 s, p90 7.76 s, max 8.40 s.
# Relative: n=31, median 3.37 s, p90 6.16 s, max 7.60 s -- a wider spread, since a
# relative move covers anything from a nudge to a full traverse.
# These replace fm/timing.py's unmeasured DEFAULT_STAGE_MOVE_TIME = 5.0.
STAGE_MOVE_ABSOLUTE_S = 8.0
STAGE_MOVE_RELATIVE_S = 6.5

# Not measured yet, and deliberately not invented here: autofocus, beam/current
# switching, and the per-task setup either side of the operations above. Tasks that
# need them should say so rather than have this module guess -- fm/timing.py's
# DEFAULT_AUTOFOCUS_TIME = 5.0 is an assumption of exactly that kind.


def image_cost(settings: Optional[ImageSettings], count: int = 1) -> float:
    """Seconds to acquire ``count`` images with these settings.

    Scan time is arithmetic (``pixels x dwell x integration``); the rest is the
    fixed per-acquisition overhead, which dominates at typical settings.
    """
    if settings is None or count <= 0:
        return 0.0
    return count * (settings.estimated_time + IMAGE_OVERHEAD_S)


def reference_image_cost(params: Optional[ReferenceImageParameters]) -> float:
    """Seconds to acquire one set of reference images.

    One image per selected field of view per selected beam, which is what
    ``ReferenceImageParameters.estimated_time`` counts -- this adds the
    per-acquisition overhead that property omits.
    """
    if params is None:
        return 0.0
    n_fovs = sum([params.acquire_image1, params.acquire_image2])
    n_beams = sum([params.acquire_sem, params.acquire_fib])
    return image_cost(params.imaging, n_fovs * n_beams)


def stage_move_cost(count: int = 1, absolute: bool = True) -> float:
    """Seconds for ``count`` stage moves."""
    if count <= 0:
        return 0.0
    return count * (STAGE_MOVE_ABSOLUTE_S if absolute else STAGE_MOVE_RELATIVE_S)


def alignment_cost(alignment: Optional[MillingAlignment]) -> float:
    """Seconds for one drift-correction pass.

    ``steps`` images, and nothing when alignment is switched off. The interval
    re-alignment (``interval_enabled``) is not counted: how many times it fires
    depends on how long the milling runs, so the caller has to fold it in against
    its own milling estimate rather than have this guess.
    """
    if alignment is None or not alignment.enabled:
        return 0.0
    return image_cost(alignment.imaging, alignment.steps)


def milling_cost(stages: Optional[Iterable["FibsemMillingStage"]]) -> float:
    """Seconds to mill these stages, counting only the enabled ones.

    Uses each stage's own ``estimated_time`` -- the local formula in
    ``milling/base.py``, not ``microscope.estimate_milling_time()``. The two
    disagree: on the measured run they matched on a plain rectangle (26.7 s vs
    27.3 s) but diverged by more than 2x on a trench (236.5 s vs 102.4 s), with
    the actual 146.2 s falling between them. The local formula is used here
    because it needs no microscope connection, so a pre-run dialog can quote a
    time before anything is set up, and because its bias on trenches is the
    conservative direction.

    No correction factor is applied yet. Two stages is not enough to calibrate
    one, and a wrong factor is worse than none.
    """
    if stages is None:
        return 0.0
    return sum(stage.estimated_time for stage in stages if stage.enabled)


def milling_task_cost(config) -> float:
    """Seconds for one ``FibsemMillingTaskConfig``: milling, alignment, acquisition.

    Takes the config structurally rather than by import, so this module does not
    depend on the milling package.
    """
    if config is None:
        return 0.0
    total = milling_cost(getattr(config, "stages", None))
    total += alignment_cost(getattr(config, "alignment", None))
    acquisition = getattr(config, "acquisition", None)
    if acquisition is not None and getattr(acquisition, "enabled", False):
        n_beams = sum([acquisition.acquire_sem, acquisition.acquire_fib])
        total += image_cost(acquisition.imaging, n_beams)
    elif acquisition is not None and getattr(acquisition, "acquire_final_image", False):
        # the post-task refresh: one FIB image, only when the task acquired none itself
        total += image_cost(acquisition.imaging, 1)
    return total


def log_estimate(label: str, seconds: float) -> None:
    """Debug-log an estimate in the same shape everywhere, for later comparison
    against the durations the experiment record already stores."""
    logging.debug({"msg": "duration_estimate", "label": label, "seconds": seconds})
