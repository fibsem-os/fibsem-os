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

# Objective insert and retract. Opaque operations -- insert()/retract() are single API
# calls into the vendor connection, with no distance or speed to reason from -- but they
# bracket themselves in the log ("Inserting objective..." / "Objective inserted."), so
# they are timeable after all.
# Insert:  n=4, 18.32 / 18.32 / 18.35 / 18.38 s -- the same figure from the UI worker
#          and from inside the fluorescence task, spread under 60 ms.
# Retract: n=2, 19.53 / 19.57 s.
# Retraction is consistently the slower of the two, so they are separate constants
# rather than one shared value.
OBJECTIVE_INSERT_S = 18.5
OBJECTIVE_RETRACT_S = 19.6

# Driving the objective to a focus position, after insertion. n=3, 4.06 / 4.17 / 4.17 s.
# Distance-independent as far as this sample shows, so a flat cost rather than a rate.
OBJECTIVE_FOCUS_MOVE_S = 4.3

# Changing the ion beam current. n=4: 20.2 s setting the milling current, and 20.2 /
# 20.1 / 20.1 s restoring the imaging current afterwards. A milling task pays this
# twice -- once into the milling current, once back out -- so roughly 40 s per task
# that milling's own estimate never sees.
BEAM_CURRENT_CHANGE_S = 20.5

# Drift-correction imaging. These are reduced-area frames and are far cheaper than the
# full-frame acquisitions IMAGE_OVERHEAD_S describes: measured 0.77-0.82 s each, against
# ~3.5 s for a full frame. n=16 across two milling stages.
ALIGNMENT_IMAGE_S = 0.85
# Autocontrast, once per alignment pass. n=3: 0.9 / 1.3 / 1.7 s.
ALIGNMENT_AUTOCONTRAST_S = 1.7

# Aligning to a stored reference image (`_align_reference_image`): one autocontrast plus
# a short burst of reduced-area frames. n=2, 4.5 / 4.6 s end to end. Charged whole rather
# than composed, because the number of frames is not configurable the way a milling
# alignment's `steps` is.
REFERENCE_ALIGNMENT_S = 4.8

# Per spot-burn point, on top of the exposure: blank, park the beam, unblank. n=21,
# median 0.69 / 0.70 s across two runs -- small, but it is paid once per coordinate.
SPOT_BURN_POINT_OVERHEAD_S = 0.75

# Still not counted, and deliberately not invented here: the pattern setup between the
# milling estimate and the first stroke (0 s on a rectangle, 15 s on a trench in the one
# run measured -- too variable to pick a number from), and the per-task setup either
# side of the operations above.


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


def alignment_cost(alignment: Optional[MillingAlignment], passes: int = 2) -> float:
    """Seconds for drift correction, over ``passes`` alignment passes.

    A pass is one autocontrast plus ``steps + 1`` reduced-area frames, matching the
    log: a stage configured for 3 steps acquired 4 images per pass, and ran two
    passes -- once at the imaging current, then again after switching to the milling
    current ("FIB Aligning at Milling Current"). Costing a single pass of ``steps``
    frames understated it by half.

    The frames are charged at :data:`ALIGNMENT_IMAGE_S` rather than through
    :func:`image_cost`, because they are reduced-area: measured ~0.8 s against ~3.5 s
    for the full frame the alignment's own ``ImageSettings`` describes.

    The interval re-alignment (``interval_enabled``) is not counted: how many times it
    fires depends on how long the milling runs, so the caller has to fold it in against
    its own milling estimate rather than have this guess.
    """
    if alignment is None or not alignment.enabled or passes <= 0:
        return 0.0
    per_pass = ALIGNMENT_AUTOCONTRAST_S + (alignment.steps + 1) * ALIGNMENT_IMAGE_S
    return passes * per_pass


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
    stages = getattr(config, "stages", None)
    total = milling_cost(stages)
    total += alignment_cost(getattr(config, "alignment", None))
    # Switching into the milling current and back out again. Charged once for the task
    # rather than per stage: the log shows one set and one restore bracketing the whole
    # milling run, not a pair per stage.
    if stages and any(stage.enabled for stage in stages):
        total += 2 * BEAM_CURRENT_CHANGE_S
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
