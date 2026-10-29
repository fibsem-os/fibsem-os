"""SEM to FM calibration for an offset FM (METEOR, iFLM): where the stage travels
between the beams and the objective, measured against the centre of slot 1.

Two parts, in the order the move that matters makes them: SEM to FIB is a half
turn at the beams, so the rotation centre comes first
(`StageSystemSettings.rotation_centre_correction`); FIB to FM is a travel, measured
as where the same point sits at the beams and under the objective
(`DeviceCalibration`). One site measures it when the FM images within 5° of the FIB
pose; more sites are its check. An FM further off needs three, spread in y. The stage then travels by what was measured, and
travel to the FM is refused until it is (`StageMovement.fm_calibration_needed`).

The steps are here, Qt-free; `fibsem.ui.widgets.sem_to_fm_dialog` walks the
operator through them. Every move goes through the stage movement service, so the
objective guard and the safe routes apply. The agent has no path to it.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fibsem import config as cfg
from fibsem import utils
from fibsem.structures.settings import BeamType, FibsemStagePosition
from fibsem.structures.system import BEAMS_STAGE_DEVICE, DEVICE_AXES, DeviceCalibration
from fibsem.util.timestamps import now, now_iso

# How far the FM's pose may differ from FIB's, in rotation and in tilt, and still be
# calibrated by re-posing at the beams; more is a configuration error. Degrees.
MAX_FM_POSE_DIFFERENCE = 15.0
# Within this of FIB's pose, in rotation and in tilt, one site calibrates the travel
# (unchecked until a second agrees); further off needs `SITES`. Degrees.
ONE_SITE_POSE_DIFFERENCE = 5.0
# With a tilt, the sites must spread at least this far in y for a travel that
# varies with position to show. Metres.
MIN_SITE_SPREAD_Y = 3.0e-3
# Sites whose travels differ by more than this disagree: the rotation centre is
# wrong, and the calibration is not offered for saving. Metres.
MAX_TRAVEL_SPREAD = 20.0e-6
SITES = 3


class CalibrationRefused(ValueError):
    """A step can't be taken where the stage is, or the result can't be saved."""


@dataclass
class Site:
    """One point, centred at the beams in the FM's pose and under the objective."""

    name: str
    beams: FibsemStagePosition
    fm: Optional[FibsemStagePosition] = None
    # After the round trip back to the beams: how far the point is from where it
    # was centred, as the stage move that would centre it again (x, y), metres.
    residual: Optional[Tuple[float, float]] = None
    coincidence: Dict[str, Any] = field(default_factory=dict)

    def travel(self) -> Optional[FibsemStagePosition]:
        """The stage move from this site at the beams to it under the objective."""
        if self.fm is None:
            return None
        return FibsemStagePosition(
            **{
                axis: getattr(self.fm, axis) - getattr(self.beams, axis)
                for axis in DEVICE_AXES
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        travel = self.travel()
        return {
            "name": self.name,
            "beams": _position_dict(self.beams),
            "fm": None if self.fm is None else _position_dict(self.fm),
            "travel": None if travel is None else _position_dict(travel),
            "residual": None if self.residual is None else list(self.residual),
            "coincidence": dict(self.coincidence),
        }


def _position_dict(position: FibsemStagePosition) -> Dict[str, float]:
    return {
        axis: float(getattr(position, axis))
        for axis in ("x", "y", "z", "r", "t")
        if getattr(position, axis) is not None
    }


class SemToFmCalibration:
    """The SEM to FM calibration's steps, run against one microscope.

    Nothing is written until `save`: the rotation centre applied in step 1 is the
    session's, and Cancel puts it back (`restore`).
    """

    def __init__(self, microscope: Any, configuration_path: Optional[str] = None):
        self.microscope = microscope
        self.movement = microscope.stage_movement
        self.configuration_path = configuration_path or getattr(
            microscope, "configuration_path", None
        )
        self.folder = os.path.join(
            cfg.LOG_PATH, "calibration", "sem-to-fm", now().strftime("%Y-%m-%d-%H%M%S")
        )
        stage = microscope.system.stage
        self._correction_before = tuple(stage.rotation_centre_correction)
        self.rotation_reference: Optional[FibsemStagePosition] = None
        self.rotation_centre: Optional[Tuple[float, float]] = None
        self.sites: List[Site] = []

    # -- where the stage is -------------------------------------------------------

    def fm_orientation(self) -> str:
        """The orientation the FM images from: the first its device names."""
        return self.microscope.system.stage.devices["FM"].available_orientations[0]

    def fm_pose_difference(self) -> Tuple[float, float]:
        """How far the FM's pose is from FIB's, (rotation, tilt) in degrees, refused
        when either is more than `MAX_FM_POSE_DIFFERENCE`."""
        fib = self.microscope.get_orientation("FIB")
        fm = self.microscope.get_orientation(self.fm_orientation())
        dr = float(np.degrees(np.arctan2(np.sin(fm.r - fib.r), np.cos(fm.r - fib.r))))
        dt = float(np.degrees(fm.t - fib.t))
        for what, difference in (("rotation", dr), ("tilt", dt)):
            if abs(difference) > MAX_FM_POSE_DIFFERENCE:
                raise CalibrationRefused(
                    f"The FM's pose is {difference:.1f}° from FIB's in {what}, more "
                    f"than {MAX_FM_POSE_DIFFERENCE:.0f}°: check the FM entry's pose."
                )
        return dr, dt

    def fm_tilt_difference(self) -> float:
        """How far the FM's pose is tilted from FIB's, in degrees."""
        return self.fm_pose_difference()[1]

    def sites_needed(self) -> int:
        """1 when the FM images within `ONE_SITE_POSE_DIFFERENCE` of FIB's pose, else
        `SITES`."""
        dr, dt = self.fm_pose_difference()
        return 1 if max(abs(dr), abs(dt)) < ONE_SITE_POSE_DIFFERENCE else SITES

    def _require_at_beams(self, orientation: Optional[str] = None) -> None:
        if self.movement.current_device() != BEAMS_STAGE_DEVICE:
            raise CalibrationRefused("Move the stage to the beams first.")
        if orientation is not None:
            now_at = self.movement.stage_orientation()
            if now_at != orientation:
                raise CalibrationRefused(
                    f"The stage is at {now_at}; this step is taken at {orientation}."
                )

    # -- 1. the rotation centre ---------------------------------------------------

    def capture_rotation_reference(self) -> FibsemStagePosition:
        """Where a mark is centred in SEM, before the half turn."""
        self._require_at_beams("SEM")
        self.rotation_reference = self.microscope.get_stage_position()
        return deepcopy(self.rotation_reference)

    def turn(self, orientation: str = "FIB") -> FibsemStagePosition:
        """Turn to *orientation* at the beams, keeping the point under the beam."""
        self._require_at_beams()
        target = self.movement.target_position(
            self.microscope.get_stage_position(),
            orientation,
            target_device=BEAMS_STAGE_DEVICE,
        )
        return self.movement.safe_absolute_stage_movement(target)

    def measure_rotation_centre(self, dx: float, dy: float) -> Tuple[float, float]:
        """The rotation centre, from where the mark is in an SEM image after the
        half turn: (*dx*, *dy*) metres from the image centre, +y up.

        Centred at p0, turned, and centred again at p1, the centre is their
        midpoint. p1 is where a stable move to the mark would take the stage."""
        if self.rotation_reference is None:
            raise CalibrationRefused("Capture the SEM reference first.")
        here = self.microscope.get_stage_position()
        centred = self.movement.project_stable_move(dx, dy, BeamType.ELECTRON, here)
        p0 = self.rotation_reference
        self.rotation_centre = (
            float((p0.x + centred.x) / 2),
            float((p0.y + centred.y) / 2),
        )
        return self.rotation_centre

    def rotation_centre_correction(self) -> Tuple[float, float]:
        """The correction that puts the half turn about the measured centre."""
        if self.rotation_centre is None:
            raise CalibrationRefused("Measure the rotation centre first.")
        current = self.microscope.rotation_centre or (0.0, 0.0)
        cx, cy = self.microscope.system.stage.rotation_centre_correction
        return (
            float(cx + self.rotation_centre[0] - current[0]),
            float(cy + self.rotation_centre[1] - current[1]),
        )

    def apply_rotation_centre(self) -> Tuple[float, float]:
        """Use the measured centre for this session's half turns."""
        correction = self.rotation_centre_correction()
        self.microscope.system.stage.rotation_centre_correction = correction
        return correction

    # -- 2 to 5. the sites --------------------------------------------------------

    def capture_beams(self) -> Site:
        """The next site at the beams, in the FM's pose, with the SEM/FIB
        coincidence measured for the operator to judge. The measurement is not a
        gate: a capture is never refused for it."""
        self._require_at_beams(self.fm_orientation())
        if len(self.sites) >= SITES:
            raise CalibrationRefused(f"{SITES} sites are captured.")
        site = Site(
            name=f"site-{len(self.sites) + 1}",
            beams=self.microscope.get_stage_position(),
        )
        site.coincidence = self.measure_coincidence(site.name)
        self.sites.append(site)
        return site

    def measure_coincidence(self, name: str) -> Dict[str, Any]:
        """The alignment service's coincidence check here, saved with the record:
        its dy, dz and score, or the reason it gave none."""
        alignment = getattr(self.microscope, "alignment", None)
        if alignment is None:
            return {}
        try:
            frame = alignment.check_coincidence(path=os.path.join(self.folder, name))
        except Exception as e:  # noqa: BLE001 - a label, never a gate
            logging.warning(f"Coincidence not measured: {e}")
            return {"error": str(e)}
        keep = ("dy", "dz", "dx", "score", "warnings", "sem_path", "fib_path")
        return {key: frame.metadata[key] for key in keep if key in frame.metadata}

    def travel_to_fm(self) -> None:
        """Retract, travel to the FM and insert: by the calibration there is, else
        the FM entry's nominal origin. The one travel the gate lets through."""
        self._require_at_beams(self.fm_orientation())
        with self.movement.calibrating():
            self.movement.move_to_device("FM")

    def capture_fm(self) -> Site:
        """The site under the objective, once the operator has centred it."""
        if not self.sites or self.sites[-1].fm is not None:
            raise CalibrationRefused("Capture the site at the beams first.")
        if self.movement.current_device() != "FM":
            raise CalibrationRefused("Travel to the FM first.")
        site = self.sites[-1]
        site.fm = self.microscope.get_stage_position()
        return site

    def return_to_beams(self) -> None:
        """Back to the beams in the FM's pose, by what this site measured, so the
        residual there is this site's error."""
        site = self.sites[-1] if self.sites else None
        if site is None or site.fm is None:
            raise CalibrationRefused("Capture the site under the objective first.")
        stage = self.microscope.system.stage
        saved = stage.device_calibration
        stage.device_calibration = DeviceCalibration(fibsem=site.beams, fm=site.fm)
        try:
            self.movement.move_to_device(
                BEAMS_STAGE_DEVICE, orientation=self.fm_orientation()
            )
        finally:
            stage.device_calibration = saved

    def record_residual(
        self, dx: float, dy: float, beam_type: BeamType = BeamType.ELECTRON
    ) -> Tuple[float, float]:
        """Where the mark is after the round trip, (*dx*, *dy*) metres from the image
        centre in *beam_type*, as the stage move that would centre it."""
        site = self.sites[-1] if self.sites else None
        if site is None or site.fm is None:
            raise CalibrationRefused("There is no round trip to measure.")
        here = self.microscope.get_stage_position()
        centred = self.movement.project_stable_move(dx, dy, beam_type, here)
        site.residual = (float(centred.x - here.x), float(centred.y - here.y))
        return site.residual

    def discard_site(self) -> None:
        """Drop the last site, to take it again."""
        if self.sites:
            self.sites.pop()

    # -- 6. check and save --------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """The sites' travels, their mean and spread, and whether it can be saved.
        ``checked`` is whether a second site agreed with the first."""
        done = [site for site in self.sites if site.fm is not None]
        travels = np.array(
            [[getattr(site.travel(), axis) for axis in DEVICE_AXES] for site in done]
        )
        problems: List[str] = []
        mean = spread = None
        try:
            needed = self.sites_needed()
        except CalibrationRefused as e:
            problems.append(str(e))
            needed = SITES
        if len(done) == 0:
            problems.append("No site is captured at both ends.")
        else:
            mean = travels.mean(axis=0)
        if len(done) > 1:
            spread = float(np.abs(travels - mean)[:, :2].max())
            if spread > MAX_TRAVEL_SPREAD:
                problems.append(
                    f"The travel differs by {spread * 1e6:.0f} µm between sites, more "
                    f"than {MAX_TRAVEL_SPREAD * 1e6:.0f} µm: the rotation centre is "
                    "likely wrong. Recalibrate it and capture the sites again."
                )
        if len(done) < needed:
            if done:
                problems.append(f"{len(done)} of {needed} sites are captured.")
        elif needed > 1:
            ys = [site.beams.y for site in done]
            if max(ys) - min(ys) < MIN_SITE_SPREAD_Y:
                problems.append(
                    f"The sites span {(max(ys) - min(ys)) * 1e3:.1f} mm in y; one must "
                    f"be at least {MIN_SITE_SPREAD_Y * 1e3:.0f} mm off."
                )
        return {
            "sites": [site.to_dict() for site in self.sites],
            "travel": None
            if mean is None
            else {axis: float(v) for axis, v in zip(DEVICE_AXES, mean)},
            "spread": spread,
            "checked": len(done) > 1,
            "problems": problems,
            "can_save": not problems,
        }

    def result(self) -> DeviceCalibration:
        """The calibration the sites give: slot 1 (the first site) at the beams, and
        under the objective by the sites' mean travel."""
        summary = self.summary()
        if not summary["can_save"]:
            raise CalibrationRefused(" ".join(summary["problems"]))
        beams = deepcopy(self.sites[0].beams)
        fm = deepcopy(self.sites[0].fm)
        for axis in DEVICE_AXES:
            setattr(fm, axis, getattr(beams, axis) + summary["travel"][axis])
        stage = self.microscope.system.stage
        return DeviceCalibration(
            fibsem=beams,
            fm=fm,
            holder=stage.active_holder or None,
            when=now_iso(),
            sites=summary["sites"],
            checked=summary["checked"],
        )

    def save(self) -> DeviceCalibration:
        """Write the calibration, and the rotation centre it was measured with, to
        the configuration, and use them from now on."""
        calibration = self.result()
        stage = self.microscope.system.stage
        if self.configuration_path:
            utils.write_device_calibration(
                self.configuration_path,
                calibration.to_dict(),
                tuple(stage.rotation_centre_correction),
            )
        stage.device_calibration = calibration
        self._correction_before = tuple(stage.rotation_centre_correction)
        logging.info(
            f"Saved the SEM to FM calibration: travel {calibration.translation()}"
            + ("." if calibration.checked else ", from one site, not cross-checked.")
        )
        return calibration

    def restore(self) -> None:
        """Put back the rotation centre this session started with (Cancel)."""
        self.microscope.system.stage.rotation_centre_correction = (
            self._correction_before
        )

    def check(self) -> Tuple[FibsemStagePosition, FibsemStagePosition]:
        """SEM to FM to SEM, by the saved calibration: where the stage started and
        where it came back to, for the residual to be read from the image."""
        start = self.microscope.get_stage_position()
        self.movement.move_to_device("FM")
        self.movement.move_to_device(BEAMS_STAGE_DEVICE, orientation="SEM")
        return start, self.microscope.get_stage_position()
