"""Bench check of ThermoFisher's Real Time Monitor (RTM), for FIB-1174.

Mills six short jobs with RTM on, one per question (see CASES), prints what each
returned, and saves the arrays. It mills at the current stage position: move to a
sacrificial area first.

    python scripts/rtm_bench.py
    python scripts/rtm_bench.py --cases baseline cleaning

Writes <out>/<timestamp>/summary.json and one .npz per case: `times`, then
`values_<pattern_id>` (one row per frame) and `positions_<pixels|dac>_<pattern_id>`.
"""

import argparse
import datetime
import json
import os
import time

import autoscript_sdb_microscope_client
import numpy as np
from autoscript_sdb_microscope_client.enumerations import RtmCoordinateSystem, RtmMode
from autoscript_sdb_microscope_client.structures import (
    GetRtmDataSettings,
    GetRtmPositionSettings,
)

from fibsem import utils
from fibsem.structures import CrossSectionPattern as CS
from fibsem.structures import FibsemMillingSettings, FibsemRectangleSettings

# name: (RTM mode, patterning mode, layout, wait_for_next_data)
CASES = {
    # pattern ids and point counts against what fibsem drew, point 0, the frame
    # interval, and whether polling returns the same pass again
    "baseline": ("low", "Serial", "two", False),
    "high_resolution": ("high", "Serial", "two", False),  # frame size and rate
    "parallel": ("low", "Parallel", "two", False),  # do both patterns report?
    "cross_section": ("low", "Serial", "cross_section", False),
    # AST-555: no data from a cleaning cross-section; does the rectangle still report?
    "cleaning": ("low", "Serial", "cleaning", False),
    # one read per pass? None once the job is over? Last, in case it hangs.
    "wait": ("low", "Serial", "two", True),
}

# width, height, centre x; the sizes differ so point counts tell the patterns apart
LEFT, RIGHT, CENTRE = (6e-6, 6e-6, -5e-6), (4e-6, 8e-6, 5e-6), (10e-6, 10e-6, 0)
LAYOUTS = {
    "two": [(LEFT, CS.Rectangle), (RIGHT, CS.Rectangle)],
    "cross_section": [(CENTRE, CS.RegularCrossSection)],
    "cleaning": [(LEFT, CS.CleaningCrossSection), (RIGHT, CS.Rectangle)],
}


def run_case(microscope, case, args):
    mode, patterning_mode, layout, wait = CASES[case]
    patterning = microscope.connection.patterning
    rtm = patterning.real_time_monitor

    microscope.setup_milling(
        FibsemMillingSettings(
            milling_current=args.current, hfw=args.hfw, patterning_mode=patterning_mode
        )
    )
    microscope.draw_patterns(
        [
            FibsemRectangleSettings(w, h, args.depth, x, 0, cross_section=cs)
            for (w, h, x), cs in LAYOUTS[layout]
        ]
    )
    result = {"drawn_ids": [p.id for p in microscope.milling._patterns]}

    rtm.mode = RtmMode.HIGH_RESOLUTION if mode == "high" else RtmMode.LOW_RESOLUTION
    settings = GetRtmDataSettings(None, wait)
    frames = []  # (seconds since start, {pattern_id: values})
    positions = {}  # "pixels" / "dac" -> {pattern_id: positions}
    rtm.start()
    t0 = time.monotonic()
    try:
        patterning.start()
        while patterning.state == "Idle" and time.monotonic() - t0 < 5:
            time.sleep(0.05)  # start() returns before the job is running
        while patterning.state != "Idle" and time.monotonic() - t0 < args.duration:
            data = rtm.get_data(settings)
            if data is None:
                result["none_during_job"] = True
                break
            if data:
                values = {d.pattern_id: np.asarray(d.values) for d in data}
                frames.append((time.monotonic() - t0, values))
                for name, cs in [
                    ("pixels", RtmCoordinateSystem.IMAGE_PIXELS),
                    ("dac", RtmCoordinateSystem.DAC),
                ]:
                    if name not in positions:
                        sets = rtm.get_positions(GetRtmPositionSettings(None, cs))
                        positions[name] = {
                            s.pattern_id: np.asarray(s.positions) for s in sets
                        }
            if not wait:
                time.sleep(args.poll)
        if patterning.state != "Idle":
            patterning.stop()
        if wait:  # None once the job is over? If this hangs, that's the answer.
            after_job = rtm.get_data(settings)
            result["wait_after_job"] = None if after_job is None else len(after_job)
    finally:
        if patterning.state != "Idle":
            patterning.stop()
        rtm.stop()
    after_stop = rtm.get_data(GetRtmDataSettings(None, False))
    result["readable_after_stop"] = len(after_stop or [])
    microscope.clear_patterns()

    times = [t for t, _ in frames]
    gaps = np.diff(times)
    pixels = positions.get("pixels", {})
    result.update(
        frames=len(frames),
        first_frame_s=times[0] if times else None,
        interval_median_s=float(np.median(gaps)) if gaps.size else None,
        interval_max_s=float(gaps.max()) if gaps.size else None,
        repeats=sum(
            a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)
            for (_, a), (_, b) in zip(frames, frames[1:])
        ),
        data_ids=sorted({k for _, f in frames for k in f}),
        position_ids={name: sorted(p) for name, p in positions.items()},
        patterns={
            pid: {
                "values": int(v.size),
                "dtype": str(v.dtype),
                "bytes": int(v.nbytes),
                "range": [float(v.min()), float(v.max())] if v.size else None,
                "positions": len(pixels.get(pid, [])),
                "points_0_1": pixels[pid][:2].tolist() if pid in pixels else None,
                "values_0_1": v[:2].tolist(),
            }
            for pid, v in (frames[-1][1] if frames else {}).items()
        },
    )
    return frames, positions, result


def save(path, frames, positions):
    arrays = {
        f"positions_{name}_{pid}": p
        for name, sets in positions.items()
        for pid, p in sets.items()
    }
    for pid in {k for _, f in frames for k in f}:
        rows = [f[pid] for _, f in frames if pid in f]
        if len({r.size for r in rows}) == 1:
            arrays[f"values_{pid}"] = np.stack(rows)
    np.savez_compressed(path, times=np.array([t for t, _ in frames]), **arrays)


def main():
    parser = argparse.ArgumentParser(description="Bench check of the RTM (FIB-1174)")
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--duration", type=float, default=20.0, help="s per case")
    parser.add_argument("--depth", type=float, default=0.2e-6, help="m")
    parser.add_argument("--current", type=float, default=100e-12, help="A")
    parser.add_argument("--hfw", type=float, default=80e-6, help="m")
    parser.add_argument("--poll", type=float, default=0.1, help="s between polls")
    parser.add_argument("--config", default=None, help="microscope configuration")
    parser.add_argument("--out", default="rtm-bench")
    parser.add_argument("--yes", action="store_true", help="don't ask before milling")
    args = parser.parse_args()

    print(f"Cases: {', '.join(args.cases)}; up to {args.duration:.0f} s each.")
    if not args.yes:
        input("This mills at the current stage position. Enter to start. ")

    microscope, _ = utils.setup_session(config_path=args.config)
    out = os.path.join(args.out, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(out, exist_ok=True)
    version = autoscript_sdb_microscope_client.build_information.INFO_VERSIONSHORT
    summary = {"autoscript": version, "args": vars(args), "cases": {}}
    try:
        for case in args.cases:
            try:
                frames, positions, result = run_case(microscope, case, args)
                save(os.path.join(out, f"{case}.npz"), frames, positions)
            except Exception as e:
                result = {"error": repr(e)}
            summary["cases"][case] = result
            print(f"{case}: {json.dumps(result, default=str)}")
            with open(os.path.join(out, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2, default=str)  # after every case
    finally:
        microscope.finish_milling()
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
