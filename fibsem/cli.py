"""fibsem-cli — command-line interface for fibsemOS microscope control."""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fibsem import acquire, utils
from fibsem.structures import (
    BeamSettings,
    BeamType,
    FibsemDetectorSettings,
    FibsemStagePosition,
    ImageSettings,
)

# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------


def _um_to_m(v: Optional[float]) -> Optional[float]:
    return v * 1e-6 if v is not None else None


def _deg_to_rad(v: Optional[float]) -> Optional[float]:
    return math.radians(v) if v is not None else None


def _m_to_mm(v: Optional[float]) -> Optional[float]:
    return v * 1e3 if v is not None else None


def _m_to_um(v: Optional[float]) -> Optional[float]:
    return v * 1e6 if v is not None else None


def _rad_to_deg(v: Optional[float]) -> Optional[float]:
    return math.degrees(v) if v is not None else None


def _parse_resolution(s: str) -> tuple:
    """Parse '1536x1024' into (1536, 1024). Used as argparse type=."""
    try:
        w, h = s.lower().split("x")
        return (int(w), int(h))
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(
            f"Resolution must be WxH, e.g. '1536x1024', got: {s!r}"
        )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_stage_position(pos: FibsemStagePosition) -> None:
    print("Stage Position")
    for axis, val, unit, is_angle in [
        ("x", pos.x, "mm", False),
        ("y", pos.y, "mm", False),
        ("z", pos.z, "mm", False),
        ("rotation", pos.r, "deg", True),
        ("tilt", pos.t, "deg", True),
    ]:
        if val is None:
            print(f"  {axis:<12}: None")
        elif is_angle:
            print(f"  {axis:<12}: {_rad_to_deg(val):8.2f} deg")
        else:
            mm = _m_to_mm(val)
            um = _m_to_um(val)
            print(f"  {axis:<12}: {mm:10.3f} mm  ({um:10.3f} um)")
    if pos.coordinate_system:
        print(f"  coordinate_system: {pos.coordinate_system}")


def _fmt_current(amps: Optional[float]) -> str:
    if amps is None:
        return "None"
    if abs(amps) < 1e-9:
        return f"{amps * 1e12:.2f} pA"
    if abs(amps) < 1e-6:
        return f"{amps * 1e9:.2f} nA"
    return f"{amps * 1e6:.2f} uA"


def _print_beam_settings(
    label: str,
    beam: BeamSettings,
    detector: FibsemDetectorSettings,
    is_on: Optional[bool],
    is_blanked: Optional[bool],
) -> None:
    print(f"{label} Beam")
    wd = (
        f"{beam.working_distance * 1e3:.3f} mm"
        if beam.working_distance is not None
        else "None"
    )
    hfw = f"{beam.hfw * 1e6:.2f} um" if beam.hfw is not None else "None"
    kv = f"{beam.voltage / 1e3:.2f} kV" if beam.voltage is not None else "None"
    dt = f"{beam.dwell_time * 1e6:.2f} us" if beam.dwell_time is not None else "None"
    res = (
        f"{beam.resolution[0]} x {beam.resolution[1]}"
        if beam.resolution is not None
        else "None"
    )
    rot = (
        f"{_rad_to_deg(beam.scan_rotation):.2f} deg"
        if beam.scan_rotation is not None
        else "None"
    )
    print(f"  voltage:           {kv}")
    print(f"  current:           {_fmt_current(beam.beam_current)}")
    print(f"  working_distance:  {wd}")
    print(f"  hfw:               {hfw}")
    print(f"  resolution:        {res}")
    print(f"  dwell_time:        {dt}")
    print(f"  scan_rotation:     {rot}")
    if beam.preset is not None:
        print(f"  preset:            {beam.preset}")
    print("  Detector")
    print(f"    type:       {detector.type}")
    print(f"    mode:       {detector.mode}")
    print(
        f"    contrast:   {detector.contrast:.2f}    brightness: {detector.brightness:.2f}"
    )
    on_str = "ON" if is_on else ("OFF" if is_on is not None else "unknown")
    blanked_str = (
        "yes" if is_blanked else ("no" if is_blanked is not None else "unknown")
    )
    print(f"  power:             {on_str}")
    print(f"  blanked:           {blanked_str}")


def _print_image_result(image, beam_type: BeamType) -> None:
    h, w = image.data.shape[:2]
    print(f"Acquired {beam_type.name} image")
    print(f"  shape:       {w} x {h}")
    md = image.metadata
    if md and md.image_settings:
        s = md.image_settings
        print(f"  hfw:         {s.hfw * 1e6:.2f} um")
        print(f"  dwell_time:  {s.dwell_time * 1e6:.2f} us")
        if s.save and s.path:
            suffix = "eb" if beam_type == BeamType.ELECTRON else "ib"
            saved = Path(s.path) / f"{s.filename}_{suffix}.tif"
            print(f"  saved:       {saved}")
        else:
            print("  saved:       False")


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_acquire(microscope, settings, args) -> int:
    res_w, res_h = args.resolution

    image_settings = ImageSettings(
        resolution=(res_w, res_h),
        dwell_time=args.dwell_time,
        hfw=args.hfw,
        autocontrast=args.autocontrast,
        save=args.save,
        filename=args.filename,
        path=args.path,
    )

    beams = []
    if args.beam in ("electron", "both"):
        beams.append(BeamType.ELECTRON)
    if args.beam in ("ion", "both"):
        beams.append(BeamType.ION)

    for beam_type in beams:
        image_settings.beam_type = beam_type
        image = acquire.new_image(microscope, image_settings)
        _print_image_result(image, beam_type)

    return 0


def cmd_move(microscope, settings, args) -> int:
    if args.mode == "absolute":
        current = microscope.get_stage_position()
        target = FibsemStagePosition(
            x=_um_to_m(args.x) if args.x is not None else current.x,
            y=_um_to_m(args.y) if args.y is not None else current.y,
            z=_um_to_m(args.z) if args.z is not None else current.z,
            r=_deg_to_rad(args.rotation) if args.rotation is not None else current.r,
            t=_deg_to_rad(args.tilt) if args.tilt is not None else current.t,
        )
        microscope.move_stage_absolute(target)
    else:
        delta = FibsemStagePosition(
            x=_um_to_m(args.x),
            y=_um_to_m(args.y),
            z=_um_to_m(args.z),
            r=_deg_to_rad(args.rotation),
            t=_deg_to_rad(args.tilt),
        )
        microscope.move_stage_relative(delta)

    pos = microscope.get_stage_position()
    print("Move complete.")
    _print_stage_position(pos)
    return 0


def cmd_position(microscope, settings, args) -> int:
    pos = microscope.get_stage_position()
    _print_stage_position(pos)
    return 0


def cmd_beam(microscope, settings, args) -> int:
    beams = []
    if args.beam in ("electron", "both"):
        beams.append(BeamType.ELECTRON)
    if args.beam in ("ion", "both"):
        beams.append(BeamType.ION)

    for beam_type in beams:
        name = beam_type.name
        if args.action == "on":
            microscope.turn_on(beam_type)
        elif args.action == "off":
            microscope.turn_off(beam_type)
        elif args.action == "blank":
            microscope.blank(beam_type)
        elif args.action == "unblank":
            microscope.unblank(beam_type)
        on = microscope.is_on(beam_type)
        blanked = microscope.is_blanked(beam_type)
        on_str = "ON" if on else "OFF"
        blanked_str = "blanked" if blanked else "unblanked"
        print(f"{name}: {on_str}, {blanked_str}")

    return 0


def cmd_autofocus(microscope, _settings, args) -> int:
    beam_type = BeamType.ELECTRON if args.beam == "electron" else BeamType.ION
    print(f"Running autofocus on {beam_type.name} beam...")
    microscope.auto_focus(beam_type)
    wd = microscope.get_working_distance(beam_type)
    print(f"Done. Working distance: {wd * 1e3:.4f} mm")
    return 0


def cmd_autocontrast(microscope, _settings, args) -> int:
    beams = []
    if args.beam in ("electron", "both"):
        beams.append(BeamType.ELECTRON)
    if args.beam in ("ion", "both"):
        beams.append(BeamType.ION)

    for beam_type in beams:
        print(f"Running autocontrast on {beam_type.name} beam...")
        microscope.autocontrast(beam_type)
        det = microscope.get_detector_settings(beam_type)
        print(f"Done. contrast: {det.contrast:.3f}  brightness: {det.brightness:.3f}")

    return 0


def cmd_mill_angle(microscope, _settings, args) -> int:
    milling_angle = _deg_to_rad(args.angle)
    rotation = _deg_to_rad(args.rotation)
    print(f"Moving to milling angle {args.angle:.2f} deg...")
    at_target = microscope.move_to_milling_angle(milling_angle, rotation)
    pos = microscope.get_stage_position()
    status = "at target" if at_target else "WARNING: not at target angle"
    print(f"Done ({status}).")
    _print_stage_position(pos)
    return 0


def cmd_set_beam(microscope, _settings, args) -> int:
    beam_type = BeamType.ELECTRON if args.beam == "electron" else BeamType.ION

    if args.voltage is not None:
        v = args.voltage * 1e3  # kV → V
        actual = microscope.set_beam_voltage(v, beam_type)
        print(f"voltage:  {actual / 1e3:.2f} kV")

    if args.current is not None:
        actual = microscope.set_beam_current(args.current, beam_type)
        print(f"current:  {_fmt_current(actual)}")

    if args.hfw is not None:
        hfw = _um_to_m(args.hfw)
        actual = microscope.set_field_of_view(hfw, beam_type)
        print(f"hfw:      {actual * 1e6:.2f} um")

    return 0


def cmd_info(microscope, _settings, _args) -> int:
    state = microscope.get_microscope_state()
    ts = datetime.fromtimestamp(state.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    print(f"Microscope State  [{ts}]")
    print()

    if state.stage_position is not None:
        _print_stage_position(state.stage_position)
        print()

    for beam_type, beam_settings, detector in [
        (BeamType.ELECTRON, state.electron_beam, state.electron_detector),
        (BeamType.ION, state.ion_beam, state.ion_detector),
    ]:
        if beam_settings is None:
            continue
        try:
            is_on = microscope.is_on(beam_type)
            is_blanked = microscope.is_blanked(beam_type)
        except Exception:
            is_on, is_blanked = None, None
        _print_beam_settings(beam_type.name, beam_settings, detector, is_on, is_blanked)
        print()

    return 0


def cmd_plugins(args) -> int:
    """Print what resolved under each entry point group.

    A plugin registered under a mistyped group name fails completely silently:
    nothing iterates unknown groups, so there is no error to log and the user
    sees only that their plugin never appeared. Printing the literal group
    strings is the whole diagnostic -- they can diff them against their own
    pyproject in the same terminal.

    Takes no microscope: see _add_plugins_parser.
    """
    from fibsem.plugins.report import collect_extensions, render_report

    # Loading the registries logs a full traceback per broken plugin. Those are
    # the very failures this command reports in one readable line each, and
    # dumping them above the report would bury it. Nothing is lost: --debug
    # brings the tracebacks back for when the one-line reason is not enough.
    #
    # Safe to do here only because importing fibsem.cli does not pull in any of
    # the three registries, so nothing has loaded yet at this point.
    if not args.debug:
        logging.disable(logging.CRITICAL)
    try:
        groups = collect_extensions()
    finally:
        logging.disable(logging.NOTSET)

    print(render_report(groups, show_builtins=args.show_builtins))
    return 0


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------


def _build_connection_parser(*, defaults: bool = True) -> argparse.ArgumentParser:
    """The connection arguments, shared by the root parser and every subcommand.

    Built twice, and the difference is load-bearing. argparse parses a subcommand
    into a fresh namespace and then copies every attribute of it over the root's,
    so an argument the subparser merely *defaulted* silently overwrites the same
    argument the root actually *parsed*. With one copy of this parser as a parent
    of both, every connection flag given before the subcommand reverted without a
    word: ``fibsem-cli --manufacturer Thermo info`` connected to the Demo
    simulator, and ``--config`` was ignored outright.

    So the root parser gets the copy carrying the real defaults -- that copy is
    what guarantees the attribute exists at all -- and the subparsers get the copy
    whose defaults are ``SUPPRESS``, which leaves each attribute absent unless
    the flag was actually passed. Both positions then work, and when a flag is
    given in both the one after the subcommand wins.
    """
    p = argparse.ArgumentParser(add_help=False)

    def default(value):
        """The declared default, or nothing at all in the subparsers' copy."""
        return value if defaults else argparse.SUPPRESS

    p.add_argument(
        "--manufacturer",
        default=default("Demo"),
        choices=["Demo", "Thermo", "Tescan", "Odemis"],
        help="Microscope manufacturer (default: Demo)",
    )
    p.add_argument(
        "--ip-address",
        default=default("localhost"),
        dest="ip_address",
        help="Microscope IP address (default: localhost)",
    )
    p.add_argument(
        "--config",
        default=default(None),
        dest="config_path",
        type=Path,
        metavar="PATH",
        help="Path to microscope config YAML",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        default=default(False),
        help="Enable debug logging",
    )
    return p


def _add_acquire_parser(sub, conn) -> None:
    p = sub.add_parser(
        "acquire", parents=[conn], help="Acquire one or both beam images"
    )
    p.add_argument(
        "--beam",
        choices=["electron", "ion", "both"],
        default="electron",
        help="Which beam(s) to image (default: electron)",
    )
    p.add_argument(
        "--hfw",
        type=float,
        default=150e-6,
        metavar="METRES",
        help="Horizontal field width in metres (default: 150e-6)",
    )
    p.add_argument(
        "--resolution",
        type=_parse_resolution,
        default="1536x1024",
        metavar="WxH",
        help="Image resolution as WxH (default: 1536x1024)",
    )
    p.add_argument(
        "--dwell-time",
        type=float,
        default=1e-6,
        dest="dwell_time",
        metavar="SECONDS",
        help="Pixel dwell time in seconds (default: 1e-6)",
    )
    p.add_argument(
        "--autocontrast",
        action="store_true",
        default=False,
        help="Apply automatic contrast enhancement",
    )
    p.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save image(s) to disk (requires --path)",
    )
    p.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Directory to save images (required if --save)",
    )
    p.add_argument(
        "--filename",
        type=str,
        default="image",
        help="Output filename stem (default: image)",
    )
    p.set_defaults(func=cmd_acquire)


def _add_move_parser(sub, conn) -> None:
    p = sub.add_parser(
        "move", parents=[conn], help="Move the stage (absolute or relative)"
    )
    p.add_argument(
        "--mode",
        choices=["absolute", "relative"],
        default="relative",
        help="Move mode (default: relative)",
    )
    p.add_argument(
        "--x",
        type=float,
        default=None,
        metavar="MICROMETRES",
        help="X position/delta in micrometres",
    )
    p.add_argument(
        "--y",
        type=float,
        default=None,
        metavar="MICROMETRES",
        help="Y position/delta in micrometres",
    )
    p.add_argument(
        "--z",
        type=float,
        default=None,
        metavar="MICROMETRES",
        help="Z position/delta in micrometres",
    )
    p.add_argument(
        "--rotation",
        type=float,
        default=None,
        metavar="DEGREES",
        help="Rotation (r) in degrees",
    )
    p.add_argument(
        "--tilt",
        type=float,
        default=None,
        metavar="DEGREES",
        help="Tilt (t) in degrees",
    )
    p.set_defaults(func=cmd_move)


def _add_position_parser(sub, conn) -> None:
    p = sub.add_parser("position", parents=[conn], help="Print current stage position")
    p.set_defaults(func=cmd_position)


def _add_beam_parser(sub, conn) -> None:
    p = sub.add_parser("beam", parents=[conn], help="Control beam power and blanking")
    p.add_argument(
        "action",
        choices=["on", "off", "blank", "unblank"],
        help="Action to perform",
    )
    p.add_argument(
        "--beam",
        choices=["electron", "ion", "both"],
        default="electron",
        help="Which beam(s) to control (default: electron)",
    )
    p.set_defaults(func=cmd_beam)


def _add_autofocus_parser(sub, conn) -> None:
    p = sub.add_parser("autofocus", parents=[conn], help="Run autofocus on a beam")
    p.add_argument(
        "--beam",
        choices=["electron", "ion"],
        default="electron",
        help="Which beam to autofocus (default: electron)",
    )
    p.set_defaults(func=cmd_autofocus)


def _add_autocontrast_parser(sub, conn) -> None:
    p = sub.add_parser(
        "autocontrast", parents=[conn], help="Run autocontrast on a beam"
    )
    p.add_argument(
        "--beam",
        choices=["electron", "ion", "both"],
        default="electron",
        help="Which beam(s) to run autocontrast on (default: electron)",
    )
    p.set_defaults(func=cmd_autocontrast)


def _add_info_parser(sub, conn) -> None:
    p = sub.add_parser("info", parents=[conn], help="Print full microscope state")
    p.set_defaults(func=cmd_info)


def _add_mill_angle_parser(sub, conn) -> None:
    p = sub.add_parser("mill-angle", parents=[conn], help="Move stage to milling angle")
    p.add_argument(
        "angle", type=float, metavar="DEGREES", help="Target milling angle in degrees"
    )
    p.add_argument(
        "--rotation",
        type=float,
        default=None,
        metavar="DEGREES",
        help="Target rotation in degrees (default: uses rotation_reference from config)",
    )
    p.set_defaults(func=cmd_mill_angle)


def _experiment_roots(args) -> "list[str]":
    """Where to look, given what the user asked for.

    Defaults to the experiment directory they have configured, falling back to
    the packaged default. Neither is authoritative -- experiments can be created
    anywhere -- which is why --root exists and why the command says how many
    places it actually looked.
    """
    from fibsem import config as cfg

    if args.root:
        return list(args.root)

    prefs = cfg.load_user_preferences()
    configured = prefs.experiment.default_experiment_directory
    return [configured or cfg.AUTOLAMELLA_LOG_PATH]


def _since_timestamp(args) -> Optional[float]:
    """Resolve --days / --since into one lower bound, or None."""
    if args.days is not None and args.since is not None:
        raise ValueError("--days and --since both set a start; use one or the other")
    if args.days is not None:
        return (datetime.now() - timedelta(days=args.days)).timestamp()
    if args.since is not None:
        return datetime.strptime(args.since, "%Y-%m-%d").timestamp()
    return None


def cmd_experiments(args) -> int:
    """List experiments, optionally narrowed to one instrument or time window.

    Answers "what did this microscope do this week?" for the roots it is given.
    It cannot answer it for the whole facility: nothing records where experiments
    are created, so this is only ever as complete as the places it looked. The
    summary line says how many that was, so an empty answer reads as "not here"
    rather than "never happened". See FIB-457.

    Takes no microscope: this is a question about files.
    """
    from fibsem.applications.autolamella.tools.experiments import (
        discover_experiments,
        filter_experiments,
        group_by_instrument,
        recent_experiments,
    )

    try:
        since = _since_timestamp(args)
    except ValueError as e:
        print(f"error: {e}")
        return 2

    roots = _experiment_roots(args)
    found = {}
    for root in roots:
        for experiment in discover_experiments(root):
            found[os.path.realpath(experiment.path)] = experiment
    if args.recent:
        for experiment in recent_experiments():
            found.setdefault(os.path.realpath(experiment.path), experiment)

    experiments = filter_experiments(
        sorted(found.values(), key=lambda e: e.created_at, reverse=True),
        instrument=args.instrument,
        operator=args.operator,
        since=since,
        until=(
            datetime.strptime(args.until, "%Y-%m-%d").timestamp()
            if args.until
            else None
        ),
    )

    if not experiments:
        print(f"No experiments found in {len(roots)} location(s): {', '.join(roots)}")
        return 0

    if args.group:
        for serial, group in group_by_instrument(experiments).items():
            model = next(
                (e.instrument_model for e in group if e.instrument_model), None
            )
            heading = f"{model} ({serial})" if serial else "No session recorded"
            print(f"\n{heading} — {len(group)} experiment(s)")
            _print_experiments(group)
    else:
        _print_experiments(experiments)

    print(f"\n{len(experiments)} experiment(s) in {len(roots)} location(s)")
    return 0


def _print_experiments(experiments) -> None:
    """One row each: when, what, where it ran, who ran it, how much was in it."""
    for e in experiments:
        when = (
            datetime.fromtimestamp(e.created_at).strftime("%Y-%m-%d %H:%M")
            if e.created_at
            else "unknown"
        )
        # An experiment written before the session record has no instrument and no
        # operator. Shown as "-" rather than blank so the column stays readable and
        # missing does not look like empty.
        instrument = e.instrument_model or "-"
        if e.instrument_serial:
            instrument = f"{instrument}/{e.instrument_serial}"
        note = "" if e.available else "  [unreadable]"
        print(
            f"  {when}  {e.name[:28]:28s} {instrument[:24]:24s} "
            f"{(e.operator or '-')[:16]:16s} {e.num_lamella:3d} lamella{note}"
        )
        print(f"      {os.path.dirname(e.path)}")


def _add_experiments_parser(sub) -> None:
    """Enumerate experiments on disk.

    Deliberately takes no connection arguments, for the same reason as `plugins`:
    this is a question about files, usually asked away from the instrument, and
    requiring a microscope to answer "what ran last week" would make it useless
    where it is most wanted.
    """
    p = sub.add_parser(
        "experiments",
        help="List experiments on disk, by instrument or time window",
    )
    p.add_argument(
        "--root",
        action="append",
        metavar="PATH",
        help="Directory to search; repeatable. Defaults to your configured "
        "experiment directory",
    )
    p.add_argument(
        "--recent",
        action="store_true",
        help="Also include experiments this machine has opened, which may live "
        "outside any searched root (last 10 only)",
    )
    p.add_argument(
        "--instrument",
        metavar="SERIAL|MODEL",
        help="Only experiments run on this instrument",
    )
    p.add_argument(
        "--operator", metavar="NAME", help="Only experiments run by this operator"
    )
    p.add_argument("--days", type=int, metavar="N", help="Only the last N days")
    p.add_argument("--since", metavar="YYYY-MM-DD", help="Only on or after this date")
    p.add_argument("--until", metavar="YYYY-MM-DD", help="Only on or before this date")
    p.add_argument(
        "--group", action="store_true", help="Group the listing by instrument"
    )
    p.set_defaults(func=cmd_experiments, needs_microscope=False)


def _add_plugins_parser(sub) -> None:
    """List what resolved under each entry point group.

    Deliberately takes no connection arguments: diagnosing a plugin is usually
    done away from the instrument, and requiring a microscope to answer "did my
    plugin load?" would make the command useless in exactly the situation it
    exists for. ``main()`` dispatches it before connecting.
    """
    p = sub.add_parser(
        "plugins",
        help="List registered patterns, strategies and tasks, and where they came from",
    )
    p.add_argument(
        "--all",
        action="store_true",
        dest="show_builtins",
        help="Include built-in extensions, which are counted but hidden by default",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        # Declared here because taking no connection arguments means `plugins
        # --debug` -- the position every other subcommand accepts, and the one a
        # user reaches for -- would otherwise be an unrecognized argument. The
        # root parser still supplies the default, hence SUPPRESS: same reason as
        # the subparsers' copy of the connection arguments, see
        # _build_connection_parser.
        default=argparse.SUPPRESS,
        help="Show the full traceback for each plugin that failed to load",
    )
    p.set_defaults(func=cmd_plugins, needs_microscope=False)


def _add_set_beam_parser(sub, conn) -> None:
    p = sub.add_parser(
        "set-beam", parents=[conn], help="Set beam voltage, current, or HFW"
    )
    p.add_argument(
        "--beam",
        choices=["electron", "ion"],
        default="electron",
        help="Which beam to configure (default: electron)",
    )
    p.add_argument(
        "--voltage", type=float, default=None, metavar="KV", help="Beam voltage in kV"
    )
    p.add_argument(
        "--current",
        type=float,
        default=None,
        metavar="AMPS",
        help="Beam current in amps (e.g. 100e-12 for 100 pA)",
    )
    p.add_argument(
        "--hfw",
        type=float,
        default=None,
        metavar="MICROMETRES",
        help="Horizontal field width in µm",
    )
    p.set_defaults(func=cmd_set_beam)


class _VersionAction(argparse.Action):
    """Print the version, revision and branch, then exit.

    A custom action rather than ``action="version"`` so the git lookup happens
    only when --version is actually passed, instead of on every CLI invocation.
    It exits during parsing, before the required-subcommand check.
    """

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        from fibsem.versioning import get_branch, get_version_string

        print(f"fibsemOS {get_version_string()}")
        branch = get_branch()
        if branch:
            print(f"branch: {branch}")
        parser.exit()


def _build_parser() -> argparse.ArgumentParser:
    # Two copies on purpose, not an oversight: the root parser holds the real
    # defaults, the subcommands hold suppressed ones so they cannot overwrite a
    # flag the root already parsed. See _build_connection_parser.
    conn = _build_connection_parser(defaults=False)
    root = argparse.ArgumentParser(
        prog="fibsem-cli",
        description="fibsemOS command-line interface for FIB/SEM microscope control",
        parents=[_build_connection_parser()],
    )
    root.add_argument(
        "--version",
        action=_VersionAction,
        help="Print the fibsemOS version and exit",
    )
    root.set_defaults(needs_microscope=True)
    sub = root.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")
    sub.required = True
    _add_acquire_parser(sub, conn)
    _add_move_parser(sub, conn)
    _add_position_parser(sub, conn)
    _add_beam_parser(sub, conn)
    _add_autofocus_parser(sub, conn)
    _add_autocontrast_parser(sub, conn)
    _add_info_parser(sub, conn)
    _add_mill_angle_parser(sub, conn)
    _add_set_beam_parser(sub, conn)
    _add_experiments_parser(sub)
    _add_plugins_parser(sub)
    return root


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.subcommand == "acquire" and args.save and args.path is None:
        parser.error("--save requires --path to be specified")

    if args.subcommand == "move":
        if all(v is None for v in [args.x, args.y, args.z, args.rotation, args.tilt]):
            parser.error(
                "move: at least one of --x, --y, --z, --rotation, --tilt must be specified"
            )

    # Subcommands that answer a question about this install rather than about a
    # microscope run before the connection attempt. `plugins` in particular has
    # to work with nothing plugged in -- it exists to explain why something is
    # missing, and demanding hardware first would make it useless.
    if not args.needs_microscope:
        try:
            sys.exit(args.func(args))
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        microscope, settings = utils.setup_session(
            manufacturer=args.manufacturer,
            ip_address=args.ip_address,
            config_path=args.config_path,
            debug=args.debug,
        )
    except Exception as e:
        print(f"error: failed to connect to microscope: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        exit_code = args.func(microscope, settings, args)
    except Exception as e:
        if args.debug:
            raise
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    else:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
