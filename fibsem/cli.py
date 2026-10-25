"""fibsem-cli: the microscope from a terminal, through the App Server.

Every microscope command is a call on the App Server, as the operator: the routes
the agent's tools take, with the server's checks, blocklist and record. The
server is the one the flags name, else the one running on this machine.
``experiment list-all`` asks the app the server hosts; ``plugins`` and ``--version``
describe this install and need no server.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from fibsem.structures.image import FibsemImage
from fibsem.structures.settings import (
    BeamType,
    FibsemStagePosition,
    ImageSettings,
    MicroscopeState,
    Resolution,
)

BEAMS = {"electron": BeamType.ELECTRON, "ion": BeamType.ION}

NO_SERVER = (
    "no fibsem host found. Open the app with Tools > Host serving, or run "
    "fibsem-autolamella-server; or give --url and --token (the operator token: "
    "Tools > Host > Copy operator token)."
)
MICROSCOPE_OFF = (
    "the host doesn't serve the microscope: turn on its microscope setting "
    "in Tools > Host."
)
STOP_REASON = "stopped from fibsem-cli"

# ---------------------------------------------------------------------------
# Units and output
# ---------------------------------------------------------------------------


def _micro(v: Optional[float]) -> Optional[float]:
    return v * 1e-6 if v is not None else None


def _rad(v: Optional[float]) -> Optional[float]:
    return math.radians(v) if v is not None else None


def _parse_resolution(s: str) -> tuple:
    """Parse '1536x1024' into (1536, 1024). Used as argparse type=."""
    try:
        return tuple(Resolution.parse(s))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Resolution must be WxH, e.g. '1536x1024', got: {s!r}"
        )


def _print_stage_position(pos: FibsemStagePosition) -> None:
    print("Stage Position")
    for axis, val, is_angle in [
        ("x", pos.x, False),
        ("y", pos.y, False),
        ("z", pos.z, False),
        ("r", pos.r, True),
        ("t", pos.t, True),
    ]:
        if val is None:
            print(f"  {axis}: -")
        elif is_angle:
            print(f"  {axis}: {math.degrees(val):.4f} deg")
        else:
            print(f"  {axis}: {val * 1e6:.2f} um")


def _fmt_current(amps: Optional[float]) -> str:
    if amps is None:
        return "-"
    for unit, scale in (("nA", 1e-9), ("pA", 1e-12)):
        if abs(amps) >= scale:
            return f"{amps / scale:.2f} {unit}"
    return f"{amps:.3e} A"


def _print_beam(name: str, state: MicroscopeState, beam: str, on, blanked) -> None:
    settings = getattr(state, f"{beam}_beam")
    detector = getattr(state, f"{beam}_detector")
    if settings is None:
        return
    print(name)
    if settings.voltage is not None:
        print(f"  voltage:    {settings.voltage / 1e3:.2f} kV")
    print(f"  current:    {_fmt_current(settings.beam_current)}")
    if settings.hfw is not None:
        print(f"  hfw:        {settings.hfw * 1e6:.2f} um")
    if settings.working_distance is not None:
        print(f"  wd:         {settings.working_distance * 1e3:.4f} mm")
    if detector is not None:
        print(f"  detector:   {detector.type} / {detector.mode}")
        print(f"  contrast:   {detector.contrast:.3f}")
        print(f"  brightness: {detector.brightness:.3f}")
    print(f"  on:         {_yes_no(on)}")
    print(f"  blanked:    {_yes_no(blanked)}")


def _yes_no(value: Optional[bool]) -> str:
    return "unknown" if value is None else ("yes" if value else "no")


def _print_image(image: FibsemImage, label: str) -> None:
    h, w = image.data.shape[:2]
    print(f"Acquired {label} image")
    print(f"  shape:       {w} x {h}")
    s = image.metadata.image_settings if image.metadata else None
    if s is not None:
        print(f"  hfw:         {s.hfw * 1e6:.2f} um")
        print(f"  dwell_time:  {s.dwell_time * 1e6:.2f} us")


def _show(result: Any) -> None:
    """A command's result, as readably as its type allows."""
    from fibsem.devices.wire import to_wire

    if result is None:
        print("done")
    elif isinstance(result, FibsemStagePosition):
        _print_stage_position(result)
    elif isinstance(result, FibsemImage):
        _print_image(result, "an")
    else:
        try:
            print(json.dumps(to_wire(result), indent=2, default=str))
        except Exception:
            print(result)


def _parse_value(text: str) -> Any:
    """A value given on the command line: JSON when it reads as JSON
    (``1e-9``, ``true``, ``[1536, 1024]``), else the text itself."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def _parse_kwargs(text: Optional[str]) -> Dict[str, Any]:
    """``--kwargs``: a JSON object, or ``@file`` with one (``.json``, ``.yaml``)."""
    if text is None:
        return {}
    if text.startswith("@"):
        path = Path(text[1:])
        content = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            import yaml

            kwargs = yaml.safe_load(content)
        else:
            kwargs = json.loads(content)
    else:
        kwargs = json.loads(text)
    if not isinstance(kwargs, dict):
        raise ValueError("--kwargs is an object of argument names to values")
    return kwargs


# ---------------------------------------------------------------------------
# Connecting
# ---------------------------------------------------------------------------


def _address(args) -> tuple:
    """(url, token): the App Server's, with the operator's token."""
    from fibsem.client.transport import resolve_operator

    resolved = resolve_operator(args.url, args.token)
    if resolved is None:
        raise _CliError(NO_SERVER)
    return resolved


def _client(args):
    from fibsem import __version__
    from fibsem.client import FibsemClient

    url, token = _address(args)
    return FibsemClient(url, token, user_agent=f"fibsem-cli/{__version__}")


class _CliError(RuntimeError):
    """A refusal this command words itself."""


# ---------------------------------------------------------------------------
# Microscope commands: each one route the agent's tools take
# ---------------------------------------------------------------------------


def _beams(beam_type: str):
    """(device name, BeamType) for ``--beam-type``: ELECTRON, ION or BOTH."""
    return [(n, b) for n, b in BEAMS.items() if beam_type in (b.name, "BOTH")]


def cmd_microscope_get_info(client, args) -> int:
    info = client.info()
    print(
        f"{info.get('manufacturer')} {info.get('model')}  "
        f"serial {info.get('serial_number')}  software {info.get('software_version')}"
    )
    return 0


def cmd_microscope_get_state(client, args) -> int:
    state = client.state()
    ts = datetime.fromtimestamp(state.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    print(f"Microscope State  [{ts}]")
    print()
    if state.stage_position is not None:
        _print_stage_position(state.stage_position)
        print()
    for name, beam_type in BEAMS.items():
        if name not in client.devices:
            continue
        beam = client.devices[name]
        on = beam.get("on") if "on" in beam.parameters else None
        blanked = beam.get("blanked") if "blanked" in beam.parameters else None
        _print_beam(beam_type.name, state, name, on, blanked)
        print()
    return 0


def cmd_stage_get_position(client, args) -> int:
    _print_stage_position(client.devices["stage"].get("position"))
    return 0


def _axes(args) -> Dict[str, Optional[float]]:
    return {
        "x": _micro(args.x),
        "y": _micro(args.y),
        "z": _micro(args.z),
        "r": _rad(args.r),
        "t": _rad(args.t),
    }


def cmd_stage_move_absolute(client, args) -> int:
    """The safe move (`StageMovement.safe_absolute_stage_movement`): flat for a
    large rotation, rotation first, the traverse to another device. An axis
    left out stays where it is."""
    current = client.devices["stage"].get("position")
    target = FibsemStagePosition(
        **{
            axis: getattr(current, axis) if value is None else value
            for axis, value in _axes(args).items()
        }
    )
    movement = client.services["stage_movement"]
    _print_stage_position(movement.safe_absolute_stage_movement(stage_position=target))
    return 0


def cmd_stage_move_relative(client, args) -> int:
    movement = client.services["stage_movement"]
    _print_stage_position(
        movement.move_relative(delta=FibsemStagePosition(**_axes(args)))
    )
    return 0


def cmd_stage_move_to_orientation(client, args) -> int:
    movement = client.services["stage_movement"]
    _print_stage_position(movement.move_to_orientation(orientation=args.orientation))
    return 0


def cmd_stage_move_to_device(client, args) -> int:
    client.services["stage_movement"].move_to_device(device=args.device)
    _print_stage_position(client.devices["stage"].get("position"))
    return 0


def cmd_stage_move_to_milling_angle(client, args) -> int:
    print(f"Moving to milling angle {args.milling_angle:.2f} deg...")
    at_target = client.services["stage_movement"].move_to_milling_angle(
        milling_angle=args.milling_angle, rotation=args.rotation
    )
    print(f"Done ({'at target' if at_target else 'WARNING: not at target angle'}).")
    _print_stage_position(client.devices["stage"].get("position"))
    return 0


def cmd_beam_acquire_image(client, args) -> int:
    from fibsem.services.imaging import image_save_path

    imaging = client.services["imaging"]
    for _, beam_type in _beams(args.beam_type):
        settings = ImageSettings(
            resolution=args.resolution,
            dwell_time=_micro(args.dwell_time),
            hfw=_micro(args.hfw),
            autocontrast=args.autocontrast,
            beam_type=beam_type,
        )
        image = imaging.acquire(beam_type=beam_type, image_settings=settings)
        _print_image(image, beam_type.name)
        if args.out is not None:
            settings.path, settings.filename = str(args.out), args.filename
            print(f"  saved:       {image.save(path=image_save_path(settings))}")
    return 0


def cmd_beam_auto_focus(client, args) -> int:
    name, beam_type = _beams(args.beam_type)[0]
    print(f"Running autofocus on {beam_type.name} beam...")
    client.services["imaging"].auto_focus(beam_type=beam_type)
    wd = client.devices[name].get("working_distance")
    print(f"Done. Working distance: {wd * 1e3:.4f} mm")
    return 0


def cmd_beam_auto_contrast(client, args) -> int:
    for name, beam_type in _beams(args.beam_type):
        print(f"Running autocontrast on {beam_type.name} beam...")
        client.services["imaging"].autocontrast(beam_type=beam_type)
        detector = getattr(client.state(), f"{name}_detector")
        print(
            f"Done. contrast: {detector.contrast:.3f}  "
            f"brightness: {detector.brightness:.3f}"
        )
    return 0


def cmd_beam_blank(client, args) -> int:
    """The beam's own blank or unblank command (``args.command``)."""
    for name, beam_type in _beams(args.beam_type):
        beam = client.devices[name]
        beam.call(args.command)
        print(f"{beam_type.name}: {'blanked' if beam.get('blanked') else 'unblanked'}")
    return 0


def cmd_beam_turn(client, args) -> int:
    """The beam's own turn_on or turn_off command (``args.command``)."""
    for name, beam_type in _beams(args.beam_type):
        on = client.devices[name].call(args.command)
        print(f"{beam_type.name}: {'on' if on else 'off'}")
    return 0


def cmd_beam_set_parameters(client, args) -> int:
    """Each write is the server's: checked against the beam's limits and
    choices, a read-only one refused."""
    beam = client.devices[_beams(args.beam_type)[0][0]]
    for parameter, value, show in [
        (
            "voltage",
            None if args.voltage is None else args.voltage * 1e3,
            lambda v: f"{v / 1e3:.2f} kV",
        ),
        ("current", args.current, _fmt_current),
        ("hfw", _micro(args.hfw), lambda v: f"{v * 1e6:.2f} um"),
    ]:
        if value is None:
            continue
        written = beam.set(parameter, value)
        clipped = "" if math.isclose(written, value) else "  (clipped to its limits)"
        print(f"{parameter + ':':9s} {show(written)}{clipped}")
    return 0


def cmd_hardware_check(client, args) -> int:
    """Run the hardware check on the server's devices and print its report.
    Exits 1 if a step failed."""
    from fibsem.client.remote_microscope import RemoteMicroscope
    from fibsem.tools.hardware_check import Level, run_hardware_check

    on_frame = None
    if args.save_frame is not None:

        def on_frame(image) -> None:
            image.save(str(args.save_frame))
            print(f"frame saved: {args.save_frame}")

    microscope = RemoteMicroscope.connect(*_address(args))
    try:
        report = run_hardware_check(microscope, Level(args.level), on_frame=on_frame)
    finally:
        microscope.disconnect()
    print(report.format())
    if args.json is not None:
        args.json.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"report saved: {args.json}")
    return 0 if report.passed else 1


# --- the server's protocol, generically: devices and services by name ----------


def _remotes(client, args):
    return client.devices if args.noun == "device" else client.services


def cmd_remote_list(client, args) -> int:
    for name, remote in _remotes(client, args).items():
        print(f"  {name:20s} {remote.description.get('class', '')}")
    return 0


def cmd_remote_describe(client, args) -> int:
    remote = _remotes(client, args)[args.name]
    print(f"{args.name} ({remote.description.get('class', '')})")
    print("parameters:")
    for parameter, value in remote.values().items():
        settable = remote.parameters.get(parameter, {}).get("settable", True)
        print(f"  {parameter:24s} {value!r}{'' if settable else '  (read-only)'}")
    for kind, names in (("commands", remote.commands), ("queries", remote.queries)):
        print(f"{kind}: {', '.join(sorted(names)) or '-'}")
    return 0


def cmd_remote_get_parameter(client, args) -> int:
    _show(_remotes(client, args)[args.name].get(args.parameter))
    return 0


def cmd_remote_set_parameter(client, args) -> int:
    remote = _remotes(client, args)[args.name]
    _show(remote.set(args.parameter, _parse_value(args.value)))
    return 0


def cmd_remote_call_query(client, args) -> int:
    remote = _remotes(client, args)[args.name]
    _show(remote.ask(args.query, **_parse_kwargs(args.kwargs)))
    return 0


def cmd_remote_call_command(client, args) -> int:
    """A command; a long one runs on the server, and is waited for here. Ctrl-C
    stops it with the service's own stop."""
    from fibsem.client import Run

    remote = _remotes(client, args)[args.name]
    result = remote.call(args.command, **_parse_kwargs(args.kwargs))
    if isinstance(result, Run):
        print(f"{result}: running (Ctrl-C stops it)")
        try:
            while result.refresh(wait=5.0) == "running":
                pass
        except KeyboardInterrupt:
            if "stop" not in remote.commands:
                raise
            remote.call("stop", reason=STOP_REASON)
            print(f"{result}: stop sent")
            result.refresh(wait=30.0)
        print(f"{result}: {result.state}")
        result = result.wait()
    _show(result)
    return 0


# --- the app's experiments ---------------------------------------------------


def _since(args) -> Optional[str]:
    """--days / --since as one lower bound (an ISO time), or None."""
    if args.days is not None and args.since is not None:
        raise ValueError("--days and --since both set a start; use one or the other")
    if args.days is not None:
        return (datetime.now() - timedelta(days=args.days)).isoformat()
    return args.since


def cmd_experiment_list_all(client, args) -> int:
    """The experiments under the experiments folder of the app the server
    hosts (`experiment_list_all`), narrowed to an instrument, an operator or a
    time window. Answers "what did this microscope do this week?"."""
    try:
        since = _since(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    filters = {
        "instrument": args.instrument,
        "operator": args.operator,
        "since": since,
        "until": args.until,
    }
    answer = client.app.fetch(
        "experiment_list_all", **{k: v for k, v in filters.items() if v is not None}
    )
    experiments = answer["items"]
    folder = answer.get("folder")

    if not experiments:
        print(f"No experiments found in {folder}")
        return 0
    if args.group:
        groups: Dict[Optional[str], list] = {}
        for e in experiments:
            groups.setdefault(e.get("instrument_serial"), []).append(e)
        for serial, group in groups.items():
            model = next(
                (e["instrument_model"] for e in group if e["instrument_model"]), None
            )
            heading = f"{model} ({serial})" if serial else "No session recorded"
            print(f"\n{heading} — {len(group)} experiment(s)")
            _print_experiments(group)
    else:
        _print_experiments(experiments)
    print(f"\n{len(experiments)} experiment(s) in {folder}")
    return 0


def _print_experiments(experiments) -> None:
    """One row each: when, what, which app, where it ran, who ran it, how many
    items it has."""
    from fibsem.util.timestamps import format_time, to_datetime

    for e in experiments:
        created = to_datetime(e.get("created_at"))
        when = format_time(created, "%Y-%m-%d %H:%M") if created else "unknown"
        # Written before the session record: no instrument and no operator, shown
        # as "-" so missing doesn't look like empty.
        instrument = e.get("instrument_model") or "-"
        if e.get("instrument_serial"):
            instrument = f"{instrument}/{e['instrument_serial']}"
        note = "" if e.get("available", True) else "  [unreadable]"
        print(
            f"  {when}  {e['name'][:28]:28s} {(e.get('app') or '-')[:12]:12s} "
            f"{instrument[:24]:24s} {(e.get('operator') or '-')[:16]:16s} "
            f"{e.get('num_items', 0):3d} items{note}"
        )
        print(f"      {e.get('relative_path') or e['path']}")


# ---------------------------------------------------------------------------
# This install: no server
# ---------------------------------------------------------------------------


def cmd_plugins(args) -> int:
    """Print what resolved under each entry point group.

    A plugin registered under a mistyped group name fails silently: nothing
    iterates unknown groups. Printing the literal group strings is the whole
    diagnostic -- they can diff them against their own pyproject.
    """
    from fibsem.plugins.report import collect_extensions, render_report

    # Loading the registries logs a traceback per broken plugin, the failures
    # this command reports in one line each; --debug brings them back.
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

    Built twice on purpose. argparse copies a subcommand's namespace over the
    root's, so a flag the subparser merely defaulted would overwrite the same
    flag given before the subcommand. The root's copy holds the real defaults;
    the subparsers' holds ``SUPPRESS``, leaving the attribute absent unless the
    flag was passed. Both positions then work; after the subcommand wins.
    """
    p = argparse.ArgumentParser(add_help=False)

    def default(value):
        return value if defaults else argparse.SUPPRESS

    p.add_argument(
        "--url",
        default=default(None),
        help="The App Server (default: FIBSEM_SERVER_URL, else the one running "
        "on this machine)",
    )
    p.add_argument(
        "--token",
        default=default(None),
        help="The operator token (default: FIBSEM_OPERATOR_TOKEN, else this "
        "install's operator token)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        default=default(False),
        help="Enable debug logging and show tracebacks",
    )
    return p


def _beam_type(p, *, both: bool) -> None:
    choices = ["ELECTRON", "ION"] + (["BOTH"] if both else [])
    p.add_argument(
        "--beam-type",
        type=str.upper,
        choices=choices,
        default="ELECTRON",
        help=f"{', '.join(choices)} (default: ELECTRON)",
    )


def _add_axes(p) -> None:
    for axis in ("x", "y", "z"):
        p.add_argument(f"--{axis}", type=float, metavar="UM", help=f"{axis} in µm")
    p.add_argument("--r", type=float, metavar="DEG", help="rotation in degrees")
    p.add_argument("--t", type=float, metavar="DEG", help="tilt in degrees")


def _group(sub, conn, name: str, help: str):
    """A command group (``stage``, ``beam``, ...) and the adder of its commands,
    each with the connection flags."""
    group = sub.add_parser(name, parents=[conn], help=help)
    verbs = group.add_subparsers(dest="command_name", metavar="COMMAND", required=True)

    def add(verb: str, func, help: str, **kwargs):
        p = verbs.add_parser(verb, parents=[conn], help=help, **kwargs)
        p.set_defaults(func=func)
        return p

    return group, add


# Commands are named as the agent's tools are (`<group>_<verb>[_<object>]`,
# fibsem.ai.agent.catalog), hyphenated: `stage move-absolute` is stage_move_absolute.


def _add_microscope_parsers(sub, conn) -> None:
    _, add = _group(sub, conn, "microscope", "The microscope: who it is, its state")
    add("get-info", cmd_microscope_get_info, "Manufacturer, model, serial, software")
    add("get-state", cmd_microscope_get_state, "The stage position and both beams")

    _, add = _group(sub, conn, "stage", "The stage: where it is, moves")
    add("get-position", cmd_stage_get_position, "The stage position")
    _add_axes(
        add(
            "move-absolute",
            cmd_stage_move_absolute,
            "Move safely to a position; axes left out stay. Crosses to another "
            "device by the traverse",
        )
    )
    _add_axes(
        add(
            "move-relative", cmd_stage_move_relative, "Move the stage axes by an offset"
        )
    )
    p = add(
        "move-to-orientation",
        cmd_stage_move_to_orientation,
        "Move to an orientation (SEM, FIB, ...)",
    )
    p.add_argument("orientation")
    p = add("move-to-device", cmd_stage_move_to_device, "Travel to a device (FM, ...)")
    p.add_argument("device")
    p = add(
        "move-to-milling-angle",
        cmd_stage_move_to_milling_angle,
        "Tilt the stage to a milling angle",
    )
    p.add_argument("milling_angle", type=float, metavar="DEG", help="in degrees")
    p.add_argument(
        "--rotation",
        type=float,
        metavar="DEG",
        help="rotation in degrees (default: the rotation reference)",
    )

    _, add = _group(sub, conn, "beam", "The beams: images, focus, contrast, settings")
    p = add(
        "acquire-image", cmd_beam_acquire_image, "Acquire an image of one or both beams"
    )
    _beam_type(p, both=True)
    p.add_argument(
        "--hfw", type=float, default=150.0, metavar="UM", help="in µm (default: 150)"
    )
    p.add_argument(
        "--resolution",
        type=_parse_resolution,
        default="1536x1024",
        metavar="WxH",
        help="(default: 1536x1024)",
    )
    p.add_argument(
        "--dwell-time",
        type=float,
        default=1.0,
        dest="dwell_time",
        metavar="US",
        help="in µs (default: 1)",
    )
    p.add_argument("--autocontrast", action="store_true", help="Autocontrast first")
    p.add_argument(
        "--out",
        type=Path,
        metavar="DIR",
        help="Save the image(s) here, on this machine",
    )
    p.add_argument(
        "--filename", default="image", help="Saved file stem (default: image)"
    )
    _beam_type(add("auto-focus", cmd_beam_auto_focus, "Focus a beam"), both=False)
    _beam_type(
        add("auto-contrast", cmd_beam_auto_contrast, "The instrument's autocontrast"),
        both=True,
    )
    for verb in ("blank", "unblank"):
        p = add(verb, cmd_beam_blank, f"{verb.capitalize()} a beam")
        p.set_defaults(command=verb)
        _beam_type(p, both=True)
    for state in ("on", "off"):
        p = add(f"turn-{state}", cmd_beam_turn, f"Switch a beam {state}")
        p.set_defaults(command=f"turn_{state}")
        _beam_type(p, both=True)
    p = add(
        "set-parameters",
        cmd_beam_set_parameters,
        "Set a beam's voltage, current or field width",
    )
    _beam_type(p, both=False)
    p.add_argument("--voltage", type=float, metavar="KV", help="in kV")
    p.add_argument(
        "--current", type=float, metavar="AMPS", help="in A (100e-12 is 100 pA)"
    )
    p.add_argument("--hfw", type=float, metavar="UM", help="in µm")

    p = sub.add_parser(
        "hardware-check",
        parents=[conn],
        help="Check that fibsemOS drives the hardware correctly",
        description="Check each device the App Server serves: read every parameter "
        "(read); for the fluorescence microscope and devices whose driver adds "
        "steps, also change each setting, read it back and put it back "
        "(settings), and take one frame, which turns the light on (acquire). No "
        "level moves the objective or the stage.",
    )
    p.add_argument(
        "--level",
        choices=["read", "settings", "acquire"],
        default="read",
        help="How far to go (default: read, which changes nothing)",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the acquire level: the light comes on and exposes the sample",
    )
    p.add_argument(
        "--json", type=Path, metavar="PATH", help="Also save the report as JSON"
    )
    p.add_argument(
        "--save-frame",
        type=Path,
        metavar="PATH",
        help="Save the frame the acquire level takes, as a TIFF",
    )
    p.set_defaults(func=cmd_hardware_check)


def _add_remote_parsers(sub, conn) -> None:
    """``device`` and ``service``: the server's protocol, by name."""
    for noun, plural in (("device", "devices"), ("service", "services")):
        group, add = _group(sub, conn, noun, f"The microscope's {plural}, by name")
        group.set_defaults(noun=noun)
        add("list", cmd_remote_list, f"The {plural} it serves")
        add(
            "describe", cmd_remote_describe, "Its parameters' values, commands, queries"
        ).add_argument("name")
        p = add("get-parameter", cmd_remote_get_parameter, "Read a parameter")
        p.add_argument("name")
        p.add_argument("parameter")
        p = add(
            "set-parameter", cmd_remote_set_parameter, "Write a parameter (SI units)"
        )
        p.add_argument("name")
        p.add_argument("parameter")
        p.add_argument("value", help="JSON (1e-9, true, [1536, 1024]) or text")
        for verb, what, func in (
            ("call-command", "command", cmd_remote_call_command),
            ("call-query", "query", cmd_remote_call_query),
        ):
            p = add(verb, func, f"Run a {what}")
            p.add_argument("name")
            p.add_argument(what)
            p.add_argument(
                "--kwargs",
                metavar="JSON|@FILE",
                help="Its arguments: a JSON object, or @file (.json, .yaml)",
            )


def _add_experiment_parsers(sub, conn) -> None:
    _, add = _group(sub, conn, "experiment", "The app's experiments")
    p = add(
        "list-all",
        cmd_experiment_list_all,
        "The experiments folder's experiments, by instrument or time window",
    )
    p.add_argument(
        "--instrument", metavar="SERIAL|MODEL", help="Only those run on this instrument"
    )
    p.add_argument("--operator", metavar="NAME", help="Only those run by this operator")
    p.add_argument("--days", type=int, metavar="N", help="Only the last N days")
    p.add_argument("--since", metavar="YYYY-MM-DD", help="Only on or after this date")
    p.add_argument("--until", metavar="YYYY-MM-DD", help="Only on or before this date")
    p.add_argument(
        "--group", action="store_true", help="Group the listing by instrument"
    )


def _add_plugins_parser(sub) -> None:
    """Takes no connection arguments: diagnosing a plugin is usually done away
    from the instrument."""
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
        # SUPPRESS: the root parser supplies the default (_build_connection_parser).
        default=argparse.SUPPRESS,
        help="Show the full traceback for each plugin that failed to load",
    )
    p.set_defaults(func=cmd_plugins, needs_server=False)


class _VersionAction(argparse.Action):
    """Print the version, revision and branch, then exit. The git lookup
    happens only when --version is passed."""

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
    conn = _build_connection_parser(defaults=False)
    root = argparse.ArgumentParser(
        prog="fibsem-cli",
        description="fibsemOS from a terminal, through the App Server, as the operator",
        parents=[_build_connection_parser()],
    )
    root.add_argument(
        "--version",
        action=_VersionAction,
        help="Print the fibsemOS version and exit",
    )
    root.set_defaults(needs_server=True)
    sub = root.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")
    sub.required = True
    _add_microscope_parsers(sub, conn)
    _add_remote_parsers(sub, conn)
    _add_experiment_parsers(sub, conn)
    _add_plugins_parser(sub)
    return root


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _said(error: Exception) -> str:
    """What to print for a failed call: the server's refusal as it words it,
    with what to do for the ones this command can say."""
    from fibsem.client import ServerError, ServerUnreachable

    if isinstance(error, ServerUnreachable):
        return f"{error}\n{NO_SERVER}"
    if isinstance(error, ServerError) and error.error_type == "microscope_off":
        return MICROSCOPE_OFF
    return str(error)


def main(argv=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING)

    if getattr(args, "func", None) in (
        cmd_stage_move_absolute,
        cmd_stage_move_relative,
    ):
        if all(v is None for v in _axes(args).values()):
            parser.error("at least one of --x, --y, --z, --r, --t must be specified")
    if args.subcommand == "hardware-check":
        if args.level == "acquire" and not args.yes:
            parser.error(
                "hardware-check: the acquire level turns the light on and exposes whatever "
                "is under the objective; add --yes to run it"
            )
        if args.save_frame is not None and args.level != "acquire":
            parser.error("hardware-check: --save-frame needs --level acquire")

    client = None
    try:
        if not args.needs_server:
            exit_code = args.func(args)
        else:
            client = _client(args)
            exit_code = args.func(client, args)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        if args.debug:
            raise
        print(f"error: {_said(e)}", file=sys.stderr)
        sys.exit(1)
    finally:
        if client is not None:
            client.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
