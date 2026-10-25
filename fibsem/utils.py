import datetime
import functools
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set, Tuple, Union

import yaml

from fibsem import config as cfg
from fibsem import state_move
from fibsem.codec import encode
from fibsem.codec.upgrades.configuration import upgrade
from fibsem.constants import DATETIME_LOG, MICRON_SYMBOL, MU_SYMBOL
from fibsem.structures.settings import FibsemStagePosition
from fibsem.structures.system import (
    CONFIGURATION_VERSION,
    CONFIGURED_DEVICES,
    DEVICES_KEEPING_DRIVER_KEYS,
    STAGE_POSITION_KEYS,
    DeviceEntry,
    MicroscopeSettings,
)

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope


def current_timestamp(fmt: str = DATETIME_LOG) -> str:
    """Now on the local clock in *fmt* (one of the ``fibsem.constants`` formats),
    for a file or session name."""
    return datetime.datetime.now().strftime(fmt)


# `MU_SYMBOL` rather than a literal, and specifically U+00B5 MICRO SIGN rather than
# U+03BC GREEK SMALL LETTER MU. The two are indistinguishable on screen and unequal to
# every string comparison, and this table agrees with `constants.MICRON_SYMBOL`,
# which every spin-box suffix and scalebar already uses. `imaging/drawing.py` records
# the reason the house character is the micro sign: the default font on Windows has no
# glyph for the Greek letter, so the Greek one renders as a box on the platform most
# instruments are driven from.
SI_PREFIXES = {
    -12: "p",
    -9: "n",
    -6: MU_SYMBOL,
    -3: "m",
    0: "",
    3: "k",
    6: "M",
    9: "G",
    12: "T",
}

# What every formatter here renders for a value the instrument did not report. An
# em-dash rather than "None" or "0": the distinction between "not measured" and
# "measured as zero" is one an operator has to be able to make at a glance.
NOT_AVAILABLE = "—"

_MIN_SI_EXP = min(SI_PREFIXES)
_MAX_SI_EXP = max(SI_PREFIXES)


def format_resolution_as_str(resolution: List[int]) -> str:
    """Format a resolution list as a string.

    Args:
        resolution (List[int]): The resolution to format.
    Returns:
        str: The formatted resolution string.
    """
    return f"{resolution[0]} x {resolution[1]}"


def _get_scale_from_value(val: float) -> float:
    """Return the scale multiplier corresponding to the SI prefix for a value."""
    if val == 0:
        return 1.0

    exponent = int(math.floor(math.log10(abs(val))))
    exponent = (exponent // 3) * 3
    exponent = max(min(exponent, _MAX_SI_EXP), _MIN_SI_EXP)
    return 10 ** (-exponent)


def _get_prefix_from_scale(scale: float) -> Tuple[str, float]:
    """Return the SI prefix and correction factor associated with a scale multiplier."""
    if scale == 0:
        return "", 1.0

    exponent = -math.log10(scale)
    exponent = int(round(exponent / 3.0) * 3)
    exponent = max(min(exponent, _MAX_SI_EXP), _MIN_SI_EXP)
    prefix = SI_PREFIXES.get(exponent, "")
    multiplier = (10 ** (-exponent)) / scale
    return prefix, multiplier


def _get_display_unit(scale: float, unit: Optional[str] = None) -> str:
    """Return the formatted unit string using the scale-derived SI prefix."""
    unit = unit or ""
    prefix, _ = _get_prefix_from_scale(scale)
    return f"{prefix}{unit}"


def format_value(
    val: float,
    unit: Optional[str] = None,
    precision: int = 2,
    scale: Optional[float] = None,
) -> str:
    """Format a numerical value as a string with nearest SI unit.

    Args:
        val: The value to format.
        unit (str, optional): The unit of the value. Defaults to None.
        precision (int, optional): Decimal places. Defaults to 2.
        scale (float, optional): Override the auto-calculated scale multiplier.

    Returns:
        str: The formatted value with the appropriate SI prefix.
    """
    if val is None:
        return NOT_AVAILABLE
    scale = scale if scale is not None else _get_scale_from_value(val)
    prefix, multiplier = _get_prefix_from_scale(scale)
    scaled_val = val * scale * multiplier
    unit = unit or ""
    return f"{scaled_val:.{precision}f} {prefix}{unit}"


# ---------------------------------------------------------------------------
# Named formatting policies
#
# `format_value` is the primitive: one precision, whichever SI prefix the magnitude
# lands on. These are the conventions that were being written out by hand instead,
# each in more than one place, because they vary precision *per band* -- which is a
# real readability rule and not something a single `precision` argument can express.
# Milling currents are the clearest case: three orders of magnitude, and rendering
# everything in pA turns the common nA reading into a four-digit number.
# ---------------------------------------------------------------------------


def format_angle(radians: Optional[float], precision: int = 1) -> str:
    """An angle in radians, as degrees.

    Not a `format_value` call: SI prefixes are the wrong vocabulary for an angle, and
    nobody wants to read a stage tilt in milliradians.
    """
    if radians is None:
        return NOT_AVAILABLE
    return f"{math.degrees(radians):.{precision}f}°"


def format_distance(metres: Optional[float]) -> str:
    """A distance, with more decimals the larger the unit.

    A stage move is interesting to the nanometre and a stage position to the micron,
    so the precision follows the prefix rather than being fixed. Zero is rendered in
    nanometres: `format_value` would land it on bare metres, and "0.00 m" beside a
    column of micron readings reads as a different quantity.
    """
    if metres is None:
        return NOT_AVAILABLE
    magnitude = abs(metres)
    if magnitude == 0:
        return "0 nm"
    if magnitude < 1e-6:
        return f"{metres * 1e9:.1f} nm"
    if magnitude < 1e-3:
        return f"{metres * 1e6:.2f} {MICRON_SYMBOL}"
    if magnitude < 1.0:
        return f"{metres * 1e3:.3f} mm"
    return f"{metres:.4f} m"


def format_bytes(count: float) -> str:
    """`3.4 GB`, `880 MB`, `12 kB`. Decimal units, because a disk is sold in them.

    One decimal place above a gigabyte and none below: the callers are estimates of
    what a run will write and readings of what a volume has left, and quoting either
    to four figures would claim a precision neither has.

    Lives here rather than in the pre-flight dialogs that first needed it because the
    experiment dialogs quote free disk space with it, and a formatter in a module of
    Qt furniture is not one a non-UI caller can reach.
    """
    for unit, step in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if count >= step:
            value = count / step
            return (
                f"{value:.1f} {unit}" if unit in ("TB", "GB") else f"{value:.0f} {unit}"
            )
    return f"{int(count)} B"


def format_current(amps: Optional[float]) -> str:
    """A beam current: `1.0 nA`, `60 pA`."""
    if amps is None:
        return NOT_AVAILABLE
    if abs(amps) >= 1e-9:
        return format_value(amps, "A", precision=1, scale=1e9)
    return format_value(amps, "A", precision=0, scale=1e12)


def format_stage_position(position: Optional["FibsemStagePosition"]) -> str:
    """A stage position on one line, in the units each axis deserves.

    Deliberately *not* `FibsemStagePosition.pretty`, which is fixed millimetres to two
    decimals and stays that way. That is the right choice where `pretty` is used --
    five `logging.info` calls and an error message -- because a log column in one unit
    can be read down, and one that alternates micrometres and millimetres cannot. It
    is the wrong choice on screen: at the milling pose `pretty` renders every
    translation axis as "0.00mm", where the operator's move was 42 micrometres.

    So both exist, and the difference between them is the medium rather than an
    oversight.
    """
    if position is None:
        return NOT_AVAILABLE
    return (
        f"X:{format_distance(position.x)}, "
        f"Y:{format_distance(position.y)}, "
        f"Z:{format_distance(position.z)}, "
        f"R:{format_angle(position.r)}, "
        f"T:{format_angle(position.t)}"
    )


def format_voltage(volts: Optional[float]) -> str:
    """A beam voltage, pinned to kV.

    Pinned rather than auto-scaled so a column of voltages stays comparable: a 500 V
    landing energy beside a 30 kV one should read `0.50 kV`, not switch units halfway
    down the list.
    """
    return format_value(volts, "V", precision=2, scale=1e-3)


def configure_logging(
    path: Path = "",
    log_filename="logfile",
    log_level=logging.DEBUG,
    _DEBUG: bool = False,
):
    """Log to the terminal and to file simultaneously."""
    logfile = os.path.join(path, f"{log_filename}.log")

    file_handler = logging.FileHandler(logfile, encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO if _DEBUG is False else logging.DEBUG)

    logging.basicConfig(
        format="%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s",
        level=log_level,
        # Multiple handlers can be added to your logging configuration.
        # By default log messages are appended to the file if it exists already
        handlers=[file_handler, stream_handler],
        force=True,
    )

    # disable some loggers
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("napari").setLevel(logging.WARNING)

    state_move.log_pending()
    return logfile


def load_yaml(fname: Path) -> dict:
    """load yaml file

    Args:
        fname (Path): yaml file path

    Returns:
        dict: Items in yaml
    """
    with open(fname) as f:
        config = yaml.safe_load(f)

    return config


def save_yaml(path: Path, data: dict) -> None:
    """Saves a python dictionary object to a yaml file

    Args:
        path (Path): path location to save yaml file
        data (dict): dictionary object
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    path = Path(path).with_suffix(".yaml")
    with open(path, "w") as f:
        yaml.dump(data, f, indent=4)


def save_json(path: Union[Path, os.PathLike, str], data: dict) -> None:
    """Saves a python dictionary object to a json file
    Args:
        path (Path): path location to save json file
        data (dict): dictionary object
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def setup_session(
    session_path: Path = None,
    config_path: Path = None,
    protocol_path: Path = None,
    setup_logging: bool = True,
    ip_address: str = None,
    manufacturer: str = None,
    debug: bool = False,
    apply_defaults: Optional[bool] = None,
    beams_on: Optional[bool] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple["FibsemMicroscope", "MicroscopeSettings"]:
    """Setup microscope session

    Args:
        session_path (Path): path to logging directory
        config_path (Path): path to config directory
        protocol_path (Path): path to protocol file
        apply_defaults (bool, optional): set the columns to the configured defaults
            once connected. None (the default) does what the configuration's
            `defaults.apply_on_connect` says.
        beams_on (bool, optional): turn the beams on once connected. None (the
            default) does what `defaults.beams_on_at_connect` says.
        progress (callable, optional): called with a short description of each
            step as it starts ("Turning the beams on…"), for a caller that shows
            it. Called on the thread this runs on.

    Returns:
        tuple: microscope, settings
    """

    # load settings
    settings = load_microscope_configuration(config_path, protocol_path)

    # create session directories
    session = f"{settings.protocol.get('name', 'fibsem-os')}_{current_timestamp()}"
    if protocol_path is None:
        protocol_path = os.getcwd()

    # configure paths
    if session_path is None:
        session_path = cfg.LOG_PATH
    os.makedirs(session_path, exist_ok=True)

    # configure logging
    if setup_logging:
        configure_logging(session_path, _DEBUG=debug)

    # connect to microscope
    # cheap overloading
    if ip_address:
        settings.system.info.ip_address = ip_address

    if manufacturer:
        # normalise the override the same way SystemInfo.from_dict normalises the
        # config value, so system.info always carries the canonical spelling
        from fibsem.drivers.registry import normalize_name

        settings.system.info.manufacturer = normalize_name(manufacturer)

    _report(progress, f"Connecting to {settings.system.info.ip_address}…")

    from fibsem.drivers.registry import connect_microscope

    microscope = connect_microscope(settings.system)

    # The planning stack estimates milling time without a microscope in scope, so
    # the milling service's model is installed for the session here, on connect.
    # Without a milling service (no ion beam) stages use the shared table.
    from fibsem.milling.base import set_milling_time_estimator

    # The service class's model, not the service: the session keeps it after a
    # disconnect, and a bound method would keep the microscope alive with it.
    milling = getattr(microscope, "milling", None)
    set_milling_time_estimator(
        type(milling).estimate_stage if milling is not None else None
    )

    # set default image_settings path
    settings.image.path = session_path

    # Remembered so a calibration action can write back to the file it came from.
    microscope.configuration_path = str(
        config_path if config_path is not None else cfg.DEFAULT_CONFIGURATION_PATH
    )
    # The sample handling was built during the connect, before the configuration was
    # known, so the session state it restores from could not be found yet.
    handling = getattr(microscope, "sample_handling", None)
    if handling is not None:
        handling.restore_occupancy()

    # What connecting did beyond connecting, as the file (or the caller) asked:
    # {"beams_on": worked, "defaults": worked}, for each one asked for. A failure
    # does not fail the connection -- the microscope is connected, only a column was
    # not set -- so it is recorded for the caller to report rather than raised.
    if beams_on is None:
        beams_on = settings.system.beams_on_at_connect
    if apply_defaults is None:
        apply_defaults = settings.system.apply_defaults_on_connect
    microscope.connect_actions = {}
    # The beams first, so the defaults are set on a live column.
    if beams_on:
        _report(progress, "Turning the beams on…")
        microscope.connect_actions["beams_on"] = _at_connect(
            "turn the beams on", microscope.turn_beams_on
        )
    if apply_defaults:
        _report(progress, "Applying the defaults…")
        microscope.connect_actions["defaults"] = _at_connect(
            "apply the configured defaults", microscope.apply_defaults
        )

    logging.info(f"Finished setup for session: {session}")

    return microscope, settings


def _report(progress: Optional[Callable[[str], None]], step: str) -> None:
    """Tell *progress* which step is starting. Never fails the connection."""
    if progress is None:
        return
    try:
        progress(step)
    except Exception as e:
        logging.debug(f"Progress callback failed on {step!r}: {e}")


def _at_connect(what: str, action) -> bool:
    """Run one of the things a configuration asks for at connect; whether it worked."""
    try:
        action()
        return True
    except Exception as e:
        logging.error(f"Could not {what} at connect: {e}")
        return False


def load_microscope_configuration(
    config_path: Path = None, protocol_path: Path = None
) -> MicroscopeSettings:
    """Load microscope settings from configuration files

    Args:
        config_path (Path, optional): path to config directory. Defaults to None.
        protocol_path (Path, optional): path to protocol file. Defaults to None.

    Returns:
        MicroscopeSettings: microscope settings
    """
    if config_path is None:
        from fibsem.config import DEFAULT_CONFIGURATION_PATH

        config_path = DEFAULT_CONFIGURATION_PATH

    # load config
    config = _load_configuration_file(config_path)

    report_unrecognised_configuration_keys(config, source=str(config_path))

    # load protocol
    protocol = load_protocol(protocol_path)

    # create settings
    settings = MicroscopeSettings.from_dict(config, protocol=protocol)

    # The FM working state for this configuration: session state, read-only here
    # (a script reads what the operator left; only the application writes it).
    from fibsem.fm.config import load_fm_configuration
    from fibsem.session_state import SessionState

    settings.fm = load_fm_configuration(SessionState(config_path))

    return settings


# Blocks accepted wholesale. `sim:` is a plain dict the simulator reads with `.get()`
# rather than a dataclass, and `protocol:` is the application's; policing either would
# invent warnings every time a backend gains a key. `calibration.holders` is keyed by
# holder name, so its keys are whatever a site called its shuttles.
OPEN_CONFIGURATION_BLOCKS = ("sim", "protocol", "calibration.holders")

# The sections, policed one level deeper: `hardware.electron` is a block of keys, and
# a typo in it should be reported the way a typo in the old flat `electron:` is.
SECTION_BLOCKS = ("hardware", "calibration", "defaults")

# Configuration version 2's list of devices, which is policed by device name.
DEVICE_LIST = "hardware.devices"


@functools.lru_cache(maxsize=1)
def written_configuration_keys() -> Set[str]:
    """Every dotted path `MicroscopeSettings.to_dict` writes.

    This *is* the schema. There is no hand-written table of known keys, because a
    table is a second copy of what the writer does and the two drift: the first
    attempt at one was written from the shipped YAML files and rejected 23 keys that
    `to_dict` writes on every save. Derived from the writer, the set of keys that will
    be saved back is by construction the set of keys that are saved back.

    One level deep, blocks and their keys -- two for the `SECTION_BLOCKS`. A value
    that is itself a dict below that (`stage.devices`) is accepted wholesale under
    its key. The device list is read by name: each device it writes is
    `hardware.devices.<name>`, with its keys under that.
    """
    written = MicroscopeSettings.from_dict({}).to_dict()
    # Written only when they are set.
    keys: Set[str] = {"info.port", "calibration.devices"}
    keys.update(
        f"calibration.devices.{k}" for k in ("fibsem", "fm", "holder", "when", "sites")
    )
    for block, value in written.items():
        keys.add(block)
        if not isinstance(value, dict):
            continue
        for key, sub in value.items():
            path = f"{block}.{key}"
            keys.add(path)
            if path == DEVICE_LIST:
                for entry in sub:
                    device = f"{path}.{entry['name']}"
                    keys.add(device)
                    # Written only when the file has them: a device's roles, and
                    # where the stage travels for it (the beams' is never written).
                    keys.add(f"{device}.roles")
                    keys.update(f"{device}.{k}" for k in STAGE_POSITION_KEYS)
                    if entry["name"] == "fm":
                        keys.add(f"{device}.mount_transform")
                        keys.add(f"{device}.camera_tilt")
                    keys.update(
                        f"{device}.{k}"
                        for k in entry
                        if k not in DeviceEntry.IDENTITY_KEYS
                    )
                continue
            if (
                block in SECTION_BLOCKS
                and isinstance(sub, dict)
                and path not in OPEN_CONFIGURATION_BLOCKS
            ):
                keys.update(f"{path}.{k}" for k in sub)
    return keys


def unrecognised_configuration_keys(config: dict) -> List[str]:
    """Dotted paths in *config* that this version will not write back.

    A configuration may hold others -- one written before a key was removed, or
    hand-edited with a guess -- and those are ignored, which is the contract that
    lets old files keep working. But ignored *silently* is how
    `imaging.imaging_current` came to be a setting a user could type, save, reload
    and never see again: `ImageSettings` has no such field, so it was dropped on load
    and nothing said so. One line at load is the difference between "my setting
    vanished" and "my setting is not supported".
    """
    known = written_configuration_keys()

    def is_known(path: str) -> bool:
        return path in known

    unknown: List[str] = []
    for block, value in (config or {}).items():
        if block in OPEN_CONFIGURATION_BLOCKS:
            continue
        if not is_known(block):
            unknown.append(block)
            continue
        if not isinstance(value, dict):
            continue
        for key, sub in value.items():
            path = f"{block}.{key}"
            if path in OPEN_CONFIGURATION_BLOCKS:
                continue
            if path == DEVICE_LIST and isinstance(sub, list):
                unknown.extend(_unrecognised_device_keys(sub, known))
                continue
            if not is_known(path):
                unknown.append(path)
            elif block in SECTION_BLOCKS and isinstance(sub, dict):
                unknown.extend(
                    f"{path}.{k}" for k in sub if not is_known(f"{path}.{k}")
                )
    return sorted(unknown)


def _unrecognised_device_keys(entries: list, known: Set[str]) -> List[str]:
    """The keys of `hardware.devices` entries this version will not write back.

    Only the devices with records of their own are policed. Any other entry -- a
    manipulator given a driver, a plugin's device -- carries its driver's keys, which
    are the driver's to know, and is written back as it was. So does a beam or the
    stage that names a driver (`DEVICES_KEEPING_DRIVER_KEYS`), and their `required:`
    is kept too.
    """
    unknown: List[str] = []
    for item in entries:
        if not isinstance(item, dict):
            unknown.append(DEVICE_LIST)
            continue
        try:
            name = DeviceEntry.from_dict(item).name
        except ValueError:
            unknown.append(DEVICE_LIST)
            continue
        device = f"{DEVICE_LIST}.{name}"
        if device not in known:
            continue
        kept: Set[str] = set()
        if name in DEVICES_KEEPING_DRIVER_KEYS:
            if item.get("driver") is not None:
                continue
            kept = {"required"}
        unknown.extend(
            f"{device}.{k}"
            for k in item
            if k not in DeviceEntry.IDENTITY_KEYS
            and k not in kept
            and f"{device}.{k}" not in known
        )
    return unknown


def report_unrecognised_configuration_keys(config: dict, source: str = "") -> List[str]:
    """Log the keys this version ignores, once, and return them."""
    unknown = unrecognised_configuration_keys(config)
    if unknown:
        where = f" in {source}" if source else ""
        logging.info(
            f"Configuration{where} contains {len(unknown)} key(s) this version does "
            f"not read and will not save back: {', '.join(unknown)}"
        )
    return unknown


def _deep_update(target: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def write_configuration(path: Union[str, Path], updates: dict) -> None:
    """Write *updates* into the configuration file at *path*, and nothing else.

    The file is read, the nested keys in *updates* are set (a dict merges into the
    block it names; anything else replaces the value there), and it is written back.
    The rest of the file -- including whatever a person wrote there by hand -- is
    preserved at the parsed level. This is the one writer of the file from the
    application: a calibration action writes `calibration.*`, the defaults panel
    writes `defaults.*`, and neither can disturb the other's section.

    The file is also brought to the current version on the way (`upgrade`),
    so a site's file takes the device list the first time anything saves it.
    """
    config = _read_for_writing(path)
    _deep_update(config, updates)
    _write_configuration_file(path, config)


def _read_for_writing(path: Union[str, Path]) -> dict:
    """The configuration at *path*, upgraded, with a copy kept of an older version."""
    config = load_yaml(os.path.join(path)) or {}
    _keep_an_older_version(path, config)
    return upgrade(config)


def _keep_an_older_version(path: Union[str, Path], config: dict) -> bool:
    """Keep a copy of the file at *path* when *config*, read from it, is from an older
    version; whether it is."""
    version = config.get("version")
    if version is None:
        _keep_the_file_as_it_was(path)
    elif version < CONFIGURATION_VERSION:
        _keep_the_file_as_it_was(path, before=CONFIGURATION_VERSION)
    else:
        return False
    return True


def _load_configuration_file(path: Union[str, Path]) -> dict:
    """The configuration at *path*, in the current version.

    A file from an older version is saved back upgraded, after a copy of it is kept,
    so it is upgraded once rather than on every load. Where it can't be written (a
    read-only share), it is upgraded on every load instead.
    """
    config = load_yaml(os.path.join(path)) or {}
    upgraded = upgrade(config)
    try:
        if _keep_an_older_version(path, config):
            _write_configuration_file(path, upgraded)
            logging.info(
                f"Saved {path} in configuration version {CONFIGURATION_VERSION}."
            )
    except OSError as e:
        logging.warning(
            f"Could not save {path} in configuration version {CONFIGURATION_VERSION} "
            f"({e}); it is upgraded each time it loads."
        )
    return upgraded


def configuration_device(config: dict, name: str, create: bool = False) -> dict:
    """The `hardware.devices` entry for *name* in an upgraded configuration dict.

    For code that reads or edits the file itself rather than through
    `MicroscopeSettings` -- the setup wizard filling in a shipped file. With *create*, an
    entry is added when the list has none, so a write lands in the file; without it, a
    missing device reads as an empty dict.
    """
    hardware = config.get("hardware")
    if not isinstance(hardware, dict):
        if not create:
            return {}
        hardware = config["hardware"] = {}
    listed = hardware.get("devices")
    if not isinstance(listed, list):
        if not create:
            return {}
        listed = hardware["devices"] = []
    for item in listed:
        if isinstance(item, dict):
            try:
                if DeviceEntry.from_dict(item).name == name:
                    return item
            except ValueError:
                continue
    if not create:
        return {}
    entry = {"name": name, "type": CONFIGURED_DEVICES.get(name, name)}
    listed.append(entry)
    return entry


def configuration_backup_path(path: Union[str, Path], before: int = 1) -> Path:
    """Where a configuration written before version *before* is kept."""
    return Path(f"{path}.before-v{before}")


def _keep_the_file_as_it_was(path: Union[str, Path], before: int = 1) -> None:
    """Copy a file written before version *before*, once, before it is changed.

    The upgrade (`fibsem.codec.upgrades.configuration`) writes the current version,
    which a version from before *before* cannot read: before 1, it reads none of the
    sections; at 1, it does not read the device list and would load every device's
    geometry as the defaults. The copy is what a site going back to that version
    restores. Made once: a second write must not replace the original with a
    half-converted file.
    """
    backup = configuration_backup_path(path, before)
    if backup.exists() or not os.path.exists(path):
        return
    import shutil

    shutil.copy2(path, backup)
    logging.info(
        f"Kept a copy of {path} as {backup} before changing it; a version of "
        f"fibsem-os from before configuration v{before} can read the copy."
    )


def write_objective_calibration(
    path: Union[str, Path],
    focus_position: Optional[float],
    limit_position: Optional[float],
) -> None:
    """Record the objective's calibration in the configuration file at *path*."""
    write_configuration(
        path,
        {
            "calibration": {
                "objective": {
                    "focus_position": focus_position,
                    "limit_position": limit_position,
                }
            }
        },
    )


def write_holder_calibration(
    path: Union[str, Path], holders: Dict[str, dict], active_holder: str
) -> None:
    """Record the sample holders and which one is fitted in the configuration at *path*.

    `calibration.holders` is replaced whole rather than merged: a holder renamed in
    the wizard must not leave its old entry behind, and the entries are the
    holders the session knows, which include every one the file had.
    """
    config = _read_for_writing(path)
    calibration = config.get("calibration")
    if not isinstance(calibration, dict):
        calibration = config["calibration"] = {}
    calibration["holders"] = holders
    calibration["active_holder"] = active_holder
    _write_configuration_file(path, config)


def write_device_calibration(
    path: Union[str, Path],
    devices: dict,
    rotation_centre_correction: Tuple[float, float],
) -> None:
    """Record the SEM to FM calibration (`calibration.devices`) and the rotation
    centre it was measured with in the configuration at *path*."""
    config = _read_for_writing(path)
    calibration = config.get("calibration")
    if not isinstance(calibration, dict):
        calibration = config["calibration"] = {}
    calibration["devices"] = devices
    calibration["rotation_centre_correction"] = [
        float(v) for v in rotation_centre_correction
    ]
    _write_configuration_file(path, config)


def _write_configuration_file(path: Union[str, Path], config: dict) -> None:
    """Write a configuration dict to exactly *path*.

    Not `save_yaml`: that forces a `.yaml` suffix, so a site whose file is
    `site.yml` would get a sibling `site.yaml` written and its own file left
    untouched; it sorts keys, which turns the sections into alphabetical order;
    and it uses the unsafe dumper, which writes numpy scalars as tags that
    `safe_load` cannot read back.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(encode(config), f, sort_keys=False, indent=4)


def load_protocol(protocol_path: Path = None) -> dict:
    """Load the protocol file from yaml

    Args:
        protocol_path (Path, optional): path to protocol file. Defaults to None.

    Returns:
        dict: protocol dictionary
    """
    if protocol_path is not None:
        protocol = load_yaml(protocol_path)
    else:
        protocol = {"name": "demo"}

    # protocol = _format_dictionary(protocol)

    return protocol


def _format_dictionary(dictionary: dict) -> dict:
    """Recursively traverse dictionary and covert all numeric values to flaot.

    Parameters
    ----------
    dictionary : dict
        Any arbitrarily structured python dictionary.

    Returns
    -------
    dictionary
        The input dictionary, with all numeric values converted to float type.
    """
    for key, item in dictionary.items():
        if isinstance(item, dict):
            _format_dictionary(item)
        elif isinstance(item, list):
            dictionary[key] = [
                _format_dictionary(i)
                for i in item
                if isinstance(i, list) or isinstance(i, dict)
            ]
        else:
            if item is not None:
                try:
                    dictionary[key] = float(dictionary[key])
                except ValueError:
                    pass
    return dictionary


def get_params(main_str: str) -> list:
    """Helper function to access relevant metadata parameters from sub field

    Args:
        main_str (str): Sub string of relevant metadata

    Returns:
        list: Parameters covered by metadata
    """
    cats = []
    cat_str = ""

    i = main_str.find("\n")
    i += 1
    while i < len(main_str):
        if main_str[i] == "=":
            cats.append(cat_str)
            cat_str = ""
            i += main_str[i:].find("\n")
        else:
            cat_str += main_str[i]

        i += 1
    return cats


def _register_metadata(
    microscope: "FibsemMicroscope",
    application_software: str,
    experiment_name: str,
    experiment_id: Optional[str] = None,
) -> None:
    """Stamp user, experiment and application identity onto what ``microscope`` acquires.

    ``experiment_id`` is the stable join key and should always be supplied; it is
    optional only so a caller that has no ID yet still registers something.

    Which application is running goes on ``SystemInfo``, not on the experiment
    reference. It is a property of the running system, and putting it in one place
    is what lets the reference carry identity only.

    There is no application *version* to record. An application shipped inside
    fibsem has no version of its own -- passing ``fibsem.__version__`` for it only
    made two fields tautologically equal -- and ``info.fibsem_revision`` already
    pins the exact commit doing the work.
    """
    from fibsem.structures.system import FibsemExperimentRef, FibsemUser

    microscope.user = FibsemUser.from_environment()
    microscope.experiment = FibsemExperimentRef(
        id=experiment_id,
        name=experiment_name,
    )

    # info carries fibsem_version and fibsem_revision already, set when it is built.
    # Reached defensively at both levels: a stand-in microscope (a test double, or a
    # caller wiring up a run without hardware) may have neither attribute, and
    # failing to record the application is not worth an AttributeError.
    info = getattr(getattr(microscope, "system", None), "info", None)
    if info is not None:
        info.application = application_software
