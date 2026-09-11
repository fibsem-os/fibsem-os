import datetime
import functools
import glob
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Set, Tuple, Union

import yaml
from PIL import Image

from fibsem import config as cfg
from fibsem import manufacturers
from fibsem.constants import DATETIME_LOG, MICRON_SYMBOL, MU_SYMBOL, TIME_FILE
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemStagePosition,
    MicroscopeSettings,
)

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope


def current_timestamp():
    """Returns current time in a specific string format

    Returns:
        String: Current time
    """
    return datetime.datetime.fromtimestamp(time.time()).strftime(
        DATETIME_LOG
    )  # PM/AM doesnt work?


def current_timestamp_v2():
    """Returns current time in a specific string format

    Returns:
        String: Current time
    """
    return str(time.time()).replace(".", "_")


def current_timestamp_v3(timeonly: bool = True) -> str:
    """Return the current time in a specific string formats: HH-MM-SS or YYYY-MM-DD-HH-MM-SSAM/PM"""
    now = datetime.datetime.now()
    if timeonly:
        return now.strftime(TIME_FILE)
    return now.strftime(DATETIME_LOG)


def _format_time_seconds(seconds: float) -> str:
    """Format a time delta in seconds to proper string format."""
    return str(datetime.timedelta(seconds=seconds)).split(".")[0]


def format_duration(seconds: float) -> str:
    """Format a duration given in seconds into a human-readable string (hours, minutes, seconds)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {seconds:.2f}s"
    elif minutes > 0:
        return f"{minutes}m {seconds:.2f}s"
    else:
        return f"{seconds:.2f}s"


def format_time_remaining(seconds: float, pad: bool = False) -> str:
    """A duration in whole units: `1h 01m`, `4m 12s`, `5s`.

    Rounded rather than truncated, so a value a hair under a boundary reads as the
    boundary -- 59.6 s is `1m 00s`, not `59s`. Rounding *before* the branch is what makes
    that work: 60 is no longer under a minute, so it takes the minutes arm.

    *pad* zero-fills the second unit, which is what a column of durations wants -- a
    figure that changes width as it counts down drags everything after it sideways. Left
    off by default because a log line does not care.

    Not `format_duration` above, which carries two decimal places: that is the right
    shape for a milling estimate quoted to the second and the wrong one for a figure
    already presented as approximate. Not `task_summary_formatting.format_duration_short`
    either -- that is a fixed-width `MMm:SSs` for a table column, blank on a missing
    value, which is a third job again.
    """
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    width = "02d" if pad else "d"
    if hours:
        return f"{hours}h {minutes:{width}}m"
    if minutes:
        return f"{minutes}m {secs:{width}}s"
    return f"{secs}s"


# `MU_SYMBOL` rather than a literal, and specifically U+00B5 MICRO SIGN rather than
# U+03BC GREEK SMALL LETTER MU. The two are indistinguishable on screen and unequal to
# every string comparison, and this table used to disagree with `constants.MICRON_SYMBOL`
# -- which every spin-box suffix and scalebar already uses. `imaging/drawing.py` records
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


def make_logging_directory(path: Optional[Path] = None, name="run"):
    """
    Create a logging directory with the specified name at the specified file path.
    If no path is given, it creates the directory at the default base path.

    Args:
        path (Path, optional): The file path to create the logging directory at. If None, default base path is used.
        name (str, optional): The name of the logging directory to create. Default is "run".

    Returns:
        str: The file path to the created logging directory.
    """

    if path is None:
        path = os.path.join(cfg.BASE_PATH, "log")
    directory = os.path.join(path, name)
    os.makedirs(directory, exist_ok=True)
    return directory


# TODO: better logs: https://www.toptal.com/python/in-depth-python-logging
# https://stackoverflow.com/questions/61483056/save-logging-debug-and-show-only-logging-info-python
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

    return logfile


def load_yaml(fname: Path) -> dict:
    """load yaml file

    Args:
        fname (Path): yaml file path

    Returns:
        dict: Items in yaml
    """
    with open(fname, "r") as f:
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


def create_gif(path: Path, search: str, gif_fname: str, loop: int = 0) -> None:
    """Creates a GIF from a set of images. Images must be in same folder

    Args:
        path (Path): Path to images folder
        search (str): search name
        gif_fname (str): name to save gif file
        loop (int, optional): _description_. Defaults to 0.
    """
    filenames = glob.glob(os.path.join(path, search))

    imgs = [Image.fromarray(FibsemImage.load(fname).data) for fname in filenames]

    print(f"{len(filenames)} images added to gif.")
    imgs[0].save(
        os.path.join(path, f"{gif_fname}.gif"),
        save_all=True,
        append_images=imgs[1:],
        loop=loop,
    )


def setup_session(
    session_path: Path = None,
    config_path: Path = None,
    protocol_path: Path = None,
    setup_logging: bool = True,
    ip_address: str = None,
    manufacturer: str = None,
    debug: bool = False,
) -> Tuple["FibsemMicroscope", "MicroscopeSettings"]:
    """Setup microscope session

    Args:
        session_path (Path): path to logging directory
        config_path (Path): path to config directory
        protocol_path (Path): path to protocol file

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
        settings.system.info.manufacturer = manufacturers.normalize_manufacturer(
            manufacturer
        )

    manufacturer = settings.system.info.manufacturer
    ip_address = settings.system.info.ip_address

    if manufacturer == manufacturers.THERMOFISHER:
        from fibsem.microscopes.autoscript import ThermoMicroscope

        microscope = ThermoMicroscope(settings.system)
        microscope.connect_to_microscope(ip_address=ip_address, port=7520)

    elif manufacturer == manufacturers.TESCAN:
        from fibsem.microscopes.tescan import TescanMicroscope

        microscope = TescanMicroscope(settings.system)
        microscope.connect_to_microscope(ip_address=ip_address, port=8300)
    elif manufacturer == manufacturers.ODEMIS:
        from fibsem.microscopes.odemis_microscope import OdemisThermoMicroscope

        microscope = OdemisThermoMicroscope(settings.system)

    elif manufacturer == manufacturers.DEMO:
        from fibsem.microscopes.simulator import DemoMicroscope

        microscope = DemoMicroscope(settings.system)
        microscope.connect_to_microscope(ip_address, port=7520)

    else:
        raise NotImplementedError(f"Manufacturer {manufacturer} not supported.")

    # set default image_settings path
    settings.image.path = session_path

    logging.info(f"Finished setup for session: {session}")

    return microscope, settings


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
    config = load_yaml(os.path.join(config_path))

    report_unrecognised_configuration_keys(config, source=str(config_path))

    # load protocol
    protocol = load_protocol(protocol_path)

    # create settings
    settings = MicroscopeSettings.from_dict(config, protocol=protocol)

    return settings


# Keys a configuration may carry that this version reads for migration and never
# writes back. Empty today. When a key moves house -- `stage.shuttle_pre_tilt` onto
# the holder, the beam defaults into their own block -- the old spelling is still read
# so existing files load, and is listed here so it is not reported as unrecognised.
LEGACY_CONFIGURATION_KEYS: Set[str] = set()

# Blocks accepted wholesale. `sim:` is a plain dict the simulator reads with `.get()`
# rather than a dataclass, and `protocol:` is the application's; policing either would
# invent warnings every time a backend gains a key.
OPEN_CONFIGURATION_BLOCKS = ("sim", "protocol")


@functools.lru_cache(maxsize=1)
def written_configuration_keys() -> Set[str]:
    """Every dotted path `MicroscopeSettings.to_dict` writes.

    This *is* the schema. There is no hand-written table of known keys, because a
    table is a second copy of what the writer does and the two drift: the first
    attempt at one was written from the shipped YAML files and rejected 23 keys that
    `to_dict` writes on every save. Derived from the writer, the set of keys that will
    be saved back is by construction the set of keys that are saved back.

    One level deep, blocks and their keys. A value that is itself a dict (`stage.devices`)
    is accepted wholesale under its key.
    """
    written = MicroscopeSettings.from_dict({}).to_dict()
    keys: Set[str] = set()
    for block, value in written.items():
        keys.add(block)
        if isinstance(value, dict):
            keys.update(f"{block}.{key}" for key in value)
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
    known = written_configuration_keys() | LEGACY_CONFIGURATION_KEYS
    unknown: List[str] = []
    for block, value in (config or {}).items():
        if block in OPEN_CONFIGURATION_BLOCKS:
            continue
        if block not in known:
            unknown.append(block)
            continue
        if not isinstance(value, dict):
            continue
        unknown.extend(
            f"{block}.{key}" for key in value if f"{block}.{key}" not in known
        )
    return sorted(unknown)


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


def _get_position(name: str):

    import os

    from fibsem import config as cfg
    from fibsem.structures import FibsemStagePosition

    ddict = load_yaml(fname=os.path.join(cfg.CONFIG_PATH, "positions.yaml"))
    # get position from save positions?
    for d in ddict:
        if d["name"] == name:
            return FibsemStagePosition.from_dict(d)
    return None


def _get_positions(fname: str = None) -> List[str]:

    import os

    from fibsem import config as cfg

    if fname is None:
        fname = os.path.join(cfg.CONFIG_PATH, "positions.yaml")

    ddict = load_yaml(fname=fname)

    return [d["name"] for d in ddict]


def save_positions(positions: list, path: str = None, overwrite: bool = False) -> None:
    """save the list of positions to file"""

    from fibsem import config as cfg

    # convert single position to list
    if not isinstance(positions, list):
        positions = [positions]

    # default path
    if path is None:
        path = cfg.POSITION_PATH

    # get existing positions
    pdict = []
    if not overwrite:
        pdict = load_yaml(fname=path)

    # append new positions
    for position in positions:
        pdict.append(position.to_dict())

    # save
    save_yaml(path, pdict)


# TODO: re-think this, dont like the pop ups
def _register_metadata(
    microscope: "FibsemMicroscope",
    application_software: str,
    experiment_name: str,
    experiment_id: Optional[str] = None,
) -> None:
    """Stamp user, experiment and application identity onto what ``microscope`` acquires.

    ``experiment_id`` is the stable join key and should always be supplied; it is
    optional only so a caller that has no ID yet still registers something. It used
    to be omitted entirely and ``id`` carried the name, which meant a rename silently
    broke the link from every image written before it. See FIB-446.

    Which application is running goes on ``SystemInfo``, not on the experiment
    reference. It is a property of the running system, and putting it in one place
    is what lets the reference carry identity only (FIB-445 D1, FIB-448).

    There is no application *version* to record. An application shipped inside
    fibsem has no version of its own -- passing ``fibsem.__version__`` for it only
    made two fields tautologically equal -- and ``info.fibsem_revision`` already
    pins the exact commit doing the work. See FIB-448.
    """
    from fibsem.structures import FibsemExperimentRef, FibsemUser

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
