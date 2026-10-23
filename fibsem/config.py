import copy
import dataclasses
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

import fibsem
import fibsem.util.numpy_yaml  # noqa: F401  numpy scalars dump as numbers
from fibsem.util.timestamps import to_datetime
from fibsem.workflows.saved_format import saved_app

# Documentation for a human reading a file, not a parsing switch -- from_dict does not
# branch on it, and additive changes are detected from field presence instead (FIB-445
# D3). Bumped to v4 when FibsemExperimentRef gained `name` and `id` became the UUID;
# to v5 when it dropped its duplicates of the application/version fields, which
# SystemInfo owns, and when `application_version` went entirely -- it was null in every
# file ever written (FIB-448). Those keys are still present in v4-and-earlier files.
# v6 replaced the embedded `system` (a whole SystemSettings, 1683 bytes) with
# `system_info` and `hardware_geometry` (FIB-481). from_dict recovers both from a v5-
# and-earlier `system` blob, so older files still load -- including their compustage
# flag, which up to v5 had to be inferred from the model name.
# v7 dropped `autogamma` from the recorded image settings along with the feature
# itself (FIB-505). Older files still carry the key; it is ignored.
# v8 added `workflow` -- which item and which task an image was acquired for
# (FIB-466). Absent in earlier files, and absent in any image acquired outside a
# workflow, which is a real answer rather than a missing one.
# v9 added `hardware_geometry.poses` -- the pose the stage declared for each
# orientation name (FIB-1101). Absent in earlier files, where
# `FibsemHardwareGeometry.declared_poses` rebuilds them from `rotation_180` and
# `is_compustage`.
# v10 added `hardware_geometry.stage_frame` -- the frame the stage position is in,
# "fibsem" or a driver's name for its instrument's own frame (FIB-1114). Absent in
# earlier files, whose positions are in the frame their driver reported then.
# v11 added `acquisition_datetime` -- when the image was acquired, as ISO 8601 with
# the acquiring machine's UTC offset (FIB-1190). Absent in earlier files, whose time
# is `microscope_state.timestamp`: a POSIX float, or AutoScript's naive
# `%m/%d/%Y %H:%M:%S` string on a ThermoFisher image. Read either through
# `fibsem.util.timestamps.acquisition_datetime_of`.
# v12 added `roles` -- for each role the acquiring beam had filled (its `scanner`),
# the device in it: name, class, driver, its parameters' last known values and its
# facts. Absent in earlier files, and empty for a beam with no role filled: the
# image came from the vendor's own scan.
METADATA_VERSION = "v12"
# What an unversioned file is. Absent means written before versioning existed, i.e.
# older than v1 -- not "current", which is what defaulting to METADATA_VERSION claimed.
UNVERSIONED_METADATA = "v0"


REFERENCE_HFW_WIDE = 2750e-6
REFERENCE_HFW_LOW = 900e-6
REFERENCE_HFW_MEDIUM = 400e-6
REFERENCE_HFW_HIGH = 150e-6
REFERENCE_HFW_SUPER = 80e-6
REFERENCE_HFW_ULTRA = 50e-6

# The resolution the overview widgets default to; the driver snaps it to one its beam
# lists. A beam's own resolutions are its `resolution` choices
# (`fibsem.devices.beam.STANDARD_RESOLUTIONS` for a driver that reports none).
DEFAULT_STANDARD_RESOLUTION_LIST = [1536, 1024]

# The field of view drawn around each marked position, in metres: the Overview and
# Positions canvases, the grid report and the markers picture. A fixed size rather
# than the current imaging settings: the box says how much sample a lamella occupies,
# which does not change when somebody adjusts the HFW for the next overview -- and a
# box that resized itself under a settings change would look like the positions had
# moved. 100 um at the standard resolution's aspect, so it is the frame a normal
# image is.
POSITION_FOV_WIDTH = 100e-6
POSITION_FOV_HEIGHT = POSITION_FOV_WIDTH * (
    DEFAULT_STANDARD_RESOLUTION_LIST[1] / DEFAULT_STANDARD_RESOLUTION_LIST[0]
)


BASE_PATH = os.path.dirname(fibsem.__path__[0])
CONFIG_PATH = os.path.join(BASE_PATH, "fibsem", "config")
LOG_PATH = os.path.join(BASE_PATH, "fibsem", "log")
DATA_PATH = os.path.join(LOG_PATH, "data")
DATA_ML_PATH: str = os.path.join(DATA_PATH, "ml")
DATA_CC_PATH: str = os.path.join(DATA_PATH, "crosscorrelation")
# Saved positions are session state (`fibsem.saved_positions`); these two files are
# where they lived before, read once to import them and never written.
POSITION_PATH = os.path.join(CONFIG_PATH, "saved-positions.yaml")
LEGACY_POSITIONS_PATH = os.path.join(CONFIG_PATH, "positions.yaml")
USER_PREFERENCES_PATH = os.path.join(CONFIG_PATH, "user-preferences.yaml")
MODELS_PATH = os.path.join(BASE_PATH, "fibsem", "segmentation", "models")
MICROSCOPE_CONFIGURATION_PATH = os.path.join(
    CONFIG_PATH, "microscope-configuration.yaml"
)
# fluorescence settings persistence
FM_RECENT_CHANNELS_PATH = os.path.join(
    CONFIG_PATH, "fm-recent-channels.yaml"
)  # recently-used channels
SAMPLE_HOLDER_CONFIGURATION_PATH = os.path.join(CONFIG_PATH, "sample-holder.yaml")
# Which grid is in which holder slot right now. Kept apart from the calibration
# above because it changes every session while the calibration changes once per
# holder; on an autoloader the hardware keeps this itself and the file is unused.
SAMPLE_HOLDER_OCCUPANCY_PATH = os.path.join(CONFIG_PATH, "sample-holder-occupancy.yaml")

# Alignment reference image filename
REFERENCE_FILENAME = "alignment_reference"


os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(DATA_ML_PATH, exist_ok=True)
os.makedirs(DATA_CC_PATH, exist_ok=True)


def load_yaml(fname):
    """
    Load a YAML file and return its contents as a dictionary.

    Args:
        fname (str): The path to the YAML file to be loaded.

    Returns:
        dict: A dictionary containing the contents of the YAML file.

    Raises:
        IOError: If the file cannot be opened or read.
        yaml.YAMLError: If the file is not valid YAML.
    """
    with open(fname) as f:
        config = yaml.safe_load(f)

    return config


DEFAULT_IP_ADDRESS = "192.168.0.1"


# user configurations -> move to fibsem.db eventually
DEFAULT_USER_CONFIGURATION_YAML: dict = {
    "configurations": {"default-configuration": {"path": None}},
    "default": "default-configuration",
}
USER_CONFIGURATIONS_PATH = os.path.join(CONFIG_PATH, "user-configurations.yaml")
if os.path.exists(USER_CONFIGURATIONS_PATH):
    USER_CONFIGURATIONS_YAML = load_yaml(USER_CONFIGURATIONS_PATH)
else:
    USER_CONFIGURATIONS_YAML = DEFAULT_USER_CONFIGURATION_YAML
USER_CONFIGURATIONS = USER_CONFIGURATIONS_YAML["configurations"]
DEFAULT_CONFIGURATION_NAME = USER_CONFIGURATIONS_YAML["default"]
DEFAULT_CONFIGURATION_PATH = USER_CONFIGURATIONS[DEFAULT_CONFIGURATION_NAME]["path"]


if DEFAULT_CONFIGURATION_PATH is None:
    USER_CONFIGURATIONS[DEFAULT_CONFIGURATION_NAME]["path"] = (
        MICROSCOPE_CONFIGURATION_PATH
    )
    DEFAULT_CONFIGURATION_PATH = MICROSCOPE_CONFIGURATION_PATH

if not os.path.exists(DEFAULT_CONFIGURATION_PATH):
    DEFAULT_CONFIGURATION_NAME = "default-configuration"
    USER_CONFIGURATIONS[DEFAULT_CONFIGURATION_NAME]["path"] = (
        MICROSCOPE_CONFIGURATION_PATH
    )
    DEFAULT_CONFIGURATION_PATH = MICROSCOPE_CONFIGURATION_PATH


def _is_same_path(path: Optional[str], other: Optional[str]) -> bool:
    """Compare two configuration paths, either of which may be unset."""
    if path is None or other is None:
        return False
    return os.path.normpath(path) == os.path.normpath(other)


def add_configuration(configuration_name: str, path: str):
    """Add a new configuration to the user configurations file."""
    if configuration_name in USER_CONFIGURATIONS:
        raise ValueError(f"Configuration name '{configuration_name}' already exists.")

    USER_CONFIGURATIONS[configuration_name] = {"path": path}
    USER_CONFIGURATIONS_YAML["configurations"] = USER_CONFIGURATIONS
    with open(USER_CONFIGURATIONS_PATH, "w") as f:
        yaml.dump(USER_CONFIGURATIONS_YAML, f)


def register_configuration(path: str, configuration_name: Optional[str] = None) -> str:
    """Register a configuration file, and return the name it was registered under.

    Registering is the only thing that makes a configuration selectable, so unlike
    add_configuration this never refuses: a name already taken by a different file
    is given a numeric suffix instead of raising.
    """
    if configuration_name is None:
        configuration_name = os.path.splitext(os.path.basename(path))[0]

    name, index = configuration_name, 2
    while name in USER_CONFIGURATIONS:
        if _is_same_path(USER_CONFIGURATIONS[name].get("path"), path):
            return name  # already registered
        name = f"{configuration_name} ({index})"
        index += 1

    add_configuration(configuration_name=name, path=path)
    return name


def remove_configuration(configuration_name: str):
    """Remove a configuration from the user configurations file."""
    if configuration_name not in USER_CONFIGURATIONS:
        raise ValueError(f"Configuration name '{configuration_name}' does not exist.")

    del USER_CONFIGURATIONS[configuration_name]
    USER_CONFIGURATIONS_YAML["configurations"] = USER_CONFIGURATIONS
    with open(USER_CONFIGURATIONS_PATH, "w") as f:
        yaml.dump(USER_CONFIGURATIONS_YAML, f)


def set_default_configuration(configuration_name: str):
    """Set the default configuration in the user configurations file."""
    if configuration_name not in USER_CONFIGURATIONS:
        raise ValueError(f"Configuration name '{configuration_name}' does not exist.")

    USER_CONFIGURATIONS_YAML["default"] = configuration_name
    with open(USER_CONFIGURATIONS_PATH, "w") as f:
        yaml.dump(USER_CONFIGURATIONS_YAML, f)


# machine learning
HUGGINFACE_REPO = "patrickcleeve/autolamella"
DEFAULT_CHECKPOINT = "autolamella-mega-20240107.pt"


# ---------------------------------------------------------------------------
# User Preferences
# ---------------------------------------------------------------------------

# The lamella strip's three arrangements, densest last. Strings rather than an Enum
# because they are persisted straight into the yaml preferences file: safe_dump raises
# RepresenterError on an Enum, and save_user_preferences swallows that in a bare
# `except Exception: logging.warning` -- so the failure would reach the user as "my
# preferences do not save", with nothing pointing at the cause.
#
# Here rather than beside the widget that draws them so the preferences dialog can offer
# the choice without fibsem.ui importing AutoLamella, which is the dependency direction
# FIB-553/554/555 removed.
# Named for density rather than for what they draw, and ordered roomiest first, the
# way Discord and Gmail order theirs. "Cozy" is the roomy one everywhere it appears in
# a UI, so it cannot be the name of the tightest layout here -- someone reaching for it
# expecting breathing room would get the densest thing on offer.
#
# The stored value is the label lowercased, deliberately: a preferences file that says
# `compact` and a dialog that says "Standard" are a support conversation waiting to
# happen.
MODE_COZY = "cozy"  # large thumbnail tile
MODE_STANDARD = "standard"  # small thumbnail beside the name
MODE_COMPACT = "compact"  # one line, no thumbnail
CARD_MODES = (MODE_COZY, MODE_STANDARD, MODE_COMPACT)
_CARD_MODE_LABELS = {
    MODE_COZY: "Cozy",
    MODE_STANDARD: "Standard",
    MODE_COMPACT: "Compact",
}


def card_mode_label(mode: str) -> str:
    """Human-readable name for a card mode, for the preferences dialog."""
    return _CARD_MODE_LABELS.get(mode, mode)


@dataclass
class SharedPreferences:
    """What both apps follow: the Qt window, and the server for the web app."""

    # Which fields the bar under each view shows, by view kind ("SEM", "FIB", "FM"),
    # in order (FIB-1186). Only a kind someone has customised has an entry: an absent
    # kind takes the bar's defaults, so a later change to those defaults still reaches
    # everyone who never chose -- which a stored default would not, since every save
    # writes every key.
    info_bar_fields: Dict[str, List[str]] = field(default_factory=dict)
    default_experiment_directory: str = ""
    default_protocol_path: str = ""
    # Pre-filled into a new experiment's metadata.
    user: str = ""
    project: str = ""
    organisation: str = ""


@dataclass
class QtPreferences:
    """What only the Qt window reads."""

    sound_enabled: bool = False
    # No `toasts_enabled` here: toasts are how the application talks, not a nicety
    # sitting beside sound. It was a preference defaulting to False, which meant 165
    # call sites emitted feedback that reached nowhere at all. Removed rather than
    # flipped (FIB-781); older files still carry the key and `_sub_from_dict` drops it.
    border_enabled: bool = True
    # Only errors, warnings and a run's end pop up as toasts; the rest go to the
    # notification bell. Not named `toasts_enabled`: older files still carry that
    # key as false.
    toast_problems_only: bool = False
    dev_mode: bool = False
    # How the lamella strip draws each card: "cozy" (large thumbnail), "standard"
    # (small thumbnail beside the name) or "compact" (no thumbnail, one line).
    #
    # A plain string rather than an Enum on purpose: preferences are written with
    # yaml.safe_dump, which raises RepresenterError on an Enum -- and the write is
    # wrapped in a bare `except Exception: logging.warning`, so the failure would
    # surface as "my preferences do not save" with nothing pointing at the cause.
    lamella_card_mode: str = MODE_COZY
    # Retake images after the operator moves the stage from the Qt stage controls.
    acquire_sem_after_stage_movement: bool = True
    acquire_fib_after_stage_movement: bool = True
    # Pre-filled into Help -> Report an Issue.
    contact_email: str = ""


# Maximum number of recent experiments to remember for quick-select
MAX_RECENT_EXPERIMENTS = 10
# Their section in the per-account session state.
RECENT_EXPERIMENTS = "recent_experiments"


@dataclass
class AppServerPreferences:
    """The App Server: the HTTP API inside AutoLamella that the agent, the
    dashboard and ``FibsemClient`` reach (FIB-1035).

    Off by default and fail-closed (CONTRIBUTING's network rule): a missing or
    unreadable section reads as these defaults, so the server stays off. When on,
    it starts when a microscope connects, listens on localhost only and requires
    a token. Monitoring -- every read of the session -- is always served while
    it runs. What the agent may do is its level, which is never stored here: it
    resets to Watch at every launch.
    """

    enabled: bool = False
    # Whether a new session serves the microscope: every route under
    # /microscope, local devices or a Device Server's, reads included. Off by
    # default: a read of the instrument is hardware access, not monitoring.
    # (Was `devices`, never released; a saved `devices` is ignored.)
    microscope: bool = False
    # Allow an agent on this PC that asks to connect without the operator's
    # click (fibsem.server.connections). Off by default.
    allow_local_agents: bool = False


@dataclass
class InstallPreferences:
    """About the program on this PC rather than how it looks."""

    # May the application make outbound network calls at start-up? Opt-in: these
    # run on instrument PCs, sometimes on institutional networks where an
    # unannounced connection is a compliance question. Skipped entirely for source
    # installs regardless of this setting (see fibsem.update_check.is_enabled).
    update_check_enabled: bool = False
    app_server: AppServerPreferences = field(default_factory=AppServerPreferences)


# Keys an older file kept under its sections (display, movement, experiment,
# reporting, app_server) and where they live now. Anything not listed was retired,
# and a retired name is never reused for a new key: an older file still carries it,
# and every save writes every key (FIB-781). Retired so far: every `features.*` flag
# (sample_holder_widget, bug_report_enabled, scripts_enabled, grid_workflow,
# agent_server_enabled, napari_overview_tab, overview_canvas_tab,
# proposer_reviewer_workflow_enabled, coincidence_milling_enabled, connection_chip),
# `display.toasts_enabled`, `experiment.last_experiment_path`,
# `reporting.crash_reporting_enabled`, `reporting.sentry_dsn`, `app_server.devices`
# and `hooks`.
_OLD_SECTIONS = {
    "display": {
        "sound_enabled": "qt",
        "border_enabled": "qt",
        "toast_problems_only": "qt",
        "dev_mode": "qt",
        "lamella_card_mode": "qt",
        "info_bar_fields": "shared",
    },
    "movement": {
        "acquire_sem_after_stage_movement": "qt",
        "acquire_fib_after_stage_movement": "qt",
    },
    "experiment": {
        "default_experiment_directory": "shared",
        "default_protocol_path": "shared",
        "user": "shared",
        "project": "shared",
        "organisation": "shared",
    },
    "reporting": {
        "contact_email": "qt",
        "update_check_enabled": "install",
    },
}

# What an older file kept that the application remembers rather than the person
# chooses. It moves to the per-account session state (`user_state`) on first load.
_OLD_STATE = {
    ("experiment", "recent_experiments"): "recent_experiments",
    ("display", "guided_setup_dismissed"): "guided_setup_dismissed",
    ("display", "history_filter"): "history_filter",
}


@dataclass
class UserPreferences:
    shared: SharedPreferences = field(default_factory=SharedPreferences)
    qt: QtPreferences = field(default_factory=QtPreferences)
    install: InstallPreferences = field(default_factory=InstallPreferences)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "UserPreferences":
        """Reconstruct from a dict: this shape, the older sectioned one, or the
        oldest flat one."""
        sections: Dict[str, dict] = {"shared": {}, "qt": {}, "install": {}}
        for old, keys in _OLD_SECTIONS.items():
            for key, value in (d.get(old) or {}).items():
                if key in keys:
                    sections[keys[key]][key] = value
        app_server = dict(d.get("app_server") or {})
        # Legacy flat format
        if "acquire_sem_after_stage_movement" in d:
            sections["qt"]["acquire_sem_after_stage_movement"] = d[
                "acquire_sem_after_stage_movement"
            ]
        if "acquire_fib_after_stage_movement" in d:
            sections["qt"]["acquire_fib_after_stage_movement"] = d[
                "acquire_fib_after_stage_movement"
            ]
        if "experiment_directory" in d:
            sections["shared"]["default_experiment_directory"] = d[
                "experiment_directory"
            ]
        # This shape wins over an older one where a file has both.
        for name in sections:
            sections[name].update(d.get(name) or {})
        app_server.update(sections["install"].pop("app_server", None) or {})
        install = _sub_from_dict(InstallPreferences, sections["install"])
        install.app_server = _sub_from_dict(AppServerPreferences, app_server)
        return cls(
            shared=_sub_from_dict(SharedPreferences, sections["shared"]),
            qt=_sub_from_dict(QtPreferences, sections["qt"]),
            install=install,
        )


def _old_state(d: dict) -> Dict[str, Any]:
    """The remembered values an older preferences file holds, by state section."""
    found: Dict[str, Any] = {}
    for (section, key), name in _OLD_STATE.items():
        value = (d.get(section) or {}).get(key)
        if value:
            found[name] = value
    return found


def _move_old_state(d: dict) -> None:
    """Copy what an older preferences file remembered into the per-account session
    state, where a section is not there yet. The next preferences save drops it
    from the preferences file."""
    found = _old_state(d)
    if not found:
        return
    from fibsem.session_state import user_state

    state = user_state(writable=True)
    held = state.sections()
    for name, value in found.items():
        if name not in held:
            state.save_section(name, value)


def _sub_from_dict(cls, d: dict):
    """Create a dataclass instance from a dict, ignoring unknown keys."""
    if not d:
        return cls()
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in known})


# The last file read, keyed by its modification time and size, so a call does not
# re-parse an unchanged file; an edit from outside is still seen on the next call.
_preferences_cache: Optional[tuple] = None


def _file_key(path: str) -> Optional[tuple]:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def load_user_preferences() -> UserPreferences:
    """Load persisted user preferences, returning a UserPreferences dataclass.

    Each call returns its own copy, which the caller may change."""
    global _preferences_cache
    key = _file_key(USER_PREFERENCES_PATH)
    if key is None:
        return UserPreferences()
    cached = _preferences_cache
    if cached is not None and cached[0] == (USER_PREFERENCES_PATH, key):
        return copy.deepcopy(cached[1])

    try:
        with open(USER_PREFERENCES_PATH) as f:
            loaded = yaml.safe_load(f) or {}
        preferences = UserPreferences.from_dict(loaded)
    except Exception as e:
        logging.warning(
            f"Failed to load user preferences from {USER_PREFERENCES_PATH}: {e}"
        )
        return UserPreferences()
    try:
        _move_old_state(loaded)
    except Exception as e:
        logging.warning(f"Could not move remembered values out of preferences: {e}")
    _preferences_cache = ((USER_PREFERENCES_PATH, key), preferences)
    return copy.deepcopy(preferences)


def save_user_preferences(preferences) -> None:
    """Persist user preferences to disk. Accepts UserPreferences or dict."""
    global _preferences_cache
    _preferences_cache = None
    try:
        os.makedirs(CONFIG_PATH, exist_ok=True)
        if isinstance(preferences, UserPreferences):
            data = preferences.to_dict()
        else:
            data = preferences  # legacy dict callers
        with open(USER_PREFERENCES_PATH, "w") as f:
            yaml.safe_dump(data, f)
    except Exception as e:
        logging.warning(
            f"Failed to save user preferences to {USER_PREFERENCES_PATH}: {e}"
        )


def update_user_preferences(
    change: Callable[[UserPreferences], None],
) -> UserPreferences:
    """Apply *change* to the preferences as they are on disk now, save, and return them.

    For a widget that owns one setting. Reading just before writing means it keeps
    whatever another window saved meanwhile, where saving a copy held since start-up
    would put the rest of that copy back over it.
    """
    preferences = load_user_preferences()
    change(preferences)
    save_user_preferences(preferences)
    return preferences


@dataclass
class ExperimentSummary:
    """What an experiment says about itself, read without loading it.

    Was ``RecentExperimentInfo``, when the recents list was the only caller.
    Enumerating a directory asks the same question of a file nobody has opened
    (FIB-457), so the name no longer claims the file is recent.

    The instrument and operator come from the ``session`` record (FIB-451) and
    are absent from anything written before it, or from an experiment no session
    has adopted. That is a real answer -- nothing is known -- so they stay None
    rather than defaulting to a string that would group with other unknowns.
    """

    path: str  # path to the experiment.yaml file
    name: str
    app: Optional[str] = None  # the application that wrote it; None if unread
    created_at: Optional[datetime] = None  # None when the file does not say
    num_lamella: int = 0
    exists: bool = True
    available: bool = True  # False if the file is missing or could not be read
    instrument_serial: Optional[str] = None
    instrument_model: Optional[str] = None
    operator: Optional[str] = None
    software_version: Optional[str] = None


def peek_experiment(experiment_yaml_path: str) -> ExperimentSummary:
    """Read an experiment.yaml's summary without fully loading it.

    Falls back to the parent directory name if the file is missing or unreadable.
    A file that exists but cannot be parsed is kept (exists=True) but flagged
    as unavailable so the caller can show it as such rather than silently
    pruning it.

    Reads only, and never raises: this runs over directories of files written by
    other versions, and one unparseable experiment must not take out the
    enumeration around it.
    """
    fallback_name = (
        os.path.basename(os.path.dirname(experiment_yaml_path)) or experiment_yaml_path
    )
    if not os.path.exists(experiment_yaml_path):
        return ExperimentSummary(
            path=experiment_yaml_path, name=fallback_name, exists=False, available=False
        )

    try:
        with open(experiment_yaml_path) as f:
            ddict = yaml.safe_load(f) or {}
        # `or {}` at each level, not `.get(k, {})`: a key present but null is how
        # an experiment no session has adopted serialises, and that has to read
        # the same as the key being absent entirely.
        session = ddict.get("session") or {}
        system = session.get("system") or {}
        user = session.get("user") or {}
        return ExperimentSummary(
            path=experiment_yaml_path,
            name=ddict.get("name") or fallback_name,
            app=saved_app(ddict),
            created_at=to_datetime(ddict.get("created_at")),
            num_lamella=len(ddict.get("positions") or []),
            exists=True,
            available=True,
            instrument_serial=system.get("serial_number"),
            instrument_model=system.get("model"),
            operator=user.get("name"),
            software_version=system.get("fibsem_version"),
        )
    except Exception as e:
        logging.warning(
            f"Failed to read experiment info from {experiment_yaml_path}: {e}"
        )
        return ExperimentSummary(
            path=experiment_yaml_path, name=fallback_name, exists=True, available=False
        )


def _recent_state(writable: bool = False):
    from fibsem.session_state import user_state

    return user_state(writable=writable)


def recent_experiment_paths() -> List[str]:
    """The ``experiment.yaml`` paths of the recent experiments, most-recent-first."""
    return [str(p) for p in _recent_state().load_section(RECENT_EXPERIMENTS, []) or []]


def record_recent_experiment(experiment_yaml_path: str) -> None:
    """Record an experiment as recently used, moving it to the front of the list
    and keeping at most ``MAX_RECENT_EXPERIMENTS``.

    Args:
        experiment_yaml_path: Path to the experiment.yaml file.
    """
    if not experiment_yaml_path:
        return

    path = os.path.normpath(str(experiment_yaml_path))
    recent = [p for p in recent_experiment_paths() if os.path.normpath(p) != path]
    recent.insert(0, path)
    _recent_state(writable=True).save_section(
        RECENT_EXPERIMENTS, recent[:MAX_RECENT_EXPERIMENTS]
    )


def get_recent_experiments(prune_missing: bool = True) -> List[ExperimentSummary]:
    """Return display info for recent experiments, most-recent-first.

    Args:
        prune_missing: If True, drop paths that no longer exist on disk and
            save the pruned list.
    """
    infos = [peek_experiment(p) for p in recent_experiment_paths()]

    if prune_missing:
        kept = [info for info in infos if info.exists]
        if len(kept) != len(infos):
            _recent_state(writable=True).save_section(
                RECENT_EXPERIMENTS, [info.path for info in kept]
            )
        return kept

    return infos


def get_last_experiment_file() -> Optional[str]:
    """Path to the ``experiment.yaml`` of the newest recent experiment still on
    disk, or None."""
    for path in recent_experiment_paths():
        if os.path.exists(path):
            return path
    return None


### AUTOLAMELLA APPLICATION PATHS
AUTOLAMELLA_BASE_PATH: Path = os.path.join(
    os.path.dirname(__file__), "applications", "autolamella"
)
AUTOLAMELLA_LOG_PATH: Path = os.path.join(AUTOLAMELLA_BASE_PATH, "log")
AUTOLAMELLA_CONFIG_PATH: Path = os.path.join(AUTOLAMELLA_BASE_PATH)
AUTOLAMELLA_PROTOCOL_PATH: Path = os.path.join(
    AUTOLAMELLA_BASE_PATH, "protocol", "legacy", "protocol-on-grid.yaml"
)
AUTOLAMELLA_TASK_PROTOCOL_PATH: Path = os.path.join(
    AUTOLAMELLA_BASE_PATH, "protocol", "task-protocol.yaml"
)
AUTOLAMELLA_EXPERIMENT_NAME = "AutoLamella"

os.makedirs(AUTOLAMELLA_LOG_PATH, exist_ok=True)
