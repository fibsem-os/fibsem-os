import dataclasses
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

import fibsem

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
METADATA_VERSION = "v8"
# What an unversioned file is. Absent means written before versioning existed, i.e.
# older than v1 -- not "current", which is what defaulting to METADATA_VERSION claimed.
UNVERSIONED_METADATA = "v0"

SUPPORTED_COORDINATE_SYSTEMS = [
    "RAW",
    "SPECIMEN",
    "STAGE",
    "Raw",
    "raw",
    "specimen",
    "Specimen",
    "Stage",
    "stage",
]

REFERENCE_HFW_WIDE = 2750e-6
REFERENCE_HFW_LOW = 900e-6
REFERENCE_HFW_MEDIUM = 400e-6
REFERENCE_HFW_HIGH = 150e-6
REFERENCE_HFW_SUPER = 80e-6
REFERENCE_HFW_ULTRA = 50e-6

REFERENCE_RES_SQUARE = [1024, 1024]
REFERENCE_RES_LOW = [768, 512]
REFERENCE_RES_MEDIUM = [1536, 1024]
REFERENCE_RES_HIGH = [3072, 2048]
REFERENCE_RES_SUPER = [6144, 4096]

# standard imaging resolutions
STANDARD_RESOLUTIONS = [
    "384x256",
    "768x512",
    "1536x1024",
    "3072x2048",
    "6144x4096",
]
SQUARE_RESOLUTIONS = [
    "256x256",
    "512x512",
    "1024x1024",
    "2048x2048",
    "4096x4096",
    "8192x8192",
]
STANDARD_RESOLUTIONS_LIST = [
    [int(x) for x in res.split("x")] for res in STANDARD_RESOLUTIONS
]
SQUARE_RESOLUTIONS_LIST = [
    [int(x) for x in res.split("x")] for res in SQUARE_RESOLUTIONS
]
AVAILABLE_RESOLUTIONS = SQUARE_RESOLUTIONS + STANDARD_RESOLUTIONS
DEFAULT_STANDARD_RESOLUTION = "1536x1024"
DEFAULT_SQUARE_RESOLUTION = "1024x1024"
SQUARE_RESOLUTIONS_ZIP = list(zip(SQUARE_RESOLUTIONS, SQUARE_RESOLUTIONS_LIST))
STANDARD_RESOLUTIONS_ZIP = list(zip(STANDARD_RESOLUTIONS, STANDARD_RESOLUTIONS_LIST))
AVAILABLE_RESOLUTIONS_ZIP = list(zip(AVAILABLE_RESOLUTIONS, [
    [int(x) for x in r.split("x")] for r in AVAILABLE_RESOLUTIONS
]))


BASE_PATH = os.path.dirname(fibsem.__path__[0])
CONFIG_PATH = os.path.join(BASE_PATH, "fibsem", "config")
LOG_PATH = os.path.join(BASE_PATH, "fibsem", "log")
DATA_PATH = os.path.join(LOG_PATH, "data")
DATA_ML_PATH: str = os.path.join(DATA_PATH, "ml")
DATA_CC_PATH: str = os.path.join(DATA_PATH, "crosscorrelation")
POSITION_PATH = os.path.join(CONFIG_PATH, "saved-positions.yaml")
USER_PREFERENCES_PATH = os.path.join(CONFIG_PATH, "user-preferences.yaml")
MODELS_PATH = os.path.join(BASE_PATH, "fibsem", "segmentation", "models")
MICROSCOPE_CONFIGURATION_PATH = os.path.join(
    CONFIG_PATH, "microscope-configuration.yaml"
)
# fluorescence settings persistence
FM_CONFIGURATION_PATH = os.path.join(CONFIG_PATH, "fm-configuration.yaml")  # working state
FM_RECENT_CHANNELS_PATH = os.path.join(CONFIG_PATH, "fm-recent-channels.yaml")  # recently-used channels
COINCIDENCE_MILLING_CONFIG_PATH = os.path.join(CONFIG_PATH, "coincidence-milling-config.yaml")  # milling default
SAMPLE_HOLDER_CONFIGURATION_PATH = os.path.join(
    CONFIG_PATH, "sample-holder.yaml"
)
DEFAULT_SAMPLE_HOLDER_CONFIGURATION_PATH = os.path.join(
    CONFIG_PATH, "default-sample-holder.yaml"
)

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
    with open(fname, "r") as f:
        config = yaml.safe_load(f)

    return config

AVAILABLE_MANUFACTURERS = ["Thermo", "Tescan", "Demo"]
DEFAULT_MANUFACTURER = "Thermo"
DEFAULT_IP_ADDRESS = "192.168.0.1"
SUPPORTED_PLASMA_GASES = ["Argon", "Oxygen", "Nitrogen", "Xenon"]

def get_default_user_config() -> dict:
    """Return the default configuration."""
    return {
        "name":                           "default-configuration",       # a descriptive name for your configuration 
        "ip_address":                     DEFAULT_IP_ADDRESS,            # the ip address of the microscope PC
        "manufacturer":                   DEFAULT_MANUFACTURER,          # the microscope manufactuer, Thermo, Tescan or Demo                       
        "rotation-reference":             0,                             # the reference rotation value (rotation when loading)  [degrees]
        "shuttle-pre-tilt":               35,                            # the pre-tilt of the shuttle                           [degrees]
        "electron-beam-eucentric-height": 7.0e-3,                        # the eucentric height of the electron beam             [metres]
        "ion-beam-eucentric-height":      16.5e-3,                       # the eucentric height of the ion beam                  [metres]
    }


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
    USER_CONFIGURATIONS[DEFAULT_CONFIGURATION_NAME][
        "path"
    ] = MICROSCOPE_CONFIGURATION_PATH
    DEFAULT_CONFIGURATION_PATH = MICROSCOPE_CONFIGURATION_PATH

if not os.path.exists(DEFAULT_CONFIGURATION_PATH):
    DEFAULT_CONFIGURATION_NAME = "default-configuration"
    USER_CONFIGURATIONS[DEFAULT_CONFIGURATION_NAME][
        "path"
    ] = MICROSCOPE_CONFIGURATION_PATH
    DEFAULT_CONFIGURATION_PATH = MICROSCOPE_CONFIGURATION_PATH
        
print(f"Default configuration {DEFAULT_CONFIGURATION_NAME}. Configuration Path: {DEFAULT_CONFIGURATION_PATH}")

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


# default configuration values
DEFAULT_CONFIGURATION_VALUES = {
    "Thermo": {
        "ion-column-tilt": 52,
        "electron-column-tilt": 0,
    },
    "Tescan": {
        "ion-column-tilt": 55,
        "electron-column-tilt": 0,
    },
    "Demo": {
        "ion-column-tilt": 52,
        "electron-column-tilt": 0,
    },
}


# machine learning
HUGGINFACE_REPO = "patrickcleeve/autolamella"
DEFAULT_CHECKPOINT = "autolamella-mega-20240107.pt"

# feature flags
APPLY_CONFIGURATION_ENABLED = True

# tescan manipulator

TESCAN_MANIPULATOR_CALIBRATION_PATH = os.path.join(CONFIG_PATH, "tescan_manipulator.yaml")

def load_tescan_manipulator_calibration() -> dict:
    """Load the tescan manipulator calibration"""
    from fibsem.utils import load_yaml
    config = load_yaml(TESCAN_MANIPULATOR_CALIBRATION_PATH)
    return config

def save_tescan_manipulator_calibration(config: dict) -> None:
    """Save the tescan manipulator calibration"""
    from fibsem.utils import save_yaml
    save_yaml(TESCAN_MANIPULATOR_CALIBRATION_PATH, config)
    return None

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
MODE_COZY = "cozy"          # large thumbnail tile
MODE_STANDARD = "standard"  # small thumbnail beside the name
MODE_COMPACT = "compact"    # one line, no thumbnail
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
class DisplayPreferences:
    sound_enabled: bool = False
    toasts_enabled: bool = False
    border_enabled: bool = True
    dev_mode: bool = False
    # How the lamella strip draws each card: "cozy" (large thumbnail), "standard"
    # (small thumbnail beside the name) or "compact" (no thumbnail, one line).
    #
    # A plain string rather than an Enum on purpose: preferences are written with
    # yaml.safe_dump, which raises RepresenterError on an Enum -- and the write is
    # wrapped in a bare `except Exception: logging.warning`, so the failure would
    # surface as "my preferences do not save" with nothing pointing at the cause.
    # Cozy, because it is what the strip drew before the other two existed: a new
    # layout is offered, never imposed on an upgrade.
    lamella_card_mode: str = MODE_COZY

@dataclass
class FeatureFlags:
    coincidence_milling_enabled: bool = False
    sample_holder_widget: bool = False
    bug_report_enabled: bool = False
    # Tools -> Scripts. A user script runs with the application's own access to the
    # microscope and none of its guard rails, so the menu is not offered to anyone
    # who has not asked for it (FIB-338).
    scripts_enabled: bool = False
    # The rebuilt FIB/SEM Overview tab, on the real-space canvas. Off while it sits
    # *beside* the napari Overview tab rather than replacing it: the two drive the same
    # instrument, and the existing tab is the one people are relying on until this has
    # had bench time (FIB-413, FIB-405).
    #
    # The only staging flag left after FIB-609, and it goes the same way the others did
    # -- deleted when the tab replaces the napari one, not kept as a preference.
    overview_canvas_tab: bool = False

@dataclass
class MovementPreferences:
    acquire_sem_after_stage_movement: bool = True
    acquire_fib_after_stage_movement: bool = True

# Maximum number of recent experiments to remember for quick-select
MAX_RECENT_EXPERIMENTS = 10


@dataclass
class ExperimentPreferences:
    default_experiment_directory: str = ""
    default_protocol_path: str = ""
    last_experiment_path: str = ""
    recent_experiments: List[str] = field(default_factory=list)
    user: str = ""
    project: str = ""
    organisation: str = ""

@dataclass
class ReportingPreferences:
    contact_email: str = ""
    crash_reporting_enabled: bool = False
    sentry_dsn: str = ""
    # Grouped with crash reporting because it is the same question: may the
    # application make outbound network calls? Opt-in for the same reason —
    # these run on instrument PCs, sometimes on institutional networks where an
    # unannounced connection at startup is a compliance question and the
    # operator would have no idea it was happening. Skipped entirely for source
    # installs regardless of this setting (see fibsem.update_check.is_enabled).
    update_check_enabled: bool = False

@dataclass
class AlignmentPreferences:
    # Validation is configured PER CONTEXT, because the cost of refusing a shift
    # differs sharply between them (measured, FIB-807):
    #
    #   workflow  catches 11/12 bad shifts, but refuses 22/96 good ones (23%)
    #   milling   catches  7/11 bad shifts, but refuses  7/85 good ones ( 8%)
    #
    # Workflow alignment aligns *to* a reference and legitimately makes ~130px
    # first-step corrections, so refusing large shifts is expensive there. Milling
    # drift correction runs from an already-aligned position where a large shift
    # really is suspect. One global setting cannot serve both.
    #
    # Each is a key of fibsem.alignment.VALIDATION_MODES. "none" is the historical
    # path: the measured shift is always applied. "verified" requires an independent
    # band-passed second opinion to agree within 3px, and applies no shift when they
    # disagree. Both use the SAME estimator by design -- validation never silently
    # changes what measures the shift; swapping the estimator is a deliberate code
    # change, not a config toggle. Not imported from fibsem.alignment here -- config
    # is imported by almost everything, and alignment imports config.
    validation_workflow: str = "none"
    validation_milling: str = "none"
    # What to do when an alignment cannot be trusted. False (default) keeps the
    # historical behaviour: log a warning and carry on, milling at whatever position
    # the stage happens to be in. True aborts the task instead, so the lamella is
    # recorded as failed rather than milled at an unverified position.
    #
    # Only has an effect where validation is on; with validation off nothing ever
    # reports failure, so this is inert under "none".
    abort_on_failure: bool = False
    # Bound every applied shift to this fraction of the half-ROI. Unlike validation
    # this CLIPS rather than refuses, so a correct large shift is not lost -- the
    # step applies the bounded move and the next step walks the remainder. That is
    # why it is on by default and validation is not.
    #
    # At 0.8 it flags 11 steps of which 10 are genuinely bad (10:1), against the
    # 1:1 of an absolute pixel threshold -- magnitude alone does not separate good
    # from bad. 0 disables it.
    clip_fraction: float = 0.8


@dataclass
class UserPreferences:
    display: DisplayPreferences = field(default_factory=DisplayPreferences)
    alignment: AlignmentPreferences = field(default_factory=AlignmentPreferences)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    movement: MovementPreferences = field(default_factory=MovementPreferences)
    experiment: ExperimentPreferences = field(default_factory=ExperimentPreferences)
    reporting: ReportingPreferences = field(default_factory=ReportingPreferences)
    # Serialized hooks, as HookManager.to_dict()["hooks"] produces them. Held as plain
    # dicts rather than Hook objects on purpose: to_dict() below is dataclasses.asdict,
    # which would recurse into a Hook, strip the "type" key that reconstructs it, and
    # then hit NotificationHook._notify -- a callable, which yaml.safe_dump refuses.
    # Keeping them opaque also keeps this module free of any hook imports.
    #
    # None and [] are different, and the difference is the whole feature: None means
    # never configured, so the application uses its defaults, while [] means the user
    # turned everything off. Conflating them either resurrects hooks someone deleted or
    # silences a fresh install. See FIB-497.
    hooks: Optional[List[dict]] = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "UserPreferences":
        """Reconstruct from a dict, handling both nested and legacy flat formats."""
        if any(k in d for k in ("display", "features", "movement", "experiment",
                                "reporting", "alignment", "hooks")):
            return cls(
                display=_sub_from_dict(DisplayPreferences, d.get("display", {})),
                alignment=_sub_from_dict(AlignmentPreferences, d.get("alignment", {})),
                features=_sub_from_dict(FeatureFlags, d.get("features", {})),
                movement=_sub_from_dict(MovementPreferences, d.get("movement", {})),
                experiment=_sub_from_dict(ExperimentPreferences, d.get("experiment", {})),
                reporting=_sub_from_dict(ReportingPreferences, d.get("reporting", {})),
                # .get() rather than a default, so an absent key stays None
                hooks=d.get("hooks"),
            )
        # Legacy flat format
        prefs = cls()
        if "acquire_sem_after_stage_movement" in d:
            prefs.movement.acquire_sem_after_stage_movement = d["acquire_sem_after_stage_movement"]
        if "acquire_fib_after_stage_movement" in d:
            prefs.movement.acquire_fib_after_stage_movement = d["acquire_fib_after_stage_movement"]
        if "experiment_directory" in d:
            prefs.experiment.default_experiment_directory = d["experiment_directory"]
        if "last_experiment_path" in d:
            prefs.experiment.last_experiment_path = d["last_experiment_path"]
        return prefs


def _sub_from_dict(cls, d: dict):
    """Create a dataclass instance from a dict, ignoring unknown keys."""
    if not d:
        return cls()
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in known})


def load_user_preferences() -> UserPreferences:
    """Load persisted user preferences, returning a UserPreferences dataclass."""
    if not os.path.exists(USER_PREFERENCES_PATH):
        return UserPreferences()

    try:
        with open(USER_PREFERENCES_PATH, "r") as f:
            loaded = yaml.safe_load(f) or {}
        return UserPreferences.from_dict(loaded)
    except Exception as e:
        logging.warning(f"Failed to load user preferences from {USER_PREFERENCES_PATH}: {e}")
        return UserPreferences()


def save_user_preferences(preferences) -> None:
    """Persist user preferences to disk. Accepts UserPreferences or dict."""
    try:
        os.makedirs(CONFIG_PATH, exist_ok=True)
        if isinstance(preferences, UserPreferences):
            data = preferences.to_dict()
        else:
            data = preferences  # legacy dict callers
        with open(USER_PREFERENCES_PATH, "w") as f:
            yaml.safe_dump(data, f)
    except Exception as e:
        logging.warning(f"Failed to save user preferences to {USER_PREFERENCES_PATH}: {e}")


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
    created_at: float = 0.0
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
    fallback_name = os.path.basename(os.path.dirname(experiment_yaml_path)) or experiment_yaml_path
    if not os.path.exists(experiment_yaml_path):
        return ExperimentSummary(
            path=experiment_yaml_path, name=fallback_name, exists=False, available=False
        )

    try:
        with open(experiment_yaml_path, "r") as f:
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
            created_at=ddict.get("created_at") or 0.0,
            num_lamella=len(ddict.get("positions") or []),
            exists=True,
            available=True,
            instrument_serial=system.get("serial_number"),
            instrument_model=system.get("model"),
            operator=user.get("name"),
            software_version=system.get("fibsem_version"),
        )
    except Exception as e:
        logging.warning(f"Failed to read experiment info from {experiment_yaml_path}: {e}")
        return ExperimentSummary(
            path=experiment_yaml_path, name=fallback_name, exists=True, available=False
        )


def add_recent_experiment(prefs: UserPreferences, experiment_yaml_path: str) -> None:
    """Update ``prefs.experiment.recent_experiments`` in place (does not save).

    Moves the path to the front, de-duplicates, and truncates to
    ``MAX_RECENT_EXPERIMENTS``. Use this when the caller already holds a prefs
    object it intends to save, to avoid a redundant load/save cycle.

    Args:
        prefs: The preferences object to mutate.
        experiment_yaml_path: Path to the experiment.yaml file.
    """
    if not experiment_yaml_path:
        return

    path = os.path.normpath(str(experiment_yaml_path))
    recent = [p for p in prefs.experiment.recent_experiments if os.path.normpath(str(p)) != path]
    recent.insert(0, path)
    prefs.experiment.recent_experiments = recent[:MAX_RECENT_EXPERIMENTS]


def record_recent_experiment(experiment_yaml_path: str) -> None:
    """Record an experiment as recently used, moving it to the front of the list.

    Args:
        experiment_yaml_path: Path to the experiment.yaml file.
    """
    if not experiment_yaml_path:
        return

    prefs = load_user_preferences()
    add_recent_experiment(prefs, experiment_yaml_path)
    save_user_preferences(prefs)


def get_recent_experiments(prune_missing: bool = True) -> List[ExperimentSummary]:
    """Return display info for recent experiments, most-recent-first.

    Args:
        prune_missing: If True, drop paths that no longer exist on disk and
            persist the pruned list back to preferences.
    """
    prefs = load_user_preferences()
    infos = [peek_experiment(str(p)) for p in prefs.experiment.recent_experiments]

    if prune_missing:
        kept = [info for info in infos if info.exists]
        if len(kept) != len(infos):
            prefs.experiment.recent_experiments = [info.path for info in kept]
            save_user_preferences(prefs)
        return kept

    return infos


def get_last_experiment_file() -> Optional[str]:
    """Path to the ``experiment.yaml`` of the last experiment opened, or None.

    ``last_experiment_path`` is the experiment *directory* -- that is how both the
    create and the load dialog record it -- so callers do not have to know where
    the file sits inside it.

    Falls back to the newest recent entry still on disk, because the two lists go
    stale differently: recents are pruned when their file disappears and this one
    never is, so the last experiment opened is routinely a scratch experiment that
    has since been deleted. The next-newest one is the useful answer there, not
    nothing.
    """
    prefs = load_user_preferences()

    last_path = str(prefs.experiment.last_experiment_path or "")
    if last_path:
        candidate = os.path.join(last_path, "experiment.yaml")
        if os.path.exists(candidate):
            return candidate

    for path in prefs.experiment.recent_experiments:
        if os.path.exists(str(path)):
            return str(path)

    return None


def apply_feature_flags(prefs: UserPreferences) -> None:
    """Update module-level FEATURE_* constants from user preferences.

    Down to one. The others either went away with their features or, in the case of
    coincidence milling, turned out to have no reader -- that one gates on
    `prefs.features.coincidence_milling_enabled` directly, which is the simpler
    thing to do when the caller already holds a preferences object.

    The global survives here because its caller does not: `FibsemMovementWidget`
    reads it while building a widget, and the alternative is re-reading
    user-preferences.yaml from disk on every construction.
    """
    global FEATURE_SAMPLE_HOLDER_WIDGET_ENABLED
    f = prefs.features
    FEATURE_SAMPLE_HOLDER_WIDGET_ENABLED = f.sample_holder_widget


### AUTOLAMELLA APPLICATION PATHS
AUTOLAMELLA_BASE_PATH: Path = os.path.join(os.path.dirname(__file__), "applications", "autolamella")
AUTOLAMELLA_LOG_PATH: Path = os.path.join(AUTOLAMELLA_BASE_PATH, 'log')
AUTOLAMELLA_CONFIG_PATH: Path = os.path.join(AUTOLAMELLA_BASE_PATH)
AUTOLAMELLA_PROTOCOL_PATH: Path = os.path.join(AUTOLAMELLA_BASE_PATH, "protocol", "legacy", "protocol-on-grid.yaml")
AUTOLAMELLA_TASK_PROTOCOL_PATH: Path = os.path.join(AUTOLAMELLA_BASE_PATH, "protocol", "task-protocol.yaml")
AUTOLAMELLA_EXPERIMENT_NAME = "AutoLamella"

os.makedirs(AUTOLAMELLA_LOG_PATH, exist_ok=True)

####### FEATURE FLAGS
FEATURE_SAMPLE_HOLDER_WIDGET_ENABLED = False
