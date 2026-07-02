import dataclasses
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml

import fibsem

METADATA_VERSION = "v3"

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

def add_configuration(configuration_name: str, path: str):
    """Add a new configuration to the user configurations file."""
    if configuration_name in USER_CONFIGURATIONS:
        raise ValueError(f"Configuration name '{configuration_name}' already exists.")

    USER_CONFIGURATIONS[configuration_name] = {"path": path}
    USER_CONFIGURATIONS_YAML["configurations"] = USER_CONFIGURATIONS
    with open(USER_CONFIGURATIONS_PATH, "w") as f:
        yaml.dump(USER_CONFIGURATIONS_YAML, f)


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

@dataclass
class DisplayPreferences:
    sound_enabled: bool = False
    toasts_enabled: bool = False
    border_enabled: bool = True
    dev_mode: bool = False

@dataclass
class FeatureFlags:
    lamella_position_on_live_view: bool = False
    viewer_movement_events: bool = False
    coincidence_milling_enabled: bool = False
    sample_holder_widget: bool = False
    scheduled_tasks: bool = False

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
class UserPreferences:
    display: DisplayPreferences = field(default_factory=DisplayPreferences)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    movement: MovementPreferences = field(default_factory=MovementPreferences)
    experiment: ExperimentPreferences = field(default_factory=ExperimentPreferences)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "UserPreferences":
        """Reconstruct from a dict, handling both nested and legacy flat formats."""
        if any(k in d for k in ("display", "features", "movement", "experiment")):
            return cls(
                display=_sub_from_dict(DisplayPreferences, d.get("display", {})),
                features=_sub_from_dict(FeatureFlags, d.get("features", {})),
                movement=_sub_from_dict(MovementPreferences, d.get("movement", {})),
                experiment=_sub_from_dict(ExperimentPreferences, d.get("experiment", {})),
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
class RecentExperimentInfo:
    """Display information for a recently used experiment."""
    path: str  # path to the experiment.yaml file
    name: str
    created_at: float = 0.0
    num_lamella: int = 0
    exists: bool = True
    available: bool = True  # False if the file is missing or could not be read


def _peek_experiment_yaml(experiment_yaml_path: str) -> RecentExperimentInfo:
    """Read minimal display info from an experiment.yaml without fully loading it.

    Falls back to the parent directory name if the file is missing or unreadable.
    A file that exists but cannot be parsed is kept (exists=True) but flagged
    as unavailable so the UI can show it as such rather than silently pruning it.
    """
    fallback_name = os.path.basename(os.path.dirname(experiment_yaml_path)) or experiment_yaml_path
    if not os.path.exists(experiment_yaml_path):
        return RecentExperimentInfo(
            path=experiment_yaml_path, name=fallback_name, exists=False, available=False
        )

    try:
        with open(experiment_yaml_path, "r") as f:
            ddict = yaml.safe_load(f) or {}
        return RecentExperimentInfo(
            path=experiment_yaml_path,
            name=ddict.get("name") or fallback_name,
            created_at=ddict.get("created_at") or 0.0,
            num_lamella=len(ddict.get("positions") or []),
            exists=True,
            available=True,
        )
    except Exception as e:
        logging.warning(f"Failed to read experiment info from {experiment_yaml_path}: {e}")
        return RecentExperimentInfo(
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


def get_recent_experiments(prune_missing: bool = True) -> List[RecentExperimentInfo]:
    """Return display info for recent experiments, most-recent-first.

    Args:
        prune_missing: If True, drop paths that no longer exist on disk and
            persist the pruned list back to preferences.
    """
    prefs = load_user_preferences()
    infos = [_peek_experiment_yaml(str(p)) for p in prefs.experiment.recent_experiments]

    if prune_missing:
        kept = [info for info in infos if info.exists]
        if len(kept) != len(infos):
            prefs.experiment.recent_experiments = [info.path for info in kept]
            save_user_preferences(prefs)
        return kept

    return infos


def apply_feature_flags(prefs: UserPreferences) -> None:
    """Update module-level FEATURE_* constants from user preferences."""
    import fibsem.config as _self
    global FEATURE_LAMELLA_POSITION_ON_LIVE_VIEW_ENABLED
    global FEATURE_VIEWER_MOVEMENT_EVENTS
    global FEATURE_COINCIDENCE_MILLING_ENABLED
    global FEATURE_SAMPLE_HOLDER_WIDGET_ENABLED
    global FEATURE_SCHEDULED_TASKS_ENABLED
    f = prefs.features
    FEATURE_LAMELLA_POSITION_ON_LIVE_VIEW_ENABLED = f.lamella_position_on_live_view
    FEATURE_VIEWER_MOVEMENT_EVENTS = f.viewer_movement_events
    FEATURE_COINCIDENCE_MILLING_ENABLED = f.coincidence_milling_enabled
    FEATURE_SAMPLE_HOLDER_WIDGET_ENABLED = f.sample_holder_widget
    FEATURE_SCHEDULED_TASKS_ENABLED = f.scheduled_tasks

    # Also update the autolamella config module which re-exports these
    try:
        import fibsem.applications.autolamella.config as al_cfg
        al_cfg.FEATURE_LAMELLA_POSITION_ON_LIVE_VIEW_ENABLED = f.lamella_position_on_live_view
        al_cfg.FEATURE_COINCIDENCE_MILLING_ENABLED = f.coincidence_milling_enabled
        al_cfg.FEATURE_SCHEDULED_TASKS_ENABLED = f.scheduled_tasks
    except ImportError:
        pass


### AUTOLAMELLA APPLICATION PATHS
AUTOLAMELLA_BASE_PATH: Path = os.path.join(os.path.dirname(__file__), "applications", "autolamella")
AUTOLAMELLA_LOG_PATH: Path = os.path.join(AUTOLAMELLA_BASE_PATH, 'log')
AUTOLAMELLA_CONFIG_PATH: Path = os.path.join(AUTOLAMELLA_BASE_PATH)
AUTOLAMELLA_PROTOCOL_PATH: Path = os.path.join(AUTOLAMELLA_BASE_PATH, "protocol", "legacy", "protocol-on-grid.yaml")
AUTOLAMELLA_TASK_PROTOCOL_PATH: Path = os.path.join(AUTOLAMELLA_BASE_PATH, "protocol", "task-protocol.yaml")
AUTOLAMELLA_EXPERIMENT_NAME = "AutoLamella"

os.makedirs(AUTOLAMELLA_LOG_PATH, exist_ok=True)

####### FEATURE FLAGS
FEATURE_LAMELLA_POSITION_ON_LIVE_VIEW_ENABLED = False
FEATURE_VIEWER_MOVEMENT_EVENTS = False
FEATURE_COINCIDENCE_MILLING_ENABLED = False
FEATURE_SAMPLE_HOLDER_WIDGET_ENABLED = False
FEATURE_SCHEDULED_TASKS_ENABLED = False