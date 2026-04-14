import dataclasses
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

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


BASE_PATH = os.path.dirname(fibsem.__path__[0])
CONFIG_PATH = os.path.join(BASE_PATH, "fibsem", "config")
LOG_PATH = os.path.join(BASE_PATH, "fibsem", "log")
DATA_PATH = os.path.join(LOG_PATH, "data")
DATA_ML_PATH: str = os.path.join(DATA_PATH, "ml")
DATA_CC_PATH: str = os.path.join(DATA_PATH, "crosscorrelation")
POSITION_PATH = os.path.join(CONFIG_PATH, "positions.yaml")
USER_PREFERENCES_PATH = os.path.join(CONFIG_PATH, "user-preferences.yaml")
MODELS_PATH = os.path.join(BASE_PATH, "fibsem", "segmentation", "models")
MICROSCOPE_CONFIGURATION_PATH = os.path.join(
    CONFIG_PATH, "microscope-configuration.yaml"
)
SAMPLE_HOLDER_CONFIGURATION_PATH = os.path.join(
    CONFIG_PATH, "sample-holder.yaml"
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
    workflow_timeline_enabled: bool = True
    dev_mode: bool = False

@dataclass
class FeatureFlags:
    minimap_plot_widget: bool = True
    lamella_position_on_live_view: bool = False
    pose_controls: bool = False
    display_grid_center_marker: bool = False
    right_click_context_menu: bool = True
    display_point_of_interest: bool = True

@dataclass
class PathPreferences:
    default_experiment_directory: str = ""
    last_experiment_path: str = ""
    default_protocol_path: str = ""

@dataclass
class MovementPreferences:
    acquire_sem_after_stage_movement: bool = True
    acquire_fib_after_stage_movement: bool = True

@dataclass
class UserPreferences:
    display: DisplayPreferences = field(default_factory=DisplayPreferences)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    paths: PathPreferences = field(default_factory=PathPreferences)
    movement: MovementPreferences = field(default_factory=MovementPreferences)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "UserPreferences":
        """Reconstruct from a dict, handling both nested and legacy flat formats."""
        # If the dict has nested sub-dicts, reconstruct directly
        if any(k in d for k in ("display", "features", "paths", "movement")):
            return cls(
                display=_sub_from_dict(DisplayPreferences, d.get("display", {})),
                features=_sub_from_dict(FeatureFlags, d.get("features", {})),
                paths=_sub_from_dict(PathPreferences, d.get("paths", {})),
                movement=_sub_from_dict(MovementPreferences, d.get("movement", {})),
            )
        # Legacy flat format (4-key YAML from previous version)
        prefs = cls()
        if "acquire_sem_after_stage_movement" in d:
            prefs.movement.acquire_sem_after_stage_movement = d["acquire_sem_after_stage_movement"]
        if "acquire_fib_after_stage_movement" in d:
            prefs.movement.acquire_fib_after_stage_movement = d["acquire_fib_after_stage_movement"]
        if "experiment_directory" in d:
            prefs.paths.default_experiment_directory = d["experiment_directory"]
        if "last_experiment_path" in d:
            prefs.paths.last_experiment_path = d["last_experiment_path"]
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


def apply_feature_flags(prefs: UserPreferences) -> None:
    """Update module-level FEATURE_* constants from user preferences."""
    import fibsem.config as _self
    global FEATURE_MINIMAP_PLOT_WIDGET_ENABLED
    global FEATURE_LAMELLA_POSITION_ON_LIVE_VIEW_ENABLED
    global FEATURE_POSE_CONTROLS_ENABLED
    global FEATURE_DISPLAY_GRID_CENTER_MARKER
    global FEATURE_RIGHT_CLICK_CONTEXT_MENU_ENABLED
    global FEATURE_DISPLAY_POINT_OF_INTEREST_ENABLED

    f = prefs.features
    FEATURE_MINIMAP_PLOT_WIDGET_ENABLED = f.minimap_plot_widget
    FEATURE_LAMELLA_POSITION_ON_LIVE_VIEW_ENABLED = f.lamella_position_on_live_view
    FEATURE_POSE_CONTROLS_ENABLED = f.pose_controls
    FEATURE_DISPLAY_GRID_CENTER_MARKER = f.display_grid_center_marker
    FEATURE_RIGHT_CLICK_CONTEXT_MENU_ENABLED = f.right_click_context_menu
    FEATURE_DISPLAY_POINT_OF_INTEREST_ENABLED = f.display_point_of_interest

    # Also update the autolamella config module which re-exports these
    try:
        import fibsem.applications.autolamella.config as al_cfg
        al_cfg.FEATURE_MINIMAP_PLOT_WIDGET_ENABLED = f.minimap_plot_widget
        al_cfg.FEATURE_LAMELLA_POSITION_ON_LIVE_VIEW_ENABLED = f.lamella_position_on_live_view
        al_cfg.FEATURE_POSE_CONTROLS_ENABLED = f.pose_controls
        al_cfg.FEATURE_DISPLAY_GRID_CENTER_MARKER = f.display_grid_center_marker
        al_cfg.FEATURE_RIGHT_CLICK_CONTEXT_MENU_ENABLED = f.right_click_context_menu
        al_cfg.FEATURE_DISPLAY_POINT_OF_INTEREST_ENABLED = f.display_point_of_interest
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
FEATURE_MINIMAP_PLOT_WIDGET_ENABLED = True
FEATURE_LAMELLA_POSITION_ON_LIVE_VIEW_ENABLED = False
FEATURE_POSE_CONTROLS_ENABLED = False
FEATURE_DISPLAY_GRID_CENTER_MARKER = False
FEATURE_RIGHT_CLICK_CONTEXT_MENU_ENABLED = True
FEATURE_DISPLAY_POINT_OF_INTEREST_ENABLED = True
