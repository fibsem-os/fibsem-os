import datetime
import glob

import logging
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple, Optional


import requests
import yaml
from packaging import version
from PIL import Image

from fibsem import config as cfg
from fibsem.structures import (
    BeamType,
    FibsemImage,
    MicroscopeSettings,
)
if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope


def current_timestamp():
    """Returns current time in a specific string format

    Returns:
        String: Current time
    """
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%I-%M-%S%p") #PM/AM doesnt work?


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
        return now.strftime("%H-%M-%S")
    return now.strftime("%Y-%m-%d-%I-%M-%S%p")

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

def format_value(val: float, unit: Optional[str] = None, precision: int = 2) -> str:
    """Format a numerical value as a string with nearest SI unit.
    Args:
        val: The value to format.
        unit (str, optional): The unit of the value. Defaults to None.
        precision (int, optional): The number of decimal places to include. Defaults to 2.
    Returns:
        str: The formatted value with the appropriate SI prefix.
    """
    if val < 1e-9:
        scale = 1e12
        prefix = "p"
    elif val < 1e-6:
        scale = 1e9
        prefix = "n"
    elif val < 1e-3:
        scale = 1e6
        prefix = "u"
    elif val < 1:
        scale = 1e3
        prefix = "m"
    elif val < 1e3:
        scale = 1
        prefix = ""
    elif val < 1e6:
        scale = 1e-3
        prefix = "k"
    elif val < 1e9:
        scale = 1e-6
        prefix = "M"
    elif val < 1e12:
        scale = 1e-9
        prefix = "G"
    else:
        scale = 1e-12
        prefix = "T"

    if unit is None:
        unit = ""
    return f"{val*scale:.{precision}f} {prefix}{unit}"

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
def configure_logging(path: Path = "", log_filename="logfile", log_level=logging.DEBUG, _DEBUG: bool = False):
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

VALID_THERMO_FISHER = ["Thermo", "Thermo Fisher Scientific", "Thermo Fisher Scientific"]
VALID_TESCAN = ["Tescan", "TESCAN" ]

def setup_session(
    session_path: Path = None,
    config_path: Path = None,
    protocol_path: Path = None,
    setup_logging: bool = True,
    ip_address: str = None,
    manufacturer: str = None,
    debug: bool = False,
) -> Tuple['FibsemMicroscope', 'MicroscopeSettings']:
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
    session = f'{settings.protocol.get("name", "openfibsem")}_{current_timestamp()}'
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
    import fibsem.microscope as fibsem_microscope

    # cheap overloading
    if ip_address:
        settings.system.info.ip_address = ip_address
    
    if manufacturer:
        settings.system.info.manufacturer = manufacturer

    manufacturer = settings.system.info.manufacturer
    ip_address = settings.system.info.ip_address

    if manufacturer in VALID_THERMO_FISHER:
        microscope = fibsem_microscope.ThermoMicroscope(settings.system)
        microscope.connect_to_microscope(
            ip_address=ip_address, port=7520
        )

    elif manufacturer in VALID_TESCAN:
        from fibsem.microscopes.tescan import TescanMicroscope
        microscope = TescanMicroscope(settings.system)
        microscope.connect_to_microscope(
            ip_address=ip_address, port=8300
        )
    elif manufacturer in ["Odemis"]:
        from fibsem.microscopes.odemis_microscope import OdemisMicroscope
        microscope = OdemisMicroscope(settings.system)

    elif manufacturer == "Demo":
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

    # load protocol
    protocol = load_protocol(protocol_path)

    # create settings
    settings = MicroscopeSettings.from_dict(config, protocol=protocol)

    return settings

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

    #protocol = _format_dictionary(protocol)

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
def _register_metadata(microscope: 'FibsemMicroscope',
                       application_software: str,
                       application_software_version: str,
                       experiment_name: str,
                       experiment_method: str) -> None:
    import fibsem
    from fibsem.structures import FibsemExperiment, FibsemUser

    user = FibsemUser.from_environment()

    experiment = FibsemExperiment(
        id = experiment_name,
        method=experiment_method,
        application=application_software, 
        fibsem_version=fibsem.__version__,
        application_version=application_software_version,
    )
    microscope.user = user
    microscope.experiment = experiment


def get_pypi_versions(package_name: str = "fibsem") -> List[str]:
    """Get all available versions from PyPI for the specified package.
    
    Args:
        package_name: Name of the package to check. Defaults to "fibsem".
        
    Returns:
        List of available versions sorted by version number (latest first).
        Returns empty list if unable to fetch versions.
    """
    try:
        url = f'https://pypi.org/pypi/{package_name}/json'
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        versions = list(data['releases'].keys())
        # Filter out pre-releases and dev versions for cleaner output
        stable_versions = [v for v in versions if not any(pre in v.lower() for pre in ['a', 'b', 'rc', 'dev'])]
        return sorted(stable_versions, key=version.parse, reverse=True)
    except Exception as e:
        logging.warning(f'Error fetching PyPI data for {package_name}: {e}')
        return []


def check_for_updates(package_name: str = "fibsem") -> dict:
    """Check if a newer version of the package is available on PyPI.
    
    Args:
        package_name: Name of the package to check. Defaults to "fibsem".
        
    Returns:
        Dictionary containing:
        - 'current_version': Currently installed version
        - 'latest_version': Latest version on PyPI (None if unable to fetch)
        - 'update_available': Boolean indicating if update is available
        - 'status': Text description of the status
    """
    import fibsem
    
    result = {
        'current_version': fibsem.__version__,
        'latest_version': None,
        'update_available': False,
        'status': 'Unknown'
    }
    
    # Get PyPI versions
    pypi_versions = get_pypi_versions(package_name)
    if pypi_versions:
        result['latest_version'] = pypi_versions[0]
    
    # Compare versions
    if result['latest_version']:
        try:
            current_ver = version.parse(result['current_version'])
            latest_ver = version.parse(result['latest_version'])
            
            if current_ver < latest_ver:
                result['update_available'] = True
                result['status'] = f'Newer version available: {result["latest_version"]} (you have {result["current_version"]})'
            elif current_ver == latest_ver:
                result['status'] = f'You have the latest version: {result["current_version"]}'
            else:
                result['status'] = f'You have a newer version than PyPI: {result["current_version"]} > {result["latest_version"]}'
        except Exception as e:
            result['status'] = f'Error comparing versions: {e}'
    else:
        result['status'] = f'Could not fetch PyPI versions for {package_name}'
    
    return result
