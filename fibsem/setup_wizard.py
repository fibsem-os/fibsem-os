"""What the first-run setup wizard decides, and what it writes.

Deliberately Qt-free. Everything here either answers a question the wizard asks or
writes a file the application will read on its next start, and both are worth testing
on CI -- which installs ``.[test]``, not ``.[ui]``, so anything importing PyQt5 is
skipped there. The dialog that collects these answers lives in
``fibsem.ui.widgets.setup_wizard_dialog``.

The wizard's whole output is one microscope configuration file plus two registry
edits (``register_configuration`` and, optionally, ``set_default_configuration``). It
starts from a *shipped* configuration rather than building one from nothing, so every
value it does not ask about is the value the project already ships for that model.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from fibsem import config as cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Where this computer sits
# ---------------------------------------------------------------------------

# Which computer the application is installed on. This decides the address, and it is
# the one thing no configuration file can imply: a shipped file records a *model*, and
# the address is a property of the site.
LOCATION_SUPPORT = "support"
LOCATION_MICROSCOPE = "microscope"
LOCATION_OFFLINE = "offline"


@dataclass(frozen=True)
class ComputerLocation:
    key: str
    label: str
    summary: str
    icon: str
    # What to put in the address field when this is chosen. Empty means the question
    # does not apply -- there is nothing to reach.
    address: str


# Support PC first, and selected by default: it is where the software is meant to run,
# and the only one of the three whose address has to be typed rather than assumed.
COMPUTER_LOCATIONS: List[ComputerLocation] = [
    ComputerLocation(
        key=LOCATION_SUPPORT,
        label="Support PC",
        summary="A workstation on the same network as the instrument. The usual place to run from.",
        icon="mdi:laptop",
        address=cfg.DEFAULT_IP_ADDRESS,
    ),
    ComputerLocation(
        key=LOCATION_MICROSCOPE,
        label="Microscope PC",
        summary="The instrument's own computer, alongside xT or Essence.",
        icon="mdi:desktop-tower",
        address="localhost",
    ),
    ComputerLocation(
        key=LOCATION_OFFLINE,
        label="Offline PC",
        summary="No instrument reachable. Everything runs against the simulator.",
        icon="mdi:cloud-off-outline",
        address="",
    ),
]

_LOCATIONS_BY_KEY: Dict[str, ComputerLocation] = {
    location.key: location for location in COMPUTER_LOCATIONS
}


def get_location(key: str) -> ComputerLocation:
    """The location with this key, falling back to Support PC.

    A fallback rather than a KeyError because this is read while drawing a dialog:
    a stale key should give the default answer, not take the window down.
    """
    return _LOCATIONS_BY_KEY.get(key, COMPUTER_LOCATIONS[0])


# ---------------------------------------------------------------------------
# Which instrument
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MicroscopeModel:
    """A shipped configuration, offered as a starting point.

    ``knows_stage`` is what lets the wizard skip a step. It is not "is this an
    Arctis" -- every shipped model file carries both ``rotation_reference`` and
    ``shuttle_pre_tilt``, so the rule is simply *a recognised model answers the stage
    question*. Only "start blank" has to ask.

    ``is_compustage`` is carried separately because nothing in the file says so. The
    simulator reads ``sim.is_compustage`` and overwrites ``info.model`` with
    "DemoMicroscope" the moment it connects, so a simulated Arctis is only a
    compustage if that flag is written. Real backends read the capability from the
    instrument and ignore this entirely.
    """

    key: str
    label: str
    summary: str
    filename: str
    is_compustage: bool = False
    knows_stage: bool = True
    # Whether the shipped file's manufacturer is the right one. False for the blank
    # start, whose base file is a Demo configuration -- taking that at face value on a
    # support PC produces a setup that can never reach an instrument, which is a
    # failure at connect time with nothing pointing back at this choice.
    knows_manufacturer: bool = True

    @property
    def path(self) -> str:
        return os.path.join(cfg.CONFIG_PATH, self.filename)


# One icon for every model, rather than one per manufacturer. A distinct glyph per card
# reads as a distinction between the instruments, and there is none to draw: these are
# five starting points for the same dialog, differing only in which file they open. It
# also spared inventing a visual identity for someone else's product.
MODEL_ICON = "mdi:microscope"

# There is no "Demo / simulator" entry, though a simulator configuration ships: the
# Offline PC location already means Demo, and asking the same question in two steps
# invites answering it inconsistently. Odemis is absent for the opposite reason -- it
# is a different control stack, reached without the address this wizard collects.
MICROSCOPE_MODELS: List[MicroscopeModel] = [
    MicroscopeModel(
        key="tfs-arctis",
        label="ThermoFisher Arctis",
        summary="Compustage, no shuttle pre-tilt",
        filename="tfs-arctis-configuration.yaml",
        is_compustage=True,
    ),
    MicroscopeModel(
        key="tfs-hydra",
        label="ThermoFisher Hydra",
        summary="Plasma FIB, 35° pre-tilted shuttle",
        filename="tfs-hydra-configuration.yaml",
    ),
    MicroscopeModel(
        key="tfs-aquilos2",
        label="ThermoFisher Aquilos 2",
        summary="Cryo-FIB, 35° pre-tilted shuttle",
        filename="tfs-aquilos2-configuration.yaml",
    ),
    MicroscopeModel(
        key="tescan",
        label="Tescan",
        summary="Presets rather than beam currents",
        filename="tescan-configuration.yaml",
    ),
    MicroscopeModel(
        key="other",
        label="Other / start blank",
        summary="Begin from the default configuration and enter the stage settings.",
        filename="microscope-configuration.yaml",
        knows_stage=False,
        knows_manufacturer=False,
    ),
]

# Offered when the model is not recognised. Odemis is absent for the same reason it is
# absent above: it is reached differently, and this wizard does not collect what it
# would need.
AVAILABLE_MANUFACTURERS: List[str] = list(cfg.AVAILABLE_MANUFACTURERS)

_MODELS_BY_KEY: Dict[str, MicroscopeModel] = {
    model.key: model for model in MICROSCOPE_MODELS
}


def get_model(key: str) -> MicroscopeModel:
    """The model with this key, falling back to the first offered."""
    return _MODELS_BY_KEY.get(key, MICROSCOPE_MODELS[0])


# ---------------------------------------------------------------------------
# The answers
# ---------------------------------------------------------------------------


@dataclass
class SetupChoices:
    """Everything the wizard collects, and nothing else.

    ``rotation_reference`` and ``shuttle_pre_tilt`` are Optional because None is a
    real answer here: it means *the wizard did not ask*, so the shipped file's own
    values stand. That matters more than it sounds -- ``tfs-arctis`` ships
    ``rotation_reference: 0`` with ``rotation_180: 0``, because a compustage has no
    180-degree rotation to reach, and re-deriving the pair would silently give it one.
    """

    model_key: str = MICROSCOPE_MODELS[0].key
    location_key: str = LOCATION_SUPPORT
    address: str = cfg.DEFAULT_IP_ADDRESS
    # None means the wizard did not ask, so the shipped file's manufacturer stands.
    # Only the blank start asks; offline overrides whatever this says.
    manufacturer: Optional[str] = None
    rotation_reference: Optional[float] = None
    shuttle_pre_tilt: Optional[float] = None
    # Where the configuration this wizard writes is saved. Empty means the shipped
    # configuration folder, which is also where the model files are read from.
    #
    # Asked separately from the experiment directory because they are separate
    # questions, and a wizard that asks only about experiments while silently writing
    # a configuration somewhere else invites the two being answered as one.
    configuration_directory: str = ""
    experiment_directory: str = ""
    protocol_path: str = ""
    name: str = ""
    set_as_default: bool = True

    @property
    def model(self) -> MicroscopeModel:
        return get_model(self.model_key)

    @property
    def location(self) -> ComputerLocation:
        return get_location(self.location_key)

    @property
    def is_offline(self) -> bool:
        return self.location_key == LOCATION_OFFLINE

    @property
    def resolved_configuration_directory(self) -> str:
        """Where the configuration will actually be written.

        Resolved rather than left implicit so the review screen can name the path. A
        wizard whose entire output is one file should say where that file is going.
        """
        return self.configuration_directory or cfg.CONFIG_PATH

    @property
    def needs_stage_step(self) -> bool:
        """Whether the stage question is still open.

        Being offline is not a reason to skip it -- a simulated instrument still has a
        pre-tilt, and a run set up offline should behave like the one it stands in for.
        Only a recognised model answers it.
        """
        return not self.model.knows_stage

    @property
    def needs_manufacturer(self) -> bool:
        """Whether the manufacturer is still an open question.

        Offline closes it -- the simulator is the manufacturer -- so the blank start
        only has to ask when something real is meant to answer.
        """
        return not self.model.knows_manufacturer and not self.is_offline


def default_address(location_key: str) -> str:
    """The address to offer for a location.

    From the location, not from the chosen model's file. Two shipped files
    (``tfs-arctis``, ``tescan``) record ``localhost``, which describes a computer
    rather than an instrument -- seeding a support PC from either would prefill an
    address that can only ever work somewhere else.
    """
    return get_location(location_key).address


# ---------------------------------------------------------------------------
# What gets written
# ---------------------------------------------------------------------------

# Anything that is not a letter or a digit becomes a dash, so a name a person typed
# becomes a filename a person can still recognise.
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def configuration_slug(name: str) -> str:
    """A filename stem for a configuration name.

    Derived rather than asked, because the name is the thing with meaning: it is what
    ``user-configurations.yaml`` keys on and what the connection tab lists. Asking for
    both invites them to disagree.
    """
    slug = _SLUG_STRIP.sub("-", (name or "").strip().lower()).strip("-")
    return slug or "microscope-configuration"


def configuration_path(name: str, directory: Optional[str] = None) -> str:
    """Where a configuration of this name would be written, without taking the name.

    Collisions get a numeric suffix rather than overwriting: the file already there
    belongs to a configuration someone may be using.
    """
    directory = directory or cfg.CONFIG_PATH
    stem = configuration_slug(name)
    path = os.path.join(directory, f"{stem}.yaml")
    index = 2
    while os.path.exists(path):
        path = os.path.join(directory, f"{stem}-{index}.yaml")
        index += 1
    return path


def build_configuration(choices: SetupChoices) -> dict:
    """The configuration dictionary these answers produce.

    Starts from the shipped file for the chosen model and overwrites only what was
    asked. Everything else -- beam eucentric heights, column tilts, detector defaults,
    the subsystem enables -- is the project's answer for that model, which is a better
    answer than anything a first-run wizard could collect.
    """
    from fibsem import utils

    model = choices.model
    config = utils.load_yaml(model.path)

    info = config.setdefault("info", {})
    if choices.name:
        info["name"] = choices.name
    info["ip_address"] = choices.address
    # Recorded so the connection tab can name the instrument before anything has
    # connected. Both real backends overwrite it from the instrument on connect, and
    # the simulator replaces it with "DemoMicroscope", so it is a starting value
    # rather than a claim.
    info["model"] = model.label
    if choices.manufacturer:
        info["manufacturer"] = choices.manufacturer
        # The blank start's base file is a Demo configuration, so its column tilts are
        # Thermo's. Column tilt is geometry -- every projection between the two beams
        # runs through it -- and a wrong one mis-places patterns without ever looking
        # like a configuration problem, so it follows the manufacturer rather than the
        # file that happened to be the starting point.
        defaults = cfg.DEFAULT_CONFIGURATION_VALUES.get(choices.manufacturer)
        if defaults:
            config.setdefault("ion", {})["column_tilt"] = defaults["ion-column-tilt"]
            config.setdefault("electron", {})["column_tilt"] = defaults[
                "electron-column-tilt"
            ]
    if choices.is_offline:
        # Last, and unconditional: offline means the simulator whatever else was said.
        info["manufacturer"] = "Demo"
        # Only meaningful to the simulator, and only written when the simulator is
        # what will read it. On a Thermo or Tescan configuration this key would be
        # inert noise in a file people read by hand.
        config.setdefault("sim", {})["is_compustage"] = model.is_compustage

    stage = config.setdefault("stage", {})
    if choices.rotation_reference is not None:
        reference = float(choices.rotation_reference)
        stage["rotation_reference"] = reference
        # The derivation the file's own comment documents. Reached only when the user
        # supplied the reference, i.e. for a model the project does not ship -- a
        # recognised model keeps its shipped pair, which for a compustage is not this.
        stage["rotation_180"] = (reference + 180) % 360
    if choices.shuttle_pre_tilt is not None:
        stage["shuttle_pre_tilt"] = float(choices.shuttle_pre_tilt)

    return config


@dataclass
class SetupResult:
    """What the wizard did, in the order a person would want it reported."""

    path: str
    configuration_name: str
    is_default: bool = False
    preferences_saved: bool = False
    warnings: List[str] = field(default_factory=list)


def apply_setup(choices: SetupChoices) -> SetupResult:
    """Write the configuration, register it, and record the folder preferences.

    Registration is the step that makes a configuration *selectable*; a file written
    and not registered is a file nobody can choose. It runs before the preferences
    write for the same reason -- if the second half fails, the configuration still
    exists and is still usable.

    The destination comes from the choices rather than a parameter, so there is one
    way to say where this goes. Anywhere on disk works: ``register_configuration``
    stores an absolute path and ``load_configuration`` loads from it, which is what
    the import-from-file flow on the connection tab has always relied on.
    """
    from fibsem import utils

    directory = choices.resolved_configuration_directory
    os.makedirs(directory, exist_ok=True)

    config = build_configuration(choices)
    path = configuration_path(choices.name, directory)
    utils.save_yaml(path, config)

    # register_configuration never refuses -- a taken name gets a numeric suffix --
    # so the name it returns is the one to report, not the one that was asked for.
    name = cfg.register_configuration(
        path=path, configuration_name=choices.name or None
    )
    result = SetupResult(path=path, configuration_name=name)

    if choices.set_as_default:
        cfg.set_default_configuration(name)
        result.is_default = True

    result.preferences_saved = _save_folder_preferences(choices, result.warnings)
    return result


def _save_folder_preferences(choices: SetupChoices, warnings: List[str]) -> bool:
    """Record the folder answers.

    Only the values; what stops the offer reappearing is the registration above, which
    is what :func:`is_first_run` reads. So a failure here loses two paths and nothing
    else -- it cannot leave the wizard offering itself forever.

    Never fatal. A configuration that saved and preferences that did not is a partly
    finished setup; raising here would report the whole thing as failed.
    """
    try:
        preferences = cfg.load_user_preferences()
        if choices.experiment_directory:
            preferences.experiment.default_experiment_directory = (
                choices.experiment_directory
            )
        if choices.protocol_path:
            preferences.experiment.default_protocol_path = choices.protocol_path
        cfg.save_user_preferences(preferences)
        return os.path.exists(cfg.USER_PREFERENCES_PATH)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Could not save the folder preferences: {e}")
        warnings.append(f"Folder preferences were not saved: {e}")
        return False


# ---------------------------------------------------------------------------
# Is this a fresh install?
# ---------------------------------------------------------------------------


def is_first_run() -> bool:
    """Whether a microscope configuration has ever been registered on this machine.

    The absence of ``user-configurations.yaml``. It is gitignored, so no install ships
    one, and nothing writes it at startup -- ``config.py`` falls back to an in-memory
    default without touching the disk. Every writer is someone importing a
    configuration, setting a default, or finishing this wizard. So its absence means
    this machine has never had a microscope set up, which is the question being asked.

    **Not** the absence of ``user-preferences.yaml``, which is what this used to be and
    was wrong in a way no test caught: the feature flag that gates the offer lives in
    that file, so turning the flag on created it, and the offer could never appear. Any
    signal a preference write can extinguish is the wrong signal for "nothing has been
    configured yet".
    """
    return not os.path.exists(cfg.USER_CONFIGURATIONS_PATH)


def is_offer_dismissed() -> bool:
    """Whether the offer was waved away.

    Separate from :func:`is_first_run` because they answer different questions --
    "has anything been configured" and "does this person want to be asked" -- and
    conflating them is what broke the first attempt.
    """
    try:
        return bool(cfg.load_user_preferences().display.setup_wizard_dismissed)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Could not read the setup wizard dismissal: {e}")
        return False


def dismiss_first_run() -> None:
    """Record that the offer was seen and declined."""
    try:
        preferences = cfg.load_user_preferences()
        preferences.display.setup_wizard_dismissed = True
        cfg.save_user_preferences(preferences)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Could not record the setup wizard dismissal: {e}")
