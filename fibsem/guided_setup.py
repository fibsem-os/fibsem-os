"""What the first-run guided setup decides, and what it writes.

Deliberately Qt-free. Everything here either answers a question the wizard asks or
writes a file the application will read on its next start, and both are worth testing
on CI -- which installs ``.[test]``, not ``.[ui]``, so anything importing PyQt5 is
skipped there. The dialog that collects these answers lives in
``fibsem.ui.widgets.guided_setup_dialog``.

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
from typing import Any, Dict, List, Mapping, Optional, Tuple

from fibsem import config as cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Where this computer sits
# ---------------------------------------------------------------------------

# Which computer the application is installed on. This decides the address, and it is
# the one thing no configuration file can imply: a shipped file records a *model*, and
# the address is a property of the site.
#
# There is no "Offline PC" here. Wanting the simulator is a statement about what you
# are connecting to, not about where you are sitting, and it is asked as a
# manufacturer instead. Having it in both places meant two ways to say the same thing
# that could disagree -- an Offline PC with an instrument's manufacturer chosen silently
# became a Demo, discarding the answer to the question just above it.
LOCATION_SUPPORT = "support"
LOCATION_MICROSCOPE = "microscope"


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
# and the only one whose address has to be typed rather than assumed.
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
        # The manufacturer's own name for its software is substituted in by the
        # dialog; this is what the card says before one has been chosen.
        summary="The instrument's own computer, alongside {control_software}.",
        icon="mdi:desktop-tower",
        address="localhost",
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


def location_summary(location: ComputerLocation, manufacturer: "Manufacturer") -> str:
    """A location's card text, with the chosen manufacturer's software named.

    Every card goes through here rather than only the one with a placeholder, so a
    placeholder added to another later is substituted rather than printed as literal
    braces.
    """
    return location.summary.format(control_software=manufacturer.control_software)


# ---------------------------------------------------------------------------
# Which manufacturer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Manufacturer:
    """Who made the instrument, asked before which instrument it is.

    Asked first because it is the question people can always answer, and because it
    settles two things the instrument list alone could not. It fixes
    ``info.manufacturer``, which decides the column tilts and therefore every
    projection between the two beams; and it decides which vendor API has to be
    installed for the connection to work at all.

    One per driver the registry offers for setup (``DriverEntry.setup``), so what each
    one says lives with its driver. The simulator is offered here rather than as a
    computer location: as far as the configuration is concerned it *is* a manufacturer,
    and asking once means the answer cannot contradict itself.
    """

    key: str
    """The driver's registry name, which is also what ``info.manufacturer`` says."""

    label: str
    summary: str
    # What this manufacturer's own control software is called, for the places the
    # wizard has to refer to it. The default is the generic phrase, so a manufacturer
    # added without one still reads sensibly rather than naming somebody else's
    # product.
    control_software: str = "the microscope control software"
    # The module the backend imports. None where there is nothing to install. Checked
    # with ``find_spec`` rather than imported: the question is whether the package is
    # present, and importing a vendor SDK to find out is slow and has side effects.
    api_module: Optional[str] = None
    api_label: str = ""
    simulated: bool = False
    # The driver's device entries merged into the written file (``DriverSetup.devices``).
    devices: Tuple[Mapping[str, Any], ...] = ()

    @property
    def config_value(self) -> str:
        return self.key

    @property
    def is_simulator(self) -> bool:
        return self.simulated


# One icon for every manufacturer, for the reason given at MODEL_ICON below: a distinct
# glyph per card reads as a distinction being drawn, and inventing a visual identity for
# someone else's company is not this dialog's job.
MANUFACTURER_ICON = "mdi:domain"


@dataclass(frozen=True)
class MicroscopeModel:
    """A shipped configuration, offered as a starting point.

    ``knows_stage`` is what makes the stage step read-only. Nothing is ever skipped:
    the step is shown either way, because a filled-in page that cannot be edited says
    the answer is already known, and an absent one says nothing at all.

    Every shipped file carries ``rotation_reference`` and ``shuttle_pre_tilt``, but
    carrying a value is not the same as the value being right for the instrument in
    front of you: on a pre-tilted-shuttle system the pre-tilt belongs to whichever
    shuttle is fitted, and the reference rotation to how the sample is loaded at that
    site. Both are site facts, so the shipped numbers are a starting point and the step
    still has to be confirmed. A stage that reaches the other side of the sample by
    tilting rather than rotating has neither, by design, so its configuration can
    answer outright.

    ``config_overrides`` are what the driver writes beyond the shipped file, such as a
    simulator's ``sim`` block, which nothing else in the file implies.
    """

    key: str
    manufacturer_key: str
    label: str
    summary: str
    filename: str
    knows_stage: bool = True
    stage_note: str = ""
    config_overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @property
    def path(self) -> str:
        return os.path.join(cfg.CONFIG_PATH, self.filename)

    @property
    def manufacturer(self) -> Manufacturer:
        return get_manufacturer(self.manufacturer_key)


# One icon for every model, rather than one per manufacturer. A distinct glyph per card
# reads as a distinction between the instruments, and there is none to draw: these are
# starting points for the same dialog, differing only in which file they open. It also
# spares inventing a visual identity for someone else's product.
MODEL_ICON = "mdi:microscope"


def _setups():
    from fibsem.drivers.registry import driver_setups

    return driver_setups()


def manufacturers() -> List[Manufacturer]:
    """The manufacturers offered, one per driver with a setup, in registry order.

    A driver without one (Odemis, reached without the address this wizard collects) is
    not offered.
    """
    return [
        Manufacturer(
            key=name,
            label=setup.label,
            summary=setup.summary,
            control_software=setup.control_software,
            api_module=setup.api_module,
            api_label=setup.api_label,
            simulated=setup.simulated,
            devices=setup.devices,
        )
        for name, setup in _setups()
    ]


def microscope_models() -> List[MicroscopeModel]:
    """Every instrument offered, grouped by manufacturer in the order shown."""
    return [
        MicroscopeModel(
            key=model.key,
            manufacturer_key=name,
            label=model.label,
            summary=model.summary,
            filename=model.filename,
            knows_stage=model.knows_stage,
            stage_note=model.stage_note,
            config_overrides=model.config_overrides,
        )
        for name, setup in _setups()
        for model in setup.models
    ]


def get_manufacturer(key: str) -> Manufacturer:
    """The manufacturer with this key, falling back to the first offered.

    A fallback rather than a KeyError for the same reason as ``get_location``: this is
    read while drawing a dialog.
    """
    offered = manufacturers()
    return next((m for m in offered if m.key == key), offered[0])


def api_is_installed(manufacturer: Manufacturer) -> Optional[bool]:
    """Whether this manufacturer's API can be imported here.

    None where the question does not apply -- the simulator needs nothing. Otherwise
    True or False, so the dialog can say what it found rather than warning in general
    terms about software the user may already have.

    Never raises: a broken package on the path is a reason to say "not found", not a
    reason to take the wizard down before it has asked anything.
    """
    if not manufacturer.api_module:
        return None
    from importlib.util import find_spec

    try:
        return find_spec(manufacturer.api_module) is not None
    except Exception as e:  # pragma: no cover - a package that breaks on inspection
        logger.warning(f"Could not check for {manufacturer.api_label}: {e}")
        return False


# ---------------------------------------------------------------------------
# Which instrument
# ---------------------------------------------------------------------------


def get_model(key: str) -> MicroscopeModel:
    """The model with this key, falling back to the first offered."""
    offered = microscope_models()
    return next((m for m in offered if m.key == key), offered[0])


def models_for(manufacturer_key: str) -> List[MicroscopeModel]:
    """The instruments offered under a manufacturer, in the order they are shown.

    Never empty for a real manufacturer key, and the caller relies on that: picking a
    manufacturer selects its first instrument, so an empty list would leave the step
    with nothing chosen and the Next button enabled.
    """
    return [m for m in microscope_models() if m.manufacturer_key == manufacturer_key]


def shipped_stage_values(model: MicroscopeModel) -> Dict[str, float]:
    """The stage values this model's configuration ships, for prefilling the step.

    A starting point rather than an answer. Offering them beats an empty field or a
    generic 35: someone confirming "yes, that is my shuttle" is doing something much
    easier than recalling a number, and the ones who differ are the ones who know they
    differ.

    Falls back to the field defaults rather than raising, because this is read while
    drawing a dialog.
    """
    from fibsem import utils

    try:
        config = utils.upgrade_configuration(utils.load_yaml(model.path))
        stage = utils.configuration_device(config, "stage")
        calibration = config.get("calibration") or {}
    except Exception as e:  # pragma: no cover - unreadable shipped file
        logger.warning(f"Could not read the stage values for {model.label}: {e}")
        stage, calibration = {}, {}
    return {
        "rotation_reference": float(stage.get("rotation_reference", 0.0)),
        "shuttle_pre_tilt": float(calibration.get("shuttle_pre_tilt", 0.0)),
    }


# ---------------------------------------------------------------------------
# The answers
# ---------------------------------------------------------------------------


@dataclass
class SetupChoices:
    """Everything the wizard collects, and nothing else.

    ``rotation_reference`` and ``shuttle_pre_tilt`` are Optional because None is a
    real answer here: it means *the wizard did not ask*, so the shipped file's own
    values stand -- which for a stage that tilts to the other side includes
    ``rotation: false``, the flag that says there is no 180-degree rotation to reach
    (FIB-834). Overwriting
    the reference on a model the wizard never asked about would leave that flag
    describing a stage the file no longer describes.
    """

    manufacturer_key: str = field(default_factory=lambda: manufacturers()[0].key)
    model_key: str = field(default_factory=lambda: microscope_models()[0].key)
    location_key: str = LOCATION_SUPPORT
    address: str = cfg.DEFAULT_IP_ADDRESS
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
    def manufacturer(self) -> Manufacturer:
        return get_manufacturer(self.manufacturer_key)

    @property
    def is_simulator(self) -> bool:
        """Whether this setup is for the built-in simulator rather than an instrument.

        Asked once, in the first step, and everything downstream reads it from here:
        the connection step has nothing to ask, no vendor API is needed, and the
        configuration is written with ``manufacturer: Demo``.
        """
        return self.manufacturer.is_simulator

    @property
    def resolved_configuration_directory(self) -> str:
        """Where the configuration will actually be written.

        Resolved rather than left implicit so the review screen can name the path. A
        wizard whose entire output is one file should say where that file is going.
        """
        return self.configuration_directory or cfg.CONFIG_PATH

    @property
    def stage_is_readonly(self) -> bool:
        """Whether the stage step is shown but not editable.

        True for a model whose configuration answers the stage questions outright (a
        stage that tilts to the other side), and the step is *shown* rather than skipped so
        the diagram can say why there is nothing to enter: a flat holder, no wedge, no
        rotation to measure. Skipping left that as an absence, which reads as an
        omission rather than an answer.
        """
        return self.model.knows_stage

    @property
    def stage_is_editable(self) -> bool:
        """Whether the stage step is asking rather than explaining.

        Choosing the simulator is not a reason to close the question -- a simulated
        instrument still has a pre-tilt, and a run set up against the simulator should
        behave like the one it stands in for.
        """
        return not self.stage_is_readonly

    @property
    def stage_coordinate_note(self) -> str:
        """A caution about the coordinate frame, where the driver gives one.

        A driver may read the stage in a different frame from the one the instrument's
        own software shows, so the rotation read here can differ from the number on
        its screen. That is not an error -- but someone comparing the two without
        knowing will reasonably assume it is, and "correct" a value that was right.
        Empty where the driver gives no note for the model.
        """
        return self.model.stage_note

    @property
    def connection_is_editable(self) -> bool:
        """Whether the connection step has anything to ask.

        False for the simulator, which has no address to reach and nothing to test.
        The step is still shown -- see ``stage_is_readonly`` for why an answered step
        beats an absent one.
        """
        return not self.is_simulator


def default_address(location_key: str) -> str:
    """The address to offer for a location.

    From the location, not from the chosen model's file. Some shipped files record
    ``localhost``, which describes a computer
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
    # In the current version whatever the shipped file is in, so a new site's file
    # starts with the device list.
    config = utils.upgrade_configuration(utils.load_yaml(model.path))

    manufacturer = choices.manufacturer

    info = config.setdefault("info", {})
    if choices.name:
        info["name"] = choices.name
    info["ip_address"] = choices.address
    # Recorded so the connection tab can name the instrument before anything has
    # connected. Both real backends overwrite it from the instrument on connect, and
    # the simulator replaces it with "DemoMicroscope", so it is a starting value
    # rather than a claim.
    info["model"] = model.label
    # Written unconditionally, because the manufacturer is now always asked. It agrees
    # with the shipped file for every model that has one of its own; the exception is
    # a driver's "another instrument" model, whose base is the generic (Demo)
    # configuration and would otherwise claim to be a Demo.
    info["manufacturer"] = manufacturer.config_value

    # The driver's device values (the column tilts) follow the manufacturer rather
    # than the file that happened to be the starting point. Column tilt is geometry --
    # every projection between the two beams runs through it -- and a wrong one
    # mis-places patterns without ever looking like a configuration problem. This is a
    # no-op for every shipped file, all of which already agree with their driver; it
    # exists for the generic base.
    for entry in manufacturer.devices:
        values = {k: v for k, v in entry.items() if k != "name"}
        utils.configuration_device(config, entry["name"], create=True).update(values)

    # What the driver writes for this model beyond the shipped file (a simulator's
    # ``sim`` block). Only the model's own driver reads it, so nothing is written for
    # a model that has none.
    for section, values in model.config_overrides.items():
        config.setdefault(section, {}).update(values)

    stage = utils.configuration_device(config, "stage", create=True)
    if choices.rotation_reference is not None:
        stage["rotation_reference"] = float(choices.rotation_reference)
    # Written by no one now: `rotation_180` is derived from the reference and the
    # `rotation` capability (FIB-834). Popped rather than left alone because the
    # template this dict was copied from is a shipped file that still carries the key,
    # and a stale number sitting beside the reference it no longer agrees with is
    # exactly what someone would go on to edit.
    stage.pop("rotation_180", None)
    if choices.shuttle_pre_tilt is not None:
        # Calibration, not hardware: it is the shuttle's, and it moves onto the
        # holder the first time one is named.
        config.setdefault("calibration", {})["shuttle_pre_tilt"] = float(
            choices.shuttle_pre_tilt
        )

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
        return bool(cfg.load_user_preferences().display.guided_setup_dismissed)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Could not read the guided setup dismissal: {e}")
        return False


def dismiss_first_run() -> None:
    """Record that the offer was seen and declined."""
    try:
        preferences = cfg.load_user_preferences()
        preferences.display.guided_setup_dismissed = True
        cfg.save_user_preferences(preferences)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Could not record the guided setup dismissal: {e}")
