"""What a driver tells the first-run guided setup about itself (FIB-1350).

Each driver that can be set up from the wizard names a :class:`DriverSetup` on its
``DRIVER`` record (``DriverEntry.setup``, as ``"module:attribute"``). The wizard
(``fibsem.guided_setup``) lists the registered drivers that have one, in registry order,
so the manufacturers, their software and their instruments live with their driver and
a plugin driver can offer itself the same way. Nothing here imports a driver.

A driver that only builds devices (a plugin's scan generator, say) names a
:class:`SetupDevice` instead, or a tuple of them: the wizard's "Other devices" step
offers each, and writes the chosen ones as ``hardware.devices`` entries. Its
:attr:`SetupDevice.keys` are also the entry's schema, which building the device checks
(``fibsem.devices.entries.build_device_entries``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple


@dataclass(frozen=True)
class SetupModel:
    """One instrument the wizard offers under a driver: a shipped configuration to
    start from."""

    key: str
    """Stable across releases: the wizard's answer is saved by it."""

    label: str
    """The card's title, without the manufacturer (that was the question before)."""

    summary: str
    filename: str
    """The configuration to start from, in ``fibsem.config.CONFIG_PATH``."""

    knows_stage: bool = True
    """Whether the configuration's stage values are right for every such instrument,
    so the stage step explains rather than asks. False where they are a site fact (a
    shuttle's pre-tilt, the reference rotation)."""

    stage_note: str = ""
    """A caution shown on the stage step, such as the frame the driver reads the stage
    in when it differs from the one the instrument's own software shows."""

    config_overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    """Top-level configuration sections merged into the written file, key by key (a
    simulator's ``sim`` block, say)."""


@dataclass(frozen=True)
class DriverSetup:
    """How the wizard offers a driver as a manufacturer, and its instruments."""

    label: str
    summary: str
    models: Tuple[SetupModel, ...]
    """Never empty: choosing the manufacturer selects its first instrument."""

    control_software: str = "the microscope control software"
    """What the instrument's own software is called ("alongside xT"). The default is
    the generic phrase, so a driver without one reads sensibly."""

    api_module: Optional[str] = None
    """The vendor package the driver imports, checked with ``find_spec``. None where
    there is nothing to install."""

    api_label: str = ""

    simulated: bool = False
    """No instrument: nothing to reach, so the connection step has nothing to ask."""

    devices: Tuple[Mapping[str, Any], ...] = ()
    """Device entries in the configuration file's layout (``name`` and the keys to
    set), merged by name into the written file's ``hardware.devices``: the values
    every instrument of this driver has, such as each column's ``column_tilt``."""


_KINDS = {bool: "true or false", int: "a whole number", float: "a number", str: "text"}


@dataclass(frozen=True)
class SetupKey:
    """One key a device entry takes beside ``name``, ``type`` and ``driver``."""

    name: str
    default: Any
    """What an entry without the key gets. Its type is the key's type: an entry
    giving another is refused (an int is taken where the default is a float)."""

    label: str = ""
    """What the wizard calls it; the name where empty."""

    doc: str = ""

    def check(self, value: Any) -> Any:
        """*value*, if it is this key's type; raises ``ValueError`` if not."""
        if self.default is None or value is None:
            return value
        wanted = type(self.default)
        if wanted is float and isinstance(value, int) and not isinstance(value, bool):
            return float(value)
        if (wanted is not bool and isinstance(value, bool)) or not isinstance(
            value, wanted
        ):
            kind = _KINDS.get(wanted, wanted.__name__)
            raise ValueError(f"'{self.name}' takes {kind}, not {value!r}")
        return value


@dataclass(frozen=True)
class SetupDevice:
    """A device a device-only driver offers in the guided setup, and the keys its
    entry takes."""

    type: str
    """The entry's ``type``, which the driver has a builder for."""

    label: str
    summary: str
    name: str
    """The entry's ``name`` the wizard writes."""

    keys: Tuple[SetupKey, ...] = ()

    role: Optional[Tuple[str, str]] = None
    """The usual binding, as (the entry that has the role, the role): ``("electron",
    "scanner")`` for a scan generator the electron beam scans with. The wizard offers
    it as its own checkbox, ticked."""

    def defaults(self) -> Dict[str, Any]:
        return {key.name: key.default for key in self.keys}

    def options(self, given: Mapping[str, Any], driver: str = "") -> Dict[str, Any]:
        """An entry's keys checked against :attr:`keys`, with the defaults filled in.
        Raises ``ValueError`` naming an unknown key, or one of the wrong type."""
        known = {key.name: key for key in self.keys}
        unknown = sorted(set(given) - set(known))
        if unknown:
            raise ValueError(
                f"unknown keys for a {driver or 'this'} {self.type}: {unknown} "
                f"(it takes {sorted(known) or 'none'})"
            )
        return {
            name: key.check(given.get(name, key.default)) for name, key in known.items()
        }

    def entry(self, driver: str, values: Optional[Mapping[str, Any]] = None) -> dict:
        """The ``hardware.devices`` entry the wizard writes, in the file's layout."""
        return {
            "name": self.name,
            "type": self.type,
            "driver": driver,
            **self.options(values or {}, driver),
        }
