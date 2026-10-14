"""What a driver tells the first-run guided setup about itself (FIB-1350).

Each driver that can be set up from the wizard names a :class:`DriverSetup` on its
``DRIVER`` record (``DriverEntry.setup``, as ``"module:attribute"``). The wizard
(``fibsem.guided_setup``) lists the registered drivers that have one, in registry order,
so the manufacturers, their software and their instruments live with their driver and
a plugin driver can offer itself the same way. Nothing here imports a driver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple


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
