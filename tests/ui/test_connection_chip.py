"""The header names the instrument this session attached to, and offers the way back.

Identity, not liveness. Nothing in the application watches the link —
`connected_signal` is emitted off `bool(self.microscope)`, a Python reference, not
the connection (FIB-777) — so a chip claiming "Connected" would go on claiming it
through a pulled cable, which is exactly when someone would be reading it. What it
connected *to* is established once and does not decay.

The window cannot be constructed offscreen (napari, vispy), so the two header
handlers are borrowed onto a host carrying the real buttons. The microscope is a
real Demo session.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QPushButton  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.applications.autolamella.structures import Experiment  # noqa: E402
from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (  # noqa: E402
    CONNECTION_CHIP_MAX_WIDTH,
    AutoLamellaSingleWindowUI,
)


class _UIStub:
    def __init__(self, microscope=None, experiment=None):
        self.microscope = microscope
        self.experiment = experiment


class _HeaderHost:
    _update_connection_chip = AutoLamellaSingleWindowUI._update_connection_chip
    _update_experiment_header = AutoLamellaSingleWindowUI._update_experiment_header

    def __init__(self, chip_enabled: bool = True):
        # The tests exercise the flag on, which is the point of landing it: the code
        # has to be right before it is switched on, not after. The off branch has a
        # test of its own below.
        self._connection_chip_enabled = chip_enabled
        self.btn_connection = QPushButton()
        self.btn_connection.setMaximumWidth(CONNECTION_CHIP_MAX_WIDTH)
        self.btn_create_experiment = QPushButton("Create Experiment")
        self.btn_load_experiment = QPushButton("Load Experiment")
        self.btn_experiment_menu = QPushButton()
        self.autolamella_ui = _UIStub()

    def close(self):
        for btn in (
            self.btn_connection,
            self.btn_create_experiment,
            self.btn_load_experiment,
            self.btn_experiment_menu,
        ):
            btn.deleteLater()


@pytest.fixture
def host(qapp):
    host = _HeaderHost(chip_enabled=True)
    yield host
    host.close()


@pytest.fixture
def host_without_chip(qapp):
    """The shipping default: flag off, Connection tab still the way to connect."""
    host = _HeaderHost(chip_enabled=False)
    yield host
    host.close()


@pytest.fixture(scope="module")
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    yield microscope
    microscope.disconnect()


def test_disconnected_the_chip_asks_to_connect(host):
    """Worded as the Connection tab's button and the File menu entry are.

    Three doors to one action, each naming it differently, is three actions as
    far as a reader is concerned.
    """
    host._update_connection_chip()

    assert host.btn_connection.text() == "Connect to Microscope"


def test_connected_the_chip_names_the_instrument(host, microscope):
    host.autolamella_ui.microscope = microscope

    host._update_connection_chip()

    info = microscope.system.info
    assert info.manufacturer in host.btn_connection.text()
    assert info.ip_address in host.btn_connection.text()
    # And the rest, including the serial number, one hover away.
    assert info.serial_number in host.btn_connection.toolTip()


def test_the_chip_never_claims_the_link_is_up(host, microscope):
    """It says what it attached to, not that it is still attached.

    Nothing polls the instrument, so "Connected" would be a claim the application
    cannot back, and it would survive the cable being pulled.
    """
    host.autolamella_ui.microscope = microscope
    host._update_connection_chip()

    text = host.btn_connection.text().lower()
    assert "connected" not in text
    assert "online" not in text


def test_the_chip_stays_within_its_width(host, microscope):
    """A long manufacturer must not spend the room #490 reclaimed."""
    host.autolamella_ui.microscope = microscope
    original = microscope.system.info.manufacturer
    microscope.system.info.manufacturer = "A Very Long Instrument Manufacturer Name"
    try:
        host._update_connection_chip()
        assert host.btn_connection.sizeHint().width() <= CONNECTION_CHIP_MAX_WIDTH
        assert "…" in host.btn_connection.text()
    finally:
        microscope.system.info.manufacturer = original


def test_the_experiment_buttons_wait_for_a_microscope(host, microscope, tmp_path):
    """Neither can do anything without one, so neither is on screen without one.

    A greyed-out pair beside the Connect chip is the spent chrome #490 removed
    from the loaded state, moved to the disconnected one.
    """
    host._update_experiment_header()
    assert host.btn_create_experiment.isHidden()
    assert host.btn_load_experiment.isHidden()

    host.autolamella_ui.microscope = microscope
    host._update_experiment_header()
    assert not host.btn_create_experiment.isHidden()
    assert host.btn_create_experiment.isEnabled()
    assert host.btn_load_experiment.isEnabled()

    # And they give way to the experiment name once there is one.
    host.autolamella_ui.experiment = Experiment(path=str(tmp_path), name="loaded")
    host._update_experiment_header()
    assert host.btn_create_experiment.isHidden()
    assert not host.btn_experiment_menu.isHidden()


def test_with_the_flag_off_nothing_changes(host_without_chip, microscope):
    """The release default, and the reason this can land now.

    With no chip in the header there is nothing else there to connect from, so the
    experiment buttons keep the behaviour they have shipped with: on screen, and
    greyed out until a microscope arrives.
    """
    host = host_without_chip

    host._update_connection_chip()
    assert host.btn_connection.text() == ""

    host._update_experiment_header()
    assert not host.btn_create_experiment.isHidden()
    assert not host.btn_create_experiment.isEnabled()

    host.autolamella_ui.microscope = microscope
    host._update_experiment_header()
    assert not host.btn_create_experiment.isHidden()
    assert host.btn_create_experiment.isEnabled()
