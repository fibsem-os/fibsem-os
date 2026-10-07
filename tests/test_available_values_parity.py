"""get_available_values gives the answers it gave before it read the device choices.

Each backend that builds beam devices answers a beam key's values from the beam
parameter's choices; only the keys with no device home stay in the backend
(``_get_available_values``). The answers the backends gave before, for every key on
no beam type and on each beam, over the fake SDKs, are pinned in
``tests/fixtures/available_values_pins.json``; each must still be the same, except
the ones in ``CHANGED``, which say what they answer now and why. Thermo runs in its
own interpreter (``tests/fixtures/autoscript_available_values.py``).

Not caught by the pins, because each case is a fresh microscope: a beam key's values
are now the choices the device read when it was built (and again when a dependency
changes), not a live read on every call (``test_a_beam_key_is_answered_without_an_sdk_call``).
Nothing here has run on an instrument.

Regenerate the pins (only ever from the code they pin) with
``FIBSEM_PIN_AVAILABLE_VALUES=1 pytest tests/test_available_values_parity.py``.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import BeamType

FIXTURES = Path(__file__).parent / "fixtures"
PINS = FIXTURES / "available_values_pins.json"
REGENERATE = os.environ.get("FIBSEM_PIN_AVAILABLE_VALUES") == "1"

KEYS = [
    "current",
    "voltage",
    "plasma_gas",
    "preset",
    "detector_type",
    "detector_mode",
    "application_file",
    "scan_direction",
    "not_a_key",
]
BEAM_TYPES = (None, BeamType.ELECTRON, BeamType.ION)


def _plain(value):
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (set, frozenset)):  # no order to pin, but the type counts
        return {"set": sorted(_plain(v) for v in value)}
    if isinstance(value, (int, str, bool, type(None))):
        return value
    return repr(value)


def _answers(prefix, microscope):
    out = {}
    for beam_type in BEAM_TYPES:
        name = "None" if beam_type is None else beam_type.name
        for key in KEYS:
            try:
                answer = _plain(microscope.get_available_values(key, beam_type))
            except Exception as e:  # a raise is an answer too
                answer = f"EXC {type(e).__name__}: {e}"
            out[f"{prefix} {name} {key}"] = answer
    return out


def _thermo(tmp_path):
    out = tmp_path / "thermo.json"
    env = dict(os.environ, FIBSEM_SIM_NO_DELAY="1")
    subprocess.run(
        [
            sys.executable,
            str(FIXTURES / "autoscript_available_values.py"),
            str(out),
            json.dumps(KEYS),
        ],
        env=env,
        check=True,
    )
    return json.loads(out.read_text())


def _tescan():
    from tests.fixtures.tescan_sdk import connect

    system = utils.load_microscope_configuration(
        os.path.join(cfg.CONFIG_PATH, "tescan-configuration.yaml")
    ).system
    with pytest.MonkeyPatch.context() as monkeypatch:
        microscope, _ = connect(monkeypatch, system)
        return _answers("tescan", microscope)


def _odemis():
    import tests.test_odemis_devices as od

    stubs = od.stubs
    saved = {}
    for name in stubs.ODEMIS_MODULE_NAMES + stubs.FIBSEM_ODEMIS_MODULE_NAMES:
        if name in sys.modules:
            saved[name] = sys.modules.pop(name)
    sys.modules.pop("fibsem.devices.drivers.odemis", None)
    stubs.install_odemis_stubs()
    try:
        from fibsem.microscopes.odemis_microscope import OdemisThermoMicroscope

        return _answers("odemis", od.make(OdemisThermoMicroscope))
    finally:
        stubs.remove_odemis_stubs()
        sys.modules.update(saved)


def _demo():
    from fibsem import utils as fibsem_utils

    microscope, _ = fibsem_utils.setup_session(manufacturer="Demo")
    return _answers("demo", microscope)


# A beam key asked with no beam type has no beam device to answer it, so it now
# gets no values, as the keys with no device home already did, where before the
# backend raised or (Thermo's detector types, the Demo's tables) answered for the
# active channel.
_NO_BEAM = {
    "demo None current": "EXC KeyError: None",
    "demo None detector_mode": ["SecondaryElectrons", "BackscatteredElectrons", "EDS"],
    "demo None detector_type": ["ETD", "TLD", "EDS"],
    "odemis None current": "EXC KeyError: None",
    "odemis None detector_type": "EXC KeyError: None",
    "odemis None voltage": "EXC KeyError: None",
    "tescan None preset": "EXC ValueError: Invalid beam type: None",
    "thermo plasma=False None detector_type": ["ETD", "TLD", "ICE"],
    "thermo plasma=False None voltage": "EXC ValueError: Unknown beam type: None",
    "thermo plasma=True None detector_type": ["ETD", "TLD", "ICE"],
    "thermo plasma=True None voltage": "EXC ValueError: Unknown beam type: None",
}
# The Odemis client gives the detector types as a set; the device lists them, in the
# set's order.
_ODEMIS_DETECTOR_TYPES = {
    "odemis ELECTRON detector_type": {"set": ["ETD", "TLD"]},
    "odemis ION detector_type": {"set": ["ETD", "ICE"]},
}
CHANGED = {**_NO_BEAM, **_ODEMIS_DETECTOR_TYPES}


@pytest.fixture(scope="module")
def answers(tmp_path_factory):
    out = {}
    out.update(_thermo(tmp_path_factory.mktemp("available_values")))
    out.update(_tescan())
    out.update(_odemis())
    out.update(_demo())
    return out


def test_every_answer_is_the_pinned_one(answers):
    if REGENERATE:
        PINS.write_text(json.dumps(answers, indent=1, sort_keys=True) + "\n")
    pinned = json.loads(PINS.read_text())
    assert sorted(answers) == sorted(pinned)
    different = {k: (pinned[k], answers[k]) for k in pinned if answers[k] != pinned[k]}
    assert sorted(different) == sorted(CHANGED)
    for key, (before, now) in different.items():
        assert before == CHANGED[key], key
        if key in _NO_BEAM:
            assert now == [], key
        else:
            assert sorted(now) == before["set"], key


def test_a_beam_key_is_answered_without_an_sdk_call(monkeypatch):
    """The values are the device's choices, read when the beam was built: asking
    again makes no instrument call, and changing the choices changes the answer."""
    from fibsem.devices.core import ParameterMetadata
    from tests.fixtures.tescan_sdk import connect

    system = utils.load_microscope_configuration(
        os.path.join(cfg.CONFIG_PATH, "tescan-configuration.yaml")
    ).system
    microscope, fake = connect(monkeypatch, system)
    param = microscope.beams[BeamType.ION].parameters["detector_type"]
    fake.log.clear()
    assert microscope.get_available_values("detector_type", BeamType.ION) == list(
        param.choices
    )
    assert fake.log == []
    param.metadata = ParameterMetadata(choices=["SE"])
    assert microscope.get_available_values("detector_type", BeamType.ION) == ["SE"]
