"""A configuration written before v1 still means what it meant.

`fixtures/configuration_v0/` holds the four manufacturer configurations as they shipped
before configuration v1 (comments stripped), in the old flat layout. Sites' own files
are edits of these. The table below is what the release before v1 read from each, and
the new reader has to agree.

Where the old reader was wrong, the table has what the file says instead: it read the
detector as "Unknown" from every file (it looked for the bare key names), and it took
the "no gas" spelling `plasma_gas: None` as the *string* "None".
"""

import math
import os
import shutil

import pytest
import yaml

from fibsem import utils
from fibsem.structures.system import CONFIGURATION_VERSION

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "configuration_v0")

# fmt: off
READS_AS = {
    "tfs-aquilos2-configuration.yaml": dict(
        rotation_reference=0, rotation_180=180, shuttle_pre_tilt=35.0,
        electron_column_tilt=0, ion_column_tilt=52,
        electron_eucentric_height=7.0e-3, ion_eucentric_height=19.0e-3,
        electron_voltage=2000, electron_current=50.0e-12,
        ion_voltage=30000, ion_current=2.0e-11,
        plasma_gas=None,
        electron_detector=("ETD", "SecondaryElectrons"),
        ion_detector=("ETD", "SecondaryElectrons"),
        imaging=("ELECTRON", 150.0e-6, [1536, 1024]),
    ),
    "tfs-arctis-configuration.yaml": dict(
        rotation_reference=0, rotation_180=180, shuttle_pre_tilt=0.0,
        electron_column_tilt=0, ion_column_tilt=52,
        electron_eucentric_height=10.0e-3, ion_eucentric_height=16.5e-3,
        electron_voltage=2000, electron_current=50.0e-12,
        ion_voltage=30000, ion_current=2.0e-11,
        plasma_gas="Xenon",
        electron_detector=("ETD", "SecondaryElectrons"),
        ion_detector=("ETD", "SecondaryElectrons"),
        imaging=("ELECTRON", 150.0e-6, [1536, 1024]),
    ),
    "tfs-hydra-configuration.yaml": dict(
        rotation_reference=0, rotation_180=180, shuttle_pre_tilt=35.0,
        electron_column_tilt=0, ion_column_tilt=52,
        electron_eucentric_height=4.0e-3, ion_eucentric_height=16.5e-3,
        electron_voltage=2000, electron_current=50.0e-12,
        ion_voltage=30000, ion_current=2.0e-11,
        plasma_gas=None,
        electron_detector=("ETD", "SecondaryElectrons"),
        ion_detector=("ETD", "SecondaryElectrons"),
        imaging=("ELECTRON", 150.0e-6, [1536, 1024]),
    ),
    "tescan-configuration.yaml": dict(
        rotation_reference=180, rotation_180=0, shuttle_pre_tilt=0.0,
        electron_column_tilt=0, ion_column_tilt=55,
        electron_eucentric_height=7.0e-3, ion_eucentric_height=16.5e-3,
        electron_voltage=2000, electron_current=50.0e-12,
        ion_voltage=30000, ion_current=2.0e-11,
        plasma_gas=None,
        electron_detector=("E-T", None),
        ion_detector=("SE", None),
        imaging=("ELECTRON", 150.0e-6, [1536, 1024]),
    ),
}

# Connected on the simulator: (r, t) in degrees, as the release before v1 computed them.
ORIENTATIONS = {
    "tfs-aquilos2-configuration.yaml": {"SEM": (0, 35), "FIB": (180, 17), "MILLING": (0, 12)},
    "tfs-arctis-configuration.yaml": {"SEM": (0, 0), "FIB": (0, -128), "MILLING": (0, -23)},
    "tfs-hydra-configuration.yaml": {"SEM": (0, 35), "FIB": (180, 17), "MILLING": (0, 12)},
    "tescan-configuration.yaml": {"SEM": (180, 0), "FIB": (0, 55), "MILLING": (180, -20)},
}
# fmt: on


def _reads_as(settings) -> dict:
    system = settings.system
    stage, electron, ion, image = (
        system.stage,
        system.electron,
        system.ion,
        settings.image,
    )
    return dict(
        rotation_reference=stage.rotation_reference,
        rotation_180=stage.rotation_180,
        shuttle_pre_tilt=stage.shuttle_pre_tilt,
        electron_column_tilt=electron.column_tilt,
        ion_column_tilt=ion.column_tilt,
        electron_eucentric_height=electron.eucentric_height,
        ion_eucentric_height=ion.eucentric_height,
        electron_voltage=electron.beam.voltage,
        electron_current=electron.beam.beam_current,
        ion_voltage=ion.beam.voltage,
        ion_current=ion.beam.beam_current,
        plasma_gas=ion.plasma_gas,
        electron_detector=(electron.detector.type, electron.detector.mode),
        ion_detector=(ion.detector.type, ion.detector.mode),
        imaging=(image.beam_type.name, image.hfw, list(image.resolution)),
    )


def _load(path):
    return utils.load_microscope_configuration(str(path))


def _copy(filename, tmp_path):
    """The fixture copied out, since loading an old file saves it back upgraded."""
    path = tmp_path / filename
    shutil.copy(os.path.join(FIXTURES, filename), path)
    return path


@pytest.mark.parametrize("filename", sorted(READS_AS))
def test_an_old_file_reads_as_it_did(filename, tmp_path):
    settings = _load(_copy(filename, tmp_path))

    assert _reads_as(settings) == READS_AS[filename]


@pytest.mark.parametrize("filename", sorted(READS_AS))
def test_an_old_file_is_saved_upgraded_and_kept_as_it_was(filename, tmp_path):
    path = _copy(filename, tmp_path)
    _load(path)

    assert utils.load_yaml(path)["version"] == CONFIGURATION_VERSION
    kept = utils.configuration_backup_path(path, before=1)
    with open(os.path.join(FIXTURES, filename)) as f:
        assert kept.read_text() == f.read()
    assert _reads_as(_load(path)) == READS_AS[filename]


@pytest.mark.parametrize("filename", sorted(READS_AS))
def test_an_old_file_saved_by_this_version_still_reads_the_same(filename, tmp_path):
    saved = tmp_path / "saved" / filename
    saved.parent.mkdir()
    saved.write_text(
        yaml.safe_dump(utils._plain(_load(_copy(filename, tmp_path)).to_dict()))
    )

    assert _reads_as(_load(saved)) == READS_AS[filename]


@pytest.mark.parametrize("filename", sorted(ORIENTATIONS))
def test_an_old_file_gives_the_same_orientations(filename, tmp_path, monkeypatch):
    """On the simulator: no real instrument, and no site holder file."""
    from fibsem.services import sample_handling

    monkeypatch.setattr(
        sample_handling,
        "SAMPLE_HOLDER_CONFIGURATION_PATH",
        str(tmp_path / "absent.yaml"),
    )
    config = utils.load_yaml(os.path.join(FIXTURES, filename))
    config["info"] = dict(config["info"], manufacturer="Demo", ip_address="localhost")
    if "arctis" in filename:
        # The simulator's stand-in for the compustage the real backend detects.
        config["sim"] = {"is_compustage": True}
    path = tmp_path / filename
    path.write_text(yaml.safe_dump(config))

    microscope, _ = utils.setup_session(
        session_path=str(tmp_path), config_path=str(path), setup_logging=False
    )
    try:
        for orientation, (r, t) in ORIENTATIONS[filename].items():
            position = microscope.get_orientation(orientation)
            assert math.degrees(position.r) == pytest.approx(r, abs=1e-6), orientation
            assert math.degrees(position.t) == pytest.approx(t, abs=1e-6), orientation
    finally:
        microscope.disconnect()
