"""The GasInjector and SputterCoater devices on the Demo driver.

Each is a device type built from a ``hardware.devices`` entry; the Arctis simulator
lists one of each. The operations and grid tasks over them, and the AutoScript
drivers, are tested where they live.
"""

import os
import threading

import pytest
import yaml

import fibsem.config as cfg
from fibsem import utils
from fibsem.devices import (
    GIS_RESOURCE,
    SPUTTER_COATER_RESOURCE,
    STAGE_RESOURCE,
    GasInjector,
    ParameterReadOnly,
    Resources,
    SputterCoater,
)
from fibsem.drivers.demo.devices import (
    DemoGasInjector,
    DemoSputterCoater,
    build_demo_gis,
    build_demo_sputter_coater,
)
from fibsem.drivers.registry import BuildContext
from fibsem.structures.system import DeviceEntry, InsertableDeviceState


def _gis(**kwargs) -> DemoGasInjector:
    return DemoGasInjector(**kwargs).connect()


def _coater(**kwargs) -> DemoSputterCoater:
    return DemoSputterCoater(**kwargs).connect()


# ---------------------------------------------------------------------------
# The gas injector
# ---------------------------------------------------------------------------


class TestGasInjector:
    def test_starts_retracted_off_and_closed(self):
        gis = _gis(gas="W")
        assert gis.gas.get_value() == "W"
        assert gis.state.get_value() is InsertableDeviceState.RETRACTED
        assert gis.heated.get_value() is False
        assert gis.opened.get_value() is False
        assert gis.temperature.get_value() == 293.0

    def test_every_parameter_is_read_only(self):
        gis = _gis()
        assert sorted(gis.parameters) == [
            "gas",
            "heated",
            "opened",
            "state",
            "temperature",
        ]
        for parameter in gis.parameters.values():
            with pytest.raises(ParameterReadOnly):
                parameter.set_value(parameter.get_value())

    def test_each_command_reads_its_change_back(self):
        gis = _gis()
        gis.state.get_value()
        seen = []
        gis.state.changed.connect(seen.append)

        assert gis.insert() is InsertableDeviceState.INSERTED
        assert gis.heater_on("Pt") is True
        assert gis.temperature.cached == 310.0
        assert gis.open() is True
        assert gis.close() is False
        assert gis.heater_off() is False
        assert gis.temperature.cached == 293.0
        assert gis.retract() is InsertableDeviceState.RETRACTED
        assert seen == [InsertableDeviceState.INSERTED, InsertableDeviceState.RETRACTED]

    def test_a_gas_given_to_the_heater_becomes_its_gas(self):
        gis = _gis(gas="Pt")
        gis.heater_on("W")
        assert gis.gas.get_value() == "W"

    def test_a_purge_opens_and_closes_the_valve_with_the_needle_out(self):
        gis = _gis()
        gis.opened.get_value()
        opened = []
        gis.opened.changed.connect(opened.append)

        assert gis.purge(5) is False

        assert gis.sim_purges == [5.0]
        assert opened == [True, False]

    def test_a_purge_is_refused_with_the_needle_in(self):
        gis = _gis()
        gis.insert()
        with pytest.raises(RuntimeError, match="retract the needle"):
            gis.purge(5)
        assert gis.sim_purges == []
        assert gis.opened.get_value() is False

    def test_a_failed_purge_wait_still_closes_the_valve(self):
        gis = _gis()

        def fail(seconds):
            raise KeyboardInterrupt

        gis._wait = fail
        with pytest.raises(KeyboardInterrupt):
            gis.purge(5)
        assert gis.sim_opened is False

    def test_commands_claim_the_gis_resource(self):
        resources = Resources(groups={GIS_RESOURCE: GIS_RESOURCE})
        gis = _gis(resources=resources)
        held = []

        def check(position):
            free = threading.Thread(
                target=lambda: held.append(
                    resources.lock(GIS_RESOURCE).acquire(timeout=0.05)
                )
            )
            free.start()
            free.join()

        gis._insert = check
        gis.insert()
        assert held == [False]

    def test_the_commands_are_listed(self):
        assert sorted(_gis().commands) == [
            "close",
            "heater_off",
            "heater_on",
            "insert",
            "open",
            "purge",
            "retract",
        ]


# ---------------------------------------------------------------------------
# The sputter coater
# ---------------------------------------------------------------------------


class TestSputterCoater:
    def test_the_current_is_set_within_the_driver_s_limits(self):
        coater = _coater(current=1e-9, current_limits=(1e-9, 2e-7))
        assert coater.current.get_value() == 1e-9

        assert coater.current.set_value(1.2e-7) == 1.2e-7
        assert coater.current.set_value(5e-6) == 2e-7  # clipped, with a warning

    def test_inserted_is_read_only(self):
        coater = _coater()
        with pytest.raises(ParameterReadOnly):
            coater.inserted.set_value(True)

    def test_a_run_coats_for_its_seconds_at_the_current(self):
        coater = _coater(current=1.2e-7)
        assert coater.run(30) is False
        assert coater.sim_runs == [(30.0, None, 1.2e-7)]

    def test_a_run_of_no_time_is_refused(self):
        with pytest.raises(ValueError, match="cannot last"):
            _coater().run(0)

    def test_a_coater_that_coats_what_is_under_it_refuses_grids(self):
        coater = _coater(coats_grids=False)
        with pytest.raises(ValueError, match="takes no grids"):
            coater.run(30, [1])
        assert coater.sim_runs == []

    def test_a_coater_that_moves_to_each_grid_takes_their_numbers(self):
        coater = _coater(coats_grids=True)
        coater.run(30, [2, 3])
        assert coater.sim_runs[-1][1] == (2, 3)
        with pytest.raises(ValueError, match="numbered from 1"):
            coater.run(30, [0])

    def test_a_run_claims_the_stage_as_well_as_the_coater(self):
        resources = Resources(
            groups={
                SPUTTER_COATER_RESOURCE: SPUTTER_COATER_RESOURCE,
                STAGE_RESOURCE: STAGE_RESOURCE,
            }
        )
        coater = _coater(resources=resources)
        held = {}

        def check(seconds, grids):
            for name in (SPUTTER_COATER_RESOURCE, STAGE_RESOURCE):
                free = threading.Thread(
                    target=lambda n=name: held.__setitem__(
                        n, resources.lock(n).acquire(timeout=0.05)
                    )
                )
                free.start()
                free.join()

        coater._run = check
        coater.run(1)
        assert held == {SPUTTER_COATER_RESOURCE: False, STAGE_RESOURCE: False}

    def test_its_facts_say_whether_it_takes_grids(self):
        assert _coater(coats_grids=True).facts()["coats_grids"] is True


# ---------------------------------------------------------------------------
# Built from entries
# ---------------------------------------------------------------------------


def test_the_builders_read_their_entry_keys():
    microscope, _ = utils.setup_session(manufacturer="Demo", setup_logging=False)
    context = BuildContext(microscope=microscope)

    gis = build_demo_gis(
        DeviceEntry.from_dict({"name": "gis_w", "type": "gis", "gas": "W"}), context
    )
    coater = build_demo_sputter_coater(
        DeviceEntry.from_dict(
            {"name": "sputter_coater", "coats_grids": True, "current": 1.2e-7}
        ),
        context,
    )

    assert isinstance(gis, GasInjector) and gis.name == "gis_w"
    assert gis.gas.get_value() == "W"
    assert isinstance(coater, SputterCoater) and coater.coats_grids is True
    assert coater.current.get_value() == 1.2e-7


def test_an_entry_named_for_either_type_needs_no_type():
    assert DeviceEntry.from_dict({"name": "gis"}).type == "gis"
    assert DeviceEntry.from_dict({"name": "sputter_coater"}).type == "sputter_coater"


def test_the_demo_builds_neither_unless_an_entry_lists_it():
    microscope, _ = utils.setup_session(manufacturer="Demo", setup_logging=False)
    assert "gis" not in microscope.devices
    assert "sputter_coater" not in microscope.devices


def _arctis(tmp_path, *extra):
    with open(
        os.path.join(cfg.SHIPPED_CONFIG_PATH, "sim-arctis-configuration.yaml")
    ) as f:
        config = yaml.safe_load(f)
    config["hardware"]["devices"].extend(extra)
    path = tmp_path / "configuration.yaml"
    path.write_text(yaml.safe_dump(config))
    microscope, _ = utils.setup_session(setup_logging=False, config_path=str(path))
    return microscope


def test_the_arctis_simulator_has_a_gis_and_a_sputter_coater(tmp_path):
    microscope = _arctis(tmp_path)

    assert isinstance(microscope.devices["gis"], DemoGasInjector)
    coater = microscope.devices["sputter_coater"]
    assert isinstance(coater, DemoSputterCoater)
    assert coater.current.get_value() == 1.2e-7
    # one connect's devices share the microscope's resources
    assert coater.resources is microscope.devices["stage"].resources


def test_a_second_gis_port_is_another_entry(tmp_path):
    microscope = _arctis(tmp_path, {"name": "gis_w", "type": "gis", "gas": "W"})

    assert microscope.devices["gis"].gas.get_value() == "Pt"
    assert microscope.devices["gis_w"].gas.get_value() == "W"
