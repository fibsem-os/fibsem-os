"""The AutoScript Arctis sputter coater against a fake of the AutoScript API.

The fake follows the AutoScript reference: ``specimen.sputter_coater`` has
``is_installed``, ``current`` (``value``, preset ``available_values``; ``limits``
raises on an Arctis and a Hydra Bio, as ``is_inserted`` does) and ``run(time)``, which on an
Arctis moves the compustage to the sputter position and back itself. Nothing here has
run on hardware -- that is the Arctis bench check.
"""

from types import SimpleNamespace
from typing import List

import pytest

from fibsem import utils
from fibsem.drivers.autoscript import devices as autoscript_devices
from fibsem.drivers.autoscript.connection import AutoScriptConnection
from fibsem.drivers.registry import device_builder
from fibsem.structures.system import DeviceEntry


class FakeBeam:
    def __init__(self, calls: List[tuple], name: str, voltage: float) -> None:
        self.calls = calls
        self.name = name
        self.is_on = False
        self.high_voltage = SimpleNamespace(value=voltage)

    def turn_on(self) -> None:
        self.calls.append(("turn_on", self.name))
        self.is_on = True


PRESETS = [1.0e-9, 5.0e-8, 1.2e-7, 2.0e-7]


class FakeCurrent:
    def __init__(self) -> None:
        self.value = 1.0e-9
        self.available_values = list(PRESETS)

    @property
    def limits(self):
        raise RuntimeError("not supported on Arctis and Hydra Bio systems")


class FakeCoater:
    def __init__(self, calls: List[tuple], beams, installed: bool = True) -> None:
        self.calls = calls
        self.beams = beams
        self.is_installed = installed
        self.is_homed = True
        self.current = FakeCurrent()
        self.fail = False

    @property
    def is_inserted(self):
        raise RuntimeError("non-insertable sputter coater")

    def home(self) -> None:
        self.calls.append(("coater_home",))
        self.is_homed = True

    def run(self, time: int, *grids) -> None:
        self.calls.append(("run", time, *grids))
        # the coater changes the ion voltage (seen on a Hydra Bio)
        self.beams.ion_beam.high_voltage.value = 2000.0
        if self.fail:
            raise RuntimeError("MicroSputter operation failed")


class FakeSdk:
    def __init__(self, installed: bool = True) -> None:
        self.calls: List[tuple] = []
        self.beams = SimpleNamespace(
            electron_beam=FakeBeam(self.calls, "electron", 2000.0),
            ion_beam=FakeBeam(self.calls, "ion", 30000.0),
        )
        compustage = SimpleNamespace(home=lambda: self.calls.append(("home",)))
        self.specimen = SimpleNamespace(
            sputter_coater=FakeCoater(self.calls, self.beams, installed),
            compustage=compustage,
        )


def _microscope(sdk: FakeSdk, model: str = "Arctis"):
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.system.info.model = model
    microscope._connection = AutoScriptConnection(sdk, microscope._threading_lock)
    return microscope


def _build(microscope, **options):
    entry = DeviceEntry(name="sputter_coater", type="sputter_coater", options=options)
    context = SimpleNamespace(microscope=microscope)
    return autoscript_devices.build_autoscript_sputter_coater(entry, context)


class TestWhereItIsBuilt:
    def test_the_driver_has_a_sputter_coater_builder(self):
        assert device_builder("thermofisher", "sputter_coater") is not None

    def test_built_by_default_on_an_arctis_with_a_coater(self):
        assert autoscript_devices.sputter_coater_fitted(_microscope(FakeSdk()))

    def test_built_by_default_on_a_hydra_bio_with_a_coater(self):
        microscope = _microscope(FakeSdk(), model="Hydra Bio")
        assert autoscript_devices.sputter_coater_fitted(microscope)

    @pytest.mark.parametrize("model", ["Aquilos 2", ""])
    def test_not_built_by_default_on_another_instrument(self, model):
        microscope = _microscope(FakeSdk(), model=model)
        assert not autoscript_devices.sputter_coater_fitted(microscope)

    def test_not_built_by_default_without_a_coater(self):
        microscope = _microscope(FakeSdk(installed=False))
        assert not autoscript_devices.sputter_coater_fitted(microscope)

    def test_a_configured_coater_is_refused_on_another_instrument(self):
        with pytest.raises(ValueError, match="Arctis or a Hydra Bio only"):
            _build(_microscope(FakeSdk(), model="Aquilos 2"))

    def test_a_configured_coater_is_refused_when_none_is_fitted(self):
        with pytest.raises(ValueError, match="no sputter coater"):
            _build(_microscope(FakeSdk(installed=False)))


class TestCoater:
    def test_current_and_its_presets(self):
        sdk = FakeSdk()
        coater = _build(_microscope(sdk))
        assert coater.name == "sputter_coater"
        assert coater.facts() == {"coats_grids": False, "is_installed": True}
        assert coater.current.choices == PRESETS
        coater.current.set_value(1.2e-7)
        assert sdk.specimen.sputter_coater.current.value == 1.2e-7

    def test_run_turns_the_beams_on_and_leaves_the_stage_to_the_coater(self):
        """No compustage home: the run moves the stage and returns it itself."""
        sdk = FakeSdk()
        coater = _build(_microscope(sdk))
        assert coater.run(10) is False
        assert sdk.calls == [
            ("turn_on", "electron"),
            ("turn_on", "ion"),
            ("run", 10),
        ]

    def test_run_puts_the_ion_voltage_back_even_when_it_fails(self):
        sdk = FakeSdk()
        sdk.specimen.sputter_coater.fail = True
        coater = _build(_microscope(sdk))
        with pytest.raises(RuntimeError, match="MicroSputter"):
            coater.run(10)
        assert sdk.beams.ion_beam.high_voltage.value == 30000.0
        assert coater.inserted.get_value() is False

    def test_a_run_takes_whole_seconds(self):
        sdk = FakeSdk()
        coater = _build(_microscope(sdk))
        with pytest.raises(ValueError, match="whole seconds"):
            coater.run(0.4)
        assert ("run", 0) not in sdk.calls

    def test_it_coats_what_is_under_it(self):
        coater = _build(_microscope(FakeSdk()))
        with pytest.raises(ValueError, match="takes no grids"):
            coater.run(10, grids=[1])


class TestHydraBio:
    def _coater(self, sdk, **options):
        coater = _build(_microscope(sdk, model="Hydra Bio"), **options)
        assert isinstance(coater, autoscript_devices.AutoscriptHydraBioSputterCoater)
        return coater

    def test_it_coats_named_grids(self):
        coater = self._coater(FakeSdk())
        assert coater.facts() == {"coats_grids": True, "is_installed": True}

    def test_run_sets_the_bench_current_and_names_the_grids(self):
        sdk = FakeSdk()
        coater = self._coater(sdk)
        assert coater.run(120, grids=[1, 2]) is False
        # no compustage home: the coater moves the stage to each grid itself
        assert sdk.calls == [
            ("turn_on", "electron"),
            ("turn_on", "ion"),
            ("run", 120, [1, 2]),
        ]
        assert sdk.specimen.sputter_coater.current.value == 1.2e-7

    def test_a_current_set_by_a_caller_is_the_one_a_run_uses(self):
        sdk = FakeSdk()
        coater = self._coater(sdk)
        coater.current.set_value(5.0e-8)
        sdk.specimen.sputter_coater.current.value = 1.0e-9  # the coater's fallback
        coater.run(60, grids=[1])
        assert sdk.specimen.sputter_coater.current.value == 5.0e-8

    def test_the_entry_sets_the_run_current(self):
        sdk = FakeSdk()
        self._coater(sdk, current=2.0e-7).run(60, grids=[2])
        assert sdk.specimen.sputter_coater.current.value == 2.0e-7

    def test_a_run_names_its_grids(self):
        sdk = FakeSdk()
        with pytest.raises(ValueError, match="name the grids"):
            self._coater(sdk).run(60)
        assert not [c for c in sdk.calls if c[0] == "run"]

    def test_the_ion_voltage_is_put_back(self):
        sdk = FakeSdk()
        self._coater(sdk).run(60, grids=[1])
        assert sdk.beams.ion_beam.high_voltage.value == 30000.0

    def test_inserted_is_its_own_run(self):
        """is_inserted raises on a Hydra Bio: the coater isn't insertable."""
        assert self._coater(FakeSdk()).inserted.get_value() is False

    def test_an_unhomed_coater_is_homed_before_the_run(self):
        sdk = FakeSdk()
        sdk.specimen.sputter_coater.is_homed = False
        self._coater(sdk).run(60, grids=[1])
        runs = [c[0] for c in sdk.calls if c[0] in ("coater_home", "run")]
        assert runs == ["coater_home", "run"]

    def test_a_current_that_is_no_preset_is_refused(self):
        sdk = FakeSdk()
        with pytest.raises(ValueError, match="not one of the coater's currents"):
            self._coater(sdk, current=3.0e-7).run(60, grids=[1])
        assert not [c for c in sdk.calls if c[0] == "run"]
