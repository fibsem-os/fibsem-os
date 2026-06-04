import pytest

from fibsem.fm.structures import FocusMethod
from fibsem.fm.strategy import (
    AUTOFOCUS_STRATEGIES,
    AUTOFOCUS_STRATEGY_CONFIGS,
    DEFAULT_STRATEGY,
    DEFAULT_STRATEGY_NAME,
    get_strategy_names,
    load_autofocus_config,
)
from fibsem.fm.strategy.base import (
    AutoFocusStrategyConfig,
    get_autofocus_strategy,
    get_strategy_from_config,
)
from fibsem.fm.strategy.coarse_fine import CoarseFineAutoFocusConfig, CoarseFineAutoFocusStrategy
from fibsem.fm.strategy.iterative import IterativeAutoFocusConfig, IterativeAutoFocusStrategy
from fibsem.fm.strategy.sweep import SweepAutoFocusConfig, SweepAutoFocusStrategy

ALL_CONFIG_CLASSES = [SweepAutoFocusConfig, CoarseFineAutoFocusConfig, IterativeAutoFocusConfig]
ALL_STRATEGY_CLASSES = [SweepAutoFocusStrategy, CoarseFineAutoFocusStrategy, IterativeAutoFocusStrategy]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_contains_all_builtin_strategies():
    names = get_strategy_names()
    assert "Sweep" in names
    assert "CoarseFine" in names
    assert "Iterative" in names


def test_registry_config_and_strategy_dicts_have_same_keys():
    assert set(AUTOFOCUS_STRATEGIES.keys()) == set(AUTOFOCUS_STRATEGY_CONFIGS.keys())


def test_default_strategy():
    assert DEFAULT_STRATEGY_NAME == "Sweep"
    assert DEFAULT_STRATEGY is SweepAutoFocusStrategy


# ---------------------------------------------------------------------------
# Config — instantiation and defaults
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_CONFIG_CLASSES)
def test_config_default_instantiation(cls):
    cfg = cls()
    assert isinstance(cfg, AutoFocusStrategyConfig)
    assert cfg.name == cls.name


@pytest.mark.parametrize("cls", ALL_CONFIG_CLASSES)
def test_config_name_is_classvar(cls):
    cfg = cls()
    # name must not appear as an instance field
    assert "name" not in cls.model_fields


def test_sweep_defaults():
    cfg = SweepAutoFocusConfig()
    assert cfg.range == pytest.approx(20e-6)
    assert cfg.step == pytest.approx(1e-6)
    assert cfg.method is FocusMethod.LAPLACIAN


def test_coarse_fine_defaults():
    cfg = CoarseFineAutoFocusConfig()
    assert cfg.coarse_range == pytest.approx(50e-6)
    assert cfg.coarse_step == pytest.approx(5e-6)
    assert cfg.fine_range == pytest.approx(10e-6)
    assert cfg.fine_step == pytest.approx(1e-6)
    assert cfg.method is FocusMethod.LAPLACIAN


def test_iterative_defaults():
    cfg = IterativeAutoFocusConfig()
    assert cfg.initial_range == pytest.approx(50e-6)
    assert cfg.initial_step == pytest.approx(5e-6)
    assert cfg.num_iterations == 3
    assert isinstance(cfg.num_iterations, int)
    assert cfg.reduction_factor == pytest.approx(0.5)
    assert cfg.method is FocusMethod.LAPLACIAN


# ---------------------------------------------------------------------------
# Config — serialisation round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_CONFIG_CLASSES)
def test_config_to_dict_contains_name(cls):
    d = cls().to_dict()
    assert d["name"] == cls.name


@pytest.mark.parametrize("cls", ALL_CONFIG_CLASSES)
def test_config_round_trip(cls):
    original = cls()
    d = original.to_dict()
    restored = cls.from_dict(d)
    assert restored.to_dict() == d


@pytest.mark.parametrize("cls", ALL_CONFIG_CLASSES)
def test_config_from_dict_ignores_name_key(cls):
    d = cls().to_dict()
    assert "name" in d
    # should not raise even though name is a ClassVar, not an instance field
    restored = cls.from_dict(d)
    assert restored.name == cls.name


def test_config_method_survives_as_enum_member():
    for method in FocusMethod:
        cfg = SweepAutoFocusConfig(method=method)
        d = cfg.to_dict()
        assert d["method"] == method.value          # stored as string in dict
        restored = SweepAutoFocusConfig.from_dict(d)
        assert restored.method is method             # restored as enum member


def test_config_method_coerced_from_string():
    cfg = SweepAutoFocusConfig.from_dict({"method": "sobel"})
    assert cfg.method is FocusMethod.SOBEL


def test_iterative_num_iterations_stays_int_after_round_trip():
    cfg = IterativeAutoFocusConfig(num_iterations=5)
    restored = IterativeAutoFocusConfig.from_dict(cfg.to_dict())
    assert restored.num_iterations == 5
    assert isinstance(restored.num_iterations, int)


# ---------------------------------------------------------------------------
# Config — field metadata
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_CONFIG_CLASSES)
def test_field_metadata_has_all_fields(cls):
    cfg = cls()
    meta = cfg.field_metadata
    for field_name in cls.model_fields:
        assert field_name in meta


@pytest.mark.parametrize("cls", ALL_CONFIG_CLASSES)
def test_field_metadata_labels_populated(cls):
    cfg = cls()
    for field_name, m in cfg.field_metadata.items():
        assert m.get("label") is not None, f"{cls.name}.{field_name} missing label"


def test_sweep_range_has_scale():
    meta = SweepAutoFocusConfig().field_metadata
    assert meta["range"]["scale"] == pytest.approx(1e6)
    assert meta["range"]["unit"] == "m"


def test_method_field_has_items():
    for cls in ALL_CONFIG_CLASSES:
        meta = cls().field_metadata
        assert meta["method"]["items"] == list(FocusMethod)


def test_required_attributes_matches_model_fields():
    for cls in ALL_CONFIG_CLASSES:
        cfg = cls()
        assert set(cfg.required_attributes) == set(cls.model_fields.keys())


# ---------------------------------------------------------------------------
# Strategy — instantiation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy_cls, config_cls", zip(ALL_STRATEGY_CLASSES, ALL_CONFIG_CLASSES))
def test_strategy_default_instantiation(strategy_cls, config_cls):
    s = strategy_cls()
    assert s.name == config_cls.name
    assert isinstance(s.config, config_cls)


@pytest.mark.parametrize("strategy_cls, config_cls", zip(ALL_STRATEGY_CLASSES, ALL_CONFIG_CLASSES))
def test_strategy_name_delegates_to_config_class(strategy_cls, config_cls):
    assert strategy_cls().name == strategy_cls.config_class.name


@pytest.mark.parametrize("strategy_cls, config_cls", zip(ALL_STRATEGY_CLASSES, ALL_CONFIG_CLASSES))
def test_strategy_accepts_custom_config(strategy_cls, config_cls):
    cfg = config_cls()
    s = strategy_cls(config=cfg)
    assert s.config is cfg


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, expected_cls", [
    ("Sweep", SweepAutoFocusStrategy),
    ("CoarseFine", CoarseFineAutoFocusStrategy),
    ("Iterative", IterativeAutoFocusStrategy),
])
def test_get_autofocus_strategy_by_name(name, expected_cls):
    s = get_autofocus_strategy(name)
    assert isinstance(s, expected_cls)


def test_get_autofocus_strategy_unknown_name_falls_back_to_default():
    s = get_autofocus_strategy("DoesNotExist")
    assert isinstance(s, DEFAULT_STRATEGY)


def test_get_autofocus_strategy_with_config_dict():
    s = get_autofocus_strategy("Sweep", {"range": 50e-6, "step": 2e-6})
    assert s.config.range == pytest.approx(50e-6)
    assert s.config.step == pytest.approx(2e-6)


def test_get_strategy_from_config():
    for cls, strategy_cls in zip(ALL_CONFIG_CLASSES, ALL_STRATEGY_CLASSES):
        cfg = cls()
        s = get_strategy_from_config(cfg)
        assert isinstance(s, strategy_cls)
        assert s.config is cfg


def test_load_autofocus_config_dispatches_by_name():
    for cls in ALL_CONFIG_CLASSES:
        d = cls().to_dict()
        loaded = load_autofocus_config(d)
        assert type(loaded) is cls
        assert loaded.to_dict() == d


def test_load_autofocus_config_unknown_falls_back_to_default():
    loaded = load_autofocus_config({"name": "Unknown"})
    assert isinstance(loaded, DEFAULT_STRATEGY.config_class)
