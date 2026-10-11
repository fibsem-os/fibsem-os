"""``get_available_values`` and its cache warn that they are deprecated, naming the
device parameter whose choices replace them, and code outside the tests no longer
calls them."""

import ast
import warnings
from pathlib import Path

import pytest

from fibsem import utils
from fibsem.structures import BeamType

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ["fibsem", "example", "examples", "docs", "scripts"]
# The deprecated methods themselves.
ALLOWED = ["fibsem/microscope.py"]
DEPRECATED = (
    "get_available_values",
    "get_available_values_cached",
    "clear_available_values_cache",
)


@pytest.fixture(scope="module")
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    yield microscope
    microscope.disconnect()


def test_it_warns_and_names_the_beam_parameter(microscope):
    with pytest.warns(DeprecationWarning) as record:
        choices = microscope.get_available_values("current", BeamType.ION)
    message = str(record[0].message)
    assert message.startswith('microscope.get_available_values("current", ...)')
    assert 'microscope.beams[BeamType.ION].parameters["current"].choices' in message
    # the warning points at the caller, not at fibsem/microscope.py
    assert record[0].filename == __file__
    # and the answer is still the device's choices
    assert choices == list(microscope.beams[BeamType.ION].current.choices)


def test_a_key_with_no_beam_parameter_points_at_the_milling_service(microscope):
    with pytest.warns(DeprecationWarning, match=r"supported_settings\(\)"):
        assert microscope.get_available_values("application_file") == []


def test_the_cache_warns_once_per_call(microscope):
    with pytest.warns(DeprecationWarning) as record:
        microscope.get_available_values_cached("voltage", BeamType.ELECTRON)
    assert len(record) == 1
    assert "get_available_values_cached" in str(record[0].message)
    with pytest.warns(DeprecationWarning, match="clear_available_values_cache"):
        microscope.clear_available_values_cache()


def test_the_device_choices_do_not_warn(microscope):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert microscope.beams[BeamType.ION].current.choices


def _calls(path: Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    rel = path.relative_to(ROOT).as_posix()
    return [
        f"{rel}:{node.lineno}: .{node.func.attr}(...)"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in DEPRECATED
    ]


def test_no_code_outside_the_tests_calls_it():
    offenders = []
    for top in SCANNED:
        base = ROOT / top
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            if rel not in ALLOWED:
                offenders.extend(_calls(path))
    assert not offenders, (
        "Use the beam parameter's choices "
        '(microscope.beams[beam_type].parameters["name"].choices) instead:\n'
        + "\n".join(offenders)
    )
