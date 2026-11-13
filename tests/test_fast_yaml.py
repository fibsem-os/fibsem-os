"""The libyaml reader and writer give what the pure-Python ones give."""

import json
from pathlib import Path

import yaml

from fibsem.util import fast_yaml

GOLDEN = Path(__file__).parent / "golden" / "experiment-with-lamellae.json"

AWKWARD = {
    "multiline": "a: b\n second line",
    "unicode": "µm — é",
    "long": "word " * 60,
    "none": None,
    "small": 1e-9,
    "flag": True,
    "empty": "",
    "looks_like_a_flag": "yes",
    "looks_like_a_number": "123",
    "floats": [1.5, float("inf")],
}


def _experiment() -> dict:
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    data["awkward"] = AWKWARD
    return data


def test_the_text_written_is_the_pure_python_writers():
    data = _experiment()
    assert fast_yaml.safe_dump(data, indent=4) == yaml.safe_dump(data, indent=4)


def test_what_is_read_is_what_the_pure_python_reader_reads():
    text = yaml.safe_dump(_experiment(), indent=4)
    assert fast_yaml.safe_load(text) == yaml.safe_load(text)


def test_it_reads_an_open_file(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text(fast_yaml.safe_dump(AWKWARD, indent=4), encoding="utf-8")
    with open(path, encoding="utf-8") as f:
        assert fast_yaml.safe_load(f) == AWKWARD


def test_a_python_object_is_refused_as_the_safe_writer_refuses_it():
    class Thing:
        pass

    for dump in (fast_yaml.safe_dump, yaml.safe_dump):
        try:
            dump({"thing": Thing()})
        except yaml.YAMLError:
            continue
        raise AssertionError(f"{dump} wrote a Python object")
