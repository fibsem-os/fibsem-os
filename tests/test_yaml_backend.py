"""Reading and writing experiments through libyaml means the same thing as before.

`fibsem._yaml` swaps PyYAML's pure-Python safe loader and dumper for libyaml's C
implementations, which read a real 85-lamella experiment in 604 ms instead of 1958 ms
and write it in 311 ms instead of 798 ms (FIB-684). The file everything lives in is not
somewhere to take that on faith.

The argument that it is safe is structural: `CSafeDumper` and `SafeDumper` are built
from the same `SafeRepresenter` and `Resolver`, so the code deciding *what* each Python
object becomes is identical and only the emitter differs. These tests are the empirical
half of it.

**Equal objects, not equal bytes.** The two emitters may legitimately differ on how they
escape exotic material and where they fold a long escaped line, and the fuzz corpus below
provokes exactly that -- a third of it comes out byte-different. What must never differ
is what the text means when it is read back, so that is what is asserted.

The one thing these tests cannot cover is the platform. libyaml is a separate C library
whose version varies by install method, and CI is `ubuntu-latest` only, so a green run
here says nothing about the Windows PCs the microscopes run on. That gate is a manual
run of `check_libyaml_safe.py` on one of them, recorded on FIB-684.
"""
import glob
import importlib
import os
import random
import string
from pathlib import Path

import pytest
import yaml

from fibsem import _yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def both_ways(document):
    """(via stock PyYAML, via the backend) for one Python document."""
    return (
        yaml.safe_load(yaml.safe_dump(document, indent=4)),
        _yaml.safe_load(_yaml.safe_dump(document, indent=4)),
    )


class TestRealFiles:
    """Every real YAML document in the tree.

    Deliberately every `.yaml`, not just `experiment.yaml`: experiments live under
    `fibsem/applications/autolamella/log/`, which is gitignored, so a checkout has none
    and a corpus limited to them would **skip in CI** and quietly assert nothing. What
    CI does have is the 29 tracked protocol, legacy-protocol and microscope-configuration
    files, which are the same hand-written YAML by hand-edited hands. On a developer's
    machine this picks up their real experiments as well.
    """

    @staticmethod
    def _yaml_files():
        found = glob.glob(str(REPO_ROOT / "**" / "*.yaml"), recursive=True)
        return sorted(p for p in found if os.sep + ".git" + os.sep not in p)

    def test_the_corpus_is_not_empty(self):
        """The tracked protocols guarantee this, so an empty corpus means the glob
        broke rather than that there is nothing to check."""
        assert len(self._yaml_files()) >= 10

    def test_they_round_trip_to_the_same_object(self):
        files = self._yaml_files()

        mismatched = []
        for path in files:
            raw = Path(path).read_text(encoding="utf-8")
            try:
                original = yaml.safe_load(raw)
            except yaml.YAMLError:
                continue  # not parseable as shipped; not this change's business
            if original is None:
                continue
            if _yaml.safe_load(raw) != original:
                mismatched.append((path, "read"))
                continue
            if yaml.safe_load(_yaml.safe_dump(original, indent=4)) != original:
                mismatched.append((path, "write"))

        assert mismatched == [], f"{len(mismatched)} file(s) changed meaning: {mismatched[:3]}"

    def test_what_the_backend_writes_is_readable_by_stock_pyyaml(self, tmp_path):
        """An experiment written on one machine is opened on another, by whatever
        PyYAML that machine has -- possibly one without libyaml at all."""
        document = {"positions": [{"petname": "01-test", "task_config": {"Mill": {"x": 1.5}}}]}
        path = tmp_path / "experiment.yaml"
        with open(path, "w") as handle:
            _yaml.safe_dump(document, handle, indent=4)

        assert yaml.safe_load(path.read_text(encoding="utf-8")) == document


class TestAwkwardDocuments:
    @pytest.mark.parametrize(
        "value",
        [
            pytest.param("yes", id="bool-looking string"),
            pytest.param("~", id="null-looking string"),
            pytest.param("0123", id="octal-looking string"),
            pytest.param("12:30", id="sexagesimal-looking string"),
            pytest.param("1.2.3", id="version string"),
            pytest.param("", id="empty string"),
            pytest.param(" padded ", id="leading and trailing space"),
            pytest.param("a\nb", id="newline"),
            pytest.param("a\tb", id="tab"),
            pytest.param("- x", id="leading dash"),
            pytest.param("k: v", id="embedded colon"),
            pytest.param("# c", id="leading hash"),
            pytest.param("µm ± 0.5 — Ω", id="unicode"),
            pytest.param(-0.0, id="negative zero"),
            pytest.param(2 ** 63, id="big int"),
            pytest.param(1e-30, id="tiny float"),
            pytest.param(None, id="none"),
            pytest.param(True, id="bool"),
        ],
    )
    def test_scalars_survive_identically(self, value):
        stock, backend = both_ways({"k": value})

        assert stock == backend == {"k": value}

    def test_a_shared_container_still_reloads_as_one_object(self):
        """`AutoLamellaTaskConfig.to_dict` assigns parameters by reference, so two
        lamellae can reach one list and the emitter writes an anchor and an alias.
        Anchors are generated by `Serializer`, which the C emitter absorbs -- the one
        place the two implementations are not literally the same class."""
        shared = [1.0, 2.0, 3.0]
        document = {"positions": [{"angles": shared}, {"angles": shared}]}

        stock, backend = both_ways(document)

        assert stock == backend == document
        for label, result in (("stock", stock), ("backend", backend)):
            first, second = result["positions"]
            assert first["angles"] is second["angles"], (
                f"{label} reloaded the alias as two separate lists"
            )

    def test_a_self_reference_is_handled(self):
        document = {"k": 1}
        document["self"] = document

        assert _yaml.safe_dump(document) == yaml.safe_dump(document)


class TestFuzz:
    """Random documents over adversarial material. Fixed seed, so a failure here is
    reproducible on any machine rather than a story about one unlucky run."""

    ALPHA = string.printable[:95] + "µ°±—Ωåäö  "
    NASTY = ["yes", "no", "on", "off", "~", "null", "0123", "1e5", ".inf",
             "- x", " a ", "a\nb", "a\tb", "", "#c", "a: b", '"q"', "---"]

    @classmethod
    def _scalar(cls, rng):
        roll = rng.random()
        if roll < 0.45:
            return "".join(rng.choices(cls.ALPHA, k=rng.choice([0, 1, 2, 5, 20, 80])))
        if roll < 0.60:
            return rng.choice([True, False, None])
        if roll < 0.78:
            return rng.choice([0, -1, 2 ** 62, -(2 ** 62), rng.randint(-10 ** 9, 10 ** 9)])
        if roll < 0.95:
            return rng.choice([0.0, -0.0, 1e-30, 1e30, rng.uniform(-1e6, 1e6)])
        return rng.choice(cls.NASTY)

    @classmethod
    def _document(cls, rng, depth=0):
        if depth > 3 or rng.random() < 0.3:
            return cls._scalar(rng)
        if rng.random() < 0.5:
            return {str(cls._scalar(rng))[:40] or "k": cls._document(rng, depth + 1)
                    for _ in range(rng.randint(0, 5))}
        return [cls._document(rng, depth + 1) for _ in range(rng.randint(0, 5))]

    def test_meaning_never_changes(self):
        rng = random.Random(20260818)
        differed_in_bytes = 0
        for i in range(400):
            document = self._document(rng)
            stock_text = yaml.safe_dump(document, indent=4)
            backend_text = _yaml.safe_dump(document, indent=4)
            differed_in_bytes += stock_text != backend_text

            assert yaml.safe_load(backend_text) == document, f"fuzz #{i}: backend write"
            assert _yaml.safe_load(stock_text) == document, f"fuzz #{i}: backend read"

        # Not an assertion about the number, a check that the corpus is actually
        # provoking the disagreement this file exists to bound. If it ever drops to
        # zero the fuzz has gone toothless and the equal-objects claim is untested.
        assert differed_in_bytes > 0, "the corpus no longer produces any byte differences"


class TestTheFallback:
    """A PyYAML built without libyaml must degrade to slow, not to broken."""

    @pytest.fixture
    def without_libyaml(self, monkeypatch):
        monkeypatch.delattr(yaml, "CSafeDumper", raising=False)
        monkeypatch.delattr(yaml, "CSafeLoader", raising=False)
        importlib.reload(_yaml)
        yield _yaml
        monkeypatch.undo()
        importlib.reload(_yaml)  # restore the C implementations for everything after

    def test_it_falls_back_to_the_python_implementation(self, without_libyaml):
        assert without_libyaml.USING_LIBYAML is False
        assert without_libyaml.SafeDumper is yaml.SafeDumper
        assert without_libyaml.SafeLoader is yaml.SafeLoader

    def test_it_still_reads_and_writes(self, without_libyaml):
        document = {"positions": [{"petname": "01-test", "angles": [1.0, 2.0]}]}

        assert without_libyaml.safe_load(without_libyaml.safe_dump(document)) == document

    def test_the_c_implementation_is_restored_afterwards(self):
        """Guards the fixture itself: a leaked reload would silently make every later
        test in the session exercise the slow path instead of the shipped one."""
        assert _yaml.USING_LIBYAML == yaml.__with_libyaml__


class TestItActuallyUsesTheConfiguredImplementation:
    """The awkward corner of this change: the C and Python implementations are meant
    to be behaviourally identical, so *no* correctness test can tell them apart. A
    wrapper that quietly called stock `yaml.safe_load` would pass every other test in
    this file and simply be slow -- which is the one thing the change exists to fix,
    and the one thing nobody would notice.

    So these are white-box on purpose: they check the configured loader and dumper are
    the ones actually used, because the observable difference is only speed.
    """

    def test_safe_load_uses_the_configured_loader(self, monkeypatch):
        used = []

        class SpyLoader(_yaml.SafeLoader):
            def __init__(self, stream):
                used.append(True)
                super().__init__(stream)

        monkeypatch.setattr(_yaml, "_Loader", SpyLoader)

        assert _yaml.safe_load("a: 1") == {"a": 1}
        assert used, "safe_load bypassed the configured loader"

    def test_safe_dump_uses_the_configured_dumper(self, monkeypatch):
        used = []

        class SpyDumper(_yaml.SafeDumper):
            def __init__(self, *args, **kwargs):
                used.append(True)
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(_yaml, "_Dumper", SpyDumper)

        assert _yaml.safe_dump({"a": 1}).strip() == "a: 1"
        assert used, "safe_dump bypassed the configured dumper"


class TestItPassesCallerOptionsThrough:
    """`safe_dump` is a wrapper, and a wrapper that quietly drops its keyword
    arguments still round-trips to an equal object -- so none of the equivalence
    tests above would notice. `AutoLamellaTaskProtocol.save` passes
    `sort_keys=False` and `default_flow_style=False`, and losing those reorders and
    reflows every key in `protocol.yaml`: a large, confusing, entirely silent diff.
    """

    def test_sort_keys_is_honoured(self):
        document = {"b": 1, "a": 2}

        assert _yaml.safe_dump(document, sort_keys=False).startswith("b:")
        assert _yaml.safe_dump(document).startswith("a:")  # PyYAML sorts by default

    def test_indent_is_honoured(self):
        document = {"outer": {"inner": 1}}

        assert "\n    inner:" in _yaml.safe_dump(document, indent=4)
        assert "\n  inner:" in _yaml.safe_dump(document, indent=2)

    def test_default_flow_style_is_honoured(self):
        document = {"outer": {"inner": 1}}

        assert "{" in _yaml.safe_dump(document, default_flow_style=True)
        assert "{" not in _yaml.safe_dump(document, default_flow_style=False)


class TestItIsTheSafePairOnly:
    def test_the_backend_refuses_what_safe_dump_refuses(self):
        """`CSafeDumper` carries `SafeDumper`'s restriction, which is why it is a
        drop-in here and a behaviour change at any `yaml.dump` call site."""
        import enum

        class Beam(enum.Enum):
            ION = 1

        with pytest.raises(yaml.YAMLError):
            yaml.safe_dump({"beam": Beam.ION})
        with pytest.raises(yaml.YAMLError):
            _yaml.safe_dump({"beam": Beam.ION})

    def test_autolamella_structures_routes_through_the_backend(self):
        """The four call sites this change exists for. A new `yaml.safe_dump` added
        here later would silently keep the pure-Python cost."""
        import ast

        source = (REPO_ROOT / "fibsem" / "applications" / "autolamella" / "structures.py").read_text(encoding="utf-8")
        direct = [
            node.lineno
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"safe_load", "safe_dump"}
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "yaml"
        ]

        assert direct == [], (
            f"structures.py calls yaml.safe_* directly at lines {direct}; "
            "use fibsem._yaml so the experiment goes through libyaml"
        )
