"""YAML safe-load/safe-dump backed by libyaml where it is available.

PyYAML ships two implementations of the same format. The pure-Python one walks
every node and every character of every scalar in Python -- `analyze_scalar`
alone inspects each character to choose between plain, quoted, literal and
folded style -- which is fine for a config file and not fine for an experiment.
On a real 85-lamella `experiment.yaml` (1.8 MB) that is 1755 ms to read and
798 ms to write; libyaml's C implementation does the same work in 275 ms and
221 ms (FIB-684).

Use `safe_load` and `safe_dump` from here in place of `yaml.safe_load` and
`yaml.safe_dump` for documents big enough to notice.

**This is the safe pair, and only the safe pair.** `yaml.dump` and
`yaml.load(..., Loader=FullLoader)` accept Python objects that the safe
implementations refuse -- an Enum, a dataclass, a numpy scalar -- by emitting
`!!python/object:` tags. `CSafeDumper` is the C implementation of `SafeDumper`,
so it carries the same restriction, which makes it a drop-in *here* and a
behaviour change at any `yaml.dump` call site. Do not route those through this
module; they are all small config files with nothing to gain.

## Why swapping the implementation does not change the file

`CSafeDumper` and `SafeDumper` are built from the same `SafeRepresenter` and the
same `Resolver` -- the code deciding *what* each Python object becomes is
identical, and only the emitter that turns an already-built node tree into text
differs. No new types become serialisable and no type changes representation.

Byte-identical output is not guaranteed by that, and is not relied on. The two
emitters can differ in how they escape exotic material (raw U+2028, control
characters, long non-ASCII runs) and where they fold a long escaped line. What
is guaranteed, and tested against every experiment and protocol file in the
repo plus a fixed-seed fuzz corpus, is that the text reloads to an equal object.
See `tests/test_yaml_backend.py`.
"""
import yaml

try:  # pragma: no cover - which branch runs depends on how PyYAML was built
    from yaml import CSafeDumper as _Dumper
    from yaml import CSafeLoader as _Loader

    USING_LIBYAML = True
except ImportError:
    # A PyYAML built without libyaml degrades to today's behaviour -- slow, not
    # broken. `yaml.__with_libyaml__` reports the same thing; this import is what
    # actually decides, so it is what sets the flag.
    from yaml import SafeDumper as _Dumper
    from yaml import SafeLoader as _Loader

    USING_LIBYAML = False

SafeDumper = _Dumper
SafeLoader = _Loader


def safe_load(stream):
    """`yaml.safe_load`, through libyaml where available."""
    return yaml.load(stream, Loader=_Loader)


def safe_dump(data, stream=None, **kwargs):
    """`yaml.safe_dump`, through libyaml where available.

    Keyword arguments are passed through unchanged, so callers keep whatever
    `indent`, `sort_keys` and `default_flow_style` they already passed.
    """
    return yaml.dump(data, stream, Dumper=_Dumper, **kwargs)
