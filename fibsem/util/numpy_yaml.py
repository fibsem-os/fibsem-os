"""Let YAML write numpy scalars as the plain numbers they are.

A numpy scalar that reaches ``yaml.safe_dump`` raises "cannot represent an
object", and one stray value anywhere in an experiment made the whole file
unsaveable. Registered on import of ``fibsem.config``, which every
writer imports; both the safe and the default dumpers get it.
"""

import numpy as np
import yaml


def _float(dumper: yaml.SafeDumper, value: np.floating) -> yaml.Node:
    return dumper.represent_float(float(value))


def _int(dumper: yaml.SafeDumper, value: np.integer) -> yaml.Node:
    return dumper.represent_int(int(value))


def _bool(dumper: yaml.SafeDumper, value: np.bool_) -> yaml.Node:
    return dumper.represent_bool(bool(value))


def register() -> None:
    for dumper in (yaml.SafeDumper, yaml.Dumper):
        dumper.add_multi_representer(np.floating, _float)
        dumper.add_multi_representer(np.integer, _int)
        dumper.add_multi_representer(np.bool_, _bool)


register()
