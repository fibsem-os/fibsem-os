"""``yaml.safe_load`` and ``yaml.safe_dump`` through libyaml when PyYAML was built
with it, which its wheels are. The text written and the data read are the same as
the pure-Python reader and writer give; an experiment of 50 lamellae saves in about
a fifth of the time and loads in about a sixth.
"""

from typing import IO, Any, Union

import yaml

_Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
_Dumper = getattr(yaml, "CSafeDumper", yaml.SafeDumper)


def safe_load(stream: Union[str, bytes, IO]) -> Any:
    return yaml.load(stream, Loader=_Loader)


def safe_dump(data: Any, **options: Any) -> str:
    return yaml.dump(data, Dumper=_Dumper, **options)
