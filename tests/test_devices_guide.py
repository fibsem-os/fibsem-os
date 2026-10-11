"""The device API guide's examples, executed against the Demo microscope.

``docs/developers/devices.md`` is the guide to the device API. Its Python blocks run
here in order, in one namespace, as a reader would paste them into one session, so a
renamed parameter or command fails this test instead of leaving the guide wrong. A
block preceded by ``<!-- not run -->`` is a fragment (a class body, a plugin module)
and is skipped.
"""

import re
from pathlib import Path

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "developers" / "devices.md"

_BLOCK = re.compile(r"(<!-- not run -->\n)?```python\n(.*?)```", re.DOTALL)


def _blocks():
    return [code for skip, code in _BLOCK.findall(GUIDE.read_text()) if not skip]


def test_the_guide_has_examples_to_run():
    # A changed fence or marker would otherwise make the test below run nothing.
    assert len(_blocks()) >= 8


def test_the_guides_examples_run():
    namespace = {"__name__": "devices_guide"}
    for i, code in enumerate(_blocks()):
        try:
            exec(compile(code, f"{GUIDE.name} block {i}", "exec"), namespace)
        except Exception as e:
            raise AssertionError(f"block {i} of {GUIDE.name} failed:\n{code}") from e

    # What the prose around the blocks says happened.
    assert namespace["written"] == 150e-6
    assert "hfw" in namespace["seen"] and "shift" in namespace["seen"]
    assert namespace["knife"].angle.cached == 0.5
