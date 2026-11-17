"""The simulator's page documents the keys the scene takes: its key tables
are written by docs/developers/render_simulator_examples.py --table-only."""

import re
from pathlib import Path

from fibsem.drivers.demo.sim.scene import SampleScene

PAGE = Path(__file__).resolve().parents[3] / "docs" / "simulator.md"


def test_the_page_lists_every_sample_key_and_no_other():
    tables = re.search(
        r"<!-- sample-keys:start -->(.*)<!-- sample-keys:end -->",
        PAGE.read_text(encoding="utf-8"),
        flags=re.S,
    ).group(1)
    rows = re.findall(r"^\| `(\w+)` \| .* \| (.*) \|$", tables, flags=re.M)
    listed = [key for key, _ in rows]
    assert sorted(listed) == sorted(SampleScene.CONFIG_KEYS)
    assert len(listed) == len(set(listed))
    assert not [key for key, description in rows if not description.strip()]


def test_a_documented_fluorescence_key_is_taken():
    scene = SampleScene.from_config({"red_fraction": 0.1, "fm_blur_px_per_um": 1.0})
    assert scene.red_fraction == 0.1 and scene.fm_blur_px_per_um == 1.0
    assert scene.beam_optics and scene.surface_grain == 1.0  # as configured scenes are
