"""Regenerate the metadata fixtures from acquisition TIFFs.

The fixtures exist so the metadata schema can be tested against files produced by
real instruments and by earlier versions of this code, without committing the images
themselves. Run this only when adding a new source file; the JSON is the artifact.

    python tests/fixtures/metadata/extract_fixtures.py

The source images are deliberately not in the repository (`.gitignore`: `tests/data/`,
`fibsem/log/*`). They embed the acquiring site's directory layout, hostname and
instrument serial number, and this remote is public. SCRUB below is the full list of
what is removed; everything else is copied through untouched, including every value
the reprojection path reads.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import tifffile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

# (source image, fixture name). The real files come from two different instruments and
# the simulator files supply the metadata versions the real ones predate.
SOURCES: List[Tuple[str, str]] = [
    (
        "tests/data/ref_Setup Lamella Position_start_eb.tif",
        "thermo_arctis_compustage_v3.json",
    ),
    ("tests/data/ref_Mill Fiducial_start_eb.tif", "thermo_aquilos_v3.json"),
    # The real instruments both wrote v3. These cover the later schema versions, and
    # the simulator's is_compustage flag -- the other arm of the compustage check.
    (
        "fibsem/log/data/crosscorrelation/ref_ot_initial_alignment_1785813018_628903_ib.tif",
        "demo_simulator_v4.json",
    ),
    (
        "fibsem/log/data/crosscorrelation/ref_ot_initial_alignment_1785817135_490401_ib.tif",
        "demo_simulator_v5.json",
    ),
]

# Replaced wherever they appear. The values are placeholders of the same *shape* as
# what they replace -- a Windows path stays a Windows path -- because the shape is
# itself under test (an embedded absolute path is the point of FIB-483).
SCRUB: Dict[str, Any] = {
    "image.path": "D:\\SharedData\\user\\test-experiment\\01-test-lamella",
    "user.name": "user",
    "user.hostname": "TEST-PC",
    "experiment.id": "test-experiment",
    "system.info.name": "test-configuration",
    "system.info.ip_address": "127.0.0.1",
    "system.info.serial_number": "0000000",
}

# Never scrubbed, and asserted present afterwards: the fixtures are worthless without
# them. `model` in particular is what compustage detection string-matches on.
REQUIRED = [
    "system.info.model",
    "system.info.manufacturer",
    "system.stage.rotation_reference",
    "system.stage.rotation_180",
    "system.stage.shuttle_pre_tilt",
    "system.electron.column_tilt",
    "system.ion.column_tilt",
]


def _dig(d: Dict[str, Any], path: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Resolve a dotted path to its parent dict and final key, or (None, key)."""
    parts = path.split(".")
    node: Any = d
    for p in parts[:-1]:
        if not isinstance(node, dict) or p not in node:
            return None, parts[-1]
        node = node[p]
    if not isinstance(node, dict):
        return None, parts[-1]
    return node, parts[-1]


def scrub(metadata: Dict[str, Any]) -> Dict[str, Any]:
    for path, replacement in SCRUB.items():
        parent, key = _dig(metadata, path)
        if parent is not None and key in parent:
            parent[key] = replacement
    for path in REQUIRED:
        parent, key = _dig(metadata, path)
        if parent is None or key not in parent:
            raise AssertionError(f"{path} missing -- fixture would not be useful")
    return metadata


def main() -> None:
    for relative_source, fixture_name in SOURCES:
        source = os.path.join(REPO, relative_source)
        if not os.path.exists(source):
            print(f"skip (source not present): {relative_source}")
            continue
        with tifffile.TiffFile(source) as tif:
            metadata = json.loads(tif.pages[0].description)
        # The pixel dimensions travel with the image, not the metadata schema.
        metadata.pop("shape", None)
        destination = os.path.join(HERE, fixture_name)
        with open(destination, "w") as f:
            json.dump(scrub(metadata), f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"wrote {fixture_name}")


if __name__ == "__main__":
    main()
