# Image metadata fixtures

Metadata blocks lifted out of real acquisition TIFFs, so the metadata schema can be
tested against files this code no longer writes.

The point is backwards compatibility. `FibsemImageMetadata.from_dict` has to keep
loading images acquired by earlier versions, and nothing else in the suite exercises
that against genuine output — synthetic dicts test the shape the current code
believes in, which is exactly the assumption at risk.

## The fixtures

| File | Source | Covers |
| -- | -- | -- |
| `thermo_arctis_compustage_v3.json` | Real, Thermo Arctis | Compustage detected via the **model name**; `experiment` with no `name`, `id` holding the experiment *name* |
| `thermo_aquilos_v3.json` | Real, Thermo Aquilos | Non-compustage; a real pretilted-shuttle geometry |
| `demo_simulator_v4.json` | Simulator | Compustage detected via the **`sim` flag**; `experiment` after it gained `name` |
| `demo_simulator_v5.json` | Simulator | Current-era `experiment` (`id`/`name`/`date`) |

Between them: both arms of compustage detection, both real instruments, and three
generations of `FibsemExperimentRef`.

Two details that make these more useful than they look:

**The Arctis geometry is all zeros** — `shuttle_pre_tilt`, `rotation_reference` and
`rotation_180` are `0`, against the Aquilos's `35.0` / `110` / `290`. That is correct:
a compustage tilts the sample directly rather than using a pre-tilted shuttle, and
`imaging/tiling/reprojection.py` already assumes the rotation is always 0 for one.
It means **absence must never be inferred from zero-valued angles** — a migration
that treats all-zeros as "not recorded" breaks Arctis and nothing else.

**`demo_simulator_v5.json` carries `system.info.application_version`**, which current
code no longer writes. It was acquired mid-change. Keep it: it is the legacy case for
that field, on a file that also claims to be the current version.

## The images are not in the repository

`.gitignore` excludes `tests/data/` and `fibsem/log/*`. The real images embed the
acquiring site's directory layout, hostname and instrument serial number, and this
remote is public. Only the scrubbed metadata is committed.

`extract_fixtures.py` regenerates the JSON when a source image is present, and skips
sources that are not. Its `SCRUB` dict is the complete list of what is replaced:
`image.path`, `user.name`, `user.hostname`, `experiment.id`, `system.info.name`,
`system.info.ip_address`, `system.info.serial_number`.

Replacements keep the *shape* of what they replace — a Windows absolute path stays a
Windows absolute path, because that shape is itself a thing under test. Nothing else
is altered, and the script asserts that the model, manufacturer and all five geometry
angles survived before it writes anything.

To add a fixture, drop the image somewhere ignored, add it to `SOURCES`, and run:

```bash
python tests/fixtures/metadata/extract_fixtures.py
```

Then check the output for anything site-identifying that `SCRUB` does not yet cover.
