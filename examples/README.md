# Examples

Scripts that show the fibsem-os API at work. Each one runs from the repository root, in
an environment with fibsem-os installed (`pip install -e .`):

```bash
python examples/quickstart.py
```

The API scripts start the simulated Demo microscope unless they load a protocol, in
which case they connect to whichever microscope your configuration names. Check that
before running one on a machine wired to a real instrument.

## API scripts

| Script | Shows |
|---|---|
| `quickstart.py` | connecting and taking an image with both beams |
| `imaging.py` | image settings and acquisition |
| `movement.py` | reading the stage position, relative and absolute moves (asks before each move) |
| `milling.py` | drawing patterns and milling them |
| `slice_and_view/` | a slice and view loop, with its protocol |
| `lithography/` | milling a bitmap lens profile, with its protocol and profile |
| `autolamella_v1/` | AutoLamella V1 in about 150 lines (asks you to pick each position) |

## Script Manager scripts

`app_scripts/` holds three scripts for **Tools → Scripts** in the AutoLamella app, one
per tier: read-only (`export_summary.py`), `writes` (`describe_lamellae.py`) and
`uses_microscope` (`survey_positions.py`). Copy one into your scripts folder to run it.
[SCRIPTING.md](../SCRIPTING.md#working-examples) explains the contract, and
`tests/autolamella/test_example_scripts.py` runs all three on every test run.
