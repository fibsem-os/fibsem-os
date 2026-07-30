# Scripting AutoLamella experiments

You can read and modify an AutoLamella experiment from a plain Python script. No microscope, no GUI, and nothing to install beyond fibsem itself — an experiment on disk is enough.

This is the quickest way to do custom things with your data: export a summary, pull out timings, find the lamellae that failed, or bulk-edit something across a whole run.

AutoLamella can also run scripts itself, against the experiment you already have open — see [Running scripts from the GUI](#running-scripts-from-the-gui).

## Loading an experiment

Every experiment directory contains an `experiment.yaml`. Point `Experiment.load()` at it:

```python
from fibsem.applications.autolamella.structures import Experiment

exp = Experiment.load("/path/to/my-experiment/experiment.yaml")

print(exp.name, len(exp.positions), "lamellae")
for lamella in exp.positions:
    print(lamella.name, lamella.completed_tasks)
```

`exp.positions` is the list of lamellae, in the order they were created.

If a `protocol.yaml` sits next to `experiment.yaml`, it is loaded automatically into `exp.task_protocol`. That matters for two of the helpers below — see [Gotchas](#gotchas).

## What you get

On the experiment:

| Attribute | Description |
|---|---|
| `exp.name` | experiment name |
| `exp.path` | experiment directory |
| `exp.created_at` | creation timestamp |
| `exp.positions` | list of lamellae |
| `exp.task_protocol` | the protocol, or `None` if `protocol.yaml` was absent |

On each lamella in `exp.positions`:

| Attribute | Description |
|---|---|
| `lamella.name` | e.g. `01-deep-crane` |
| `lamella.path` | that lamella's directory, where its images live |
| `lamella.completed_tasks` | list of completed task names |
| `lamella.last_completed_task` | the most recent task state, or `None` |
| `lamella.task_history` | every task state, including repeats |
| `lamella.is_failure` | `True` if marked as a failure |
| `lamella.stage_position` | the stage position |
| `lamella.milling_angle` | milling angle |

## Ready-made tables

Three helpers return pandas DataFrames.

```python
exp.task_history_dataframe()      # one row per task run
exp.experiment_summary_dataframe()  # one row per lamella
exp.workflow_dataframe()          # one row per task in the protocol
```

`task_history_dataframe()` gives you `lamella_name`, `task_name`, `task_type`, `task_status`, `start_timestamp`, `end_timestamp`, `completed_at` and `duration`.

`experiment_summary_dataframe()` gives you `lamella_name`, `last_completed_task`, `last_completed_at`, `is_completed`, `is_failure` and `milling_angle`, plus the experiment's own name, path and id on every row.

## Examples

### Export a summary to CSV

```python
from pathlib import Path
from fibsem.applications.autolamella.structures import Experiment

exp = Experiment.load("/path/to/my-experiment/experiment.yaml")

out = Path(exp.path) / "summary.csv"
exp.experiment_summary_dataframe().to_csv(out, index=False)
print("wrote", out)
```

Writing into `exp.path` keeps the export next to the data it describes.

### List stage positions

```python
from fibsem.applications.autolamella.structures import Experiment

exp = Experiment.load("/path/to/my-experiment/experiment.yaml")

for lamella in exp.positions:
    pos = lamella.stage_position
    print(f"{lamella.name:24s} x={pos.x} y={pos.y} z={pos.z} r={pos.r} t={pos.t}")
```

### Find the lamellae that failed

```python
from fibsem.applications.autolamella.structures import Experiment

exp = Experiment.load("/path/to/my-experiment/experiment.yaml")

failed = [p for p in exp.positions if p.is_failure]
print(f"{len(failed)} of {len(exp.positions)} failed")

for lamella in failed:
    print(f"  {lamella.name}: {lamella.defect.description or 'no description'}")
    print(f"    got as far as: {lamella.completed_tasks}")
```

### How long did each task take?

```python
from fibsem.applications.autolamella.structures import Experiment

exp = Experiment.load("/path/to/my-experiment/experiment.yaml")

df = exp.task_history_dataframe()
df = df[df["duration"].notna()]

print(df.groupby("task_name")["duration"].agg(["count", "mean", "max"]).round(1))
```

### Find each lamella's images

```python
from pathlib import Path
from fibsem.applications.autolamella.structures import Experiment

exp = Experiment.load("/path/to/my-experiment/experiment.yaml")

for lamella in exp.positions:
    for image in sorted(Path(lamella.path).glob("*.tif")):
        print(lamella.name, image.name)
```

`lamella.path` is re-derived from the experiment's location when it loads, so this keeps working after you copy an experiment off the microscope PC.

## Changing an experiment

The same objects are writable. Call `exp.save()` when you are done:

```python
from fibsem.applications.autolamella.structures import Experiment, DefectType

exp = Experiment.load("/path/to/my-experiment/experiment.yaml")

for lamella in exp.positions:
    if "MillPolishing" not in lamella.completed_tasks:
        lamella.defect.set_defect("did not reach polishing", state=DefectType.FAILURE)

exp.save()
```

To clear a mark again, use `lamella.defect.clear()`.

`exp.save()` rewrites `experiment.yaml` only, leaving `protocol.yaml` untouched. Pass `exp.save(save_protocol=True)` if you also want the protocol written out.

**Back up the experiment directory before running anything that writes.** These scripts have no undo.

## Running scripts from the GUI

Everything above assumes you loaded the experiment yourself. AutoLamella can instead run a script against the experiment it already has open, from **Tools → Scripts → Manage scripts…**.

Everything happens in that dialog. The menu itself only opens it and the scripts folder — it deliberately does not run anything, because a menu item has no way to show you a script that is still going, and no way to stop it.

Drop a `.py` file in:

```
~/.autolamella/scripts
```

Set `AUTOLAMELLA_SCRIPTS_DIR` to use a different folder. **Tools → Scripts → Manage scripts…** shows which folder it resolved to; **Open folder** there creates it if it does not exist yet, and **New script…** writes a working stub into it.

### Working examples

Three complete scripts live in [`examples/scripts/`](examples/scripts), one per tier. Copy any of them into your scripts folder and it will run as-is.

| File | Flags | What it shows |
| -- | -- | -- |
| [`export_summary.py`](examples/scripts/export_summary.py) | none | reading the experiment, and how the return value becomes output |
| [`describe_lamellae.py`](examples/scripts/describe_lamellae.py) | `writes` | changing lamellae and letting the runner save |
| [`survey_positions.py`](examples/scripts/survey_positions.py) | `uses_microscope` | moving the stage, acquiring, and honouring Stop |

These are executed by the test suite (`tests/autolamella/test_example_scripts.py`) against a real experiment and a simulated microscope, so they cannot quietly rot. That is deliberate: earlier versions of the snippets below shipped with two bugs that only running them would have caught — one addressed a `lamella.state` attribute that does not exist, and one used `microscope.acquire_image`, which ignores `save=True` and writes nothing.

### The contract

One rule: a module-level `run(ctx)`.

```python
"""Export the experiment summary to CSV."""   # first line becomes the tooltip

def run(ctx):
    out = ctx.path / "summary.csv"
    ctx.experiment.experiment_summary_dataframe().to_csv(out, index=False)
    return out
```

That is all of it — no registration step, nothing to install, no restart. Files whose names start with `_` are hidden from the list.

You can import anything installed — the standard library, `numpy`, `pandas`, and all of `fibsem` itself. You **cannot** import another file from the scripts folder: the folder is not on `sys.path`, so `import _helpers` fails even with the file sitting right beside your script. To share code between scripts, put it in a small package of your own and `pip install -e` it.

`ctx` carries:

| Attribute | Description |
|---|---|
| `ctx.experiment` | the experiment currently open in the GUI |
| `ctx.path` | its directory — the default place to write output |
| `ctx.log(message)` | writes to the log, tagged with your script's name |
| `ctx.save()` | called for you; see `writes` below |
| `ctx.microscope` | the microscope, or `None` unless you declared `uses_microscope` |

### Showing your results

**Return the output.** There is no console in the app — `print()` goes to a terminal you may not have, and a packaged Windows build has none at all. A script that prints its answer and returns nothing looks like it did nothing.

What you return decides how it is shown:

| Return | What happens |
|---|---|
| a `str` | shown as a toast |
| a `DataFrame` | opened in a table |
| a `Path` | toast, and the containing folder opens |
| `None` | a "finished" toast |

`ctx.log("...")` writes to the log file, tagged with the script name so the lines stay attributable afterwards. Plain `logging.info(...)` works too and goes to the same place, just untagged.

### Flags

A script declares what it needs with module-level names, so the declaration cannot drift from the code it governs:

```python
"""Mark every lamella that never reached polishing."""
writes = True

def run(ctx):
    ...
```

| Flag | What it does |
|---|---|
| `writes = True` | asks you to confirm before running, then calls `ctx.save()` afterwards |
| `uses_microscope = True` | asks you to confirm, then runs the script on a background thread with `ctx.microscope` available |
| `background = True` | **not implemented** — parsed, but the script still runs inline |
| `on_workflow_completed = True` | **not implemented** — nothing runs it automatically |

Without `writes`, nothing your script changed is persisted — a read-only script cannot rewrite `experiment.yaml` by accident.

## Microscope scripts

`uses_microscope = True` gives you `ctx.microscope` and lets the script drive the hardware.

**Nothing validates what you do.** There are no limits, no interlocks and no checks that the state you left the microscope in is sane. A script that drives the stage into the pole piece will do exactly that. This is the same access the application's own code has, with none of its guard rails — treat it accordingly, and test on a dummy or a sacrificial grid first.

```python
"""Acquire an ion image at every lamella position."""
uses_microscope = True

from fibsem import acquire
from fibsem.structures import BeamType, ImageSettings


def run(ctx):
    settings = ImageSettings(hfw=80e-6, beam_type=BeamType.ION, save=True)

    for lamella in ctx.experiment.positions:
        ctx.raise_if_cancelled()          # let Stop actually stop it
        ctx.log(f"moving to {lamella.name}")
        ctx.microscope.safe_absolute_stage_movement(lamella.stage_position)
        settings.path, settings.filename = lamella.path, "script-survey"
        acquire.acquire_image(ctx.microscope, settings)

    return f"imaged {len(ctx.experiment.positions)} positions"
```

Use `fibsem.acquire.acquire_image(microscope, settings)`, not `microscope.acquire_image(settings)`. The method on the microscope is the raw grab and **ignores `save=True`** — the module-level function is the one that applies autocontrast, auto-gamma and writes the file.

### How these differ from data scripts

**They run on a background thread.** Hardware operations take seconds to minutes, and running one inline would freeze the window for the whole time, which is indistinguishable from a crash. The dialog's Run button becomes **Stop** while the script runs.

**So do not touch `ctx.experiment` from one.** It holds evented containers whose change handlers write straight into widgets, and those must only be touched from the GUI thread. Reading is generally fine; mutating is not. Nothing stops you — this is a convention you have to keep.

**Stop is cooperative, and it is your job.** A Python thread cannot be killed. Pressing Stop sets a flag; nothing happens until your script looks at it:

```python
ctx.raise_if_cancelled()   # raises, unwinding the script
if ctx.cancelled: ...      # or check it yourself and return
```

A script that never checks runs to completion no matter how many times Stop is pressed. Call it between steps — after each move, between images.

**One at a time, and never alongside a workflow.** A second script is refused while one is running, and starting a workflow is refused while a microscope script is running — otherwise two things end up commanding the stage at once.

### Editing and re-running

Scripts are re-read from disk on every run. Edit the file, click Run, and the new code runs; there is no reload button because there is nothing to reload.

The dialog also lists the files that failed to import, with the reason. A file with `def main(ctx)` instead of `def run(ctx)` shows up as an error row rather than quietly disappearing from the list.

### What to expect

**The window freezes while a script runs.** Scripts run on the GUI thread, so a slow loop over thousands of images locks the interface until it finishes. Keep GUI scripts short and do the heavy work headlessly.

**Scripts are unavailable with no experiment loaded, and while a workflow is running.** Run is greyed out and the dialog says which. A workflow mutates lamella state from a worker thread, so a script reading mid-run would see a torn snapshot.

**`ctx.microscope` is `None` unless you declared `uses_microscope`.** If you reach for it without the flag you get `AttributeError: 'NoneType' object has no attribute …` — add the flag rather than working around it. This is a guard against forgetting, not a sandbox: a script is arbitrary Python and can import its way to a microscope handle. Doing that skips the confirmation, runs the hardware call on the GUI thread, and lets a workflow start underneath you.

## Gotchas

**Close the experiment in the GUI first, or reload it afterwards.** AutoLamella holds the experiment in memory. If it is open while your script writes, whichever saves last wins and the other's changes are gone.

**`experiment_summary_dataframe()` and `workflow_dataframe()` need the protocol.** Both read `exp.task_protocol`, so if there is no `protocol.yaml` beside `experiment.yaml` they raise `AttributeError: 'NoneType' object has no attribute 'workflow_config'`. Guard it if you are working with experiments of unknown origin:

```python
if exp.task_protocol is None:
    raise SystemExit("no protocol.yaml next to this experiment")
```

`task_history_dataframe()` has no such requirement and always works.

**An empty experiment gives you an empty DataFrame**, not an error — so `df["duration"]` will raise a `KeyError` rather than returning nothing. Check `len(exp.positions)` first if that matters.

**Timestamps are POSIX floats**, not datetimes. Convert with `datetime.fromtimestamp(ts)` or `pd.to_datetime(df["start_timestamp"], unit="s")`.

**Lamella names are generated**, of the form `01-deep-crane`. Match on the numeric prefix or on `completed_tasks` rather than hardcoding a full name.
