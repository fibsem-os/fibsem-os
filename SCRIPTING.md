# Scripting AutoLamella experiments

You can read and modify an AutoLamella experiment from a plain Python script. No microscope, no GUI, and nothing to install beyond fibsem itself — an experiment on disk is enough.

This is the quickest way to do custom things with your data: export a summary, pull out timings, find the lamellae that failed, or bulk-edit something across a whole run.

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
| `lamella.path` | that lamella's directory — but see [Gotchas](#gotchas) before using it |
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

## Gotchas

**Don't use `lamella.path` — derive it instead.** This one bites as soon as you copy an experiment off the microscope PC, which is the usual way of working.

`exp.path` is taken from the file you loaded, so it is always right. But each `lamella.path` is stored in `experiment.yaml` exactly as it was when the lamella was created, and never updated. On a copied or moved experiment it still points at the original machine's directory:

```
exp.path      : /data/analysis/my-experiment          # correct
lamella.path  : /home/user/experiments/my-experiment/01-deep-crane   # gone
```

If the original directory happens to still exist on the same machine, this is worse than an error — you silently read the wrong experiment. Build the path yourself:

```python
from pathlib import Path

for lamella in exp.positions:
    lamella_dir = Path(exp.path) / lamella.name
    for image in sorted(lamella_dir.glob("*.tif")):
        print(lamella.name, image.name)
```

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
