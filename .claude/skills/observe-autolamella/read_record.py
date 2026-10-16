#!/usr/bin/env python3
"""Read a finished AutoLamella experiment from its folder, with no app running.

The offline half of the Observer: what happened, to which lamella, how long it
took and why it failed, from the experiment's own records. It reads and never
writes into the folder. It does not call ``Experiment.load()``, which changes
the record as it reads it (it withdraws open questions and re-points paths);
``experiment.yaml`` is read as plain YAML and ``events.jsonl`` through the same
tables the report uses.

    python .claude/skills/observe-autolamella/read_record.py <experiment folder>
    python .claude/skills/observe-autolamella/read_record.py <folder> --item 2
    python .claude/skills/observe-autolamella/read_record.py <folder> --failures
    python .claude/skills/observe-autolamella/read_record.py <folder> --preview <tif> --out <dir>

``--item`` takes a lamella's name, or its number ("2" is 02-...). ``--preview``
writes a PNG of a saved TIFF to ``--out`` (never into the experiment), for an
agent that can look at PNGs but not TIFFs.

Needs fibsem installed: run it with the Python AutoLamella runs in.
"""

import argparse
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from fibsem.applications.autolamella.tools.event_tables import read_event_tables
from fibsem.util.timestamps import wall_time_of
from fibsem.workflows.event_recording import EVENTS_FILENAME, read_events

LOG_FILENAME = "logfile.log"
# A log line's own time, at the start of every line the logger wrote.
_LOG_TIME = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} — ")
# How far either side of a failed run to show the log: the failure is often
# logged just after the run's recorded end.
_LOG_MARGIN = timedelta(seconds=5)


def _seconds(value) -> str:
    if value is None or value != value:  # None or NaN
        return "-"
    value = float(value)
    if value < 60:
        return f"{value:.0f}s"
    if value < 3600:
        return f"{value / 60:.1f}min"
    return f"{value / 3600:.2f}h"


def _text(value) -> str:
    """A table cell as text: empty for None and NaN."""
    if value is None or value != value:
        return ""
    return str(value)


def _time(value) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S") if isinstance(value, datetime) else "-"


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _items(experiment: dict) -> list:
    return list(experiment.get("positions") or [])


def _find_item(items: list, wanted: str) -> dict:
    """A lamella by name, by its number, or by the start of its name."""
    wanted = wanted.strip()
    for item in items:
        if item.get("petname") == wanted:
            return item
    if wanted.isdigit():
        for item in items:
            if item.get("number") == int(wanted):
                return item
    matches = [i for i in items if str(i.get("petname", "")).startswith(wanted)]
    if len(matches) == 1:
        return matches[0]
    names = ", ".join(str(i.get("petname")) for i in items)
    sys.exit(f"No single lamella matches {wanted!r}. Lamellae: {names}")


def _required_tasks(protocol: dict) -> list:
    workflow = protocol.get("workflow") or {}
    return [t.get("name") for t in workflow.get("tasks") or [] if t.get("required")]


def _completed(item: dict) -> set:
    return {
        t.get("name")
        for t in item.get("task_history") or []
        if t.get("status") == "Completed"
    }


def _defect(item: dict) -> str:
    defect = item.get("defect") or {}
    verdict = defect.get("verdict") or "UNASSESSED"
    if verdict == "UNASSESSED":
        return ""
    reason = f" ({defect['reason']})" if defect.get("reason") else ""
    return f"{verdict}{reason}"


def overview(folder: Path, experiment: dict, protocol: dict, tables) -> None:
    items = _items(experiment)
    required = _required_tasks(protocol)
    runs = tables.runs
    print(f"Experiment: {experiment.get('name')}  ({folder})")
    print(f"Created: {experiment.get('created_at', '-')}")
    if required:
        print(f"Required tasks: {', '.join(required)}")
    if not (folder / EVENTS_FILENAME).exists():
        print(f"No {EVENTS_FILENAME}: timings, decisions and notes are not recorded.")
    print()

    finished = 0
    print("Lamellae:")
    for item in items:
        name = item.get("petname")
        done = _completed(item)
        missing = [t for t in required if t not in done]
        failed_runs = runs[(runs["item"] == name) & (runs["outcome"] == "failed")]
        defect = _defect(item)
        if required and not missing and not defect:
            finished += 1
        state = item.get("task_state") or {}
        line = f"  {name}: last {state.get('name') or '-'} {state.get('status') or ''}".rstrip()
        if missing:
            line += f"; not done: {', '.join(missing)}"
        if len(failed_runs):
            line += f"; {len(failed_runs)} failed run(s)"
        if defect:
            line += f"; defect: {defect}"
        print(line)
    if required:
        print(
            f"\nYield: {finished} of {len(items)} lamellae completed every required task with no defect recorded."
        )

    if len(runs):
        print("\nRuns by task:")
        for task, group in runs.groupby("task", sort=False):
            outcomes = group["outcome"].value_counts().to_dict()
            ended = group[group["outcome"] == "completed"]
            print(
                f"  {task}: {', '.join(f'{n} {o}' for o, n in outcomes.items())}; "
                f"median {_seconds(ended['duration'].median())}, "
                f"waiting {_seconds(group['waiting'].sum())} in all"
            )
        start, end = runs["start"].min(), runs["end"].max()
        print(f"\nFirst run started {_time(start)}, last ended {_time(end)}.")

    failed = [
        (run["item"], run["task"], _time(run["end"]), run["reason"])
        for _, run in runs[runs["outcome"] == "failed"].iterrows()
    ]
    if not len(runs):  # no events file: the experiment's own task history
        failed = [
            (
                item.get("petname"),
                t.get("name"),
                t.get("end_timestamp"),
                t.get("status_message"),
            )
            for item in items
            for t in item.get("task_history") or []
            if t.get("status") == "Failed"
        ]
    if failed:
        print("\nFailures:")
        for item, task, when, reason in failed:
            print(f"  {item} / {task} at {when}: {reason}")

    if len(tables.actors):
        who = tables.actors.groupby("actor")["count"].sum().to_dict()
        print("\nRecords by actor: " + ", ".join(f"{a} {n}" for a, n in who.items()))


def item_story(folder: Path, experiment: dict, tables, wanted: str) -> None:
    item = _find_item(_items(experiment), wanted)
    name = item.get("petname")
    item_folder = folder / str(name)
    print(f"Lamella {name} (number {item.get('number')})  {item_folder}")
    for key in ("description", "milling_angle"):
        if item.get(key):
            print(f"  {key}: {item[key]}")
    if _defect(item):
        print(f"  defect: {_defect(item)}")
    print(f"  poi: {item.get('poi')}  alignment area: {item.get('alignment_area')}")

    runs = tables.runs[tables.runs["item"] == name]
    extras = _run_extras(folder)
    print("\nRuns:")
    if not len(runs):
        print("  none recorded")
    for _, run in runs.iterrows():
        line = (
            f"  {_time(run['start'])}  {run['task']}: {run['outcome']}, "
            f"{_seconds(run['duration'])} (waiting {_seconds(run['waiting'])})"
        )
        if _text(run["reason"]):
            line += f" -- {run['reason']}"
        print(line)
        steps = tables.steps[tables.steps["task_id"] == run["task_id"]]
        if len(steps):
            print(
                "      steps: "
                + ", ".join(
                    f"{s['step']} {_seconds(s['duration'])}"
                    for _, s in steps.iterrows()
                )
            )
        extra = extras.get(run["task_id"], {})
        if run["outcome"] == "failed" and extra.get("step"):
            print(f"      stopped at step: {extra['step']}")
        for alignment in extra.get("alignments", []):
            print(f"      alignment {alignment}")
        milling = tables.milling[tables.milling["task_id"] == run["task_id"]]
        for _, stage in milling.iterrows():
            done = "" if stage["finished"] else " (did not finish)"
            print(f"      milled {stage['stage']}: {_seconds(stage['duration'])}{done}")
        for role, files in (run["outputs"] or {}).items():
            for file in files if isinstance(files, list) else [files]:
                path = item_folder / file
                print(f"      {role}: {path}{'' if path.exists() else ' (missing)'}")

    decisions = tables.decisions[tables.decisions["item"] == name]
    if len(decisions):
        print("\nDecisions:")
        for _, d in decisions.iterrows():
            reason = f" -- {d['reason']}" if _text(d["reason"]) else ""
            print(
                f"  {_time(d['time'])}  {d['task']} {d['kind']}: {d['outcome']} by {d['actor']}{reason}"
            )

    edits = tables.edits[tables.edits["item"] == name]
    if len(edits):
        print("\nEdits:")
        for _, e in edits.iterrows():
            print(
                f"  {_time(e['time'])}  {e['target']} by {e['actor']} via {e['via']}: {e['fields']}"
            )

    notes = _notes(folder, name)
    if notes:
        print("\nNotes:")
        for time, actor, text in notes:
            print(f"  {_time(time)}  {actor}: {text}")


def _run_extras(folder: Path) -> dict:
    """What the tables leave out, by run id: the step a run ended on, and its
    alignments that did not complete (a skipped one often points back at an
    earlier task that failed to leave its reference image)."""
    events = folder / EVENTS_FILENAME
    extras: dict = {}
    if not events.exists():
        return extras
    for record in read_events(events):
        kind = record.get("kind")
        payload = record.get("payload") or {}
        run_id = (record.get("task") or {}).get("id")
        if kind == "task_failed":
            step = (payload.get("task_state") or {}).get("step")
            extras.setdefault(run_id, {})["step"] = step
        elif kind == "alignment" and payload.get("status") not in (None, "completed"):
            reason = payload.get("reason") or payload.get("error") or ""
            extras.setdefault(run_id, {}).setdefault("alignments", []).append(
                f"{payload.get('status')}: {reason}".rstrip(": ")
            )
    return extras


def _notes(folder: Path, name: str) -> list:
    events = folder / EVENTS_FILENAME
    if not events.exists():
        return []
    notes = []
    for record in read_events(events):
        if record.get("kind") != "agent_note":
            continue
        payload = record.get("payload") or {}
        if (
            payload.get("item_name") == name
            or (record.get("item") or {}).get("name") == name
        ):
            notes.append(
                (
                    wall_time_of(record.get("t")),
                    record.get("actor"),
                    payload.get("text"),
                )
            )
    return notes


def failures(folder: Path, tables) -> None:
    failed = tables.runs[tables.runs["outcome"] == "failed"]
    if not len(failed):
        print("No failed runs recorded.")
        return
    log = folder / LOG_FILENAME
    extras = _run_extras(folder)
    for _, run in failed.iterrows():
        print(f"{run['item']} / {run['task']}: {run['reason']}")
        print(
            f"  ran {_time(run['start'])} to {_time(run['end'])} ({_seconds(run['duration'])})"
        )
        steps = tables.steps[tables.steps["task_id"] == run["task_id"]]
        step = extras.get(run["task_id"], {}).get("step")
        if step or len(steps):
            print(f"  stopped at step: {step or steps.iloc[-1]['step']}")
        if log.exists() and run["end"] is not None:
            start = run["start"] if run["start"] is not None else run["end"]
            lines = _log_window(log, start - _LOG_MARGIN, run["end"] + _LOG_MARGIN)
            if lines:
                print(f"  warnings and errors in {LOG_FILENAME} around it:")
                for line in lines:
                    print(f"    {line}")
        print()
    if not log.exists():
        print(
            f"No {LOG_FILENAME} in the folder: only the recorded reasons are available."
        )


def _log_window(log: Path, start: datetime, end: datetime) -> list:
    """WARNING and ERROR lines logged between start and end, with the traceback
    lines that follow them."""
    keep, inside, showing = [], False, False
    with open(log, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = _LOG_TIME.match(line)
            if m:
                time = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                inside = start.replace(microsecond=0) <= time <= end
                showing = inside and (" — WARNING — " in line or " — ERROR — " in line)
            if showing:
                keep.append(line)
    return keep


def preview(tif: Path, out: Path) -> None:
    from fibsem.imaging.thumbnail import write_thumbnail
    from fibsem.structures import FibsemImage

    image = FibsemImage.load(str(tif))
    out.mkdir(parents=True, exist_ok=True)
    path = write_thumbnail(
        image.filtered_data, out / (tif.stem + ".png"), max_edge=1024
    )
    md = image.metadata
    print(path)
    if md is not None and md.pixel_size is not None:
        print(f"pixel size {md.pixel_size.x:.3e} m, shape {image.data.shape}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "folder", type=Path, help="the experiment folder (holds experiment.yaml)"
    )
    parser.add_argument("--item", help="one lamella's story: its name or number")
    parser.add_argument(
        "--failures",
        action="store_true",
        help="every failed run, with the log around it",
    )
    parser.add_argument("--preview", type=Path, help="a saved TIFF to write as a PNG")
    parser.add_argument(
        "--out", type=Path, help="where --preview writes (never the experiment folder)"
    )
    args = parser.parse_args(argv)

    folder = args.folder.expanduser().resolve()
    if folder.is_file():
        folder = folder.parent
    if args.preview:
        out = (args.out or Path.cwd()).expanduser().resolve()
        if out == folder or folder in out.parents:
            sys.exit(
                "--out must be outside the experiment folder: this reader never writes into it."
            )
        preview(args.preview, out)
        return
    if not (folder / "experiment.yaml").exists():
        sys.exit(f"No experiment.yaml in {folder}: not an experiment folder.")

    experiment = _read_yaml(folder / "experiment.yaml")
    protocol = _read_yaml(folder / "protocol.yaml")
    events = folder / EVENTS_FILENAME
    if events.exists():
        tables = read_event_tables(events)
    else:
        from fibsem.applications.autolamella.tools.event_tables import event_tables

        tables = event_tables([])

    if args.item:
        item_story(folder, experiment, tables, args.item)
    elif args.failures:
        failures(folder, tables)
    else:
        overview(folder, experiment, protocol, tables)


if __name__ == "__main__":
    main()
