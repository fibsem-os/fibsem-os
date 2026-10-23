from __future__ import annotations

import datetime
import logging
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from fibsem.applications.autolamella.structures import Experiment, Lamella
from fibsem.util.durations import format_duration_as_clock

# `{'msg': 'status', ...}`: the key every reported record starts with.
_MSG_KEY = re.compile(r"^\{'msg': '([^']+)'")
# The records the report reads, by their `msg`.
_REPORTED = {"status", "milling_task", "feature_detection"}
# What a lamella task's status record carries.
_STATUS_KEYS = (
    "lamella",
    "lamella_id",
    "task_name",
    "task_id",
    "task_type",
    "task_step",
)


#### TASK REFACTORING ####


def read_log_tables(fname: str) -> Tuple[List[dict], List[dict], List[dict]]:
    """The report's rows from a log: task steps, milling stages and detections.

    Each milling stage and detection carries the task step it happened in.
    Records the report wants but cannot read are counted and logged, not
    silently skipped.
    """
    from fibsem.applications.autolamella.tools.replay import (
        parse_record,
        read_log_records,
    )

    steps_data: List[dict] = []
    det_data: List[dict] = []
    milling_data2: List[dict] = []
    unreadable = 0

    stepd = None

    for record in read_log_records(Path(fname)):
        # Dispatch on the record's own `msg` key, not the function that logged it:
        # a renamed method would otherwise empty a table without a trace.
        match = _MSG_KEY.match(record.message)
        if match is None or match.group(1) not in _REPORTED:
            continue
        msgd = parse_record(record.message)
        if msgd is None:
            unreadable += 1
            continue
        # whole seconds, local time: as the report has always timed its rows
        tsd = datetime.datetime.timestamp(record.time.replace(microsecond=0))
        kind = match.group(1)

        if kind == "milling_task":
            if stepd is None:
                continue

            # add milling task data
            msgd2 = stepd
            msgd2["timestamp"] = tsd
            msgd2.update(msgd)
            milling_data2.append(deepcopy(msgd2))

        if kind == "status":
            # A lamella task's step. A grid task's names its grid instead and is
            # not in this lamella report.
            if any(key not in msgd for key in _STATUS_KEYS):
                continue
            # global data
            stepd = {
                "timestamp": tsd,
                "lamella": msgd["lamella"],
                "lamella_id": msgd["lamella_id"],
                "task_name": msgd["task_name"],
                "task_id": msgd["task_id"],
                "task_type": msgd["task_type"],
                "task_step": msgd["task_step"],
            }
            steps_data.append(deepcopy(stepd))

        if kind == "feature_detection":  # DETECTION INTERACTION
            if stepd is None:
                continue

            dmsgd2 = stepd
            dmsgd2["timestamp"] = tsd
            dmsgd2.update(msgd)
            det_data.append(deepcopy(dmsgd2))

    if unreadable:
        logging.warning(
            f"{unreadable} records in {fname} could not be read and are not in the report."
        )
    return steps_data, milling_data2, det_data


def parse_logfile(path: str, encoding="utf-8") -> Dict[str, pd.DataFrame]:
    """Updated parser for task based workflow

    The records are the dicts the task, milling and detection code log at DEBUG,
    rendered by ``str()``: Python reprs, not JSON. They are read with the replay's
    AST reader (``tools/replay.py``), which takes literals and calls on dotted
    names (``np.float64(2e-09)``) and never evaluates anything. The string
    substitutions this used before corrupted values -- ``(rough)`` became
    ``[rough]`` -- and failed outright on numpy 2 reprs and apostrophes.

    ``encoding`` is kept for callers; it is detected instead (UTF-8, or cp1252 as
    older Windows installs wrote), since a wrong guess used to empty the report.
    """
    fname = os.path.join(path, "logfile.log")
    print("-" * 80)
    print(f"Parsing {fname}")
    steps_data, milling_data2, det_data = read_log_tables(fname)

    df_tasks = pd.DataFrame(steps_data)
    df_tasks["duration"] = df_tasks["timestamp"].diff().shift(-1)
    df_tasks.fillna(0, inplace=True)

    from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol

    exp = Experiment.load(os.path.join(path, "experiment.yaml"))  # type: ignore
    exp.task_protocol = AutoLamellaTaskProtocol.load(
        os.path.join(path, "protocol.yaml")
    )
    df_task_history = exp.task_history_dataframe()

    df_exp = exp.experiment_summary_dataframe()
    df_workflow = exp.workflow_dataframe()
    df_milling2 = pd.json_normalize(milling_data2)
    df_det = pd.json_normalize(det_data)

    # TODO: save these dataframes to csv
    # TODO: add experiment name and id to all dataframes

    path = os.path.join(exp.path, "reporting")
    os.makedirs(path, exist_ok=True)

    filename = os.path.join(path, "experiment.csv")
    df_exp.to_csv(filename, mode="w", header=True, index=False)
    filename = os.path.join(path, "workflow.csv")
    df_workflow.to_csv(filename, mode="w", header=True, index=False)
    filename = os.path.join(path, "task_history.csv")
    df_task_history.to_csv(filename, mode="w", header=True, index=False)
    filename = os.path.join(path, "milling.csv")
    df_milling2.to_csv(filename, mode="w", header=True, index=False)
    filename = os.path.join(path, "detection.csv")
    df_det.to_csv(filename, mode="w", header=True, index=False)

    return {
        "experiment": df_exp,
        "workflow": df_workflow,
        "tasks": df_tasks,
        "milling": df_milling2,
        "task_history": df_task_history,
        "detection": df_det,
    }


def format_pretty_dataframes(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Format dataframes for pretty printing in notebook/report"""
    df_exp = dfs["experiment"]

    # only show lamella_name, last_complete, last_completed_at, milling_angle, is_failure
    df_exp_filtered = df_exp[
        [
            "lamella_name",
            "last_completed",
            "milling_angle",
            "is_completed",
            "is_failure",
        ]
    ]

    # rename these cols, to be pretty
    df_exp_filtered = df_exp_filtered.rename(
        columns={
            "lamella_name": "Lamella Name",
            "last_completed": "Last Completed Task",
            "milling_angle": "Milling Angle",
            "is_completed": "Completed",
            "is_failure": "Is Failure",
        }
    )

    # round milling angle to 1 decimal place (use degree symbol)
    df_exp_filtered["Milling Angle"] = df_exp_filtered["Milling Angle"].round(1)

    # WORKFLOW
    df_workflow = dfs["workflow"]
    # rename cols
    # errors="raise" so a column that stops existing fails loudly. This rename
    # asked for "task" while workflow_dataframe() produces "task_name", and pandas
    # ignores unknown labels by default -- so the PDF quietly showed the raw column
    # name instead. See FIB-458.
    df_workflow = df_workflow.rename(
        columns={
            "order": "Order",
            "task_name": "Task Name",
            "required": "Required",
            "attention": "Attention",
        },
        errors="raise",
    )

    # TASK HISTORY
    df_task_history = dfs["task_history"]

    # filter to only show lamella_name, task_name, completed_at, duration
    df_task_history_filtered = df_task_history[
        ["lamella_name", "task_name", "completed_at", "duration"]
    ]

    df_task_history_filtered = df_task_history_filtered.rename(
        columns={
            "lamella_name": "Lamella Name",
            "task_name": "Task Name",
            "task_type": "Task Type",
            "completed_at": "Completed At",
            "duration": "Duration",
        }
    )

    # format duration to be in minutes and seconds
    df_task_history_filtered["Duration"] = pd.to_timedelta(
        df_task_history_filtered["Duration"], unit="s"
    )
    # format MM:SS
    df_task_history_filtered["Duration"] = df_task_history_filtered["Duration"].apply(
        lambda x: (
            f"{int(x.total_seconds() // 60):02d}:{int(x.total_seconds() % 60):02d}"
        )
    )

    # sort by Completed At
    df_task_history_filtered = df_task_history_filtered.sort_values(
        by=["Lamella Name", "Completed At"], ascending=[True, True]
    )

    ### TASK STEP HISTORY
    df_task_step_history = dfs["tasks"]

    # filter to only show lamella_name, task_name, task_step, duration, timestamp
    df_task_step_history_filtered = df_task_step_history[
        ["lamella", "task_name", "task_step", "duration", "timestamp"]
    ]

    df_task_step_history_filtered = df_task_step_history_filtered.rename(
        columns={
            "lamella": "Lamella Name",
            "task_name": "Task Name",
            "task_step": "Task Step",
            "duration": "Duration",
            "timestamp": "Timestamp",
        }
    )

    # format timestamps fromtimestamp to datetime
    df_task_step_history_filtered["Timestamp"] = pd.to_datetime(
        df_task_step_history_filtered["Timestamp"], unit="s"
    )

    # drop task_step if STARTED, FINISHED
    df_task_step_history_filtered = df_task_step_history_filtered[
        ~df_task_step_history_filtered["Task Step"].isin(["STARTED", "FINISHED"])
    ]

    # group by Task Name and Task Step, and get mean duration
    df_task_step_summary = (
        df_task_step_history_filtered.groupby(["Task Name", "Task Step"])
        .agg({"Duration": "mean", "Lamella Name": "count"})
        .reset_index()
    )
    df_task_step_summary = df_task_step_summary.rename(
        columns={"Duration": "Mean Duration", "Lamella Name": "Count"}
    )

    # format duration to be in minutes and seconds
    df_task_step_summary["Mean Duration"] = pd.to_timedelta(
        df_task_step_summary["Mean Duration"], unit="s"
    )
    # show as total seconds
    df_task_step_summary["Mean Duration"] = df_task_step_summary["Mean Duration"].apply(
        lambda x: f"{x.total_seconds():.1f} s"
    )
    # sort by Task Name, then Task Step
    df_task_step_summary = df_task_step_summary.sort_values(
        by=["Task Name", "Task Step"], ascending=[True, True]
    )

    # MILLLING TASKS DATAFRAME
    df_milling = dfs["milling"]
    df_milling["duration"] = df_milling["end_time"] - df_milling["start_time"]
    df_milling_filtered = df_milling[
        [
            "lamella",
            "task_name",
            "milling_task_name",
            "stage.name",
            #   "timestamp",
            "duration",
            "stage.milling.milling_current",
            "stage.pattern.depth",
        ]
    ]

    df_milling_filtered = df_milling_filtered.rename(
        columns={
            "lamella": "Lamella Name",
            "task_name": "Task Name",
            "milling_task_name": "Milling Task",
            "stage.name": "Milling Stage",
            "duration": "Duration",
            "stage.milling.milling_current": "Milling Current (A)",
            "stage.pattern.depth": "Depth (m)",
        }
    )

    # format milling current in nA
    df_milling_filtered["Milling Current"] = df_milling_filtered[
        "Milling Current (A)"
    ].apply(lambda x: f"{x * 1e9:.2f} nA")

    # format depth in microns
    df_milling_filtered["Depth"] = df_milling_filtered["Depth (m)"].apply(
        lambda x: f"{x * 1e6:.2f} µm"
    )

    # remove original cols
    df_milling_filtered = df_milling_filtered.drop(
        columns=["Milling Current (A)", "Depth (m)"]
    )

    # as a clock, 4:05 (FIB-1191)
    df_milling_filtered["Duration"] = df_milling_filtered["Duration"].apply(
        format_duration_as_clock
    )

    # DETECTION DATAFRAME
    df_det = dfs["detection"]
    if not df_det.empty:
        df_det_filtered = df_det[
            [
                "lamella",
                "task_name",
                "task_step",
                "feature",
                "is_correct",
                "dm.x",
                "dm.y",
                "beam_type",
            ]
        ]
        df_det_filtered = df_det_filtered.rename(
            columns={
                "lamella": "Lamella Name",
                "task_name": "Task Name",
                "task_step": "Task Step",
                "feature": "Feature",
                "is_correct": "Is Correct",
                "dm.x": "Delta X (m)",
                "dm.y": "Delta Y (m)",
                "beam_type": "Beam Type",
                # "timestamp": "Timestamp"
            }
        )
        # format delta x and y in microns
        df_det_filtered["Delta X"] = df_det_filtered["Delta X (m)"].apply(
            lambda x: f"{x * 1e6:.2f} µm"
        )
        df_det_filtered["Delta Y"] = df_det_filtered["Delta Y (m)"].apply(
            lambda x: f"{x * 1e6:.2f} µm"
        )
        # remove original cols
        df_det_filtered = df_det_filtered.drop(columns=["Delta X (m)", "Delta Y (m)"])

        # group by Task Step, Feature and Is Correct, and get count
        df_det_summary = (
            df_det_filtered.groupby(["Task Step", "Feature", "Is Correct"])
            .agg({"Lamella Name": "count"})
            .reset_index()
        )
        df_det_summary = df_det_summary.rename(columns={"Lamella Name": "Count"})

        # calculate percentage of correct and incorrect
        total_counts = df_det_summary.groupby(["Task Step", "Feature"])[
            "Count"
        ].transform("sum")
        df_det_summary["Percentage"] = (
            df_det_summary["Count"] / total_counts * 100
        ).round(2).astype(str) + "%"

        # add a total count for each Task Step and Feature
        df_totals = (
            df_det_summary.groupby(["Task Step", "Feature"])["Count"]
            .sum()
            .reset_index()
        )
        df_totals = df_totals.rename(columns={"Count": "Total Count"})
        df_det_summary = pd.merge(
            df_det_summary, df_totals, on=["Task Step", "Feature"]
        )
        # remove rows where Is Correct is False. A real bool now: the log used to
        # be read by string substitution, which turned it into the string "True",
        # and a comparison with that string would now drop every row.
        df_det_summary = df_det_summary[df_det_summary["Is Correct"].eq(True)]
        # sort by Task Step and Feature
        df_det_summary = df_det_summary.sort_values(
            by=["Task Step", "Feature"], ascending=[True, True]
        )

        # drop Is Correct column, Count
        df_det_summary = df_det_summary.drop(columns=["Is Correct", "Count"])
    else:
        df_det_filtered = pd.DataFrame()
        df_det_summary = pd.DataFrame()

    return {
        "experiment": df_exp_filtered,
        "workflow": df_workflow,
        "task_history": df_task_history_filtered,
        "task_step_summary": df_task_step_summary,
        "milling": df_milling_filtered,
        "detection": df_det_filtered,
        "detection_summary": df_det_summary,
    }
