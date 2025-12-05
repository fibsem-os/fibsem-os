from __future__ import annotations
import datetime
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from fibsem.applications.autolamella.structures import Experiment, Lamella, LamellaState


class PythonLiteralJSONDecoder(json.JSONDecoder):
    """
    Custom JSON decoder that handles Python literals like tuples, None, True, False
    and other Python-specific syntax when parsing JSON strings.
    """
    def decode(self, s, *args, **kwargs):
        # Replace Python literals with their JSON equivalents
        replacements = {
            # Convert Python literals to JSON equivalents
            "'": '"',       # Single quotes to double quotes
            "None": "null", # Python None to JSON null
            "True": "true", # Python True to JSON true
            "False": "false", # Python False to JSON false
            # Convert tuple syntax to array syntax
            "(": "[",
            ")": "]",
        }
        
        # Apply all replacements
        for old, new in replacements.items():
            s = s.replace(old, new)
            
        try:
            return super().decode(s, *args, **kwargs)
        except json.JSONDecodeError as e:
            # Helpful error message with context
            context = s[max(0, e.pos-20):min(len(s), e.pos+20)]
            raise json.JSONDecodeError(
                f"{e.msg} | Context: '...{context}...'", 
                e.doc, 
                e.pos
            )

def parse_msg(msg: str):
    """parse message json"""
    # turn this into a loop
    # keywords = []
    # return json.loads(msg, cls=PythonLiteralJSONDecoder)
    return json.loads(msg.replace("'", '"').replace("None", '"None"').replace("True", '"True"').replace("False", '"False"').replace("(", "[").replace(")", "]"))

def get_timestamp(line: str) -> float:
    """get timestamp from line"""
    ts = line.split("—")[0].split(",")[0].strip()
    tsd = datetime.datetime.timestamp(datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"))
    return tsd

def get_function(line: str) -> str:
    """get the function name from the line"""
    return line.split("—")[-2].strip()

def get_message(line) -> str:
    """get the message from the line"""
    msg = line.split("—")[
        -1
    ].strip()  # should just be the message # TODO: need to check the delimeter character...
    return msg 

def parse_line(line: str) -> Tuple[str, str, str]:
    """parse a line from the log file into a tuple of timestamp, function, and message"""

    tsd = get_timestamp(line)
    func = get_function(line)
    msg = get_message(line)

    return tsd, func, msg


def calculate_statistics_dataframe(path: Path, encoding: str = "cp1252"):

    fname = os.path.join(path, "logfile.log")
    df_beam_shift = []
    current_lamella = "NULL" 
    current_stage = "SystemSetup"
    current_step = "SystemSetup"
    step_n = 0 
    steps_data = []
    
    state_data = []
    stage_data = []
    det_data = []
    click_data = []
    milling_data = []

    print("-" * 80)
    print(f"Parsing {fname}")
    # encoding = "cp1252" if "nt" in os.name else "cp1252" # TODO: this depends on the OS it was logged on, usually windows, need to make this more robust.
    with open(fname, encoding=encoding) as f:
        # Note: need to check the encoding as this is required for em dash (long dash) # TODO: change this delimiter so this isnt required.
        lines = f.read().splitlines()
        for i, line in enumerate(lines):

            if line == "":
                continue
            try:
                
                # get timestamp, function, and message from log line
                tsd, func, msg = parse_line(line)
                # msgd = parse_msg(msg) # TODO: enable, and remove indiviudal calls

                # TELEMETRY -> depcrecated in favour of manufacturer telemetry
                if "get_microscope_state" in func:
                    msgd = parse_msg(msg)

                    state_data.append(deepcopy(msgd["state"]))
                    
                if "get_stage_position" in func:
                    msgd = parse_msg(msg)
                    staged = msgd["stage"]
                    staged["timestamp"] = tsd
                    staged["lamella"] = current_lamella
                    staged["stage"] = current_stage
                    staged["step"] = current_step
                
                    stage_data.append(deepcopy(staged))

                if "log_status_message" in func:
                    if "STATUS" in msg:
                        continue        # skip old status messages 
                    
                    # global data
                    tsd = get_timestamp(line)
                    msgd = parse_msg(msg)
                    current_lamella = msgd["petname"]
                    current_stage = msgd["stage"]
                    current_step = msgd["step"]

                    # step data                    
                    step_d = deepcopy(msgd)
                    step_d["lamella"] = current_lamella
                    step_d["timestamp"] = tsd
                    step_d["step_n"] = step_n
                    step_n += 1
                    steps_data.append(deepcopy(step_d))

                
                if "beam_shift" in func:
                    msgd = parse_msg(msg)
                    msgd["timestamp"] = tsd
                    msgd["lamella"] = current_lamella
                    msgd["stage"] = current_stage
                    msgd["step"] = current_step
                    df_beam_shift.append(deepcopy(msgd))


                # TODO: confirm this parses the correct data
                if "confirm_button" in func or "save_ml" in func: # DETECTION INTERACTION
                    # log detection data
                    msgd = parse_msg(msg)
                    detd = deepcopy(msgd)

                    detd["px_x"] = msgd["px"]["x"]
                    detd["px_y"] = msgd["px"]["y"]
                    detd["dpx_x"] = msgd["dpx"]["x"]
                    detd["dpx_y"] = msgd["dpx"]["y"]
                    detd["dm_x"] = msgd["dm"]["x"]
                    detd["dm_y"] = msgd["dm"]["y"]
                    
                    del detd["dpx"]
                    del detd["dm"]
                    del detd["px"]

                    detd["timestamp"] = tsd
                    detd["lamella"] = current_lamella
                    detd["stage"] = current_stage
                    detd["step"] = current_step
                    det_data.append(deepcopy(detd))

                    # log detection interaction
                    if detd["is_correct"] == "False":
                        click_d = {
                            "lamella": detd["lamella"],
                            "stage": detd["stage"],
                            "step": detd["step"],
                            "type": "DET",
                            "subtype": detd["feature"],
                            "dm_x": detd["dm_x"],
                            "dm_y": detd["dm_y"],
                            "beam_type": detd["beam_type"],
                            "timestamp": detd["timestamp"],
                        }
                        click_data.append(deepcopy(click_d))    

                if "_single_click" in func: # MILLING INTERACTION
                    # log milling interaction
                    msgd = parse_msg(msg)
                    
                    clickd = {}
                    clickd["timestamp"] = tsd
                    clickd["lamella"] = current_lamella
                    clickd["stage"] = current_stage
                    clickd["step"] = current_step
                    
                    clickd["dm_x"] = msgd["dm"]["x"]
                    clickd["dm_y"] = msgd["dm"]["y"]
                    clickd["type"] = "MILL"
                    clickd["subtype"] = msgd["pattern"]
                    clickd["beam_type"] = msgd["beam_type"]

                    click_data.append(deepcopy(clickd))

                if "_double_click" in func: # MOVEMENT INTERACTION
                    
                    # log movement interaction
                    msgd = parse_msg(msg)
                    clickd = {}
                    clickd["timestamp"] = tsd
                    clickd["lamella"] = current_lamella
                    clickd["stage"] = current_stage
                    clickd["step"] = current_step

                    clickd["dm_x"] = msgd["dm"]["x"]
                    clickd["dm_y"] = msgd["dm"]["y"]
                    clickd["type"] = "MOVE"
                    clickd["subtype"] = msgd["movement_mode"]
                    clickd["beam_type"] = msgd["beam_type"]

                    click_data.append(deepcopy(clickd))

                if "mill_stages" in func:
                    msgd = parse_msg(msg)

                    milld = {}
                    milld["timestamp"] = tsd
                    milld["lamella"] = current_lamella
                    milld["stage"] = current_stage
                    milld["step"] = current_step
                    milld["name"] = msgd["stage"]["name"]
                    milld["start_time"] = msgd["start_time"]
                    milld["end_time"] = msgd["end_time"]
                    milld["duration"] = msgd["end_time"] - msgd["start_time"]
                    milld["milling_current"] = msgd["stage"]["milling"]["milling_current"]
                    milld["depth"] = msgd["stage"]["pattern"].get("depth", 0)
                    # TODO: what other attrs are useful?
                    milling_data.append(deepcopy(milld))

            except Exception as e:
                # print(e, " | ", line)
                pass
 
    # experiment
    experiment = Experiment.load(os.path.join(path, "experiment.yaml"))
    df_experiment = experiment.__to_dataframe__()
    df_history = experiment.history_dataframe()
    df_steps = pd.DataFrame(steps_data)
    df_stage = pd.DataFrame(stage_data)
    df_det = pd.DataFrame(det_data)
    df_beam_shift = pd.DataFrame.from_dict(df_beam_shift) # TODO: remove this, not used
    df_click = pd.DataFrame(click_data)
    df_milling = pd.DataFrame(milling_data)

    if not df_steps.empty:
        df_steps["duration"] = df_steps["timestamp"].diff() # TODO: fix this duration
        df_steps["duration"] = df_steps["duration"].shift(-1)


    # add date and name to all dataframes
    df_experiment["exp_name"] = experiment.name
    df_history["exp_name"] = experiment.name
    df_beam_shift["exp_name"] = experiment.name
    df_steps["exp_name"] = experiment.name
    df_stage["exp_name"] = experiment.name
    df_det["exp_name"] = experiment.name
    df_click["exp_name"] = experiment.name
    df_milling["exp_name"] = experiment.name

    # add experiment id to all df
    df_history["exp_id"] = experiment._id if experiment._id is not None else "NO_ID"
    df_beam_shift["exp_id"] = experiment._id if experiment._id is not None else "NO_ID"
    df_steps["exp_id"] = experiment._id if experiment._id is not None else "NO_ID"
    df_stage["exp_id"] = experiment._id if experiment._id is not None else "NO_ID"
    df_det["exp_id"] = experiment._id if experiment._id is not None else "NO_ID"
    df_click["exp_id"] = experiment._id if experiment._id is not None else "NO_ID"
    df_milling["exp_id"] = experiment._id if experiment._id is not None else "NO_ID"

    # write dataframes to csv, overwrite
    filename = os.path.join(path, 'history.csv')
    df_history.to_csv(filename, mode='w', header=True, index=False)
    filename = os.path.join(path, 'beam_shift.csv')
    df_beam_shift.to_csv(filename, mode='w', header=True, index=False)
    filename = os.path.join(path, 'experiment.csv')
    df_experiment.to_csv(filename, mode='w', header=True, index=False)
    filename = os.path.join(path, 'steps.csv')
    df_steps.to_csv(filename, mode='w', header=True, index=False)
    filename = os.path.join(path, 'stage.csv')
    df_stage.to_csv(filename, mode='w', header=True, index=False)
    filename = os.path.join(path, 'det.csv')
    df_det.to_csv(filename, mode='w', header=True, index=False)
    filename = os.path.join(path, 'click.csv')
    df_click.to_csv(filename, mode='w', header=True, index=False)
    filename = os.path.join(path, 'milling.csv')
    df_milling.to_csv(filename, mode='w', header=True, index=False)

    return df_experiment, df_history, df_beam_shift, df_steps, df_stage, df_det, df_click, df_milling



#### TASK REFACTORING ####

def parse_logfile(path: str, encoding="utf-8") -> Dict[str, pd.DataFrame]:
    """Updated parser for task based workflow"""

    fname = os.path.join(path, "logfile.log")
    steps_data = []
    det_data = []
    click_data = []
    milling_data2 = []

    stepd = None

    print("-" * 80)
    print(f"Parsing {fname}")
    # encoding = "cp1252" if "nt" in os.name else "cp1252" # TODO: this depends on the OS it was logged on, usually windows, need to make this more robust.
    with open(fname, encoding=encoding) as f:
        # Note: need to check the encoding as this is required for em dash (long dash) # TODO: change this delimiter so this isnt required.
        lines = f.read().splitlines()
        for i, line in enumerate(lines):

            if line == "":
                continue
            try:

                # get timestamp, function, and message from log line
                tsd, func, msg = parse_line(line)
                msgd = parse_msg(msg)

                if "milling_task" in msg:

                    if stepd is None:
                        continue

                    # add milling task data
                    msgd2 = stepd
                    msgd2["timestamp"] = tsd
                    msgd2.update(msgd)
                    milling_data2.append(deepcopy(msgd2))

                if "log_status_message" in func:
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

                if "save_ml" in func: # DETECTION INTERACTION
                    # log detection data
                    msgd = parse_msg(msg)

                    if stepd is None:
                        continue

                    dmsgd2 = stepd
                    dmsgd2["timestamp"] = tsd
                    dmsgd2.update(msgd)
                    det_data.append(deepcopy(dmsgd2))

                    # # log detection interaction
                    # if detd["is_correct"] == "False":
                    #     click_d = {
                    #         "lamella": detd["lamella"],
                    #         "stage": detd["stage"],
                    #         "step": detd["step"],
                    #         "type": "DET",
                    #         "subtype": detd["feature"],
                    #         "dm_x": detd["dm_x"],
                    #         "dm_y": detd["dm_y"],
                    #         "beam_type": detd["beam_type"],
                    #         "timestamp": detd["timestamp"],
                    #     }
                    #     click_data.append(deepcopy(click_d))    

                # if "_single_click" in func: # MILLING INTERACTION
                #     # log milling interaction
                #     msgd = parse_msg(msg)
                    
                #     clickd = {}
                #     clickd["timestamp"] = tsd
                #     clickd["lamella"] = current_lamella
                #     clickd["stage"] = current_stage
                #     clickd["step"] = current_step
                    
                #     clickd["dm_x"] = msgd["dm"]["x"]
                #     clickd["dm_y"] = msgd["dm"]["y"]
                #     clickd["type"] = "MILL"
                #     clickd["subtype"] = msgd["pattern"]
                #     clickd["beam_type"] = msgd["beam_type"]

                #     click_data.append(deepcopy(clickd))

                # if "_double_click" in func: # MOVEMENT INTERACTION
                    
                #     # log movement interaction
                #     msgd = parse_msg(msg)
                #     clickd = {}
                #     clickd["timestamp"] = tsd
                #     clickd["lamella"] = current_lamella
                #     clickd["stage"] = current_stage
                #     clickd["step"] = current_step

                #     clickd["dm_x"] = msgd["dm"]["x"]
                #     clickd["dm_y"] = msgd["dm"]["y"]
                #     clickd["type"] = "MOVE"
                #     clickd["subtype"] = msgd["movement_mode"]
                #     clickd["beam_type"] = msgd["beam_type"]


            except Exception as e:
                pass

    df_tasks = pd.DataFrame(steps_data)
    df_tasks["duration"] = df_tasks["timestamp"].diff().shift(-1)
    df_tasks.fillna(0, inplace=True)

    from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol
    exp = Experiment.load(os.path.join(path, "experiment.yaml")) # type: ignore
    exp.task_protocol = AutoLamellaTaskProtocol.load(os.path.join(path, "protocol.yaml"))
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
    df_exp.to_csv(filename, mode='w', header=True, index=False)
    filename = os.path.join(path, "workflow.csv")
    df_workflow.to_csv(filename, mode="w", header=True, index=False)
    filename = os.path.join(path, "task_history.csv")
    df_task_history.to_csv(filename, mode="w", header=True, index=False)
    filename = os.path.join(path, "milling.csv")
    df_milling2.to_csv(filename, mode="w", header=True, index=False)
    filename = os.path.join(path, "detection.csv")
    df_det.to_csv(filename, mode="w", header=True, index=False)

    return {"experiment": df_exp, "workflow": df_workflow,
            "tasks": df_tasks, 
            "milling": df_milling2,
            "task_history": df_task_history,
            "detection": df_det}


def format_pretty_dataframes(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Format dataframes for pretty printing in notebook/report"""
    df_exp = dfs["experiment"]

    # only show lamella_name, last_complete, last_completed_at, milling_angle, is_failure
    df_exp_filtered = df_exp[["lamella_name", "last_completed", "milling_angle", "is_completed", "is_failure"]]

    # rename these cols, to be pretty
    df_exp_filtered = df_exp_filtered.rename(columns={
        "lamella_name": "Lamella Name",
        "last_completed": "Last Completed Task",
        "milling_angle": "Milling Angle",
        "is_completed": "Completed",
        "is_failure": "Is Failure"
    })

    # round milling angle to 1 decimal place (use degree symbol)
    df_exp_filtered["Milling Angle"] = df_exp_filtered["Milling Angle"].round(1)


    # WORKFLOW
    df_workflow = dfs["workflow"]
    # rename cols
    df_workflow = df_workflow.rename(columns={
        "order": "Order",
        "task": "Task Name",
        "required": "Required",
        "supervised": "Supervised",
    })

    # TASK HISTORY
    df_task_history = dfs["task_history"]

    # filter to only show lamella_name, task_name, completed_at, duration
    df_task_history_filtered = df_task_history[["lamella_name", "task_name", "completed_at", "duration"]]

    df_task_history_filtered = df_task_history_filtered.rename(columns={
        "lamella_name": "Lamella Name",
        "task_name": "Task Name",
        "task_type": "Task Type",
        "completed_at": "Completed At",
        "duration": "Duration"
    })

    # format duration to be in minutes and seconds
    df_task_history_filtered["Duration"] = pd.to_timedelta(df_task_history_filtered["Duration"], unit='s')
    # format MM:SS
    df_task_history_filtered["Duration"] = df_task_history_filtered["Duration"].apply(
        lambda x: f"{int(x.total_seconds()//60):02d}:{int(x.total_seconds()%60):02d}"
    )

    # sort by Completed At
    df_task_history_filtered = df_task_history_filtered.sort_values(by=["Lamella Name", "Completed At"], ascending=[True, True])


    ### TASK STEP HISTORY
    df_task_step_history = dfs["tasks"]

    # filter to only show lamella_name, task_name, task_step, duration, timestamp
    df_task_step_history_filtered = df_task_step_history[["lamella", "task_name", "task_step", "duration", "timestamp"]]

    df_task_step_history_filtered = df_task_step_history_filtered.rename(columns={
        "lamella": "Lamella Name",
        "task_name": "Task Name",
        "task_step": "Task Step",
        "duration": "Duration",
        "timestamp": "Timestamp"
    })

    # format timestamps fromtimestamp to datetime
    df_task_step_history_filtered["Timestamp"] = pd.to_datetime(df_task_step_history_filtered["Timestamp"], unit='s')

    # drop task_step if STARTED, FINISHED
    df_task_step_history_filtered = df_task_step_history_filtered[~df_task_step_history_filtered["Task Step"].isin(["STARTED", "FINISHED"])]

    # group by Task Name and Task Step, and get mean duration
    df_task_step_summary = df_task_step_history_filtered.groupby(["Task Name", "Task Step"]).agg({"Duration": "mean", "Lamella Name": "count"}).reset_index()
    df_task_step_summary = df_task_step_summary.rename(columns={
        "Duration": "Mean Duration",
        "Lamella Name": "Count"
    })

    # format duration to be in minutes and seconds
    df_task_step_summary["Mean Duration"] = pd.to_timedelta(df_task_step_summary["Mean Duration"], unit='s')
    # show as total seconds
    df_task_step_summary["Mean Duration"] = df_task_step_summary["Mean Duration"].apply(
        lambda x: f"{x.total_seconds():.1f} s"
    )
    # sort by Task Name, then Task Step
    df_task_step_summary = df_task_step_summary.sort_values(by=["Task Name", "Task Step"], ascending=[True, True])


    # MILLLING TASKS DATAFRAME
    df_milling = dfs["milling"]
    df_milling["duration"] = df_milling["end_time"] - df_milling["start_time"]
    df_milling_filtered = df_milling[["lamella", "task_name",
                                      "milling_task_name", "stage.name",
                                    #   "timestamp", 
                                      "duration",
                                      "stage.milling.milling_current",
                                      "stage.pattern.depth"]]

    df_milling_filtered = df_milling_filtered.rename(columns={
        "lamella": "Lamella Name",
        "task_name": "Task Name",
        "milling_task_name": "Milling Task",
        "stage.name": "Milling Stage",
        "duration": "Duration",
        "stage.milling.milling_current": "Milling Current (A)",
        "stage.pattern.depth": "Depth (m)"
    })

    # format milling current in nA
    df_milling_filtered["Milling Current"] = df_milling_filtered["Milling Current (A)"].apply(
        lambda x: f"{x*1e9:.2f} nA"
    )

    # format depth in microns
    df_milling_filtered["Depth"] = df_milling_filtered["Depth (m)"].apply(
        lambda x: f"{x*1e6:.2f} µm"
    )

    # remove original cols
    df_milling_filtered = df_milling_filtered.drop(columns=["Milling Current (A)", "Depth (m)"])

    # format duration to be in minutes and seconds
    df_milling_filtered["Duration"] = pd.to_timedelta(df_milling_filtered["Duration"], unit='s')
    # format as MMm:SSs
    df_milling_filtered["Duration"] = df_milling_filtered["Duration"].apply(
        lambda x: f"{int(x.total_seconds()//60)}m:{int(x.total_seconds()%60)}s"
    )

    # DETECTION DATAFRAME
    df_det = dfs["detection"]
    if not df_det.empty:
        df_det_filtered = df_det[["lamella", "task_name", "task_step", "feature", 
                                  "is_correct", "dm.x", "dm.y", "beam_type"]]
        df_det_filtered = df_det_filtered.rename(columns={
            "lamella": "Lamella Name",
            "task_name": "Task Name",
            "task_step": "Task Step",
            "feature": "Feature",
            "is_correct": "Is Correct",
            "dm.x": "Delta X (m)",
            "dm.y": "Delta Y (m)",
            "beam_type": "Beam Type",
            # "timestamp": "Timestamp"
        })
        # format delta x and y in microns
        df_det_filtered["Delta X"] = df_det_filtered["Delta X (m)"].apply(
            lambda x: f"{x*1e6:.2f} µm"
        )
        df_det_filtered["Delta Y"] = df_det_filtered["Delta Y (m)"].apply(
            lambda x: f"{x*1e6:.2f} µm"
        )
        # remove original cols
        df_det_filtered = df_det_filtered.drop(columns=["Delta X (m)", "Delta Y (m)"])

        # group by Task Step, Feature and Is Correct, and get count
        df_det_summary = df_det_filtered.groupby(["Task Step", "Feature", "Is Correct"]).agg({"Lamella Name": "count"}).reset_index()
        df_det_summary = df_det_summary.rename(columns={
            "Lamella Name": "Count"
        })

        # calculate percentage of correct and incorrect
        total_counts = df_det_summary.groupby(["Task Step", "Feature"])["Count"].transform("sum")
        df_det_summary["Percentage"] = (df_det_summary["Count"] / total_counts * 100).round(2).astype(str) + '%'

        # add a total count for each Task Step and Feature
        df_totals = df_det_summary.groupby(["Task Step", "Feature"])["Count"].sum().reset_index()
        df_totals = df_totals.rename(columns={
            "Count": "Total Count"
        })
        df_det_summary = pd.merge(df_det_summary, df_totals, on=["Task Step", "Feature"])
        # remove rows where Is Correct is False
        df_det_summary = df_det_summary[df_det_summary["Is Correct"] == "True"]
        # sort by Task Step and Feature
        df_det_summary = df_det_summary.sort_values(by=["Task Step", "Feature"], ascending=[True, True])

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