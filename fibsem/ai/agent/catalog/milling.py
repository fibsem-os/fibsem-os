"""The milling tools. Their actions are said by the milling service's commands (`fibsem.services.milling`)."""

from typing import Tuple

from fibsem.ai.agent.catalog.spec import ToolSpec, command_body, kwargs_body

_HOLD_NOTE = (
    "As milling_stop: a mill the agent started itself it can pause and resume at "
    "Assist or Drive, never at Watch; anyone else's needs Drive, and at Assist "
    "goes in request_create for the operator to approve."
)
SPOT_BURN_SETTINGS = (
    "SpotBurnSettings dict: coordinates [{x, y, name}] (0-1 in the ion image), "
    "milling_current (A, default 100 pA), exposure_time (s per spot, 0.1-60, "
    "default 10)"
)

TOOLS: Tuple[ToolSpec, ...] = (
    ToolSpec(
        name="milling_get_state",
        description="Milling state (IDLE, RUNNING, PAUSED, ...).",
        method="GET",
        path="/microscope/services/milling/state",
        level="watch",
    ),
    ToolSpec(
        name="milling_estimate_time",
        description=(
            "How long the drawn patterns take to mill, in seconds, and the ion "
            "current they mill at (milling_current, amps). warning is set when the "
            "recipe asked for a current the beam does not list; it names the "
            "nearest listed currents. With stages (FibsemMillingStage dicts, as "
            "run_stage takes), how long those would take before anything is "
            "drawn: seconds in all, and each stage's."
        ),
        method="POST",
        path="/microscope/services/milling/queries/estimate_time",
        level="watch",
        params={"stages": "FibsemMillingStage dicts; left out, what prepare drew"},
        optional=("stages",),
        call=lambda a: ({}, kwargs_body(stages=a.get("stages"))),
    ),
    ToolSpec(
        name="milling_run_stage",
        description=(
            "Mill one stage as a task of its own: prepare its patterns, run them "
            "with the stage's strategy, and put the beam back. stage is a "
            "FibsemMillingStage dict, e.g. one from item_get_task_config; "
            "milling_preview_patterns first to see where it mills. It saves images "
            "only when the stage's milling.acquire_images is set, under path "
            "(a folder inside the open experiment; left out, the app's data "
            "folder). A long command: it answers "
            "with its run at once; service_wait_for_run milling gives how it "
            "ended and the folder it saved to; milling_stop stops it, and "
            "milling_end ends it keeping what it milled. A stage can list "
            "monitors (milling_list_options)."
        ),
        method="POST",
        path="/microscope/services/milling/commands/run_stage",
        level="drive",
        hardware=True,
        params={
            "stage": "a FibsemMillingStage dict",
            "path": "optional folder inside the open experiment to save into",
        },
        optional=("path",),
        call=lambda a: (
            {},
            kwargs_body(stage=dict(a["stage"]), path=a.get("path")),
        ),
    ),
    ToolSpec(
        name="milling_run_spot_burn",
        description=(
            "Burn spots with the ion beam, as the Spot Burn tab does: each point "
            "for exposure_time at milling_current, the beam blanked between them "
            "and its current put back after. settings.coordinates are points in "
            "the ion image, 0 to 1 on both axes ({x, y, name}); points outside "
            "it are dropped. milling_estimate_spot_burn first for how long. A long "
            "command: it answers with its run at once; service_wait_for_run "
            "spot_burn says how it ended; service_call_command spot_burn stop, "
            "with a reason, ends it."
        ),
        method="POST",
        path="/microscope/services/spot_burn/commands/run",
        level="drive",
        hardware=True,
        params={"settings": SPOT_BURN_SETTINGS},
        call=lambda a: ({}, kwargs_body(settings=dict(a["settings"]))),
    ),
    ToolSpec(
        name="milling_estimate_spot_burn",
        description=(
            "How long milling_run_spot_burn with these settings burns, in seconds: "
            "the points inside the image times exposure_time. Moves nothing."
        ),
        method="POST",
        path="/microscope/services/spot_burn/queries/estimate",
        level="watch",
        params={"settings": SPOT_BURN_SETTINGS},
        call=lambda a: ({}, kwargs_body(settings=dict(a["settings"]))),
    ),
    ToolSpec(
        name="milling_run_task",
        description=(
            "Run a milling task as the app runs a protocol's: each enabled stage "
            "in turn with its strategy, aligning to the task's reference image "
            "between stages, and the beam put back at the end. config is a "
            "FibsemMillingTaskConfig dict, e.g. one from item_get_task_config. "
            "It saves its reference and stage images under "
            "<path>/Milling/<name>, path a folder inside the open experiment "
            "(left out, the app's data folder). "
            "A long command: it answers with its run at once; "
            "service_wait_for_run milling gives the task's status, the folder "
            "and how each stage ended; milling_stop ends it."
        ),
        method="POST",
        path="/microscope/services/milling/commands/run_task",
        level="drive",
        hardware=True,
        params={
            "config": "a FibsemMillingTaskConfig dict",
            "path": "optional folder inside the open experiment to save into",
        },
        optional=("path",),
        call=lambda a: (
            {},
            kwargs_body(config=dict(a["config"]), path=a.get("path")),
        ),
    ),
    ToolSpec(
        name="milling_preview_patterns",
        description=(
            "Milling stages drawn over a saved ion image, to see where they would "
            "mill before running them: an image you can look at, the saved image's "
            "shape scaled to 1024 px wide. stages is a list of FibsemMillingStage "
            "dicts (as run_stage takes); left out, the patterns prepare drew. "
            "image_path must be inside the open experiment. Moves nothing."
        ),
        method="POST",
        path="/microscope/services/milling/queries/preview_patterns",
        level="watch",
        params={
            "image_path": "a saved ion image inside the open experiment",
            "stages": "FibsemMillingStage dicts; left out, what prepare drew",
        },
        optional=("stages",),
        call=lambda a: (
            {},
            kwargs_body(image_path=str(a["image_path"]), stages=a.get("stages")),
        ),
    ),
    ToolSpec(
        name="milling_check_stages",
        description=(
            "Check milling stages before anything is drawn: per stage, how long it "
            "would mill (seconds) and the patterns that reach past its field (its "
            "milling hfw wide, aspect of that high), which a real instrument "
            "refuses to mill. A stage that cannot define its patterns says why. "
            "Moves nothing."
        ),
        method="POST",
        path="/microscope/services/milling/queries/check_stages",
        level="watch",
        params={
            "stages": "FibsemMillingStage dicts",
            "aspect": "the image's height over width (default 2/3, a 1536 x 1024 image)",
        },
        optional=("aspect",),
        call=lambda a: (
            {},
            kwargs_body(
                stages=[dict(stage) for stage in a["stages"]],
                aspect=float(a.get("aspect") or 2 / 3),
            ),
        ),
    ),
    ToolSpec(
        name="milling_list_options",
        description=(
            "The pattern names a stage's pattern.name takes, the strategy names its "
            "strategy.name takes, and the monitors a stage can list: each name with "
            "its config's fields. A stage lists a monitor in its monitors as "
            "{name, config}; a protocol stage's monitor config is patched as "
            "stages.<stage>.monitors.<i>.config.<field>. settings: the recipe "
            "(stage.milling) fields this instrument mills with, per channel, each "
            "with its limits and choices; a driver's own are named by their path "
            "(driver.Tescan.preset), and the fields left out it ignores."
        ),
        method="POST",
        path="/microscope/services/milling/queries/list_options",
        level="watch",
        call=lambda a: ({}, kwargs_body()),
    ),
    ToolSpec(
        name="milling_end",
        description=(
            "End the milling stage now running, completed, keeping what it milled: "
            "the pattern stops, the stage starts no further step and a milling task "
            "goes on to its next stage. For a monitor's suggestion (a "
            "MILL READING line with on_trigger=suggest) you agree with. "
            "milling_stop instead stops the whole task. Refused 409 NothingToStop "
            "when nothing is milling."
        ),
        method="POST",
        path="/microscope/services/milling/commands/end",
        level="drive",
        params={"reason": "why you are ending it, for the operator (required)"},
        note="As milling_stop: a mill the agent started itself it can end at Assist or Drive, never at Watch; anyone else's needs Drive, and at Assist goes in request_create for the operator to approve.",
        call=lambda a: (
            {},
            command_body({"by": "agent"}, a.get("reason")),
        ),
    ),
    ToolSpec(
        name="milling_pause",
        description=(
            "Pause the mill now running, whoever started it: the pattern holds "
            "where it is until milling_resume. While a monitor has it paused (an "
            "SEM peek) it stays paused when the monitor is done. Use it to look "
            "before deciding; milling_stop or milling_end to finish."
        ),
        method="POST",
        path="/microscope/services/milling/commands/pause",
        level="drive",
        note=_HOLD_NOTE,
        call=lambda a: ({}, kwargs_body()),
    ),
    ToolSpec(
        name="milling_resume",
        description=(
            "Resume the mill milling_pause paused. While a monitor has it "
            "paused, it resumes when the monitor is done."
        ),
        method="POST",
        path="/microscope/services/milling/commands/resume",
        level="drive",
        note=_HOLD_NOTE,
        call=lambda a: ({}, kwargs_body()),
    ),
    ToolSpec(
        name="milling_follow_readings",
        description=(
            "The readings of the running mill's monitors, numbered after since, "
            "waiting up to timeout seconds (at most 30) for one: {seq, readings, "
            "missed}. Pass the answer's seq as the next since to follow along. "
            "Notable readings (a drop, an error) also come as MILL READING lines; "
            "this gives every reading, to judge for yourself."
        ),
        method="POST",
        path="/microscope/services/milling/queries/follow_readings",
        level="watch",
        params={
            "since": "the seq of the last reading you have (0 for all kept)",
            "timeout": "seconds to wait for a new reading, at most 30",
        },
        optional=("since", "timeout"),
        call=lambda a: (
            {},
            kwargs_body(
                since=int(a.get("since") or 0), timeout=float(a.get("timeout") or 0.0)
            ),
        ),
    ),
    ToolSpec(
        name="milling_stop",
        description="Stop an in-progress mill, with the reason the operator reads: the whole milling task, so no later stage starts. Refused 409 NothingToStop when nothing is milling.",
        method="POST",
        path="/microscope/services/milling/commands/stop",
        level="drive",
        params={"reason": "why you are stopping it, for the operator (required)"},
        note="A mill the agent started itself it can stop at Assist or Drive, never at Watch; anyone else's (the workflow's, the operator's) needs Drive, and at Assist goes in request_create for the operator to approve. Never refused for someone else being in control of the microscope.",
        # A service command or query with no arguments still sends its kwargs.
        call=lambda a: ({}, command_body(reason=a.get("reason"))),
    ),
)
