"""The coating tools. Their actions are said by the coating service's commands (`fibsem.services.coating`)."""

from typing import List, Tuple

from fibsem.ai.agent.catalog.spec import ToolSpec, arg, kwargs_body

TOOLS: Tuple[ToolSpec, ...] = (
    ToolSpec(
        name="coating_deposit",
        description=(
            "Deposit with the GIS, as the Sample tab's Deposit does: move the "
            "stage to the instrument's deposition position (service_get_parameter "
            "coating deposition_position; none means where the stage is), insert "
            "the needle, warm the gas, open the valve for seconds, close it, "
            "retract and move the stage back. The valve, needle and stage are put "
            "back however it ends. The GIS's own commands are not the agent's. A "
            "long command: it answers with its run at once; service_wait_for_run "
            "coating says how it ended; service_call_command coating stop, with a "
            "reason, closes the valve early."
        ),
        method="POST",
        path="/microscope/services/coating/commands/deposit",
        result_key="run",
        level="drive",
        hardware=True,
        params={"seconds": arg("how long the valve stays open, seconds", float)},
        call=lambda a: ({}, kwargs_body(seconds=float(a["seconds"]))),
    ),
    ToolSpec(
        name="coating_sputter",
        description=(
            "Sputter coat, as the Sample tab's Sputter does, for seconds (whole "
            "seconds on a Thermo coater) at current (amperes, one of the "
            "sputter_coater's current choices; left out, as it is). On an Arctis "
            "the coater moves the stage to it and back itself. grids, the coater's "
            "1-based grid numbers, only where it coats each grid itself "
            "(device_list: sputter_coater coats_grids). The coater's own commands "
            "are not the agent's. A long command that can't be stopped: it answers "
            "with its run at once; service_wait_for_run coating says how it ended."
        ),
        method="POST",
        path="/microscope/services/coating/commands/sputter",
        result_key="run",
        level="drive",
        hardware=True,
        params={
            "seconds": arg("how long to sputter, seconds", float),
            "current": arg("sputter current in amperes", float),
            "grids": arg(
                "the grid numbers to coat, where the coater takes them", List[int]
            ),
        },
        optional=("current", "grids"),
        call=lambda a: (
            {},
            kwargs_body(
                seconds=float(a["seconds"]),
                current=None if a.get("current") is None else float(a["current"]),
                grids=a.get("grids"),
            ),
        ),
    ),
)
