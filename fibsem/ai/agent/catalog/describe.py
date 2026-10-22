"""What each requested action does, in the operator's words (FIB-1283).

The approval popover shows every action of a request as two lines: what it acts
on (an item and task, the stage, a beam) and the change. The server writes them
from the tool and its arguments, never from the agent's own words, so an agent
cannot word an action to look like something else. Only the request's reason is
the agent's.

Each tool says its own line (``ToolSpec.line``), and each service command its
own (``command(line=...)``), whichever tool calls it. A config edit shows each
dotted path as old -> new, the old value read from the same document the edit
patches, when ``read`` can supply it. Where the app can reach the typed config
(``describe``), each path reads as its form names it, in display units: "Rough
Mill 01 · trench depth 2 µm → 2.5 µm" (FIB-1403).
"""

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from fibsem.ai.agent.catalog import CATALOG, route_call
from fibsem.util import wording

# The app's own description of an action it can say better (a config edit, from
# the typed config): a partial ``{"what", "change", "details"}``, or None.
Describer = Callable[[str, Dict[str, Any]], Optional[Dict[str, Any]]]

_TOOLS = {spec.name: spec for spec in CATALOG}
_SERVICE_COMMAND = re.compile(r"/microscope/services/(\w+)/commands/(\w+)")
_UNLOADS_FIRST = re.compile(r" \(unloads '[^']*' first\)")


def describe_actions(
    actions: Sequence[Tuple[str, Dict[str, Any], str]],
    read: Optional[wording.Reader] = None,
    describe: Optional[Describer] = None,
) -> List[Dict[str, Any]]:
    """Each of a request's ``(tool, arguments, path)`` described, in order. An
    action can read what came before it: a milling run says how long the
    patterns an earlier action prepares take.

    A line that can only fail when it runs carries ``refused``, why: a grid
    load the inventory doesn't hold, with no inventory scan before it. The
    request is refused at filing for it (FIB-1493)."""
    from fibsem.services.sample_handling import NOT_IN_INVENTORY

    described = []
    estimate: Optional[Tuple[Optional[float], Any]] = None
    unloaded = inventoried = inserted = False
    for tool, arguments, path in actions:
        line = _describe(tool, arguments, path, read, describe, estimate)
        if "_estimate" in line:
            estimate = line.pop("_estimate")
        grids = _grid_command(tool, arguments)
        if unloaded and grids == "ensure_loaded":
            # An earlier step unloads it: the load's own "(unloads 'G1'
            # first)", read from the stage as it is now, says it twice.
            line["change"] = _UNLOADS_FIRST.sub("", str(line.get("change")))
        if (
            grids == "ensure_loaded"
            and not inventoried
            and str(line.get("change")).endswith(NOT_IN_INVENTORY)
        ):
            name = _command_kwargs(tool, arguments).get("grid_name")
            line["refused"] = (
                f"grid {name!r} is not in the inventory; run an inventory first "
                "(grid_run_inventory), or load a grid it holds"
            )
        if tool == "fm_acquire_overview" and not inserted:
            state = _objective_state(read)
            if state and state != "inserted":
                line["change"] = (
                    f"{line.get('change')}\nThe objective is {state} now and no "
                    "step before this inserts it, so the overview will be refused"
                )
        unloaded = unloaded or grids == "unload"
        inventoried = inventoried or grids == "run_inventory"
        inserted = inserted or _inserts_objective(tool, arguments)
        described.append(line)
    return described


def describe_action(
    tool: str,
    arguments: Dict[str, Any],
    path: str,
    read: Optional[wording.Reader] = None,
    describe: Optional[Describer] = None,
) -> Dict[str, Any]:
    """``{"what": ..., "change": ...}`` for one action, with ``details`` where
    there is more to show. ``path`` is the route the action calls; ``read`` GETs
    a route, for the old values of a config edit; ``describe`` is the app's own
    description, for what it can say better."""
    line = _describe(tool, arguments, path, read, describe, None)
    line.pop("_estimate", None)
    return line


def describe_done(tool: Any, arguments: Any, what: str) -> Optional[str]:
    """A done call, said as done: "Confirmed Mill Rough on 02-alive-skunk".
    ``what`` is the call's recorded line, ``item · task · change``. None for a
    tool said the same way before and after."""
    if tool != "proposal_decide":
        return None
    a = arguments if isinstance(arguments, dict) else {}
    parts = [p.strip() for p in str(what or "").split(" · ")]
    item = parts[0] if len(parts) >= 3 else a.get("item_id")
    task = parts[1] if len(parts) >= 3 else a.get("task_name")
    on = f"{task} on {item}" if task else str(item or "the proposal")
    if str(a.get("outcome", "")).lower().startswith("reject"):
        return f"Rejected {on}: {a.get('reason', '')}".rstrip(": ")
    if a.get("action"):
        return f"Decided {on}: {a.get('action')}"
    line = f"Confirmed {on}"
    if a.get("values"):
        line += f" with {wording.values_text(a.get('values'))}"
    if a.get("patch"):
        line += f", setting {wording.values_text(a.get('patch'))}"
    return line


def _describe(
    tool: str,
    arguments: Dict[str, Any],
    path: str,
    read: Optional[wording.Reader],
    describe: Optional[Describer],
    estimate: Optional[Tuple[Optional[float], Any]],
) -> Dict[str, Any]:
    a = arguments or {}
    line = _line(tool, a, wording.ActionContext(path, read, estimate))
    if describe is not None:
        try:
            own = describe(tool, a)
        except Exception:  # noqa: BLE001 - the server's own lines stand
            own = None
        if own:
            line = {**line, **own}
    if line.get("details") is None:
        line.pop("details", None)
    return line


def _line(tool: str, a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    spec = _TOOLS.get(tool)
    line = None
    if spec is not None and spec.line is not None:
        line = spec.line(a, ctx)
    elif tool in ("device_call_command", "service_call_command"):
        line = _call_command(tool, a, ctx)
    else:
        command = _service_command(tool, a)
        if command is not None:
            said = _command_line(*command)
            if said is not None:
                line = said(_command_kwargs(tool, a), ctx)
    if line:
        return line
    line = {"what": wording.words(tool).capitalize(), "change": "Run"}
    details = wording.kwargs_table(a)
    if details is not None:
        line["details"] = details
    return line


def _call_command(
    tool: str, a: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    """A device or service command called by name: the command's own line
    where it has one, else its name and arguments."""
    owner = a.get("device") if tool == "device_call_command" else a.get("service")
    kwargs = a.get("kwargs") or {}
    if not isinstance(kwargs, dict):
        kwargs = {}
    if tool == "service_call_command":
        said = _command_line(str(owner), str(a.get("command")))
        line = said(kwargs, ctx) if said is not None else None
        if line:
            return line
    extra = ", ".join(
        f"{wording.words(k)} {wording.clip(wording.short(v))}"
        for k, v in kwargs.items()
    )
    if len(extra) > 2 * wording.LINE_VALUE:
        extra = wording.plural(len(kwargs), "argument")
    line = {
        "what": wording.words(owner).capitalize() if owner else "",
        "change": f"{wording.words(a.get('command', '')).capitalize()}"
        + (f": {extra}" if extra else ""),
    }
    details = wording.kwargs_table(kwargs)
    if details is not None and wording.short(kwargs) not in line["change"]:
        line["details"] = details
    return line


def _command_line(service: str, command: str) -> Optional[wording.Line]:
    from fibsem.services.core import command_line

    return command_line(service, command)


def _service_command(tool: str, a: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """The service command a call runs, as ``(service, command)``: by name, or
    the one a named tool's route is."""
    if tool == "service_call_command":
        return str(a.get("service")), str(a.get("command"))
    spec = _TOOLS.get(tool)
    found = _SERVICE_COMMAND.fullmatch(spec.path) if spec is not None else None
    return (found.group(1), found.group(2)) if found else None


def _command_kwargs(tool: str, a: Dict[str, Any]) -> Dict[str, Any]:
    """The command's arguments: as given by name, or as the tool's call makes
    them."""
    spec = _TOOLS.get(tool)
    if spec is None or tool == "service_call_command":
        kwargs = a.get("kwargs")
        return kwargs if isinstance(kwargs, dict) else {}
    try:
        _, _, body = route_call(spec, a)
    except ValueError:
        return dict(a)
    kwargs = (body or {}).get("kwargs")
    return kwargs if isinstance(kwargs, dict) else {}


def _grid_command(tool: str, a: Dict[str, Any]) -> Optional[str]:
    """The sample handling command a call runs, called either way."""
    command = _service_command(tool, a or {})
    return command[1] if command and command[0] == "sample_handling" else None


def _objective_state(read: Optional[wording.Reader]) -> Optional[str]:
    """The FM objective's state now ('inserted', 'retracted'), or None when it
    can't be read (no FM)."""
    if read is None:
        return None
    try:
        answer = read("/microscope/devices/objective/state")
    except Exception:  # noqa: BLE001 - nothing to warn about
        return None
    value = answer.get("value") if isinstance(answer, dict) else None
    return str(value).lower() if value else None


def _inserts_objective(tool: str, arguments: Dict[str, Any]) -> bool:
    return tool == "fm_insert_objective" or (
        tool == "device_call_command"
        and (arguments or {}).get("device") == "objective"
        and (arguments or {}).get("command") == "insert"
    )
