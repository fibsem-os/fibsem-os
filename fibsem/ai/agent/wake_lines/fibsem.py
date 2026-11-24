"""The FIB-SEM's own lines: its mills, whoever started them.

``MILL STAGE STARTED`` and ``MILL ENDED`` (every mill), and ``MILL READING``
(a milling monitor's trigger or error).
"""

from fibsem.ai.agent.wake_lines.base import Line, Lines, item_of
from fibsem.workflows.events import kinds as event_kinds
from fibsem.workflows.events import payloads as typed

MILL_STAGE_STARTED = "mill_stage_started"
MILL_ENDED = "mill_ended"
MILL_FAILED = "mill_failed"
MILL_READING = "mill_reading"
# A milling monitor's stop rule fired (a MILL READING with a trigger).
STOP_RULE = "stop_rule"


def _stage_started(event) -> Line:
    return MILL_STAGE_STARTED, _mill_stage_line(event, typed.read_payload(event))


def _ended(event) -> Line:
    p = typed.read_payload(event)
    kind = MILL_FAILED if p.status == "failed" else MILL_ENDED
    return kind, _mill_ended_line(event, p), {"ok": _went_as_planned(p)}


# As fibsem.milling.tasks records a stage turned off in the protocol.
_TURNED_OFF = "turned off in the protocol"


def _went_as_planned(p: typed.Milling) -> bool:
    """Whether a mill completed with every stage milled out: none skipped
    (but those turned off in the protocol), ended early, failed or on another
    current than asked."""
    if p.status != "completed" or p.error:
        return False
    for stage in p.stages or []:
        if not isinstance(stage, dict):
            return False
        if stage.get("status") == "skipped" and stage.get("reason") == _TURNED_OFF:
            continue
        end = stage.get("end") or {}
        if (
            stage.get("status") not in (None, "completed")
            or stage.get("reason")
            or stage.get("current_warning")
            or end.get("kind") not in (None, "finished")
        ):
            return False
    return True


def _reading(event) -> Line:
    data = event.get("payload") or {}
    line = _reading_line(data)
    if not line:
        return None
    reading = data.get("reading") or {}
    return (STOP_RULE if reading.get("trigger") else MILL_READING), line


LINES: Lines = {
    event_kinds.MILLING_STAGE_STARTED: _stage_started,
    event_kinds.MILLING: _ended,
    event_kinds.MILLING_READING: _reading,
}


def _mill_stage_line(event, p: typed.MillingStageStarted):
    """A mill's stage started, from ``milling_stage_started``: which of how
    many, with the current and depth it mills at, to check against the plan."""
    stage = p.stage if isinstance(p.stage, dict) else {}
    index, total = p.stage_index, p.total_stages
    place = f"{index + 1}/{total}" if isinstance(index, int) else "?"
    line = (
        f"MILL STAGE STARTED {place} '{stage.get('name')}'{item_of(event)} "
        f"task='{p.milling_task_name}'"
    )
    current = (stage.get("milling") or {}).get("milling_current")
    if current is not None:
        line += f" current={_si(current)}A"
    depth = (stage.get("pattern") or {}).get("depth")
    if depth is not None:
        line += f" depth={_si(depth)}m"
    return f"{line} run={p.milling_run_id}"


def _si(value):
    """A number in SI units, short (``2e-11``); anything else as it came."""
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _mill_ended_line(event, p: typed.Milling):
    """A mill ended, from its ``milling`` event: its status, how each stage
    ended, and the error of a failed one."""
    status = p.status
    if p.reason:
        status = f"{status} ({p.reason})"
    line = (
        f"MILL ENDED '{p.milling_task_name}'{item_of(event)} "
        f"status={status} run={p.milling_run_id}"
    )
    stages = [_stage_part(stage) for stage in p.stages or []]
    if stages:
        line += " stages: " + ", ".join(stages)
    return f"{line} ({p.error})" if p.error else line


def _stage_part(stage):
    """A stage's part of the MILL ENDED line, with why it ended other than by
    milling its patterns out: a skipped stage's reason, else its end's
    (``ended by monitor:FM Intensity: intensity drop``). A current the beam
    snapped to another it lists is said in brackets."""
    part = f"{stage.get('name')}={stage.get('status')}"
    if stage.get("current_warning"):
        part += f" [{stage['current_warning']}]"
    end = stage.get("end") or {}
    if stage.get("reason"):
        return f"{part} ({stage['reason']})"
    if end.get("kind") in (None, "finished") or not end.get("reason"):
        return part
    by = f" by {end['by']}" if end.get("by") else ""
    return f"{part} ({end['kind']}{by}: {end['reason']})"


def _reading_line(data):
    """A monitor's notable reading, from ``milling_reading``: a trigger (a
    drop) or the monitor's error. A step's own pauses and end are left to the
    MILL ENDED line."""
    reading = data.get("reading") or {}
    head = f"MILL READING monitor='{data.get('monitor')}' milled={data.get('milled_time')}s"
    if reading.get("error"):
        return f"{head} error: {reading['error']}"
    if not reading.get("trigger"):
        return None
    line = (
        f"{head} trigger='{reading['trigger']}' on_trigger={reading.get('on_trigger')}"
    )
    if reading.get("on_trigger") == "suggest":
        line += " -- the mill goes on; milling_end ends the stage if you agree"
    return line
