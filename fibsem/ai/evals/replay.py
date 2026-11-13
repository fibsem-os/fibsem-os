"""A replay of one live run: what happened on the instrument step by step,
the image the app had just taken at each step, and on the simulator the
lamella's true thickness as it went. Read from the run's own record: the
experiment's events (``events.jsonl``), the images they name, the simulator's
sample record (``sim-record.json``), the answer and its score.

``python -m fibsem.ai.evals replay <out>`` writes ``replay.html`` beside each
repeat of a run folder: a page with a slider over the steps (or the arrow
keys), to see where a run went wrong without reading its logs.
"""

import base64
import html
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from fibsem.ai.evals import visual
from fibsem.ai.evals.runner import ANSWER, RESULT

logger = logging.getLogger(__name__)

EVENTS = "events.jsonl"
SIM_RECORD = "sim-record.json"
REPLAY = "replay.html"
# images in the page are cut down to this many pixels across
THUMB = 512
# the events a step is made of; the rest (progress, beam shifts, commands)
# say the same thing again
SHOWN = {
    "agent_level_set",
    "grid_loaded",
    "stage_moved",
    "milling_stage_started",
    "milling",
    "alignment",
    "image_acquired",
    "action",
    "autofocus",
    "error",
}


def _um(v: Any) -> str:
    return "?" if v is None else f"{float(v) * 1e6:.3g} um"


def _nm(v: Any) -> Optional[float]:
    return None if v is None else round(float(v) * 1e9, 1)


def _text(kind: str, p: Dict[str, Any]) -> str:
    """One line for an event, in words."""
    if kind == "agent_level_set":
        return f"level set to {p.get('level')}"
    if kind == "grid_loaded":
        return f"loaded {p.get('grid')}"
    if kind == "stage_moved":
        request = p.get("request") or {}
        if "orientation" in request:
            return f"stage to the {request['orientation']} orientation"
        end = p.get("end") or {}
        return f"stage to x {_um(end.get('x'))}, y {_um(end.get('y'))}, z {_um(end.get('z'))}"
    if kind == "milling_stage_started":
        stage = p.get("stage") or {}
        current = (stage.get("milling") or {}).get("milling_current")
        amps = "" if current is None else f" at {current * 1e12:.3g} pA"
        return f"milling {p.get('milling_task_name')}: {stage.get('name')}{amps}"
    if kind == "milling":
        return f"{p.get('milling_task_name')} {p.get('status')}"
    if kind == "alignment":
        results = p.get("results") or [{}]
        last = results[-1]
        shift = last.get("shift") or {}
        score = last.get("score")
        return (
            f"aligned ({p.get('name')}): shift {_nm(shift.get('x'))}, "
            f"{_nm(shift.get('y'))} nm"
            + ("" if score is None else f", score {score:.2f}")
        )
    if kind == "image_acquired":
        return f"{p.get('beam_type', '')} image {p.get('filename')}, {_um(p.get('hfw'))} across"
    if kind == "action":
        return str(p.get("what"))
    return json.dumps(p)[:160]


def _thumb(path: Path) -> Optional[str]:
    try:
        image = visual._read_image(path)
    except Exception as exc:  # a missing or odd image does not stop the page
        logger.debug("replay: could not read %s: %s", path, exc)
        return None
    image, _ = visual._shrink(image, THUMB)
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray((visual._stretch(image) * 255).astype(np.uint8)).save(
        buffer, format="PNG", optimize=True
    )
    return base64.b64encode(buffer.getvalue()).decode()


def _events(home: Path) -> List[Dict[str, Any]]:
    found = sorted(home.glob(f"experiments/*/{EVENTS}"))
    events: List[Dict[str, Any]] = []
    for path in found:
        for line in path.read_text().splitlines():
            if line.strip():
                events.append(json.loads(line))
    return sorted(events, key=lambda e: e.get("timestamp", 0))


def steps_of(
    events: Iterable[Dict[str, Any]], passes: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """The steps to show: each event in words, with the image just taken and
    how many of the simulator's milling passes were done. A pass is one
    milling stage, in order (the simulator keeps no times)."""
    steps: List[Dict[str, Any]] = []
    started = done = 0
    t0 = None
    for e in events:
        kind = e.get("kind")
        if kind not in SHOWN:
            continue
        p = e.get("payload") or {}
        if kind == "milling_stage_started":
            done, started = started, started + 1
        elif kind == "milling":
            done = started
        t0 = e.get("timestamp") if t0 is None else t0
        steps.append(
            {
                "t": round((e.get("timestamp") or t0) - t0, 1),
                "actor": e.get("actor"),
                "kind": kind,
                "item": e.get("item"),
                "text": _text(kind, p),
                "image": p.get("path") if kind == "image_acquired" else None,
                "passes_done": min(done, len(passes)),
            }
        )
    return steps


def replay_html(repeat: Path) -> str:
    """The replay page for one repeat folder of a run."""
    repeat = Path(repeat)
    result = _load(repeat / RESULT) or {}
    answer = _load(repeat / "work" / ANSWER)
    sim = _load(repeat / SIM_RECORD) or {}
    passes = sim.get("passes") or []
    steps = steps_of(_events(repeat / "home"), passes)
    images: Dict[str, int] = {}
    pictures: List[str] = []
    for step in steps:
        path = step.pop("image")
        if path and path not in images and Path(path).exists():
            data = _thumb(Path(path))
            if data:
                images[path] = len(pictures)
                pictures.append(data)
        step["image"] = images.get(path) if path else None
    last = None
    for step in steps:  # each step shows the latest image so far
        last = step["image"] if step["image"] is not None else last
        step["shown"] = last
    gaps = [
        {
            "pass": p.get("pass"),
            "nm": _nm(p.get("gap_mean") or p.get("gap")),
            "damage": p.get("damage"),
        }
        for p in passes
    ]
    score = (result.get("scores") or [{}])[0]
    data = {
        "case": result.get("case", repeat.parent.name),
        "repeat": result.get("repeat"),
        "result": score.get("result"),
        "verdict": (score.get("details") or {}).get("verdict"),
        "status": result.get("status"),
        "seconds": result.get("seconds"),
        "answer": answer,
        "steps": steps,
        "gaps": gaps,
        "damage": sim.get("damage_started"),
        "target_nm": _target(result),
    }
    page = _PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    page = page.replace("__IMAGES__", json.dumps(pictures))
    return page.replace("__TITLE__", html.escape(f"Replay: {data['case']}"))


def _target(result: Dict[str, Any]) -> float:
    """The case's target thickness (nm), for the chart's line."""
    import yaml

    try:
        case = yaml.safe_load((Path(result["case_path"]) / "case.yaml").read_text())
        return float((case.get("scoring") or {}).get("target_nm", 200))
    except (KeyError, OSError, ValueError, TypeError, AttributeError):
        return 200.0


def _load(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def write_replays(out: Path, only: Optional[str] = None) -> List[Path]:
    """``replay.html`` beside each repeat of a run folder with live records."""
    written = []
    for result in sorted(Path(out).glob(f"*/*/{RESULT}")):
        repeat = result.parent
        if only and only not in repeat.parent.name:
            continue
        if not list((repeat / "home").glob(f"experiments/*/{EVENTS}")):
            continue
        (repeat / REPLAY).write_text(replay_html(repeat), encoding="utf-8")
        written.append(repeat / REPLAY)
    return written


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --bg:#fbfbfa; --fg:#1d1d1b; --muted:#6b6b66; --line:#deddd8; --card:#fff;
  --accent:#2f6fdb; --hit:#2e7d4f; --miss:#b3402b; --band:rgba(47,111,219,.12); }
@media (prefers-color-scheme: dark) { :root { --bg:#161615; --fg:#ecebe6; --muted:#9a9993;
  --line:#33332f; --card:#1f1f1d; --accent:#7aa5f0; --hit:#6cc191; --miss:#ec8a74;
  --band:rgba(122,165,240,.15); } }
* { box-sizing:border-box } body { margin:0; background:var(--bg); color:var(--fg);
  font:14px/1.45 system-ui, sans-serif } header { padding:16px; border-bottom:1px solid var(--line) }
h1 { font-size:18px; margin:0 0 4px } .muted { color:var(--muted) }
.pill { display:inline-block; padding:1px 8px; border-radius:10px; font-weight:600 }
.hit { color:var(--hit) } .miss { color:var(--miss) }
main { display:grid; grid-template-columns:minmax(0,1.2fr) minmax(0,1fr); gap:16px; padding:16px }
@media (max-width:800px) { main { grid-template-columns:1fr } }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px }
#img { width:100%; image-rendering:auto; background:#000; border-radius:4px; min-height:120px }
#controls { display:flex; gap:8px; align-items:center; margin:8px 0 }
#slider { flex:1 } button { font:inherit; padding:2px 10px }
#steps { max-height:60vh; overflow:auto; margin:0; padding:0; list-style:none }
#steps li { padding:3px 6px; border-radius:4px; cursor:pointer; display:flex; gap:8px }
#steps li.on { background:var(--band) } #steps li .t { color:var(--muted); min-width:52px;
  font-variant-numeric:tabular-nums } .agent { color:var(--accent) }
svg text { fill:var(--muted); font-size:11px } pre { white-space:pre-wrap; margin:0 }
</style></head><body>
<header><h1 id="title"></h1><div id="sub" class="muted"></div></header>
<main>
 <section>
  <div class="card"><img id="img" alt="the image the app had just taken">
   <div id="caption" class="muted"></div>
   <div id="controls"><button id="prev">&larr;</button>
    <input id="slider" type="range" min="0" value="0"><button id="next">&rarr;</button>
    <span id="pos" class="muted"></span></div>
   <div id="now"></div></div>
  <div class="card" style="margin-top:16px"><b>True thickness after each pass</b>
   <span class="muted">(the simulator's record; the line is the target)</span>
   <svg id="chart" viewBox="0 0 520 200" width="100%"></svg></div>
 </section>
 <section>
  <div class="card"><b>Steps</b> <span class="muted">(blue: the agent; grey: set-up)</span>
   <ul id="steps"></ul></div>
  <div class="card" style="margin-top:16px"><b>Answer</b><pre id="answer"></pre></div>
 </section>
</main>
<script>
const D = __DATA__, I = __IMAGES__;
const $ = id => document.getElementById(id);
$("title").textContent = `${D.case}, repeat ${D.repeat ?? ""}`;
$("sub").innerHTML = `<span class="pill ${D.result}">${D.result ?? D.status}</span> ` +
  `${D.verdict ?? ""} &middot; ${D.steps.length} steps &middot; ${Math.round(D.seconds ?? 0)} s`;
$("answer").textContent = JSON.stringify(D.answer, null, 2);
const list = $("steps");
D.steps.forEach((s, n) => { const li = document.createElement("li");
  li.innerHTML = `<span class="t">${s.t.toFixed(0)} s</span><span class="${s.actor}"></span>`;
  li.lastChild.textContent = s.text; li.onclick = () => show(n); list.appendChild(li); });
const slider = $("slider"); slider.max = Math.max(0, D.steps.length - 1);
const cut = D.gaps.filter(g => g.nm != null);
function chart(done) {
  const svg = $("chart"), W = 520, H = 200, L = 46, B = 24, T = 8;
  const ys = cut.map(g => g.nm).concat([D.target_nm]);
  const lo = Math.log(Math.min(...ys) * 0.8), hi = Math.log(Math.max(...ys) * 1.2);
  const x = i => L + (W - L - 10) * (cut.length < 2 ? 0.5 : i / (cut.length - 1));
  const y = v => T + (H - T - B) * (1 - (Math.log(v) - lo) / (hi - lo));
  let s = `<line x1="${L}" x2="${W - 10}" y1="${y(D.target_nm)}" y2="${y(D.target_nm)}"
    stroke="var(--accent)" stroke-dasharray="4 3"/>`;
  [100, 200, 500, 1000, 2000, 5000].forEach(v => { if (Math.log(v) > lo && Math.log(v) < hi)
    s += `<text x="4" y="${y(v) + 4}">${v} nm</text>`; });
  const pts = cut.map((g, i) => `${x(i)},${y(g.nm)}`).join(" ");
  s += `<polyline points="${pts}" fill="none" stroke="var(--muted)" stroke-width="1.5"/>`;
  cut.forEach((g, i) => { const on = D.gaps.indexOf(g) < done;
    s += `<circle cx="${x(i)}" cy="${y(g.nm)}" r="${on ? 5 : 3}"
      fill="${g.damage ? "var(--miss)" : on ? "var(--accent)" : "var(--line)"}"><title>pass ${g.pass}: ${g.nm} nm${g.damage ? ", " + g.damage : ""}</title></circle>`; });
  s += `<text x="${L}" y="${H - 6}">pass, in order (red: damage)</text>`;
  svg.innerHTML = s;
}
function show(n) {
  n = Math.max(0, Math.min(D.steps.length - 1, n)); slider.value = n;
  const s = D.steps[n]; if (!s) return;
  $("img").src = s.shown == null ? "" : "data:image/png;base64," + I[s.shown];
  $("caption").textContent = s.shown == null ? "no image yet" :
    (s.image == null ? "the latest image before this step" : "taken at this step");
  $("pos").textContent = `${n + 1} / ${D.steps.length}`;
  const g = D.gaps.slice(0, s.passes_done).filter(g => g.nm != null).pop();
  $("now").innerHTML = `<b>${s.text}</b><div class="muted">${s.t.toFixed(0)} s in, by the ${s.actor}` +
    (g ? `; the lamella is truly ${g.nm} nm thick` : "") + `</div>`;
  [...list.children].forEach((li, i) => li.classList.toggle("on", i === n));
  list.children[n].scrollIntoView({block: "nearest"}); chart(s.passes_done);
}
slider.oninput = () => show(+slider.value);
$("prev").onclick = () => show(+slider.value - 1); $("next").onclick = () => show(+slider.value + 1);
document.onkeydown = e => { if (e.key === "ArrowLeft") show(+slider.value - 1);
  if (e.key === "ArrowRight") show(+slider.value + 1); };
show(0);
</script></body></html>
"""
