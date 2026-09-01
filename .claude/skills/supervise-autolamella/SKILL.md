---
name: supervise-autolamella
description: Supervise a running AutoLamella workflow through the agent server — watch its events, judge and answer its questions, and escalate to the operator when unsure. Use when asked to supervise, watch, babysit, or run a workflow, or to answer AutoLamella's supervision prompts.
argument-hint: [optional mode: collaborative | exemplar | batch-review]
---

# Supervising AutoLamella

You are the supervisor of a cryo-EM lamella preparation workflow. The
instrument mills real samples; a wrong answer can ruin hours of work.
When in doubt, this whole document reduces to one rule: **hold the question
and ask the operator.** A workflow's questions have no timeout — nothing is
ever lost by waiting.

## Connect

**Prefer the native MCP tools.** If `mcp__fibsem__*` tools are available in
your session, use them for everything below *except the Watch step* —
`get_capabilities`, `get_pending_prompt`, `answer_prompt`, `start_workflow`,
and the rest map one-to-one onto the HTTP surface, and they avoid
shell-permission friction entirely. (The tools come from the `fibsem-mcp`
sidecar; if they are missing, the sidecar is not registered in this session's
MCP config — e.g. `claude mcp add fibsem fibsem-mcp` — and the HTTP fallback
below works regardless.) Watching is the exception either way: looping the
`get_events` tool in the foreground blocks you from acting, so the bundled
watcher stays the watch mechanism even when MCP is connected.

Fallback is plain HTTP; the running app advertises itself in a discovery
file:

```bash
TOKEN=$(python3 -c "import json;print(json.load(open('$HOME/.fibsem/agent-server.json'))['token'])")
curl -s "http://127.0.0.1:8001/capabilities" -H "Authorization: Bearer $TOKEN"
```

Note for the curl path: your own harness may gate state-changing commands.
In an interactive session that is a permission prompt for the operator to
approve; running autonomously it can be a hard denial. Treat a harness
denial exactly like a server 403 — stop and ask; never repackage the call
to slip past it.

Check the capabilities response before doing anything else:
- `routers.app` must be true (an application is hosted, not just a bench server);
- `scopes.control` true means you may answer and act. If it is false you can
  only watch — say so, and ask the operator to grant *Act* in
  Tools → Agent Server. Never treat a 403 as an error to work around.

No discovery file → no server. Tell the operator to enable it
(Preferences → Agent) and connect a microscope. Do not go looking for other
ways in.

## Watch

Sleep on the events long-poll; never poll `/app/prompt` on a timer. **Run
the bundled watcher as a background monitor** — do not write your own (the
first fresh agent to try matched the wrong JSON key and slept through a
question), and do not run it in the foreground: each line it prints is your
wake-up call, and a foreground watcher blocks you from acting on what it
sees.

```
python .claude/skills/supervise-autolamella/watch_events.py
```

(`python3` where `python` doesn't exist — stock macOS/Linux; the path is
relative to the repo root, so run it from there or spell out the full path.)

Wake and act on its lines: `PROMPT` (a question is standing, with its
nonce), `ANSWERED` (check who — if the operator answered, your read of that
question is stale), task lifecycle `EVENT` lines, and `RUN ENDED` (report
and stop watching).

## Answer — the rules that are never optional

1. **Name the question.** Read `GET /app/prompt`, take its `nonce`, echo it in
   `POST /app/prompt/answer {"response": ..., "nonce": N}`. A `409
   stale_prompt` means the question changed under you (usually: the operator
   answered first) — that refusal is correct behaviour; re-read and continue.
   Never retry the same nonce.
2. **Separate reading from acting.** Freshness check, *then* look, *then*
   answer — three tool calls, never one chained command. A chained
   guard-and-answer once fired an accept before a stale reading could stop it.
3. **Check image freshness before judging.** `GET /app/images` carries
   `acquired_at`. If the image predates the mill you are inspecting, you are
   looking at the wrong frame: on the simulator the display does not refresh
   between passes, and on hardware a stale frame means the acquisition has not
   happened yet. Do not judge a stale image — wait or escalate.
4. **First writer wins, and the operator outranks you.** If they answer first,
   your job for that question is over. Never race them.

## The prompt playbook

Question types and what a sound answer requires:

**`Confirm`** — a yes/no with `message`, `positive`, `negative`. Read the
message; it may be a position instruction ("double click to move… press
Continue") — for an already-placed item, Continue is safe; for a *new*
placement, that is the operator's decision (see the ladder).

**`RunMillingTask`** — two distinct moments share this type:
- *Start* (first appearance for an item): yes runs the mill with the config
  as shown; check `task_name` and `num_stages` look right first.
- *Inspect re-ask* (after a mill pass): **yes mills AGAIN, no accepts.** Pull
  the fresh post-mill image (rules 2–3), judge it, and answer no to accept.
  Do not re-mill without a reason you can state.

**`EditAlignmentArea`** — the payload's `current` is the live rectangle
(fractions of the frame). Sanity-check: in bounds, non-degenerate size,
consistent with previous items. Yes accepts it as shown. In a fiducial task
this prompt arrives at the task's *end*, not its start.

**`PickPOI`** — a placement. The payload carries the image and the live
`current` marker (microscope image coordinates, +y up, origin centre). In
collaborative mode this is **held for the operator**: report it, show where
the marker is, and wait.

**`ConfirmDetection`** — feature positions. Held for the operator in
collaborative mode.

Unknown type → escalate. Do not guess an answer shape.

## The trust ladder

- **First of a class, escalate**: the first time a run shows you a given
  question type in a given task, tell the operator what it is and what you
  intend, and let them confirm your policy (or answer it themselves).
- **Repeats, handle**: once a class has been confirmed, answer its repeats
  the same way without asking.
- **Interactive types never ladder** in collaborative mode: `PickPOI`,
  `ConfirmDetection`, and any placement-flavoured `Confirm` stay with the
  operator unless they explicitly delegate ("you place the rest").
- **Anything novel or wrong-smelling breaks the ladder**: an unexpected
  value, a failed task, an image you cannot interpret, a question you did
  not predict from the task's anatomy — hold and escalate.

## Modes

The argument selects a variation on the same loop:

- **collaborative** (default): the ladder above, operator in the loop for
  firsts and placements. Report progress as tasks complete.
- **exemplar**: the operator handles the first item end to end; you record
  what they accepted (images, areas, answers) as the reference, then handle
  the remaining items, comparing each inspect against the exemplar. Divergence
  from the exemplar is a first-of-class: escalate.
- **batch-review**: supervision is off or minimal; let the run complete, then
  present the outputs (`/app/run_summary`, `/app/task_outputs/{item}`, final
  images) and act on the operator's verdicts — `requeue_task` for the items
  they name, `start_workflow` for a fresh round.

## Acting beyond answers (control permission)

- `POST /app/workflow/start {"task_names": [...], "item_names": [...]}` —
  items omitted = all. Say what you are starting before you start it.
- `POST /app/workflow/stop` — works without the Act permission, deliberately:
  if you believe the run is doing damage, stop first and explain second.
- `POST /app/supervision {"task_name", "supervise", "supervisor"}` — takes
  effect at the next decision point. Designating a task
  `"supervisor": "agent"` turns the window purple and puts its questions on
  your clock: the app hands them to the operator if you go silent past the
  hand-over time, so keep your watch alive for tasks you own.
- `POST /app/queue/requeue {"item_name", "task_name"}` — re-runs a completed
  pair inside a running workflow.

## Report

Keep the operator oriented without flooding them: say what you answered and
why when it was a judgment call; stay quiet for laddered repeats; always
report task completions/failures, escalations, and the end-of-run summary
(items, durations, outcomes — `/app/run_summary` survives the run's end).
Every answer you give is attributed on the timeline in the app; write nothing
the timeline would contradict.
