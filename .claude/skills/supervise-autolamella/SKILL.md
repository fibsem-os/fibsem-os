---
name: supervise-autolamella
description: Supervise a running AutoLamella workflow through the App Server — watch its events, judge and answer its questions, and escalate to the operator when unsure. Use when asked to supervise, watch, babysit, or run a workflow, or to answer AutoLamella's supervision prompts.
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
MCP config — e.g. `claude mcp add -s user fibsem -- fibsem-mcp` — and the HTTP fallback
below works regardless.) Watching is the exception either way: looping the
`get_events` tool in the foreground blocks you from acting, so
`fibsem-mcp watch` stays the watch mechanism even when MCP is connected.

Fallback is plain HTTP; the running app advertises itself in a discovery
file:

```bash
TOKEN=$(python3 -c "import json;print(json.load(open('$HOME/.fibsem/app-server.json'))['token'])")
curl -s "http://127.0.0.1:8001/capabilities" -H "Authorization: Bearer $TOKEN"
```

Note for the curl path: your own harness may gate state-changing commands.
In an interactive session that is a permission prompt for the operator to
approve; running autonomously it can be a hard denial. Treat a harness
denial exactly like a server 403 — stop and ask; never repackage the call
to slip past it.

Check the capabilities response before doing anything else:
- `routers.app` must be true (an application is hosted);
- `level` is what the operator lets you do (next section). Tell them, in a
  sentence, what that level means you can and can't do this session.

No discovery file → no server. Tell the operator to enable it
(Preferences → App Server) and connect a microscope. Do not go looking for
other ways in.

## Your level

The operator sets one level for you from the Agent chip in AutoLamella's
status bar. Every launch starts at Watch; nothing in a file raises it.

- **Watch** — you see everything and can add notes; you cannot answer,
  start, stop, re-queue or edit. Watch is a complete job: say what you see,
  say what you would answer and why, and leave the click to the operator.
  What you would answer goes on the record too (see *Write down why*).
- **Drive** — you may answer questions, start and stop workflows, change
  supervision, re-queue, and edit configuration and schedules.
- **Assist** — you ask before each action and the operator approves it on
  the Agent chip; you suggest answers rather than answering. That is the
  assist-autolamella skill's job: use it at Assist. **Off** refuses
  everything, `/capabilities` included (`403 agent_off`).

At Drive, with the microscope served, you can also use the
microscope directly (see *Using the microscope*). Supervising doesn't need
it.

A refusal names its reason in `detail.error_type`:

- `level` — the action needs a higher level. Say what
  you wanted to do and that it needs Drive; ask once, then carry on watching.
- `agent_stopped` — the operator pressed *Stop agent* and you are at Watch.
  Stop acting at once: abandon what you were doing, report where you left
  it, and keep watching. Do not ask to be raised again unless they ask you
  to continue.
- `agent_off` — you are off. Report that once and stop; there is nothing to
  watch.
- `ask_first` — you are at Assist: the operator wants to approve each
  action. Switch to the assist-autolamella skill and ask through a request.
- `not_allowed` — you may never use that write or command, at any level
  (the listings mark it `agent: false`), and no request makes it allowed.
  Say what you wanted and that it is the operator's to do.

If `level` drops from Drive to Watch mid-session for any reason, the rule is
the same as `agent_stopped`: report and stop acting. Never treat a 403 as an
error to work around, and never retry it in another form.

## Watch

Sleep on the events long-poll; never poll `/app/workflow/prompt` on a timer. **Run
the watcher as a background monitor** — do not write your own (the first
fresh agent to try matched the wrong JSON key and slept through a question),
and do not run it in the foreground: each line it prints is your wake-up
call, and a foreground watcher blocks you from acting on what it sees.

```
fibsem-mcp watch
```

It comes with fibsem and finds the app the way the MCP tools do. If
`fibsem-mcp` is not on your PATH, run `python -m fibsem.mcp.sidecar watch`
with the Python fibsem is installed in, from a folder that holds no other
fibsem checkout (the current folder is searched first). Check it is the
fibsem the app runs: a watcher from before Assist prints no `REQUEST` lines
and has no `--since`.

A process that only runs in the background is not a wake-up call: its
lines go to a file, and you read them on your next turn, which may be
long after the question. In Claude Code, run the watcher in the
background writing to a file, then stream that file with the Monitor
tool, so each new line wakes you:

```
fibsem-mcp watch > watch.log 2>&1        # Bash, run_in_background
tail -n +1 -f watch.log                  # Monitor
```

A monitor expires after 30 minutes. Re-arm it each time it ends, starting
after the last line you read (`tail -n +43 -f watch.log` when you have read
42 lines), so a line written in between still reaches you. The watcher
itself keeps running. Other harnesses need the same thing: a line from the
watcher has to start your turn.

Wake and act on its lines: `PROMPT` (a question is standing, with its
`question_id`), `QUESTION` (the run is stopped on a recorded question — rule 5),
`ANSWERED` (check who — if the operator answered, your read of that
question is stale), task lifecycle `EVENT` lines, `GAP` (events were lost:
re-read `/app/workflow/prompt` and `/app/workflow/status`), and `RUN ENDED` (report and stop
watching). A run that ended badly says how: `workflow_stalled` (decisions
are pending), `workflow_failed: <error>`, or `stopped without an end event`
(it died without saying so). For the last two, read `get_log` with the
run's `task_id` and `level=WARNING` before you report. It exits 1 when the app cannot be watched; say so rather than
going idle.

If your harness cannot keep a background process alive while you are idle
(spawned subagents usually cannot — the watcher dies with your turn, and no
line will ever wake you), do not go idle expecting one. Hold the watch in
the foreground of a single bounded command instead: run the watcher with
your tool timeout as the cap, act on whatever lines it printed when the
command returns, and run it again with `--since <the last seq you saw>`
(`latest_seq` from `get_events`) so the gap between runs is not lost. A stalled watch looks exactly like a
quiet run — when in doubt, read `/app/workflow/prompt` directly.

## Answer — the rules that are never optional

1. **Name the question.** Read `GET /app/workflow/prompt`, take its `question_id`, echo it in
   `POST /app/workflow/prompt/answer {"response": ..., "question_id": N}`. A `409
   stale_prompt` means the question changed under you (usually: the operator
   answered first) — that refusal is correct behaviour; re-read and continue.
   Never retry the same `question_id`.
2. **Separate reading from acting.** Freshness check, *then* look, *then*
   answer — three tool calls, never one chained command. A chained
   guard-and-answer once fired an accept before a stale reading could stop it.
3. **Check image freshness before judging.** `GET /app/experiment/images` carries
   `acquired_at`. If the image predates the mill you are inspecting, you are
   looking at the wrong frame: on the simulator the display does not refresh
   between passes, and on hardware a stale frame means the acquisition has not
   happened yet. Do not judge a stale image — wait or escalate.
4. **First writer wins, and the operator outranks you.** If they answer first,
   your job for that question is over. Never race them.
5. **A recorded question is decided, not answered.** With the review
   preference on, a task asks a detection or a position confirmation on the
   item's record and stops on it. No prompt is raised: the watcher prints a
   `QUESTION item=… task=…` line, and `get_pending_reviews` lists it with
   `holding_the_run: true` (its `proposal_id`, `values` and image are
   there). Decide it with `decide_review`: `item_id`, `task_name`,
   `proposal_id`, the outcome, and — on Confirmed — every value the kind
   carries, as proposed or adjusted. Rejected needs a reason and fails the
   task that asked, so when unsure hold and escalate rather than reject.
   Take a `holding_the_run` review before anything else in the inbox:
   nothing runs until it is decided. (An older path still exists: a prompt
   whose payload says `answer_via: "decide"` carries a `decide` block with
   the same three fields; `answer_prompt` on it is refused with nothing
   clicked, and `decide_review` is the answer there too.)

## Write down why

Chat is not kept; the experiment folder is. Every time you act (answer,
decide, start, stop, re-queue, edit) or suggest an action at Watch, call
`add_note` with the item's name, before the action. The note lands in the
experiment's `events.jsonl` and `logfile.log`, beside what you did.

A note says what you looked at and what you saw in it, then what you did:

- *"lamella-02 Polishing: ref_polishing_ib.tif (02:14:05, after the last
  pass). Face clean; curtaining stops below the trench. Answered no: accept."*
- *"Suggest: lamella-04 Setup, POI 1.5 µm left of the marker. The marker
  sits on the grid bar edge in ref_setup_eb.tif. Left for the operator."*

Name the image, value or event you judged from. A note you cannot tie to
something you looked at is not a reason; look first, or escalate instead.
Laddered repeats get a note too, one line. Notes change nothing, so they
are allowed at Watch.

## The prompt playbook

Question types and what a sound answer requires:

**`Confirm`** — a yes/no with `message`, `positive`, `negative`. Read the
message; it may be a position instruction ("double click to move… press
Continue") — for an already-placed item, Continue is safe; for a *new*
placement, that is the operator's decision (see the ladder).

**`RunMillingTask`** — two distinct moments share this type; `passes` in
the payload tells them apart:
- *Start* (`passes: 0`): yes runs the mill with the config as shown; check
  `task_name` and `num_stages` look right first. **No continues without
  milling.** That is the operator's call; escalate rather than answer no
  here.
- *Inspect re-ask* (`passes` ≥ 1, after a mill pass): **yes mills AGAIN, no
  accepts.** Pull the fresh post-mill image (rules 2–3), judge it, and answer
  no to accept. Do not re-mill without a reason you can state.

**`EditAlignmentArea`** — the payload's `current` is the live rectangle
(fractions of the frame), and `image` is the frame it sits on (the post-mill
FIB view). Sanity-check against the image: in bounds, non-degenerate size,
covering the fiducial, consistent with previous items. Yes accepts it as
shown. In a fiducial task this prompt arrives at the task's *end*, not its
start. An answer may carry a **corrected rectangle**:
`{"response": true, "question_id": N, "value": {"left", "top", "width", "height"}}`
(fractions of the frame, origin top-left). The server validates it and
places it on the operator's screen through the same widget they would drag,
then accepts it — use this for a small correction you can justify from your
criteria, never to invent an area from scratch (that is an escalation).

**`PickPOI`** — a placement. The payload carries the image and the live
`current` marker (microscope image coordinates, +y up, origin centre). In
collaborative mode this is **held for the operator**: report it, show where
the marker is, and wait. Where placement has been delegated (exemplar mode,
or an explicit "you place the rest"), the answer may carry the point:
`"value": {"x", "y"}` in the payload's stated coordinates — validated
against the image bounds, and landed on screen before acceptance.

**`ConfirmDetection`** — feature positions. Held for the operator in
collaborative mode. With the review preference on this is a recorded
question (rule 5): a `detection` review whose `values.features` are the
model's points in image pixels, each with a name. Confirm with the features
as they are to say the model was right; move one only for a correction you
can state. Where it is a prompt rather than a review, the payload's
`current` is the live features, and an answer may move some:
`"value": {"features": [{"name", "px": {"x", "y"}}]}` (source-image pixels,
origin top-left; features you leave out stay). The model's points are kept
beside your correction, as they are when the operator drags one.

**`RunSpotBurn`** — points to burn on the FIB image. Yes burns the points
as shown and the question comes back after the burn; no continues. The
payload's `current` is the live points and `image` the frame they sit on.
Placing points is a placement: held for the operator unless delegated. When
it is, an answer may carry them: `"value": {"points": [{"x", "y"}]}`
(fractions of the frame, origin top-left). Current and exposure stay the
protocol's.

**A `state` review** — the task's position, "press Continue when ready" on
the record (rule 5). Confirming carries no values: the task reads the
instrument after the answer, so a move the operator made first is the
delta. Sanity-check its `message` against the run before confirming; the
ladder applies as for any `Confirm`.

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
- **exemplar**: the operator handles the first item end to end; you extract
  what made their answers acceptable, then handle the remaining items. The
  exemplar is an *example*, not a template: generalize each acceptance into
  the criterion behind it, and judge later items by the criteria against
  their own recorded config — never by literal match. The alignment area
  does not have to sit where item 1's sat; it has to cover *this item's*
  fiducial, stay clear of *this item's* POI, and be in bounds at a sensible
  size. One mill pass sufficed on the exemplar means one pass with a fresh
  image is acceptable — not that the pixels must match. Record from durable
  state, not the live prompt: after an `ANSWERED by=operator` line, read
  `GET /app/experiment/items/{item_name}` — the accepted POI, alignment area, and
  milling angle are there — and pair it with the display image. Never race
  the operator for the prompt's `current` value. A small drift against a
  criterion you can state is a correction, not an escalation: answer with a
  `value` carrying the fixed geometry (the alignment rectangle nudged off
  the POI, the point moved onto the feature) — the operator sees it land.
  Escalate when you cannot state the correction, when you cannot evaluate a
  criterion, or when a question type the exemplar never showed appears.
- **batch-review**: supervision is off or minimal; let the run complete, then
  present the outputs (`/app/workflow/run_summary`, `/app/experiment/items/{item}/outputs`,
  `/app/experiment/items/{item}`, final images) and act on the verdicts —
  `requeue_task` for items in a running workflow, `start_workflow` for a
  fresh round. Whose verdicts is a **policy question you settle up front**,
  never a judgment you assume: at the start of the session, ask the operator
  (or take it from your instructions) whether you **propose** reruns for
  their confirmation — the default — or **decide** them yourself. Deciding
  is an explicit delegation; even then keep it bounded (at most one rerun
  per item unless told otherwise) and report every rerun with the evidence
  behind it. A rerun spends beam time and dose on a possibly marginal
  lamella — whenever the delegation is unclear, propose.
- **grid review**: after a screening run, `GET /app/experiment/grids` lists each grid
  with its latest overview per role; `get_grid_output_image` (`GET
  /app/experiment/grids/{grid}/outputs/{file}`) is the preview and `.../markers` places the experiment's items on it in
  source pixels with their status — use it for "which items sit near the
  edge / off the grid" judgments, never re-derive stage geometry yourself.
  `quality` is the operator's verdict, not yours: say what you see, propose,
  and leave the verdict to them. Starting a screening run is a control
  action: `POST /app/workflow/grids/plan` first and show the operator the
  plan (every step, one load per grid — an exchange is the most physical
  thing this server commands), then `POST /app/workflow/grids/start` with
  the same body; `screen_all: true` inventories and runs every present grid.
  One run at a time: it is refused while a lamella run is live, and a
  lamella start is refused while a grid run is.

## Acting beyond answers (Drive)

- `POST /app/workflow/start {"task_names": [...], "item_names": [...]}` —
  items omitted = all. Say what you are starting before you start it.
- `POST /app/workflow/stop {"reason": "..."}` — at Drive, if you believe the
  run is doing damage, stop first; the reason is what the operator reads. At
  Assist, file it as a request (the assist skill). At Watch you cannot stop
  it: tell the operator at once, plainly, what you see and that they should
  press Stop.
- `POST /app/agent/supervision {"task_name", "supervise"}` — takes effect at the
  next decision point. A supervised task's questions go to the operator and
  to you alike; the first answer wins, and the operator can always answer
  first.
- `POST /app/workflow/queue/requeue {"item_name", "task_name"}` — re-runs a completed
  pair inside a running workflow.

## Using the microscope (Drive)

Moving the stage, acquiring images, autocontrast, and device parameters and
commands. Supervising never needs these: use them only when the operator asks
for something on the microscope, and say what you are about to do first.

**Check before you start.** In `/capabilities`, `level` must be `drive` and
`routers.microscope` true. If either is not, say plainly which, and that the
operator changes it (the Agent chip for the level; Tools → App Server,
*Serve the microscope*, for the microscope). Don't look for another
way in. A hardware tool missing from your tool list means the same thing.

**Get your bearings once, before the first move.** On an instrument you
have not used this session:
- `list_devices` shows what it has: the beams, detectors, stage and an FM.
- Read `get_parameter` `on` for the electron and ion beams. If one is off,
  stop and tell the operator: switching a beam on is theirs. Note it if a
  beam reads `blanked`. Some instruments have no `blanked`.
- `get_geometry` gives the stage's poses, column tilts, pre-tilt, device
  origins and the milling angle. It also gives, for each beam at the pose and
  scan rotation now, the stage move a +1 µm `stable_move` along each image
  axis makes, and which way raw +z goes. If a view says
  `scan_rotation_handled: false`, its image axes are not the stage's: don't
  `stable_move` from it.
- `get_microscope_state` gives each beam's voltage, current, working
  distance and scan rotation, the detectors' type and mode, and the stage
  position. Say what you read, and anything that looks wrong for the task
  (a milling current on the imaging beam, a detector the operator
  doesn't use).
- Take an initial image with each beam (`acquire_image_preview`) and look
  at it before moving: is the sample in view and in focus?
- Then confirm the geometry with one small `stable_move` and a fresh image,
  and check that the feature went where it says. If it did not, stop and
  tell the operator what you saw, rather than correcting by hand.

**Read the microscope before each step, not once.** The operator, a run or
a script may have moved it since you last looked. Before a move, read
`get_stage_position`; before judging what is under the beam, take a fresh
image (`acquire_image_preview`) rather than trusting the last one. Say what
you read when you act on it.

**Move with `stable_move`.** To bring something you see to the centre of
an image, call `stable_move` with its offset from the image centre in
metres, x to the right and y up, and the beam the image came from. From a
preview: `dx = (px - width/2) * hfw / width`, `dy = (height/2 - py) * hfw /
width` (pixel rows count down, so y flips). The server turns that into the
stage x, y and z for that beam's viewing angle and the stage tilt, rotation
and scan rotation, so the sample slides along its own surface at the same
height. The feature stays in focus, and at the coincidence point it stays in
view of the other beam too. `move_stage_relative` moves the raw stage axes:
on a tilted stage a y move there also changes the sample's height, so the
feature goes out of focus and drifts out of the other beam, and you have no
way to tell from the numbers where it will land. Use it only when the
operator gives you stage coordinates.

`stable_move` uses the stage and scan rotation as they are now. An offset
measured on an older image is only right if neither has changed since; after
a rotation or a scan-rotation change, take a new image and measure again.
Before a large stable move, `call_service_query` `project_stable_move` with
the same offset answers where the stage would go without moving it.

Measure on a preview with its own scale: `hfw / width` per preview pixel.
The `pixelsize` in an image answer is per pixel of the full image, which is
larger than the preview.

The rest of the moves, by when to use them:
- `vertical_move`: the feature is centred in one beam's image but off in
  the other. Pass its offset in the image where it is off, and that beam as
  `beam_type` (ION, the default, or ELECTRON); it moves the stage along the
  chamber vertical, which the beam where the feature is centred doesn't see.
- `move_to_orientation` (SEM, FIB, MILLING) and `move_to_milling_angle`:
  change the stage's rotation and tilt.
- `safe_move_stage_absolute`: go back to a saved position. It tilts flat
  before a large rotation; `move_stage_absolute` moves every axis at once.

After every move, read the position back and compare it with what you
asked for, then take a fresh image and check where the feature landed before
the next one. A move under about 1 µm may not happen at all; the position
read back says so. Move z in small steps and tilt in steps, with an image
after each.

**Coincidence: do it yourself.** To bring the beams onto the same feature,
centre it in one beam with `stable_move`, find it in the other beam's image,
and `vertical_move` by its offset there (`beam_type` = the beam you measured
in). Cap each step to a few µm, take both images again, and repeat until it
is centred in both. Work from whichever beam has it centred already. Don't
use the coincidence commands: this loop worked better on a real instrument.
The horizontal offset between the beams changes with tilt; note it.

**A refused move and a failed move are different.** A move refused before it
starts (outside the stage limits, `microscope_busy`, a bad argument, the
level or the allowlist) changed nothing: fix the request and go again. A move
that fails while moving (a hardware error, the touch alarm, stopped partway)
may have hit something. Then stop moving: don't retry it, and don't reach the
same place another way. Read the stage position, add a note saying what you
asked for and what happened, and hand back to the operator.

**Milling.** `prepare` the settings and patterns, mill with `run` (not
`start`, which reports no progress and records nothing), then `clear` and
`restore`, and check the ion beam current is back where it was. A long
`run` can outlast your tool call; read `get_service_parameter` milling
`progress` and the events to follow it.

**`409 microscope_busy` is someone else working.** Its
`detail.in_control` names who (`workflow`, `script`, or another command)
and what. Treat it as a person would: say who has the microscope and what you
wanted to do, then wait for that to finish or ask the operator. Never retry
it in a loop, and never try the same thing through another tool. While a run
waits on a question, it lends control, so your command can go through; when
the question is answered the run takes it back.

**Stop agent stops you mid-action.** If the operator presses it, your mill,
spot burn, auto focus or live imaging is stopped and you drop to Watch
(`agent_stopped`). Report where the microscope was left (read the stage
position; take no new image) and stop acting.

The operator's own buttons don't take control yet, so nothing refuses your
command while they are working at the microscope by hand. If the events
show them acting (a `stage_position_changed` or acquisition event with
`actor: operator`), stop and ask before your next command.

## Editing configuration (Drive)

At Drive you can change task parameters. At Watch you can still read them
(`get_protocol_task_config`, `get_item_task_config`).

Two levels, one dance:

- **Per-item** (`update_item_task_config`): the config this item's next run
  of that task will execute.
- **Protocol** (`update_protocol_task_config`): the defaults new items
  copy. Never affects existing items — they hold their own copies.

The dance: **read, patch against that read's `version`, report**. A
`409 stale_config` means the config changed since you read it (often: the
operator edited) — that refusal is correct; re-read and decide again on
the current state. A `409 task_running` means the task already copied its
config for the run in progress — patch after it finishes. Values are SI
(metres, amps); bounds are validated for you and refusals name the
failing path.

Scheduling is Drive too: `set_task_schedule` sets or clears
when a task may start (ISO-8601, instrument-local when naive; null
clears). The workflow reads it at each task start, so it never interrupts
anything running.

So is the run summary: by default the summary dialog opens at the end of
every run. Before you start runs yourself, turn it off with
`set_workflow_options(show_run_summary=false)` so no dialog is left over
the operator's window, and tell them you did. Each run's end still shows a
toast with a Show summary link, and it stays in the notification bell.
Read `/app/workflow/run_summary` for the outcome yourself. If the operator turns it
back on, leave it on.

Etiquette:

- **Change what was asked, exactly.** "Make the fiducial 2 µm deeper" is
  one dotted path, not an opinion about the neighbouring fields. Patches
  are small and named; never rewrite a document wholesale.
- **Unprompted edits follow the trust ladder**: propose in chat first
  unless the operator has delegated parameter changes to you.
- **Report every applied change** old → new (the response echoes it; the
  timeline records it; your report should match both).
- The operator sees your change land: the open editor rebuilds and a
  toast names the edit. Nothing you change is silent.

## Report

Keep the operator oriented without flooding them: say what you answered and
why when it was a judgment call (the note already records it); stay quiet
for laddered repeats; always report task completions/failures, escalations,
and the end-of-run summary (items, durations, outcomes — `/app/workflow/run_summary`
survives the run's end).
Every answer you give is attributed on the timeline in the app; write nothing
the timeline would contradict.
