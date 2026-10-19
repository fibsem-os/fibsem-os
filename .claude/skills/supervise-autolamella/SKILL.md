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

Read the autolamella-agent skill first: it covers connecting, the levels and
their refusals, the watcher, writing down why, session setup and using the
microscope. This skill is what supervising adds. It is for **Drive**; at
Assist use assist-autolamella, and at Watch say what you would answer and
leave the click to the operator, with a note saying why.

## Answer — the rules that are never optional

1. **Name the question.** Read `GET /app/workflow/prompt`, take its `question_id`, echo it in
   `POST /app/workflow/prompt/answer {"response": ..., "question_id": N}`. A `409
   stale_prompt` means the question changed under you (usually: the operator
   answered first) — that refusal is correct behaviour; re-read and continue.
   Never retry the same `question_id`.
2. **Separate reading from acting.** Freshness check, *then* look, *then*
   answer — three tool calls, never one chained command. A chained
   guard-and-answer once fired an accept before a stale reading could stop it.
3. **Check image freshness before judging.** An image that predates the
   mill you are inspecting is the wrong frame (autolamella-agent, *Reading
   the experiment*). Do not judge a stale image — wait or escalate.
4. **First writer wins, and the operator outranks you.** If they answer first,
   your job for that question is over. Never race them.
5. **A recorded question is decided, not answered.** A task asks a
   position, an alignment area, a detection, or a mill's or spot burn's
   pass on the item's record and stops on it; the operator sees a
   position or a pass on the prompt bar and everything else in the Review
   tab. No prompt is raised for you: the watcher prints a
   `QUESTION item=… task=…` line, and `review_list` lists it with
   `holding_the_run: true` (its `proposal_id`, `kind`, `values` and image
   are there). Decide it with `review_decide`: `item_id`, `task_name`,
   `proposal_id`, the outcome, and — on Confirmed — every value the kind
   carries, as proposed or adjusted. Rejected needs a reason and does what
   the review's `rejected_fails_task` says: true, it fails the task that
   asked; false (a `state` question), it means *don't* and the task carries
   on. When unsure, hold and escalate rather than reject. Take a
   `holding_the_run` review before anything else in the inbox: nothing
   runs until it is decided. A supervised task's result (Setup's point of
   interest) is in `review_list` too once the task ends, not holding the
   run: the tasks that require it wait on the decision.

## Write down why

Before every answer, decision, start, stop, re-queue or edit, add a note as
the autolamella-agent skill's *Write down why* says. Laddered repeats get a
note too, one line.

## The prompt playbook

Question types and what a sound answer requires:

**`Confirm`** — a yes/no with `message`, `positive`, `negative`. Read the
message; it may be a position instruction ("double click to move… press
Continue") — for an already-placed item, Continue is safe; for a *new*
placement, that is the operator's decision (see the ladder).

**A `milling` review** — one pass of the mill, a recorded question (rule
5): `values.milling_config` is the config this pass would mill, and the
image is the FIB view it sits on. Decide it with `review_decide` Confirmed
and an `action`: `provenance.actions` lists the ones allowed now, and
`provenance.passes` says how many times the mill has run for this task. No
values mills it as proposed.
- *Before a pass* (`passes: 0`): `run` mills the config; check the config's
  name and the stages look right first (`milling_preview_patterns` draws
  them). **`skip` goes on without milling.** That is the operator's call;
  escalate rather than skip.
- *After a pass* (`passes` ≥ 1): **`run` mills AGAIN, `accept` goes on.**
  Pull a fresh image (rules 2–3), judge it, and accept. Do not re-mill
  without a reason you can state. `provenance.last_error` says why the last
  run failed, if it did.
- *Edits*: change the config by `patch` (dotted paths, SI units, a stage by
  name or index: `{"action": "run", "patch": {"stages.Polish
  1.milling.milling_current": 1e-11}}`) or by a whole
  `values.milling_config` (add, remove, reorder stages, change a pattern
  type). `edit` takes the change and asks again without milling, so you can
  see it on the next pass first. Edits are the operator's call unless
  delegated, like any placement.
- Between passes the question lends control, as any other does: image,
  move or re-align before the next `run`. Never during a mill.

**An `alignment_area` review** — a recorded question (rule 5):
`values.alignment_area` is the rectangle (`left`, `top`, `width`,
`height`, fractions of the frame, origin top-left) and the image is the
frame it sits on (the post-mill FIB view in a fiducial task, where it
arrives at the task's *end*, not its start). Sanity-check against the
image: in bounds, non-degenerate size, covering the fiducial, consistent
with previous items. Confirm with the rectangle as proposed to accept it,
or with a **corrected rectangle** for a small correction you can justify
from your criteria — never invent an area from scratch (that is an
escalation). The task acquires its reference in the decided area.

**A `point_of_interest` review** — a placement, and Setup's result: it is
proposed at the end of the task on the final FIB image and does not hold
the run; the milling tasks that follow the point wait on the decision.
`values.poi` is `{"x", "y"}` in metres, milling frame (+y up, origin the
image centre). In collaborative mode this is **held for the operator**:
report it, show where the point is, and wait. Where placement has been
delegated (exemplar mode, or an explicit "you place the rest"), confirm
with the point you judge right; it is written through and the patterns
that follow it move with it.

**A `detection` review** — feature positions, a recorded question (rule
5): `values.features` are the model's points in source-image pixels
(origin top-left), each with a name. Held for the operator in
collaborative mode. Confirm with the features as they are to say the model
was right; move one only for a correction you can state
(`{"features": [{"name", "px": {"x", "y"}}]}`). The model's points are
kept beside your correction, as they are when the operator drags one.

**A `spot_burn` review** — one pass of the spot burn, the same loop as
the mill's: `values.spot_burn` is `{coordinates: [{x, y}], milling_current,
exposure_time}` (fractions of the frame, origin top-left; A; s), decided
Confirmed with an `action` from `provenance.actions`. Before a burn, `run`
burns the points as proposed and **`skip` goes on without burning** (the
operator's call). After one, `run` burns AGAIN and `accept` goes on; look at
the FIB image the burn left first. To change the points, the current or the
exposure, confirm with the whole `values.spot_burn` (all the points,
replacing the old). Placing points is a placement: held for the operator
unless delegated. The question lends control between burns.

**A `state` review** — the task's position, "press Continue when ready" on
the record (rule 5). Confirming carries no values: the task reads the
instrument after the answer, so a move the operator made first is the
delta. Rejecting is *don't*, not a failure (`rejected_fails_task: false`):
at Setup's tilt question the task does not tilt and carries on at the angle
it is at. Sanity-check its `message` against the run before confirming;
the ladder applies as for any `Confirm`.

**Before a position or a POI: check coincidence.** Setup's position
question ("Double click the image to move to the milling position… Press
Continue", a `Confirm` or a `state` review) fixes where the lamella is
milled, and the `point_of_interest` review after it fixes the point the
mills follow. Both
beams have to be looking at the same target. Before you answer, suggest or
place either:

1. Choose a target you can name: a cell, or the feature the operator asked
   for. The image centre is not a target; a grid-square edge with no cell
   on it is not one either.
2. At the position question, take fresh SEM and FIB images
   (`beam_acquire_image`) and check the target is at the centre of both. If
   it is not, run the coincidence loop (autolamella-agent, *Using the
   microscope*): centre it in one beam with `stage_stable_move`, bring the
   other beam onto it with `stage_vertical_move`, and image both again.
3. Only then answer, and show the operator both images, saying which is
   which.

Checking only the beam you moved in is not enough. A FIB stable move onto a
cell once left the SEM centred on a different one, 30 µm away. Moving the
stage after the position question invalidates the image the point is
proposed on, so do the check at the position question; at the
`point_of_interest` review, put the point on the target in its image.

Unknown type → escalate. Do not guess an answer shape.

## The trust ladder

- **First of a class, escalate**: the first time a run shows you a given
  question type in a given task, tell the operator what it is and what you
  intend, and let them confirm your policy (or answer it themselves).
- **Repeats, handle**: once a class has been confirmed, answer its repeats
  the same way without asking.
- **Interactive types never ladder** in collaborative mode: the
  `point_of_interest` and `detection` reviews, and any placement-flavoured
  `Confirm` stay with the
  operator unless they explicitly delegate ("you place the rest").
- **Anything novel or wrong-smelling breaks the ladder**: an unexpected
  value, a failed task, an image you cannot interpret, a question you did
  not predict from the task's anatomy — hold and escalate.

## When to propose auto

Supervision decides who has to be at the microscope, so it is the
operator's setting. Never switch a task to auto on your own initiative.

- **Supervise every task until one lamella passes clean.** That is one
  lamella whose mills (Mill Fiducial, Rough Milling, Polishing) were each
  accepted after a single pass, judged on fresh images. A lamella that
  needed a second pass or a fix resets the count. On a clean sample the
  first lamella can be enough; on a hard one it may never happen.
- **Then propose** switching those milling tasks to auto, with the lamella
  and the images behind it. Switch (`protocol_set_supervision`) only after
  the operator says yes, and add a note.
- **Setup stays supervised** for every lamella: each position and POI is
  its own choice (see *Before a position or a POI*).
- **Supervision comes back** on any failure or Rework, a config edit, a new
  sample or grid, or a post-mill image that looks wrong. Turn it back on for
  the milling tasks yourself and say so; then supervise the next lamella
  through again before proposing auto.

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
  the operator for a question they are answering. A small drift against a
  criterion you can state is a correction, not an escalation: confirm with
  values carrying the fixed geometry (the alignment rectangle nudged off
  the POI, the point moved onto the feature).
  Escalate when you cannot state the correction, when you cannot evaluate a
  criterion, or when a question type the exemplar never showed appears.
- **batch-review**: supervision is off or minimal; let the run complete, then
  present the outputs (`/app/workflow/summary`, `/app/experiment/items/{item}/outputs`,
  `/app/experiment/items/{item}`, final images) and act on the verdicts —
  `workflow_requeue` for items in a running workflow, `workflow_start` for a
  fresh round. Whose verdicts is a **policy question you settle up front**,
  never a judgment you assume: at the start of the session, ask the operator
  (or take it from your instructions) whether you **propose** reruns for
  their confirmation — the default — or **decide** them yourself. Deciding
  is an explicit delegation; even then keep it bounded (at most one rerun
  per item unless told otherwise) and report every rerun with the evidence
  behind it. A rerun spends beam time and dose on a possibly marginal
  lamella — whenever the delegation is unclear, propose.
- **grid review**: after a screening run, `GET /app/experiment/grids` lists each grid
  with its latest overview per role; `grid_get_output_image` (`GET
  /app/experiment/grids/{grid}/outputs/{file}`) is the preview and `.../markers` places the experiment's items on it in
  source pixels with their status — use it for "which items sit near the
  edge / off the grid" judgments, never re-derive stage geometry yourself.
  For any other file, `experiment_list_files` and `experiment_read_file` list and read the open experiment's folder (text as text, images as previews with metadata), and `experiment_crop_image` shows one region of an image at full resolution.
  `quality` is the operator's verdict, not yours: say what you see, propose,
  and leave the verdict to them. Starting a screening run is a control
  action: `POST /app/workflow/grids/plan` first and show the operator the
  plan (every step, one load per grid — an exchange is the most physical
  thing this server commands), then `POST /app/workflow/grids/start` with
  the same body; `screen_all: true` inventories and runs every present grid.
  One run at a time: it is refused while a lamella run is live, and a
  lamella start is refused while a grid run is.
- **grids by hand** (Drive): loading, unloading and recording grids, as the
  autolamella-agent skill's *Grids by hand* says.

## Acting beyond answers (Drive)

- `POST /app/workflow/start {"task_names": [...], "item_names": [...]}` —
  items omitted = all. Say what you are starting before you start it.
- `POST /app/workflow/stop {"reason": "..."}` — at Drive, if you believe the
  run is doing damage, stop first; the reason is what the operator reads. At
  Assist, file it as a request (the assist skill). At Watch you cannot stop
  it: tell the operator at once, plainly, what you see and that they should
  press Stop.
- `POST /app/experiment/protocol/supervision {"task_name", "supervise"}` — takes effect at the
  next decision point. A supervised task's questions go to the operator and
  to you alike; the first answer wins, and the operator can always answer
  first.
- `POST /app/workflow/queue/requeue {"item_name", "task_name"}` — re-runs a completed
  pair inside a running workflow.

## Using the microscope (Drive)

Supervising needs the microscope only for the coincidence check before a
position (*Before a position or a POI*). Otherwise use it when the operator
asks for something on the microscope, as the autolamella-agent skill's
*Using the microscope* says.

## Editing configuration (Drive)

At Drive you can change task parameters. At Watch you can still read them
(`protocol_get_task_config`, `item_get_task_config`).

Two levels, one dance:

- **Per-item** (`item_update_task_config`): the config this item's next run
  of that task will execute.
- **Protocol** (`protocol_update_task_config`): the defaults new items
  copy. Never affects existing items — they hold their own copies.

The dance: **read, patch against that read's `version`, report**. A
`409 stale_config` means the config changed since you read it (often: the
operator edited) — that refusal is correct; re-read and decide again on
the current state. A `409 task_running` means the task already copied its
config for the run in progress — patch after it finishes. Values are SI
(metres, amps); bounds are validated for you and refusals name the
failing path.

Scheduling is Drive too: `protocol_set_schedule` sets or clears
when a task may start (ISO-8601, instrument-local when naive; null
clears). The workflow reads it at each task start, so it never interrupts
anything running.

So is the run summary dialog: turn it off before you start workflows
yourself, as the autolamella-agent skill's *Session setup* says.

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
and the end-of-run summary (items, durations, outcomes — `/app/workflow/summary`
survives the run's end).
Every answer you give is attributed on the timeline in the app; write nothing
the timeline would contradict.

During a live run, keep each update to a line or two: what just happened and
what is next. Summarise a long log with a short script rather than reading
it out. Say what the log shows, and call a cause a guess until you have
checked it.
