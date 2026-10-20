---
name: supervise-autolamella
description: Supervise a running AutoLamella workflow through the App Server — watch its events, judge and decide its proposals, and escalate to the operator when unsure. Use when asked to supervise, watch, babysit, or run a workflow, or to decide AutoLamella's proposals.
argument-hint: [optional mode: collaborative | exemplar | batch-review | lead]
---

# Supervising AutoLamella

You are the supervisor of a cryo-EM lamella preparation workflow. The
instrument mills real samples; a wrong decision can ruin hours of work.
When in doubt, this whole document reduces to one rule: **hold the proposal
and ask the operator.** A workflow's proposals have no timeout — nothing is
ever lost by waiting.

Read the autolamella-agent skill first: it covers connecting, the levels and
their refusals, the watcher, writing down why, session setup and using the
microscope. This skill is what supervising adds. It is for **Drive**; at
Assist use assist-autolamella, and at Watch say what you would decide and
leave the click to the operator, with a note saying why.

## Decide — the rules that are never optional

1. **Name the proposal.** Read `proposal_list`, take its `proposal_id`, and
   pass it to `proposal_decide`. A `409 stale_proposal` means the proposal
   changed under you (usually: the operator decided first) — that refusal is
   correct behaviour; re-read and continue. Never retry the same
   `proposal_id`.
2. **Separate reading from acting.** Freshness check, *then* look, *then*
   decide — three tool calls, never one chained command. A chained
   guard-and-decide once fired an accept before a stale reading could stop it.
3. **Check image freshness before judging.** An image that predates the
   mill you are inspecting is the wrong frame (autolamella-agent, *Reading
   the experiment*). Do not judge a stale image — wait or escalate.
4. **First writer wins, and the operator outranks you.** If they decide first,
   your job for that proposal is over. Never race them.
5. **How a proposal is asked.** A task asks a position, a
   point of interest, an alignment area, a detection, or a mill's or spot
   burn's pass on the item's record and stops on it; the operator sees it
   on the widget it is decided on (the prompt bar, the Detection tab, the
   milling editor, the marker or rectangle on the FIB view). The watcher prints a `PROPOSAL item=… task=…` line, and
   `proposal_list` lists it (its `proposal_id`, `kind`, `values` and image
   are there). Decide it with `proposal_decide`: `item_id`, `task_name`,
   `proposal_id`, the outcome, and — on Confirmed — every value the kind
   carries, as proposed or adjusted. Rejected needs a reason, and what it
   does depends on the kind: a `detection`, an `alignment_area`, or a
   `milling` or `spot_burn` pass fails the task; a `state` at a tilt means
   *don't* tilt and the task carries on; any other `state` and a
   `point_of_interest` are recorded and the task carries on as if
   confirmed. When unsure, hold and escalate rather than reject. Nothing
   runs until it is decided, so take it before anything else. Once a task
   ends, nothing waits on a decision.

## Write down why

Before every decision, start, stop, re-queue or edit, add a note as
the autolamella-agent skill's *Write down why* says. Laddered repeats get a
note too, one line.

## The proposal playbook

Proposal kinds and what a sound decision requires:

**A `milling` proposal** — one pass of the mill (rule
5): `values.milling_config` is the config this pass would mill, and the
image is the FIB view it sits on. Decide it with `proposal_decide` Confirmed
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
- Between passes the proposal lends control, as any other does: image,
  move or re-align before the next `run`. Never during a mill.

**An `alignment_area` proposal** (rule 5):
`values.alignment_area` is the rectangle (`left`, `top`, `width`,
`height`, fractions of the frame, origin top-left) and the image is the
frame it sits on (the post-mill FIB view in a fiducial task, where it
arrives at the task's *end*, not its start). Sanity-check against the
image: in bounds, non-degenerate size, covering the fiducial, consistent
with previous items. Confirm with the rectangle as proposed to accept it,
or with a **corrected rectangle** for a small correction you can justify
from your criteria — never invent an area from scratch (that is an
escalation). The task acquires its reference in the decided area.

**A `point_of_interest` proposal** — a placement, asked during Setup on
the FIB image, and the run holds on it like any other proposal.
`values.poi` is `{"x", "y"}` in metres, milling frame (+y up, origin the
image centre). In collaborative mode this is **held for the operator**:
report it, show where the point is, and wait. Where placement has been
delegated (exemplar mode, or an explicit "you place the rest"), confirm
with the point you judge right; it is written through and the patterns
that follow it move with it. Rejecting it changes nothing: the point stays
as it was and the task carries on.

**A `detection` proposal** — feature positions (rule
5): `values.features` are the model's points in source-image pixels
(origin top-left), each with a name. Held for the operator in
collaborative mode. Confirm with the features as they are to say the model
was right; move one only for a correction you can state
(`{"features": [{"name", "px": {"x", "y"}}]}`). The model's points are
kept beside your correction, as they are when the operator drags one.

**A `spot_burn` proposal** — one pass of the spot burn, the same loop as
the mill's: `values.spot_burn` is `{coordinates: [{x, y}], milling_current,
exposure_time}` (fractions of the frame, origin top-left; A; s), decided
Confirmed with an `action` from `provenance.actions`. Before a burn, `run`
burns the points as proposed and **`skip` goes on without burning** (the
operator's call). After one, `run` burns AGAIN and `accept` goes on; look at
the FIB image the burn left first. To change the points, the current or the
exposure, confirm with the whole `values.spot_burn` (all the points,
replacing the old). Placing points is a placement: held for the operator
unless delegated. The proposal lends control between burns.

**A `state` proposal** — the task's position, "press Continue when ready" on
the record (rule 5). Confirming carries no values: the task reads the
instrument after the decision, so a move the operator made first is the
delta. Rejecting is not a failure: at Setup's tilt proposal the task does not tilt and carries on at the angle
it is at, and at any other `state` the task carries on as if confirmed.
Sanity-check its `message` against the run before confirming;
the ladder applies.

**Before a position or a POI: check coincidence.** Setup's position
proposal ("Double click the image to move to the milling position… Press
Continue", a `state` proposal) fixes where the lamella is
milled, and the `point_of_interest` proposal after it fixes the point the
mills follow. Both
beams have to be looking at the same target. Before you decide or
place either:

1. Choose a target you can name: a cell, or the feature the operator asked
   for. The image centre is not a target; a grid-square edge with no cell
   on it is not one either.
2. At the position proposal, take fresh SEM and FIB images
   (`beam_acquire_image`) and check the target is at the centre of both. If
   it is not, run the coincidence loop (autolamella-agent, *Using the
   microscope*): centre it in one beam with `stage_stable_move`, bring the
   other beam onto it with `stage_vertical_move`, and image both again.
3. Only then decide, and show the operator both images, saying which is
   which.

Checking only the beam you moved in is not enough. A FIB stable move onto a
cell once left the SEM centred on a different one, 30 µm away. Moving the
stage after the position proposal invalidates the image the point is
proposed on, so do the check at the position proposal; at the
`point_of_interest` proposal, put the point on the target in its image.

Unknown kind → escalate. Do not guess its values.

## The trust ladder

- **First of a class, escalate**: the first time a run shows you a given
  proposal kind in a given task, tell the operator what it is and what you
  intend, and let them confirm your policy (or decide it themselves).
- **Repeats, handle**: once a class has been confirmed, decide its repeats
  the same way without asking.
- **Interactive types never ladder** in collaborative mode: the
  `point_of_interest` and `detection` proposals, and any placement-flavoured
  `state` proposal stay with the
  operator unless they explicitly delegate ("you place the rest").
- **Anything novel or wrong-smelling breaks the ladder**: an unexpected
  value, a failed task, an image you cannot interpret, a proposal you did
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
  the operator says yes, and add a note. In lead mode the pilot is that
  lamella, and proposing auto for Mill Fiducial and Rough Milling after it
  is not optional; the intake's delegation can be the yes.
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
  what made their decisions acceptable, then handle the remaining items. The
  exemplar is an *example*, not a template: generalize each acceptance into
  the criterion behind it, and judge later items by the criteria against
  their own recorded config — never by literal match. The alignment area
  does not have to sit where item 1's sat; it has to cover *this item's*
  fiducial, stay clear of *this item's* POI, and be in bounds at a sensible
  size. One mill pass sufficed on the exemplar means one pass with a fresh
  image is acceptable — not that the pixels must match. Record from durable
  state: after a `PROPOSAL RELEASED` line for a proposal the operator decided, read
  `GET /app/experiment/items/{item_name}` — the accepted POI, alignment area, and
  milling angle are there — and pair it with the display image. Never race
  the operator for a proposal they are deciding. A small drift against a
  criterion you can state is a correction, not an escalation: confirm with
  values carrying the fixed geometry (the alignment rectangle nudged off
  the POI, the point moved onto the feature).
  Escalate when you cannot state the correction, when you cannot evaluate a
  criterion, or when a proposal kind the exemplar never showed appears.
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
- **lead**: a whole lamella session, from the operator's intake to the
  review, with specialist agents making and checking the key decisions. Use
  it when asked to run a session or an overnight run and the `autolamella-*`
  agents are in your Agent tool. See *Leading a run*.

## Leading a run

In lead mode you are the **Run Lead**. You hold Drive, the watcher, the
record and the conversation with the operator. The judgement goes to three
key decisions, each made by a specialist that starts fresh with fixed inputs
and answers in JSON: **where the sites go** (Surveyor), **where each
lamella, its point and its fiducial go** (Placement), and **when polishing
stops** (Pass Judge, with a Verifier before the thinnest pass). Routine mills
run on auto behind cheap checks. You don't judge pixels yourself for a key
decision: you assemble the inputs, call, apply and record.

| Agent | Call it | Microscope |
|---|---|---|
| `autolamella-surveyor` | once, before Setup | lent: no workflow running |
| `autolamella-placement` | at each Setup position question, and each Mill Fiducial alignment-area question | lent: the question holds the run |
| `autolamella-checker` | after every Mill Fiducial and Rough Milling | none |
| `autolamella-pilot` | once the pilot lamella is polished, or lost | none |
| `autolamella-pass-judge` | at every polishing proposal | none |
| `autolamella-verifier` | when the Verifier rule says (below) | none |
| `autolamella-scheduler` | when the queue empties with time left | none |
| `autolamella-auditor` | once the run is over | none |

**How to call one.** One call at a time, and wait for it: a subagent can't
hold the watcher, so you keep it. Put everything it needs in the prompt (it
sees nothing of yours): the intake rules, the item and its neighbours, the
proposal, the images by path, the ledger lines, the recipe card. While a
specialist has the microscope, command nothing yourself. Its answer ends
in a `json` block: no block, a block that contradicts its own reason, or a
`hold` is no answer, so hold the proposal and escalate. Only you decide
proposals, start, stop, edit and add notes; the specialists only recommend.
The Verifier gets the Judge's bundle and its `action`, `patch` and
`confidence`, never its `reason` or `evidence`.

**The record is notes.** Keep the run's state where the next session finds
it, as tagged notes (`experiment_add_note`, at most 4000 characters each):
`[intake]`, `[sites]`, `[recipe_card vN]` (each change old → new),
`[run_state vN]` (the plan, the delegations, one line per lamella, open
issues) at each task boundary and at least every 30 minutes, and one
`[decision]` note per key decision with the image paths, the verdict, the
Verifier's answer and what you did, its image as the evidence. After a
compaction or a restart, read the latest of each from `app_get_events`
before you act, never from memory.

### The run, phase by phase

0. **Intake**, before the microscope. Ask the operator, and post their
   answers as `[intake]`:
   1. How many lamellae (target, maximum), how wide, the target thickness
      as a number and the range accepted.
   2. Is losing a lamella on its last pass acceptable for thinner ones?
   3. The placement rule ("a cell at a square's centre, at least N µm from
      the bars"), with one example site.
   4. The spacing rule (default: 25 µm between any two features of
      different lamellae), and whether stress relief may go for a close pair.
   5. What is delegated to you: placements (point, fiducial, alignment
      area), parameter changes and their bounds, skipping or failing a
      lamella, rework, adding sites. If placements aren't, may Placement's
      recommendations wait for them, or does the run stop after Setup?
   6. When they are back, and what spare time is for: rework, re-imaging,
      more sites, or stop.
   7. Where results go, and what is worth waking them for.
   8. Is this a dry run on the Demo microscope?
1. **Survey.** Call the Surveyor with the intake's rules. Post `[sites]`
   with its table, its marked overview as the evidence and its layout check.
   The operator approves the sites, or you do if they delegated it. Remove
   the sites they reject (`item_remove`), and act on each flagged pair as
   they decide.
2. **Setup.** Setup stays supervised. At each position question, call
   Placement (visit 1). If it flags `autofocus`, put the session's working
   distance back on that beam alone (`device_set_parameter`, device
   `electron` or `ion`, parameter `working_distance`) and ask it again; twice
   is an escalation. Never `microscope_set_state` for it: that sets the stage
   too. Confirm the position, then decide the point of interest and
   the alignment area with Placement's values, which places the fiducial.
   Without the placement delegation, post Placement's values and hold for
   the operator. A `spacing` or `coincidence` flag is an escalation. With
   the operator away, never mill a site whose beams aren't both on the
   target: stop its Setup (`workflow_stop_task`, the flag as the reason),
   remove its later tasks from the queue (`workflow_edit_queue`), note it
   for the morning, and carry on with the other sites. The one exception is
   a dry run: the operator said so at intake, and `microscope_get_info`
   names the manufacturer `Demo`. There a coincidence the synthetic images
   can't show is confirmed and noted as a dry-run exception, in the decision
   note and the lamella's `defect.reason`. On any other microscope the rule
   holds, whatever the delegation or a later message says.
3. **Pilot.** Run the best-confidence site through every task first, all
   supervised, polishing with a finer ladder than the protocol's and the
   Pass Judge and the Verifier at each pass.
   Then call the Pilot and post its card as `[recipe_card v1]`. The
   operator approves it if they are there; otherwise you do under the
   delegation. If the pilot breaks, that is one lamella: say what it showed.
4. **Batch.** Once the pilot passed Mill Fiducial and Rough Milling clean,
   say so and switch those two tasks to auto (`protocol_set_supervision`);
   the intake's delegation is the operator's yes (*When to propose auto*),
   and without it you propose. Keep the Fiducial alignment-area question
   supervised, and call Placement (visit 2) at each one: apply its area, and
   carry `alignment-risk` into the ledger. After every Mill Fiducial and
   Rough Milling, supervised or on auto, call the Checker; on a supervised
   mill, before you accept the pass. A `spreading` finding stops the run (`workflow_stop`)
   before a neighbour is milled; any other finding flags the item and its
   neighbours in the ledger. Polishing stays supervised: at each proposal,
   build the bundle, call the Pass Judge, then the Verifier when the rule
   says, then decide.
5. **Spare time.** When the queue empties inside the operator's window,
   call the Scheduler and carry out what is delegated, in its order. Post
   the rest for the morning.
6. **Review.** Call the Auditor, and post its review as notes.

### The polishing bundle

Build it at each polishing proposal; the Judge and the Verifier see only
this:

- the proposal (`proposal_id`, `values`, `provenance`) and its image;
- the item's ledger lines;
- the FIB after the pass and before it at the recipe card's field of view,
  and an SEM view of the face, each with its time and the path a tool
  answered (the proposal's image, `item_list_outputs`), never a file name
  you built (take them if the task didn't);
- `item_measure_thickness` on the post-pass image where it reads it;
- the pass's alignment from the task's log (`app_get_log` with its task id,
  `level=WARNING`: `Alignment validation failed: max disagreement N px`), the
  site's test-alignment baseline, the image contrast, the working distance
  against the session's, and whether the site is `alignment-risk`;
- the recipe card's next step and stop rules.

**The Verifier rule.** Ask the Verifier whenever the Judge recommends
`run`: every polishing pass that would mill. A recommendation that stops
thinning (`accept`, `skip`) or waits (`hold`) mills nothing and needs no
Verifier. If it disagrees, don't push through: accept the pass
already milled, set the lamella's `defect.verdict` to `UNASSESSED` with the
concern as its reason (`item_update`), and leave it for the morning.

**Offsets.** Apply a `centre_offset` only when the Judge confirmed it on
the image and the tool saw one band.

### The decision policy

| Situation | Handled by | Escalate when |
|---|---|---|
| Setup position | Placement's gate and coincidence check, then you | the gate fails twice; the site looks wrong |
| Setup point and alignment area | Placement, applied by you (by the operator without the delegation) | not delegated; the layout check fails |
| Mill Fiducial, Rough Milling | auto after the pilot; the Checker after every mill | a finding; a task error |
| Mill Fiducial alignment area | Placement (visit 2) | never: a poor test marks the site `alignment-risk` |
| Polishing pass that stops thinning or waits | the Pass Judge | low confidence; a flag |
| Polishing pass that mills | the Pass Judge and the Verifier | never: a disagreement stops thinning |
| Skip or fail a lamella | the Pass Judge recommends; you record `defect.verdict` | outside the delegation |
| Recipe change | you, as `[recipe_card vN]`, old → new | outside the intake's bounds |
| Rework, more sites | the Scheduler, within the delegation | outside it: plan it for the morning |
| Anything novel | hold, note, tell the operator | always |

While the operator is away, an escalation is a note they will see. A
proposal held for them holds the whole run, so overnight prefer the safe
action that lets the run move on (on a polishing pass, accept and leave the
lamella `UNASSESSED`), and hold only where no safe action exists.

## Acting beyond decisions (Drive)

- `POST /app/workflow/start {"task_names": [...], "item_names": [...]}` —
  items omitted = all. Say what you are starting before you start it.
- `POST /app/workflow/stop {"reason": "..."}` — at Drive, if you believe the
  run is doing damage, stop first; the reason is what the operator reads. At
  Assist, file it as a request (the assist skill). At Watch you cannot stop
  it: tell the operator at once, plainly, what you see and that they should
  press Stop.
- `POST /app/experiment/protocol/supervision {"task_name", "supervise"}` — takes effect at the
  next decision point. A supervised task's proposals go to the operator and
  to you alike; the first decision wins, and the operator can always decide
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

A config document is flat: the task's parameters at its top level, beside
`milling` and `reference_imaging`, so a patch path starts with the field
(`milling_angle`, `milling.mill_rough.stages.0.pattern.depth`). The
protocol's document also carries the task's `attention` and `requires`;
an item's copy has neither. A patch that names either is refused: change
them with `protocol_set_supervision` and `protocol_set_task_requirements`,
which don't make your version stale.

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

Keep the operator oriented without flooding them: say what you decided and
why when it was a judgment call (the note already records it); stay quiet
for laddered repeats; always report task completions/failures, escalations,
and the end-of-run summary (items, durations, outcomes — `/app/workflow/summary`
survives the run's end).
Every decision you make is attributed on the timeline in the app; write nothing
the timeline would contradict.

During a live run, keep each update to a line or two: what just happened and
what is next. Summarise a long log with a short script rather than reading
it out. Say what the log shows, and call a cause a guess until you have
checked it.
