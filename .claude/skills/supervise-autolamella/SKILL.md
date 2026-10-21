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
- *Polishing passes*: the stop rules are autolamella-agent's *Polishing*:
  step the gap down gently, image FIB and SEM after every pass, and accept
  as soon as the band is under 200 nm, goes faint, or the SEM shows damage.

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

**Before a position or a POI: check focus, contrast and coincidence.**
Setup's position proposal ("Double click the image to move to the milling
position… Press Continue", a `state` proposal) fixes where the lamella is
milled, and the `point_of_interest` proposal after it fixes the point the
mills follow. Both beams have to be looking at the same target, in focus and
readable. This is not optional: skipping it because the mills are FIB only
is wrong, because the SEM is how you check the lamella later, in polishing
above all, and it has to show the same place at a small field of view.
Before you decide or place either:

1. Choose a target you can name: a cell, or the feature the operator asked
   for. The image centre is not a target; a grid-square edge with no cell
   on it is not one either.
2. At the position proposal, take fresh SEM and FIB images
   (`beam_acquire_image`). Check both are in focus and their brightness and
   contrast let you read them; fix them first if not (autolamella-agent,
   *Don't judge a poor image*), and check what Setup's own autofocus did.
3. Check the target is at the centre of both. If it is not, run the
   coincidence loop (autolamella-agent, *Using the microscope*): centre it
   in one beam with `stage_stable_move`, bring the other beam onto it with
   `stage_vertical_move`, and image both again.
4. Check it again with both beams at a 50 µm field of view: the same feature
   within 5 µm of the centre in both images (aim for 2-3 µm). Coincidence
   only shows as a vertical offset, so measure it up and down the image.
5. Only then decide, and show the operator both images, saying which is
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

**Setup and Polishing are always supervised.** Every other task (Mill
Fiducial, Rough Milling, spot burns, plugin tasks) may run on auto, and the
switch is yours to make: auto only means nobody watches the task while it
runs, not that nobody checks it.

- **Supervise every task until one lamella passes clean.** That is one
  lamella whose tasks were each accepted after a single pass, judged on
  fresh images. A lamella that needed a second pass, a fix or a Setup run
  again resets the count. On a clean sample the first lamella can be enough; on a hard one it
  may never happen.
- **Then switch** the tasks other than Setup and Polishing to auto
  (`protocol_set_supervision`), and add a note with the lamella and the
  images behind it.
- **Check every auto task when it finishes**, as *After every task* says
  (lead mode) or by the same look at its final images in any mode.
- **Supervision comes back** on any failure or Rework, a bad result, a
  config edit, or a new sample or grid. Turn it back on for that task
  yourself, find out why, and say so; then supervise the next lamella
  through it again before switching back. A config edit on an auto task
  gets a `[decision]` note (old → new, why) and, in lead mode, the
  Verifier's look at its pattern preview (`milling_preview_patterns`)
  before it runs.
- **A switch on a running task** takes effect at that task's next
  proposal, not mid-mill.

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
  For any other file, `experiment_list_files` and `experiment_read_file` list and read the open experiment's folder (text as text, images as previews with metadata), and `experiment_crop_image` shows one region of an image at full resolution. `experiment_fetch_file` saves a file as it is on your machine, for your own code. Each takes `experiment` to read a past experiment the operator named, by its `relative_path` from `experiment_list_all`; every such read goes on the record.
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
- **lead**: a whole lamella session the operator starts and leaves, from
  the intake to the review, with the `autolamella-*` agents checking the key
  decisions. Use it when asked to run a session or an overnight run and
  those agents are in your Agent tool. See *Leading a run*.

## Leading a run

In lead mode you are the **Run Lead**. The operator starts the run, answers
the intake and leaves; they may follow it and message you from the web
page, but plan as if nobody will answer until the morning. You hold Drive,
the watcher and the record. You make every decision yourself, by the
sections of this skill and autolamella-agent, and three agents help: they
start fresh, read only, and answer in JSON.

| Agent | Call it | What follows |
|---|---|---|
| `autolamella-verifier` | on the site list, on each Setup placement, and before every polishing pass | a second opinion on your decision, judged by the same rules without your reasoning (*The Verifier*) |
| `autolamella-investigator` | when a task's result is bad | the cause, its evidence and a fix for you to apply (*After every task*) |
| `autolamella-auditor` | once, when the run is over | the session review, from the record |

**How to call one.** One call at a time, and wait for it: you keep the
watcher. Put everything it needs in the prompt, since it sees nothing of
yours: the intake, the item and its neighbours, the proposal, the images by
the paths a tool answered, the ledger lines, the recipe card. Its answer ends
in a `json` block; no block, or one that contradicts its own words, is no
answer: treat it as a disagreement. Only you decide proposals, start, stop,
edit and add notes.

**The record is notes.** Keep the run's state where the next session finds
it, as tagged notes (`experiment_add_note`, at most 4000 characters each):
`[intake]`, `[sites]`, `[recipe_card vN]` (each change old → new),
`[run_state vN]` (the plan, the delegations, one line per lamella, what is
held and why) at each task boundary, at each failure and at least every 30
minutes, and one
`[decision]` note per site list, placement and polishing pass with the image
paths, your decision, the Verifier's answer and what you did, its image as
the evidence. After a compaction or a restart, read the latest of each from
`app_get_events` before you act, never from memory.

**Operator messages** win over the intake and over the Verifier. Update
`[intake]` and `[run_state]`, and say back what you will do differently.
Keep everything you hold (a site, a placement, a lamella left UNASSESSED) in
`[run_state]`, so the operator can release it from one list.

### The run, phase by phase

0. **Intake**, before the microscope, while the operator is there. Ask, and
   post the answers as `[intake]`:
   1. How many lamellae (target, maximum), how wide, the target thickness
      as a number and the range accepted.
   2. Is losing a lamella on its last pass acceptable for thinner ones?
   3. The placement rule, with one example site (default: *Finding sites*
      in autolamella-agent, one lamella per square).
   4. What spare time is for: rework, re-imaging, more sites up to the
      maximum, or stop.
   5. When they are back, and what is worth a message.
   6. Is this a dry run on the Demo microscope?
   Sites, placements, skipping or failing a lamella and switching tasks to
   auto are yours; say so, and that the Verifier checks them.
1. **Sites.** Pick them by *Finding sites*. Call the Verifier with the
   marked overview, the intake's rules and the list; drop or replace each
   site it flags, and ask it again about the replacements. Then post
   `[sites]` with the overview as the evidence.
2. **Setup** is always supervised. At each position question, check focus,
   contrast and coincidence (*Before a position or a POI*). If Setup's
   autofocus moved the working distance badly (its `AUTOFOCUS` line says
   `doubtful`), put the session's back on that beam alone
   (`device_set_parameter`, device `electron` or `ion`, parameter
   `working_distance`) and read it back; never `microscope_set_state` for
   it, which sets the stage too. Do it at the position question, after
   Setup's last autofocus: its autofocus after the tilt overwrites an
   earlier reset. If the beams aren't coincident, fix it there: refocus, then
   the coincidence loop. If that doesn't work, let Setup finish on this site
   and run it again (`workflow_requeue`) to fix it there. Place the point of
   interest and the alignment area (which places the fiducial), then call
   the Verifier with the fresh images, your values and the neighbours while
   the alignment area's proposal is still pending: once it is decided,
   Setup ends and nothing can be placed again. If it disagrees, place again
   and ask again.
   A site you still can't fix or agree on waits for the operator: stop its
   Setup (`workflow_stop_task`, the reason as given), take its later tasks
   out of the queue (`workflow_edit_queue`), note it in `[run_state]`, and
   carry on with the others. Never mill a site whose beams aren't both on
   the target. The one exception is a dry run: the operator said so at
   intake, and `microscope_get_info` names the manufacturer `Demo`. There a
   coincidence the synthetic images can't show is confirmed and noted as a
   dry-run exception, in the decision note and the lamella's
   `description` (not `defect.reason`: a reason there leaves it for
   someone to judge). On any other microscope the rule holds, whatever a
   later message says.
3. **Mill Fiducial's alignment area.** Check the area covers the milled
   fiducial, then run a test alignment: `alignment_measure_shift` between
   the new reference and a fresh image. Record its score as the site's
   baseline in `[run_state]`. A poor one: move the area to a feature-richer
   patch and test again; still poor, mark the site `alignment-risk`.
4. **The first lamella** goes through every task supervised. Polish it with
   finer steps than the protocol's, measuring after each, and post what it
   showed as `[recipe_card v1]`: the band against the gap, what faint looked
   like, the smallest gap and current that left the band intact, the stop
   rules. Then switch the other tasks to auto (*When to propose auto*).
5. **The batch.** Check every task when it finishes (*After every task*).
   Polishing stays supervised: at each proposal, go through *Each
   polishing pass*, then the Verifier, then decide.
6. **Spare time.** When the queue empties inside the operator's window, do
   what the intake allowed, in this order: rework, re-image uncertain
   lamellae without milling, more sites up to the maximum. Post the rest for
   the morning.
7. **Review.** Put supervision back as the protocol had it before the run
   (`protocol_set_supervision`), and say so. Then call the Auditor, and post
   its review as notes.

### After every task

When a task finishes, supervised or on auto, look at its result before the
lamella goes on: its final images, the wide view with its neighbours, and
the task's `ALIGNMENT` and `AUTOFOCUS` lines. Is anything torn, missing,
shifted, or damaged next door? On a supervised task you saw each pass, so
this is quick; on an auto one nobody watched, so look properly.

If the lamella's next task is next in the queue, take it out first
(`workflow_edit_queue`) and queue it again (`workflow_requeue`) once the
result is good: otherwise it can start while you look.

A bad result:

1. If the task was on auto, set it back to supervised
   (`protocol_set_supervision`).
2. Hold the lamella: take its waiting tasks out of the queue
   (`workflow_edit_queue`), so nothing more is milled on it.
3. Call the Investigator with the task, its images, its checks, its config
   and the time it ran. Post its cause as a note.
4. Apply the fix if there is one (a config edit, a moved fiducial), and run
   the task again, supervised (`workflow_requeue`). When it passes, queue
   the lamella's held tasks again.
5. Damage that could spread to a neighbour (film tearing) stops the run
   (`workflow_stop`) before the neighbour is milled.

### Each polishing pass

Gather this before you decide, and give the Verifier the same:

- the proposal (`proposal_id`, `values`, `provenance`) and its image;
- the lamella's lines in `[run_state]`;
- the FIB after the pass and before it at the recipe card's field of view,
  and an SEM view of the face, each with the path a tool answered, never a
  file name you built. Take any the task didn't with `beam_acquire_image`
  and a `path` (e.g. the lamella's folder), at the question: the run lends
  you the microscope only while it waits on one;
- `item_measure_thickness` on the post-pass image, with its warnings;
- the pass's `ALIGNMENT` line, the site's baseline and whether it is
  `alignment-risk`, the working distance against the session's;
- the recipe card's next step and stop rules.

Apply a `centre_offset` only when you have checked it on the image and the
tool saw one band.

### The Verifier

It gets what you based a decision on and the decision itself (for a pass:
the action and the values), never your reasoning, and judges by the same
section of the skill you used. Each call starts fresh, with one exception:
for one lamella's polishing passes you may carry on the Verifier you asked
about its first pass (`SendMessage` to it), which halves the time. A new
lamella gets a fresh one.

- **Sites:** a site it flags is dropped or replaced before Setup.
- **Placements:** if it disagrees, place again and ask again; still
  disputed, the site waits for the operator (Setup, step 2).
- **Polishing:** ask before every pass that would mill. A decision that
  stops thinning mills nothing and needs no Verifier. If it disagrees,
  don't push through: accept the pass already milled, set the lamella's
  `defect.verdict` to `UNASSESSED` with the concern as its reason
  (`item_update`), and leave it for the morning.

While the operator is away, a proposal held for them holds the whole run, so
prefer the safe action that lets the run move on, and hold only where no
safe action exists.

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
