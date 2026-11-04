---
name: autolamella
description: Making lamellae with AutoLamella at any level — watching and explaining a run or a finished experiment, suggesting or deciding the workflow's proposals, editing task configs, a lamella's record, and the rules for rough milling, polishing and thickness. Read fibsem-agent first, then this for any AutoLamella work, then autolamella-lead or autolamella-assistant.
---

# Working with AutoLamella

Read the fibsem-agent skill first: it covers connecting, the levels, asking
at Assist, the watcher, messages, writing down why, reading the experiment
and using the microscope. This skill is the method for making
lamellae with AutoLamella, however the run is led. Read one more skill for
who is there, and ask if you don't know: "Will you be at the microscope
during the run?"

- **autolamella-lead**: nobody is at the microscope. The main way to run: at
  Drive, a whole session the operator starts and leaves, with the
  `autolamella-*` agents checking the key decisions.
- **autolamella-assistant**: the operator is at the microscope, working with
  you.

Ask again when the level changes or the operator says they're leaving or
staying, and read the other skill. Two more build on this one:
**screen-grids** for screening grids and picking sites, and
**autolamella-correlative** for lamellae targeted with the FM.

The instrument mills real samples, and a wrong decision can ruin hours of
work. When in doubt, and the operator is there, hold the proposal and ask:
a workflow's proposals have no timeout, so nothing is lost by waiting. When
they are away, a held proposal holds the whole run: autolamella-lead says
what to do instead.

## Your level

The same run, three jobs:

- **Watch: explain, never act.** Say what is happening, what happened to
  each lamella, why a task failed, and what you would decide on a proposal
  and why; leave the click to the operator, with a note saying why. Asked to
  act, say what you'd do and why, say who can do it (the operator, or an
  agent they raise to Assist or Drive from the Agent chip), and stop there.
  If the run looks like it is doing damage, tell the operator at once to
  press Stop.
- **Assist: suggest and ask.** Suggest each proposal's decision
  (`proposal_suggest`, *Suggesting at Assist*); anything else you want done
  is a request (fibsem-agent, *Asking at Assist*).
- **Drive: decide and act.** Decide proposals, start, stop and re-run tasks,
  edit configs, by the rules below.

If you were asked only to explain a run, you don't act at any level, even
when you could: that is this skill's rule, not the server's.

## The workflow's proposals

Before you start a lamella run (`workflow_start`), plan it with
`workflow_plan` and show the operator what it would queue and how long it
would take (`workflow_plan_grids` before a grid run).

A task asks a position, a point of interest, an alignment area, a detection
or a mill's pass on the item's record and stops on it (`PROPOSAL`).
`proposal_list` lists it, and `proposal_decide` decides it: at Drive you
decide it, at Assist you suggest the decision with `proposal_suggest` for the
operator to approve or decline, at Watch you say what you would decide.

The rules that are never optional, at any level:

1. **Name the proposal.** Read `proposal_list`, take its `proposal_id`, and
   pass it to `proposal_decide`. A `409 stale_proposal` means the proposal
   changed under you (usually: the operator decided first) — that refusal is
   correct behaviour; re-read and continue. Never retry the same
   `proposal_id`.
2. **Separate reading from acting.** Freshness check, *then* look, *then*
   decide — three tool calls, never one chained command. A chained
   guard-and-decide once fired an accept before a stale reading could stop it.
3. **Check image freshness before judging.** An image that predates the
   mill you are inspecting is the wrong frame (fibsem-agent, *Reading
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
6. **Write down why.** Before every decision, start, stop, re-queue or edit,
   add a note (fibsem-agent, *Write down why*). Laddered repeats get a
   one-line note too.

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
- *Polishing passes*: the stop rules are *Polishing* below: step the gap
  down gently, image FIB and SEM after every pass, and accept by the
  stop rules there: under the operator's target, or a stop sign.

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
image centre). With the operator at the microscope this is **held for
them** (autolamella-assistant): report it, show where the point is, and
wait. Where placement is yours (a lead run, or an explicit "you place the
rest"), confirm with the point you judge right; it is written through and the patterns
that follow it move with it. Rejecting it changes nothing: the point stays
as it was and the task carries on.

**A `detection` proposal** — feature positions (rule
5): `values.features` are the model's points in source-image pixels
(origin top-left), each with a name. Held for the operator when
they are at the microscope. Confirm with the features as they are to say the model
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
   contrast let you read them; fix them first if not (fibsem-agent,
   *Don't judge a poor image*), and check what Setup's own autofocus did.
3. Check the target is at the centre of both. If it is not, run the
   coincidence loop (fibsem-agent, *Using the microscope*): centre it
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

## Suggesting at Assist

When the watcher prints `PROPOSAL item=… task=…` (a position, an alignment
area, a detection, a mill's or a spot burn's pass), read it with
`proposal_list` and judge it by the playbook above: check
image freshness, and separate reading from acting. At Assist you don't
decide it: suggest the decision with `proposal_suggest` (the same outcome,
values or patch and action `proposal_decide` takes, with your reason and,
optionally, your `evidence`). The operator sees it on your card
and approves it, which decides the proposal as you suggested, or declines it,
or decides it another way. The watcher prints `SUGGESTION APPROVED` or
`SUGGESTION DECLINED` with the operator's reason; read a decline's reason
before suggesting on the next proposal. One suggestion stands per proposal: a
new one replaces yours. When the decision needs actions first (a stage move
to check coincidence), file them and the `proposal_decide` call together as
one request instead.

The playbook's rules hold at Assist too: before you put forward a Setup
position or a POI, check focus, contrast and coincidence (*Before a position
or a POI*), with the fresh images and any `stage_stable_move` or
`stage_vertical_move` in one request, and file the decision only once both
beams agree. Switching a task to auto (*When to propose auto*) and turning
supervision back on are requests too.

When you can't judge a proposal (an image you can't read or with no
`acquired_at`, a kind you haven't seen, a proposal the record doesn't back
up, anything that smells wrong), suggest nothing. Tell the operator what you
see and that it's theirs with `experiment_message_operator`.

## The trust ladder

- **First of a class, escalate**: the first time a run shows you a given
  proposal kind in a given task, tell the operator what it is and what you
  intend, and let them confirm your policy (or decide it themselves).
- **Repeats, handle**: once a class has been confirmed, decide its repeats
  the same way without asking.
- **Interactive types never ladder** while the operator is there: the
  `point_of_interest` and `detection` proposals, and any placement-flavoured
  `state` proposal stay with the operator unless they explicitly delegate
  ("you place the rest"). In a lead run they are yours, checked by the
  Verifier.
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
- **Check every auto task when it finishes** (*After every task*).
- **Supervision comes back** on any failure or Rework, a bad result, a
  config edit, or a new sample or grid. Turn it back on for that task
  yourself, find out why, and say so; then supervise the next lamella
  through it again before switching back. A config edit on an auto task
  gets a `[decision]` note (old → new, why) and, in lead mode, the
  Verifier's look at its pattern preview (`milling_preview_patterns`)
  before it runs.
- **A switch on a running task** takes effect at that task's next
  proposal, not mid-mill.

## After every task

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
3. Find out why (*Explaining a run*, *Why did Mill Fiducial fail?*) and post
   the cause as a note. In a lead run the Investigator does this
   (autolamella-lead).
4. Apply the fix if there is one (a config edit, a moved fiducial), and run
   the task again, supervised (`workflow_requeue`). When it passes, queue
   the lamella's held tasks again. With the operator there, the fix is
   theirs to agree first.
5. Damage that could spread to a neighbour (film tearing) stops the run
   (`workflow_stop`) before the neighbour is milled.

Grid screening and review, and loading grids by hand, are the screen-grids
skill's.

## Starting, stopping and re-running (Drive)

- `POST /app/workflow/start {"task_names": [...], "item_names": [...]}` —
  items omitted = all. Say what you are starting before you start it.
- `POST /app/workflow/stop {"reason": "..."}` — at Drive, if you believe the
  run is doing damage, stop first; the reason is what the operator reads. At
  Assist, file it as a request. At Watch you cannot stop
  it: tell the operator at once, plainly, what you see and that they should
  press Stop.
- `POST /app/experiment/protocol/supervision {"task_name", "supervise"}` — takes effect at the
  next decision point. A supervised task's proposals go to the operator and
  to you alike; the first decision wins, and the operator can always decide
  first.
- `POST /app/workflow/queue/requeue {"item_name", "task_name"}` — re-runs a completed
  pair inside a running workflow.

## Editing configuration

At Drive you can change task parameters. At Assist each change is a request; at Watch you can still read them
(`protocol_get_task_config`, `item_get_task_config`).

Two levels, one dance:

- **Per-item** (`item_update_task_config`): the config this item's next run
  of that task will execute.
- **Protocol** (`protocol_update_task_config`): the defaults new items
  copy. Never affects existing items — they hold their own copies.

A config document is flat: the task's parameters at its top level, beside
`milling` and `reference_imaging`, so a patch path starts with the field
(`milling_angle`, `milling.mill_rough.stages.0.pattern.depth`). A task's
`attention` and `requires` are on its plan entry (`protocol_get`), not its
config. A patch that names either is refused: change them with
`protocol_set_supervision` and `protocol_set_task_requirements`, which
don't make your version stale.

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
yourself, as fibsem-agent's *Session setup* says.

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

## A lamella's record

Draw an item's milling as well as your own marks (fibsem-agent, *Draw what
you mean*):

- `item_draw_milling`: an item's patterns as its task configs have them, on
  its latest ion image. Draw it before a mill, and before you decide or
  suggest a milling proposal.

Where a site came from (grid, square, rank, why it was chosen) belongs on the
lamella, not in a note: `item_update` with `{"patch": {"description":
"..."}, "version": "..."}`, the version from `item_get`. When you add a lamella (`item_add`), leave `name` out so it gets the
app's usual name.

## Reading a lamella

As well as the reads in fibsem-agent's *Reading the experiment*:

| Question | Read |
|---|---|
| One lamella | `item_get`, `item_list_outputs`, `item_get_output_image` |
| How thick is it? | `item_measure_thickness` on a FIB image of the item |
| The plan | `protocol_get`, `item_get_task_config`, `protocol_get_task_config` |

**A lamella's saved images.** `item_get_output_image` with a basename from
`item_list_outputs` gives a 768 px JPEG preview. Without MCP, fetch it over
HTTP; basenames have spaces in them, so URL-encode them:

```bash
curl -s -o out.jpg -H "Authorization: Bearer $TOKEN" \
  "http://127.0.0.1:8001/app/experiment/items/01-tough-mako/outputs/ref_Mill%20Fiducial_start_ib.tif"
```

`app_get_status` gives the experiment's folder (`experiment.path`). On the
microscope PC, the offline reader (*Reading a finished experiment offline*) reads
the record from there
while the run is going.

## Explaining a run

Answer from the record, and say where each claim came from: the reason the
task recorded, a log line, an image you looked at, or your own inference.
Keep inference marked as inference ("the fiducial looks out of frame in the
start image, so alignment probably failed for that reason").

**"What happened to lamella 2?"** Its runs in order, how each ended, how
long it took, any decisions or edits on it, and its current state. Look at
the final images of the last task it completed. Lead with the outcome
("Rough Milling done; Mill Fiducial failed once and wasn't re-run").

**"Why did Mill Fiducial fail?"** For each failed run: the recorded reason,
the step it stopped at, the log around it, and the images from just before
the failure (the task's start images, or the previous task's final ones).
If the same task failed on several lamellae, say what they have in common.
Then look downstream: a later task on the same lamella may have run without
what the failed one should have left, and the reader shows that as an
alignment line under the later run (`alignment skipped: reference image ...
does not exist`). Say so; it is often the more important finding.

**When there are no images to show.** A task that failed early saved none,
and a first task has no previous one. Say that, then show what's closest:
the next task's start images on the same lamella, and the same task's start
image on a lamella where it succeeded, for comparison. Label which is which.
Images from the simulator are placeholders (noise, often with "FIB" or "SEM"
written on them) and show no sample: say the run was simulated and that the
images can't settle the question.

**Showing images.** Give the user the path of each PNG you looked at, and
attach or embed them where your interface allows. With `--preview`, use one
`--out` folder per lamella: file names repeat across lamellae.

**A run summary**, on request or when a run ends: lamellae done out of
total, failures with reasons, time per task and in all, how long the run
waited on people, and anything you'd want the operator to look at. A few
lines, then the detail if they ask.

**Following a run.** The watcher's `PROPOSAL` lines mean the run is
waiting on a person. Read the proposal (`proposal_list`)
and answer it as your level allows (*Your level*). On `WORKFLOW ENDED`, summarise (*A run summary*) and keep
watching.
When it says `workflow_failed: <error>` or `stopped without an end event`,
the run died: read `app_get_log` with the run's `task_id` and
`level=WARNING` first and say why.

## Reading a finished experiment offline

Read the folder with the bundled reader. It needs a shell and the folder:
Claude Code has both, `fibsem-agent` doesn't (fibsem-agent, *Connect*). There,
read the session through the MCP tools, and leave a folder to Claude Code. It reads `experiment.yaml`,
`protocol.yaml`, `events.jsonl` and `logfile.log`, and writes nothing into
the folder. Run it with the Python AutoLamella is installed in:

```
python .claude/skills/autolamella/read_record.py <folder>              # lamellae, yield, timings, failures
python .claude/skills/autolamella/read_record.py <folder> --item 2     # one lamella: runs, steps, milling, decisions, edits, notes, images
python .claude/skills/autolamella/read_record.py <folder> --failures   # each failed run, with the log's warnings and errors around it
python .claude/skills/autolamella/read_record.py <folder> --preview <tif> --out <scratch dir>
```

`--item` takes a name or a number ("2" finds `02-...`). `--preview` turns a
saved TIFF into a PNG you can look at; `--out` must be outside the experiment
folder. If you need something the reader doesn't print, read `events.jsonl`
directly: one JSON record per line, with `kind`, `t`, `actor`, `item`, `task`
and `payload`.

Never load the experiment with `Experiment.load()` to read it, and never
open it in AutoLamella on the user's behalf. Loading changes the record: it
withdraws pending proposals and re-points paths, and the app saves that back.

Things to know when reading a record:

- **Yield** is the reader's count of lamellae that completed every required
  task in the protocol with no defect recorded. Say how you counted if the
  user means something else (for example, only up to Rough Milling).
- **A failed run's reason** is the task's own error message. The log around
  it often says more; `--failures` shows it. A folder without `logfile.log`
  has only the reason.
- **No `events.jsonl`** (an experiment from before it was recorded): you get
  item states and task history, but no timings, decisions, edits or notes.
  Say so.
- **Times** are the instrument's wall clock.
- **Who did what**: every event carries an `actor`: `task`, `operator` or
  `agent`. Decisions say who decided.

## Milling lamellae

These add to fibsem-agent's *Using the microscope*.

**Setup's own autofocus** (`use_autofocus`, on by default) applies its best
working distance without the check fibsem-agent asks for: on dark FIB views
it has picked a wrong one, so read the working distance after Setup and
compare.

**What tasks move for you.** Some tasks edit other configs as they run, so
check the next one before you rely on an edit you made by hand. Setup moves
the fiducial to the centre of the decided alignment area
(`sync_fiducial_position`): a wide area can push it into a neighbour.
Rough Milling moves the Polishing patterns to its own position
(`sync_polishing_position`). Setup can move the fluorescence pose to follow
the lamella (`sync_fluorescence_pose`, off by default). Turn a sync off in
that task's config to keep your edit.

**Polishing.** The target is a ceiling: done means the band is under the
operator's target, 200 nm unless they gave another. How far below it to go
is the operator's answer to "stop thicker, or try thinner?":

- **Stop thicker** (the default, and whenever you can't ask): accept at the
  first pass that measures under the target. A lost lamella is to be
  avoided.
- **Try thinner**: once under the target, keep polishing while nothing says
  stop, accepting that a lamella may be lost.

Either way, stop at once on a stop sign below. The same rules apply in the Polishing task's milling proposal and by hand (one
`milling_run_stage` per step):

- Start gently and step down. Begin with a wide gap and narrow it a step at
  a time, lowering the current as it narrows; never jump to the final gap.
  Drop to 30 pA once the slab is under about 500 nm. On the
  Arctis the band ran at 0.53-0.73× the gap, so expect the target near a
  250-300 nm gap there; other instruments differ, so measure, don't assume.
- Preview every step's patterns on a fresh ion image first
  (`milling_preview_patterns`), and look at where they land.
- Polish in passes at least 1 µm deep, each placed from a fresh ion image
  (the field of view is in the table below). 0.4 µm passes at 10 pA removed
  nothing seven times out of seven.
- Align before every step. In the Polishing task, the task aligns before
  each pass: set its `alignment.full_frame_below` (80 pA, say) so the
  low-current passes align on the full frame, and don't re-align by hand as
  well. By hand (`milling_run_stage`), re-align to the reference yourself
  before every step (fibsem-agent, *Align to the image you placed on*).
  Either way, below about 300 nm, if the alignment methods disagree, align
  again before milling.
- After every step, a FIB image and an SEM image, both saved as files, not
  previews, so the operator can review them later. Read the faces only on
  FIB images 3072 px wide: on a real run, faces read on 1536 px images led
  to three passes that milled nothing.
- Measure the band, don't trust the gap, and stop on the first of: the band
  under the target (with stop thicker); the band faint or low in contrast, even if it still looks
  wide (a faint band is already thin: a lamella milled through on the next
  pass when this was read as width); the SEM shows a crack, a break, a hole,
  bending or the face deteriorating.
- On a correlative lamella, the faces follow the target:
  autolamella-correlative's *Polishing to a target*.
- The polishing patterns can't be rotated to follow a tilted band. If the
  band is tilted enough to leave a wedge, say so before thinning further.
- Report each step: current, gap, time, the alignment residual, the band
  measured, what the SEM showed.

The field of view for each job:

| Job | Field of view |
|---|---|
| Placing a polishing pass, and checking coincidence | 50 µm |
| Imaging the band to judge a pass, until it is under about 400 nm | 30 µm |
| Imaging the band once it is under about 400 nm | 15 µm |
| Comparing thickness readings (`item_measure_thickness`) | 20 µm, at similar contrast |

In a lead run, the recipe card records the fields of view the first lamella
used; use the card's where it differs from this table.

**Iterative Polishing.** Polishing one pass at a time, judging each pass on
its images and readings before the next. Use it only when the operator asks
for it, and only at Drive. You choose every pass's settings: the pattern and
where it sits, the current and depth, the images and stacks.

Each pass is one `milling_run_stage` with the strategy set:

```json
{"name": "Polish top 3",
 "milling": {"milling_current": 3e-11, "hfw": 1.5e-05},
 "pattern": {"name": "Rectangle", "point": {"x": 0.0, "y": 4.5e-07},
             "width": 1e-05, "height": 1e-06, "depth": 1e-06},
 "strategy": {"name": "Iterative Polishing",
              "config": {"face": "top", "acquire_sem_image": true,
                         "acquire_reflection_stack": false}}}
```

The pass mills the stage as given (a Rectangle for one face, a Trench for
both, `face` `both`), then takes its images. `service_wait_for_run` milling
answers with its facts and a figure (pass.png): the FIB band's top and
bottom edge (µm, +y up, the frame a pattern is placed in), its FWHM and
centre offset, and the file paths of each image. Look at the figure and the
facts before the next pass. With a coincident FM, the pass also takes FM
stacks and follows a target: coincidence-milling's *Iterative Polishing with
the FM* adds its settings and stop points.

- **Before the first pass.** Take a FIB image and measure the band.
- **Each pass needs an SEM image.** Without a coincident FM, set
  `acquire_reflection_stack` false and `acquire_sem_image` true: stacks are
  refused there.
- **Both faces.** It isn't done until the top and the bottom face have each
  reached their stop point, in either order; a `both` pass counts for both.
  If you stop a face early, tell the operator. With a trench, narrow its
  `spacing` by twice the step each pass and keep its centre where it is;
  move the centre only to correct a band centre offset, or to thin one face
  on purpose.
- **Stop and ask the operator** (guidance, not hard limits) when: about
  0.7 µm has come off a face; the FWHM nears its floor; the FIB image shows
  damage; the SEM check finds a problem.
- **The SEM check**, on every pass with an SEM image (on a system without a
  coincident FM it is the main one): the protective layer still covers the
  front; no cracks, breaks or holes; no curtaining, bending or bowing; no
  charging; an even thickness. Any of these: stop and ask.
- **The final thickness** is the operator's to measure.

**Rough milling.** Run the rough stages in the protocol's order (Rough Mill
01 before Rough Mill 02), and tell the operator before you turn one off.
Once the slab is under about 1.5 µm, don't add rough rounds: thin with the
polishing stages. An extra rough round on a real run cut through the slab
and left a double lamella.

On a correlative run, autolamella-correlative's *Rough milling to a target*
adds to this.

**How thick is the lamella?** Use `item_measure_thickness` instead of
measuring a preview yourself, on an image taken at the item's milling
position so the POI falls on the lamella. Read its `warnings` first. On
`under_resolved`, take the next image at the suggested field width. On
`low_contrast`, take a cleaner image: a ratio of 1.5 gave misleading numbers
on a real run. On `several_bands`, look for residue in a trench or a stepped
wall. Check the answer on the image before you use it:
the band it finds may be the wrong one, and then its `centre_offset` is wrong
too (a polish window moved by it landed on trench residue). If the band you
checked sits away from the POI, pass `centre` [x, y] at it rather than
moving the item. Compare readings
only across images of similar contrast, at a 20 µm field width: on a real
run one lamella read 226 nm on a high-contrast image and 305-341 nm on
low-contrast ones. A number that moves with contrast is not a measurement;
image again before acting on it.

**Finding sites.** Picking cells and squares for lamellae is the
screen-grids skill's *Finding sites*: read it before you pick.

## How you report

As fibsem-agent says, and:

- Say what you decided and why when it was a judgment call (the note already
  records it); stay quiet for laddered repeats.
- Always report task completions and failures, escalations, and the
  end-of-run summary: lamellae done out of total, failures with reasons, time
  per task and in all, how long the run waited on people (`workflow_get_summary`
  survives the run's end).
- After a request finishes, say what happened in a line, with any refusal
  and its reason.
- Overnight, report failures as they happen and the summary at the end; stay
  quiet about tasks that went as expected.
