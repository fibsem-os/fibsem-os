---
name: autolamella-lead
description: Run a whole AutoLamella session at Drive that the operator starts and leaves, from the intake to the review, with the autolamella-verifier, -investigator and -auditor agents checking the key decisions. Read fibsem-agent and autolamella-agent first, then this when asked to run a session or an overnight run on your own.
---

# Leading an AutoLamella run

Read the fibsem-agent and autolamella-agent skills first. This skill is how
to run a whole lamella session at **Drive** that the operator starts and
leaves. Use it when asked to run a session or an overnight run and the
`autolamella-*` agents are in your Agent tool; without them, supervise the
run as autolamella-agent says.

## The team and the record

In lead mode you are the **Run Lead**. The operator starts the run, answers
the intake and leaves; they may follow it and message you from the web
page, but plan as if nobody will answer until the morning. You hold Drive,
the watcher and the record. You make every decision yourself, by the
sections of this skill, fibsem-agent and autolamella-agent, and three agents help: they
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

## The run, phase by phase

0. **Intake**, before the microscope, while the operator is there. Ask, and
   post the answers as `[intake]`:
   1. How many lamellae (target, maximum), how wide, the target thickness
      as a number and the range accepted.
   2. Is losing a lamella on its last pass acceptable for thinner ones?
   3. The placement rule, with one example site (default: screen-grids'
      *Finding sites*, one lamella per square).
   4. What spare time is for: rework, re-imaging, more sites up to the
      maximum, or stop.
   5. When they are back, and what is worth a message.
   6. Is this a dry run on the Demo microscope?
   Sites, placements, skipping or failing a lamella and switching tasks to
   auto are yours; say so, and that the Verifier checks them.
1. **Sites.** Pick them by screen-grids' *Finding sites*. Call the Verifier with the
   marked overview, the intake's rules and the list; drop or replace each
   site it flags, and ask it again about the replacements. Then post
   `[sites]` with the overview as the evidence.
2. **Setup** is always supervised. At each position question, check focus,
   contrast and coincidence (autolamella-agent, *Before a position or a POI*). If Setup's
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
   rules. Then switch the other tasks to auto (autolamella-agent, *When to propose auto*).
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
3. Call the Investigator with the task, its images, its checks, its config
   and the time it ran. Post its cause as a note.
4. Apply the fix if there is one (a config edit, a moved fiducial), and run
   the task again, supervised (`workflow_requeue`). When it passes, queue
   the lamella's held tasks again.
5. Damage that could spread to a neighbour (film tearing) stops the run
   (`workflow_stop`) before the neighbour is milled.

## Each polishing pass

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

## The Verifier

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
