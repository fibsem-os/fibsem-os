---
name: autolamella-lead
description: Run a whole AutoLamella session at Drive that the operator starts and leaves, from the intake to the review, with the autolamella-verifier, -investigator and -auditor agents checking the key decisions and -site-finder looking through grids for sites. The main way AutoLamella runs. Read fibsem-agent, fibsem-microscope and autolamella first, then this when nobody will be at the microscope during the run.
---

# Leading an AutoLamella run

Read the fibsem-agent, fibsem-microscope and autolamella skills first. This skill is how
to run a whole lamella session at **Drive** that the operator starts and
leaves. Use it when nobody will be at the microscope during the run and the
`autolamella-*` agents are in your Agent tool; without them, supervise the
run as autolamella says. If the operator says they'll stay, read
autolamella-assistant instead.

## The team and the record

In lead mode you are the **Run Lead**. The operator starts the run, answers
the intake and leaves; they may follow it and message you from the web
page, but plan as if nobody will answer until the morning. You hold Drive,
the watcher and the record. You make every decision yourself, by the
sections of this skill, fibsem-agent and autolamella, and four agents help: they
start fresh, read only, and answer in JSON.

| Agent | Call it | What follows |
|---|---|---|
| `autolamella-verifier` | on the site list, on each Setup placement, and before every polishing pass | a second opinion on your decision, judged by the same rules without your reasoning (*The Verifier*) |
| `autolamella-investigator` | when a task's result is bad | the cause, its evidence and a fix for you to apply (*A bad result*) |
| `autolamella-auditor` | once, when the run is over | the session review, from the record |
| `autolamella-site-finder` | when picking sites means looking through many squares, tiles or grids | a short list of what it saw, or candidate sites with their doubts and evidence, for you to check (screen-grids, *The site finder*) |
| `fibsem-worker` | unless the operator said no at intake (question 9): each grid's sites (picked and checked), each Setup placement and each lamella's polishing, one at a time | the job done on the microscope in its own context, its proposal answered, its result and evidence for you to check (*Workers*) |

**How to call one.** One call at a time, and wait for it: you keep the
watcher. Put everything it needs in the prompt, since it sees nothing of
yours: the intake, the item and its neighbours, the proposal, the images by
the paths a tool answered, the item notes, the plan's What worked. Its answer ends
in a `json` block; no block, or one that contradicts its own words, is no
answer: treat it as a disagreement. Only you decide proposals, start, stop,
edit and write the record (a worker answers its own item's proposal:
*Workers*).

**The record is four places**, one for each question, where the next
context and the next session find it:

- **The plan** (fibsem-agent, *Keep a plan*): what we're doing and where the
  run stands. The intake's answers go in Brief and Rules, the sites in
  Steps (a `### Sites` group, one box per site), what is held and why in a
  `Held` section, the run in Progress at each task boundary and each
  failure, and **What worked**: per task, the settings and signs that worked
  on this sample, each change old → new.
- **Item notes** (autolamella, *A lamella's record*): where each lamella or
  site stands.
- **The review list** (fibsem-agent, *Keep a review list*): one item per
  key decision (the site list, each placement, each lamella's polishing),
  with its images, your decision, and the Verifier's answer as a fact.
  The Verifier puts its own `[verdict]` note on the record as well.
- **The knowledge base**, at the review, for the next run.

After a compaction, a restart or a fresh context, read the plan
(`plan_get`; `previous=true` when you take over an earlier session's run),
the review list (`review_list`) and the item notes before you act, never
from memory.

**Fresh contexts** (fibsem-agent, *A fresh context*): this run's natural
boundaries are after the sites are approved, after the first lamella's
What worked, between lamellae in the batch once a lamella's task set is done,
and before the review. Bringing the plan's Progress and the item notes up to
date first makes the handoff note short.

Rewrite a lamella's item notes with `item_update` after every task or
decision on it, so its state is on the lamella, not only in Progress.

**Operator messages** win over the intake and over the Verifier. Update
the plan's Brief, Rules and Progress, and say back what you will do
differently. Keep everything you hold (a site, a placement, a lamella left
UNASSESSED) in the plan's `Held` section, so the operator can release it
from one list.

## The run, phase by phase

0. **Intake**, before the microscope, while the operator is there. Read the
   session's goal and their messages first, and ask only what they don't
   already answer. Once you know the sample, read its knowledge base entries
   and say which change the plan. Put the answers in your plan.md
   (fibsem-agent, *Keep a plan*) before they go: the brief and its
   `Sample:` line, the rules, and the phases below as Steps:
   1. The scientific brief: the question, the sample and the target (a
      cell, a structure, a label).
   2. How many lamellae (target, maximum), how wide, the target thickness
      as a number and the range accepted.
   3. Stop thicker, or try thinner? Stop thicker accepts at the first pass
      under the target; try thinner keeps going and may lose a lamella.
      Default: stop thicker.
   4. Has the grid been screened, or should you find sites first
      (screen-grids)?
   5. The placement rule, with one example site (default: screen-grids'
      *Finding sites*, one lamella per square).
   6. What spare time is for: rework, re-imaging, more sites up to the
      maximum, or stop.
   7. When they are back, and what is worth a message.
   8. Is this a dry run on the Demo microscope?
   9. Hand each grid's sites (picked and checked), each Setup placement and each lamella's
      polishing to a worker (*Workers*)? Default: yes.
   Sites, placements, skipping or failing a lamella and switching tasks to
   auto are yours; say so, and that the Verifier checks them. Say too that
   a hardware error stops the run until they're back (fibsem-microscope, *A
   hardware error is the operator's*).
1. **Sites.** Pick them by screen-grids' *Finding sites*, and check each
   one there before you keep it, or hand each grid to a worker
   (*Workers*). Call the Verifier with the
   marked overview, the intake's rules and the list; drop or replace each
   site it flags, and ask it again about the replacements. Then write the
   sites in the plan's Steps, and add the list to the review list with the
   marked overview as its picture.
2. **Setup** is always supervised. At each position question, check focus,
   contrast and coincidence (autolamella, *Before a position or a POI*). If Setup's
   autofocus moved the working distance badly (its `AUTOFOCUS` line says
   `doubtful`), put the session's back on that beam alone
   (`device_set_parameter`, device `electron` or `ion`, parameter
   `working_distance`) and read it back; never `microscope_set_state` for
   it, which sets the stage too. Do it at the position question, after
   Setup's last autofocus: its autofocus after the tilt overwrites an
   earlier reset. If the beams aren't coincident, fix it there: refocus, then
   the coincidence loop. If that doesn't work, let Setup finish on this site
   and run it again (`workflow_requeue`) to fix it there. Place the point of
   interest and the alignment area (which places the fiducial), draw the
   site and its neighbours on the SEM overview (`grid_draw_milling`: the
   fiducial and trenches clear of bars, the gap to the nearest neighbour's
   cuts), then call the Verifier with the fresh images, your values, that
   picture and the neighbours while the alignment area's proposal is still pending: once it is decided,
   Setup ends and nothing can be placed again. If it disagrees, place again
   and ask again.
   A site you still can't fix or agree on waits for the operator: stop its
   Setup (`workflow_stop_task`, the reason as given), take its later tasks
   out of the queue (`workflow_edit_queue`), put it in the plan's `Held`, and
   carry on with the others. Never mill a site whose beams aren't both on
   the target. The one exception is a dry run: the operator said so at
   intake, and `microscope_get_info` names the manufacturer `Demo`. There a
   coincidence the synthetic images can't show is confirmed and noted as a
   dry-run exception, in its review item and the lamella's
   `description` (not `defect.reason`: a reason there leaves it for
   someone to judge). On any other microscope the rule holds, whatever a
   later message says.
3. **Mill Fiducial's alignment area.** Check the area covers the milled
   fiducial, then run a test alignment: `analysis_measure_shift` between
   the new reference and a fresh image. Record its score as the site's
   baseline in its item notes. A poor one: move the area to a feature-richer
   patch and test again; still poor, mark the site `alignment-risk`.
4. **The first lamella** goes through every task supervised. Polish it with
   finer steps than the protocol's, measuring after each, and write what it
   showed in the plan's What worked, under Polishing: the band against the
   gap, what faint looked like, the smallest gap and current that left the
   band intact, the fields of view it used, the stop rules. Switch the other tasks to auto only by
   autolamella's *When to propose auto*: once one lamella has passed clean,
   which the first may not.
5. **The batch.** Check every task when it finishes (autolamella, *After
   every task*, and *A bad result* below).
   Polishing stays supervised: at each proposal, go through *Each
   polishing pass*, then the Verifier, then decide.
6. **Spare time.** When the queue empties inside the operator's window, do
   what the intake allowed, in this order: rework, re-image uncertain
   lamellae without milling, more sites up to the maximum. Post the rest for
   the morning.
7. **Review.** Put supervision back as the protocol had it before the run
   (`protocol_set_supervision`), and say so. Then call the Auditor, and put
   its review on the review list. Then write the knowledge base (fibsem-agent, *The
   knowledge base*): the run's record from that review, and an entry for
   each lesson it names that would change the next run.

## A bad result

Check every task as autolamella's *After every task* says. On a bad result,
its step 3 is the Investigator's: call it with the task, its images, its
checks, its config and the time it ran, and put its cause in the lamella's
item notes. Then
apply its fix as that section says.

## Each polishing pass

Gather this before you decide, and give the Verifier the same:

- the proposal (`proposal_id`, `values`, `provenance`) and its image;
- the lamella's item notes;
- the FIB after the pass and before it at What worked's field of view
  (autolamella-polishing's *Fields of view*, where it says nothing),
  and an SEM view of the face, each with the path a tool answered, never a
  file name you built. Take any the task didn't with `beam_acquire_image`
  and a `path` (e.g. the lamella's folder), at the question: the run lends
  you the microscope only while it waits on one;
- the widest FIB image the lamella has (Rough Milling's, around 100 µm)
  and the latest at that field of view, for ice near the lamella;
- `item_measure_thickness` on the post-pass image, with its warnings;
- the pass's `ALIGNMENT` line, the site's baseline and whether it is
  `alignment-risk`, the working distance against the session's;
- What worked's next step and stop rules.

Apply a `centre_offset` only when you have checked it on the image and the
tool saw one band.

## Workers

Unless the operator said no at intake (question 9), hand `fibsem-worker`
each grid's sites (picked and checked), each Setup placement and each lamella's polishing
instead of doing them in your own context (fibsem-agent, *Working with a
worker*). If they said no, do them yourself, as the rest of this skill says. Fiducial, rough milling and
thinning stay the workflow's; a worker only redoes one of their results by
hand. One worker at a time: it drives the microscope and your turn waits
for it.

- **Sites**, one worker per grid: it picks the candidates from the grid's
  saved overviews and checks each on the microscope, in one job, so what it
  saw in the overview is still in its context when it looks closer. Brief
  it with: the task ("find up to 8 sites on grid 2 by the placement rule:
  pick candidates from the overview, then check each on the microscope");
  the limits (no milling; the stage within this grid); the skill
  (screen-grids, *Finding sites*); the grid, its overview by path, the
  intake's rule, example and number of sites; and what to return: for each
  candidate, keep or drop, why, its position, its coincident height and an
  image of it, and the overview marked with the kept ones. The site finder
  is not needed then. You make the list from its answer and ask the
  Verifier about it, as the run's step 1 says.
- **Setup**, one worker per site, when the site's Setup asks its first
  position question. Brief it with: the task ("place 03's lamella:
  coincidence, point of interest and alignment area, then answer its
  Setup proposals"); the limits (no milling; the stage within this site;
  the Verifier on the placement before the alignment area's proposal is
  decided, as *Setup* above says); the skill (autolamella, *Before a
  position or a POI*, and this skill's *Setup*); the site, its proposal id,
  its neighbours and its overview by path, its item notes and the plan's
  What worked; and what to return. It answers the site's proposals itself.
- **Polishing**, one worker per lamella, at its first Polishing proposal.
  Brief it with: the task ("polish 05 under 200 nm, stop thicker"); the
  limits (it may mill polishing passes only; the passes at most; the stop
  rules from What worked and the intake); the skill (autolamella-polishing,
  and this skill's *Each polishing pass*); the lamella, its proposal id and
  its last images by path, its item notes and What worked; and what to
  return. It asks the Verifier about its plan and before every pass that
  would mill, and answers the lamella's Polishing proposals itself.

On a correlative run, each lamella's FM target and its correlation are
workers' jobs too, and a coincidence-milling polish gives the Polishing worker more
in its brief: autolamella-correlative's and autolamella-coincidence-milling's
*Workers*.

When it answers, look at its first one or two evidence images yourself,
then add the review item (its images, the result, the Verifier's answer as a
fact) and update the plan. A result you disagree with: continue the same
worker with `SendMessage` and what to change, or hold the item in the
plan's `Held`. A `refused` answer names what the brief lacked: brief it
again.

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
safe action exists. A hardware error is the exception: it stops the run
until the operator is back.
