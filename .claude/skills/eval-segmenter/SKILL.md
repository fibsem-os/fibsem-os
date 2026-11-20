---
name: eval-segmenter
description: Segment an episode (the record of one run of a microscope agent) into the decisions an eval case could score, and write them to segments.yaml for a person to review. Offline, on files only.
---

# Segmenting an episode

You are segmenting the record of one run of a microscope agent (an
"episode") into the decisions an eval case could score. You work offline
on files: nothing you do reaches a microscope.

The episode folder is your working folder:
- `episode.yaml`: where the run came from, and its agent sessions.
- `experiment/<name>/events.jsonl`: every event of the run, in time order.
  `segmenter/events.txt` is the same, one line each: id, time, actor, kind,
  item, and the start of the payload. Read that first; open events.jsonl
  for a payload in full.
- `sessions/<id>/transcript.jsonl`: what each agent session saw and said.
  It is long; search it (Grep) for the times and items you need.
- The experiment's images and files, under `experiment/<name>/`.
- `segmenter/catalogue.yaml`: the case types (tasks), what moment in a run
  each scores (`decision`), the inputs a case needs, and its answer shape.

Write `segments.yaml` in the episode folder, in this form:

```yaml
format: 0
episode: <the episode folder's name>
segments:
- id: seg-01
  task: setup_placement        # a task from the catalogue, or null
  new_task:                    # only when task is null
    name: milling_current
    question: Which current for the next polishing pass, and why?
    answer_shape: choice       # one of the catalogue's no_task_yet shapes
  item: Lam-01                 # the lamella or site it is about, or null
  start_event: <event id>      # the cut: the case shows only what came before
  end_event: <event id>        # where the decision was acted on
  inputs:                      # each input the task names, as a path
    fib_image: Lam-01/ref_setup_ib.tif   # relative to experiment/<name>/
  decision: what the run decided, in a sentence
  facts:                       # only what the record states, when the task
    milling_angle_deg: 15      # needs it: target_nm, milling_angle_deg,
    target_nm: 200             # lamella_um (width, spacing)
  answer: {...}                # the run's decision in the task's answer shape,
                               # when you can say it; else leave it out
  override:                    # only when the operator changed or refused it
    what: what the operator did
    said: their words, quoted from the record
  reasons: why this is a decision worth a case, in a sentence
  status: proposed
```

## How to segment

1. One segment per decision a person could judge right or wrong from what
   was seen at the time: a placement, a stop, a site picked, a target found.
   Not every action is a decision; skip routine moves and images.
2. A task's decision text says which moment it scores. Use the task whose
   decision matches; one segment per lamella for a task decided pass by
   pass (its inputs list the passes).
3. start_event is the last event before the agent began working the
   decision out; every input must have been saved before it. Pick files
   the record names (an event's payload path), so the check can confirm it.
4. A decision worth scoring that no task covers gets task: null and a
   new_task. Name it plainly; reuse a name you already gave.
5. Overrides matter most: an operator edit, a declined suggestion or a
   message that corrects the agent. Quote what they said.
6. Write every segment with status: proposed. A person keeps or drops them.

## When you are done

Tell the person with experiment_message_operator how many segments you
proposed and how many have no task yet, then end your turn. Do not write
any other file.

The person (or the program that started you) checks segments.yaml against
the record. A MESSAGE listing its problems may follow: fix them in
segments.yaml (drop a segment you cannot fix), tell them what you changed,
and end your turn again.
