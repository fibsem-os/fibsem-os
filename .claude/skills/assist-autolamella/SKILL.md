---
name: assist-autolamella
description: Work alongside an AutoLamella operator at the Assist level — read the session and its proposals, and ask the operator to approve each action before it runs. Use when the agent level is Assist, or when asked to help with a run but act only with the operator's approval.
argument-hint: [optional: what the operator wants help with]
---

# Assisting in AutoLamella

Read the autolamella-agent skill first: it covers connecting, the levels, the
watcher, notes and using the microscope. This skill is what Assist adds.

You help an operator who is at the microscope. At **Assist** you can see
everything, but nothing you want done happens until the operator approves
it. You ask by filing a **request**. It is one or more of your own tool
calls and your reason. The operator reads it on the web Agent page, then
approves or rejects it. Approved actions run
exactly as if you had called them at Drive. They are checked, refused when
the microscope is busy, and put on the record as yours.

**Reads** need no approval. Neither does stopping a mill **you** started.
Stopping anything else (the workflow, the operator's mill) is a request like
any other action: the operator approves it, or presses Stop themselves.

The operator is looking at the same screen. Ask for little, ask clearly, and
never make them work out what you meant.

## Connect

Connect as the autolamella-agent skill says, and read
`server_get_capabilities`:

- `level: "assist"` is the level this skill is for. Tell the operator in a
  sentence what that means: you read, and you ask before
  acting, stopping included. A direct call that needs Drive is refused with
  `ask_first`; that is what you can call directly, not what you can ask for.
- `level: "watch"`: you can read and add notes, and act on nothing. Work as
  the observe skill says. If they want your help acting, they raise you to
  Assist from the Agent chip.
- `level: "drive"`: you may act directly. The supervise skill is the one for
  that.

The tools this skill uses, with their routes for plain HTTP:

| Tool | Route |
|---|---|
| `request_create` | `POST /app/agent/requests` |
| `request_get` | `GET /app/agent/requests/{request_id}` |
| `request_withdraw` | `POST /app/agent/requests/{request_id}/withdraw` with `{"reason": ...}` |
| `experiment_add_note` | `POST /app/experiment/notes` |
| `experiment_message_operator` | `POST /app/experiment/messages` |
| `workflow_stop`, `milling_stop` | `POST /app/workflow/stop`, `POST /microscope/services/milling/commands/stop`, each with `{"reason": ...}` |
| `milling_end` | `POST /microscope/services/milling/commands/end` with `{"kwargs": {"reason": ..., "by": "agent"}, "reason": ...}` |

In a request, `tool` is the tool's name without the `mcp__fibsem__`
prefix (`protocol_set_supervision`, `stage_move_relative`, `workflow_requeue`), and
`arguments` are what that tool takes. Any tool that acts can go in a
request; the server says so when it can't (`not_an_action`), or when the
agent may never use it (`403 not_allowed`: `agent: false` in `device_list`).

## The watcher

Start the watcher (autolamella-agent, *The watcher*) before you file
anything: the operator often decides within seconds. If you can only run it
bounded in the foreground, read `latest_seq` from `app_get_events` just
before you file and pass it as `--since` on the next run.

Each time your request moves, it prints a `REQUEST` line:

- `REQUEST ab12 approved`
- `REQUEST ab12 applied (2 of 2 applied)` or `failed (1 of 2 applied)`
- `REQUEST ab12 rejected: <the operator's words>`
- `REQUEST ab12 cancelled: <why>`

Don't poll `request_get` on a timer. Wait for the line, then read the
request once. The operator may start the next workflow, or approve your
request to, so keep the watcher running across workflows.

## Asking: `request_create`

```json
{"actions": [
   {"tool": "item_update_task_config",
    "arguments": {"item_name": "01-alive-dory", "task_name": "Rough Milling",
                  "patch": {"milling.mill_rough.stages.0.pattern.depth": 8e-07},
                  "version": "<from item_get_task_config>"}},
   {"tool": "workflow_requeue",
    "arguments": {"item_name": "01-alive-dory", "task_name": "Rough Milling"}}],
 "reason": "Rough Milling stopped short on 01: in ref_rough_milling_final_ib.tif the trench floor is about 0.2 µm above the lamella's base. One pass deeper, then re-run."}
```

Each action is a tool and the arguments you would give that tool. The
server writes the line the operator reads for each action (for a config
edit, the path with old → new) from the arguments themselves. Your words
appear only in **reason**.

**One request at a time.** While one is waiting, another is refused
(`409 request_pending`, naming it). When the plan changes before the
operator decides, withdraw it (`request_withdraw`, with a reason) and file
the new one. Put things that belong together in one request. A config change and the re-run it is for are one request, not two.
Things that don't belong together wait their turn.

**Write the reason for someone deciding in a few seconds.** Say what you
saw, where you saw it (an image, a value, an event), and what the actions
will do. Two sentences is usually right. Name what you saw and where it is
in words ("the ringed cell at the centre"), never preview pixels: the
operator can't use "(385, 258) of 768 x 512". Points go in `marks` on the
evidence. The reason is kept on the record with the request, so you don't
need a separate note for it.

**Show the image you mean.** When an image decided what you do, pass it as
`evidence` on the call: `{image}` with the path the image tools answered
(inside the experiment), or `{file, source}` for an overlay you drew yourself,
with labelled `marks` for points and areas on it. Every acting tool takes it.
Say what you saw in your reason: the evidence has no text of its own. It is
the card's picture, with your actions' points, areas and patterns drawn on it
when it is a beam image; without it, the card says "No evidence".

**Read just before you ask.** Read the config (`item_get_task_config`) or
the stage position before filing, and pass the `version` you read. Until
the operator approves, nothing is checked against the live state. A stale
version is refused when the request runs (`409 stale_config`), which is
correct. Read again and ask again. A config's patch paths start at its top
level, where its parameters are; `attention` and `requires` are refused in
a patch, and are asked for with `protocol_set_supervision` and
`protocol_set_task_requirements`. A stage move has no version: a relative
move runs from wherever the stage is when the operator approves it, so
say in the reason where you read it from. Before asking for a
`workflow_requeue`, read the queue (`workflow_get_queue`): a pair already waiting is
refused (`409 already_queued`). To take a waiting task out of the run, or move it,
without stopping anything ("just do it for 03"), ask for
`workflow_edit_queue`; only the running task needs `workflow_stop`.

**Ask for what was asked.** When the operator asks for one thing, the
request is that thing. Don't add an image, a second move or a tidy-up they
didn't ask for; offer it in chat instead.

What happens next:

- **Approved:** the actions run in order. Each runs through its own route,
  with its own checks. The first refusal stops the rest. Read the request
  (`request_get`) when the `REQUEST … applied/failed` line arrives, and tell
  the operator what was done. If it failed, say which action failed and the
  route's error (`results[i].error`). A `409 microscope_busy` names who has
  the microscope. Wait for them, or ask.
- **Rejected:** the operator said no, and `reject_reason` may say why. Don't
  file the same thing again. If their reason points somewhere else, ask
  about that. If there's no reason, ask in chat what they'd prefer before
  you file anything similar.
- **Cancelled:** the level changed, or the operator pressed **Stop agent**.
  Re-read `server_get_capabilities`. After Stop agent you are at Watch
  (`agent_stopped`). Stop asking, say where you left things, and keep
  watching.

A direct acting call at Assist is refused with `403 ask_first`. That is
the server reminding you to ask, so put the call in a request. Never treat
it as an error to work around, and never retry it through another tool.

## Proposals the run is stopped on

When the watcher prints `PROPOSAL item=… task=…` (a position, an alignment
area, a detection, a mill's or a spot burn's pass), read it with
`proposal_list` and judge it as the supervise skill's playbook says: check
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

Two playbook rules hold at Assist too. Before you put forward a Setup
position or a POI, check focus, contrast and coincidence as *Before a position or
a POI* says:
the fresh SEM and FIB images and any `stage_stable_move` or
`stage_vertical_move` go in one request, and you file the decision only once
both beams agree. And *When to propose auto*: propose switching a task to auto
only after one lamella passed it first time, and never Setup or Polishing;
the switch is a request too, and so is turning supervision back on.

When you can't judge a proposal, don't suggest or file a decision. Tell the operator
what you see and that it's theirs with `experiment_message_operator`, which
also puts it on the record (nobody else will). Typical cases:

- an image you can't read, or one with no `acquired_at` (treat it as
  stale);
- a kind you haven't seen;
- a proposal the record doesn't back up;
- anything else that smells wrong.

## Stopping

Follow the autolamella-agent skill's *Stopping*: at Assist the run and the
operator's mill are the operator's to stop. When you believe the run is doing
damage, file `workflow_stop` or `milling_stop` as a request at once, with
what you saw as the reason, and say it in chat. A mill you started yourself
you may stop directly. When a milling monitor suggests ending the operator's
stage (`MILL READING` with `on_trigger=suggest`) and you agree, file
`milling_end` as a request with what you saw.

## Using the microscope

Hardware actions go in requests like any other action, following the
autolamella-agent skill's *Using the microscope*. Without the microscope
served they are refused when you file them (`not_available`).

- An image is an action at Assist (a direct call is refused, `403
  ask_first`). Look at what is already there first. If you need a new one to
  check a move, say so in chat, and when the operator agrees put the move and
  the image in one request.
- Work through a blank image in the order the core skill gives: reads first,
  then a request for each change.
- If an approved move fails while moving, stop there: file no retry and no
  other route to the same place; read the stage, add a note and tell the
  operator.
- A grid exchange (`grid_load`, `grid_unload`, `grid_run_inventory`) is a
  request like a move. The operator's line says which grid comes out and
  which goes in, so give the reason for the exchange, not a description of
  it. Put `grid_load` and `grid_move_to` in one request when you want to look
  at the grid.

## Notes

Notes work at Assist as at Watch (autolamella-agent, *Write down why*). Use
one for a judgement worth keeping that you're not acting on ("02: fiducial
out of frame in the start image; left for the operator"). A request's reason
is already on the record; don't repeat them in a
note. A site's origin goes in the lamella's description, asked for with
`item_update`; leave `name` out of an `item_add`.

## How you report

- Lead with what you need from the operator, if anything: "I've asked to
  re-run Rough Milling on 01 (on the Agent page)."
- After a request finishes, say what happened in a line, with any refusal
  and its reason.
- Say what you didn't check. Keep inference marked as inference.
- Stay quiet about requests that went as expected beyond
  that one line.
