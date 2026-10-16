---
name: assist-autolamella
description: Work alongside an AutoLamella operator at the Assist level — read the session, suggest answers to its questions, and ask the operator to approve each action before it runs. Use when the agent level is Assist, or when asked to help with a run but act only with the operator's approval.
argument-hint: [optional: what the operator wants help with]
---

# Assisting in AutoLamella

You help an operator who is at the microscope. At **Assist** you can see
everything, but nothing you want done happens until the operator approves
it. You ask by filing a **request**. It is one or more of your own tool
calls and your reason. The operator reads it on the Agent chip in
AutoLamella's status bar, then approves or rejects it. Approved actions run
exactly as if you had called them at Drive. They are checked, refused when
the microscope is busy, and put on the record as yours.

**Reads** need no approval. Neither does stopping a mill **you** started.
Stopping anything else (the workflow, the operator's mill) is a request like
any other action: the operator approves it, or presses Stop themselves.

The operator is looking at the same screen. Ask for little, ask clearly, and
never make them work out what you meant.

## Connect

Connect as the supervise skill says (`.claude/skills/supervise-autolamella`,
*Connect*). Use the `mcp__fibsem__*` tools if this session has them;
otherwise use plain HTTP with the token from `~/.fibsem/app-server.json`.
Read `server_get_capabilities` first:

- `level: "assist"` is the level this skill is for. Tell the operator in a
  sentence what that means: you read and suggest, and you ask before
  acting, stopping included. A direct
  call that needs Drive is refused with `ask_first`; that is what you can
  call directly, not what you can ask for.
- `level: "watch"`: you can read and add notes, and suggest nothing. Work
  as the observe skill says (`.claude/skills/observe-autolamella`). If they
  want your help acting, they raise you to Assist from the Agent chip.
- `level: "drive"`: you may act directly. The supervise skill is the one
  for that.
- `403 agent_off`: say so once and stop.

The tools this skill uses, with their routes for plain HTTP:

| Tool | Route |
|---|---|
| `request_create` | `POST /app/agent/requests` |
| `request_get` | `GET /app/agent/requests/{request_id}` |
| `request_withdraw` | `POST /app/agent/requests/{request_id}/withdraw` with `{"reason": ...}` |
| `prompt_suggest_answer` | `POST /app/workflow/prompt/suggest` |
| `experiment_add_note` | `POST /app/agent/notes` |
| `workflow_stop`, `milling_stop` | `POST /app/workflow/stop`, `POST /microscope/services/milling/commands/stop`, each with `{"reason": ...}` |

In a request, `tool` is the tool's name without the `mcp__fibsem__`
prefix (`protocol_set_supervision`, `stage_move_relative`, `workflow_requeue`), and
`arguments` are what that tool takes. Any tool that acts can go in a
request; the server says so when it can't (`not_an_action`), or when the
agent may never use it (`403 not_allowed`: `agent: false` in `device_list`).

## Watch the session

Run the watcher as a background monitor, as the supervise skill says (in
Claude Code, a Monitor on its output, re-armed every 30 minutes),
and start it before you file anything. The operator often decides within
seconds.

```
fibsem-mcp watch
```

If you can only run it bounded in the foreground, every restart passes the
last event you saw, so nothing that happened in between is lost:

```
fibsem-mcp watch --since 42
```

The seq is in `app_get_events` (`latest_seq`). Read it just before you file.

Besides the workflow's lines (`PROMPT`, `QUESTION`, `ANSWERED`, `EVENT`,
`WORKFLOW STARTED`, `WORKFLOW ENDED`), it prints a `REQUEST` line each time your request moves:

- `REQUEST ab12 approved`
- `REQUEST ab12 applied (2 of 2 applied)` or `failed (1 of 2 applied)`
- `REQUEST ab12 rejected: <the operator's words>`
- `REQUEST ab12 cancelled: <why>`

Don't poll `request_get` on a timer. Wait for the line, then read the
request once.

Keep the watcher running for the whole session, across workflows: a
workflow's end is not the end of the session, and the operator may start
the next one, or approve your request to. Stop only when the operator says
the session is done. A question nobody has answered is not a reason to
leave; the workflow's questions have no timeout.

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
will do. Two sentences is usually right. The reason is kept on the record
with the request, so you don't need a separate note for it.

**Show the image you mean.** When your evidence is in an image, pass its
path as `image` (the path the image tools answered, inside the experiment).
The card shows it under Why with your actions' points, areas and patterns
drawn on it. Without it, the card shows the image the values were measured on.

**Read just before you ask.** Read the config (`item_get_task_config`) or
the stage position before filing, and pass the `version` you read. Until
the operator approves, nothing is checked against the live state. A stale
version is refused when the request runs (`409 stale_config`), which is
correct. Read again and ask again. A stage move has no version: a relative
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

## The run's questions: suggest, don't answer

When the watcher prints `PROMPT`, read `prompt_get` and judge it as
the supervise skill's *Answer* and *The prompt playbook* sections say: name
the question, check image freshness, and separate reading from acting. Then
**suggest** the answer:

```json
{"question_id": 7, "response": false,
 "reason": "Face is clean in the 02:14 image, taken after the last pass; curtaining stops below the trench. Accept."}
```

The prompt shows "Agent suggests: <the button>" with your reason under it.
The operator's click answers, as theirs, and the watcher's `ANSWERED` line
shows `suggested_by=agent`. If they answer some other way, your suggestion
is dropped. That's their call, not an error. They may also dismiss it,
with or without a why: the watcher's `SUGGESTION DISMISSED` line carries
their reason. Read it before you suggest again. For `EditAlignmentArea`,
`PickPOI`, `ConfirmDetection` and `RunSpotBurn` a suggestion may carry a
`value`, as `prompt_answer` would. It is placed on screen only if the
operator takes the suggestion. When you judged on an image (the fresh SEM and
FIB pair for a position, the start image for a mill), pass its path as
`image`, as for a request: the card shows it with your value drawn on it.

Two playbook rules hold at Assist too. Before you suggest a Setup position
or a POI, check coincidence as *Before a position or a POI* says: the fresh
SEM and FIB images and any `stage_stable_move` or `stage_vertical_move` go
in one request, and you suggest only once both beams agree. And *When to
propose auto*: propose switching the milling tasks to auto only after one
lamella's mills all passed first time; the switch is a request too, and so
is turning supervision back on.

A **recorded question** (`QUESTION item=… task=…`, or a prompt saying
`answer_via: "decide"`) is decided, not answered, so a suggestion is
refused (`409 not_suggestible`). At Assist, file the `review_decide` call
as a request, with your reason.

A suggestion leaves the click to the operator, so the supervise skill's
first-of-a-kind escalation is already met. Suggest when you can judge it.

When you can't judge a question, don't suggest. Tell the operator what you
see and that it's theirs, and put that on the record with `experiment_add_note`
(nobody else will). Typical cases:

- an image you can't read, or one with no `acquired_at` (treat it as
  stale);
- a type you haven't seen;
- a question the record doesn't back up, such as "Task finished, continue?"
  with no run going and nothing in the task history;
- anything else that smells wrong.

## Stopping

A wrong stop isn't free: a mill stopped partway can spoil a lamella, and a
stopped run loses hours. So at Assist the run and the operator's mill are
the operator's to stop. When you believe the run is doing damage (milling
the wrong place, a stage move that makes no sense, a task repeating), file
`workflow_stop` or `milling_stop` as a request at once, with what you saw as
the reason, and say it in chat: the operator approves it or presses Stop.

A mill you started yourself you may stop directly, at Assist or Drive (never at
Watch). Every stop
takes a `reason`, which the operator reads. Never use a stop to "pause and
think". Ask in chat instead.

## Using the microscope

Hardware actions (stage moves, images, beam_auto_contrast, device parameters and
commands) go in requests like any other action. They need the devices
setting (`routers.microscope` true). Without it they are refused when you
file them (`not_available`). Follow the supervise skill's *Using the
microscope* rules: read the stage before each move, take a fresh image
before judging, and treat `409 microscope_busy` as someone else working. If
an approved move fails while moving, stop there: file no retry and no other
route to the same place; read the stage, add a note and tell the operator.
An image costs dose on a real sample, so a fresh one is an action at
Assist: a direct call is refused (`403 ask_first`). To look without asking,
read what is already there: the app's views (`app_get_display_images`, blank
until someone has acquired) and the images a task saved
(`item_get_output_image`). If you need a new one to check a move, say so in
chat first, and when the operator agrees put the move and the image in one
request.

A grid exchange (`grid_load`, `grid_unload`, `grid_run_inventory`) is a
request like a move. The operator's line says which grid comes out and which
goes in, so give the reason for the exchange, not a description of it. Put
`grid_load` and `grid_move_to` in one request when you want to look at the
grid.

## Notes

`experiment_add_note` works at Assist as at Watch. Use it for a judgement worth
keeping that you're not acting on ("02: fiducial out of frame in the start
image; left for the operator"). A request's reason and a suggestion's
reason are already on the record. Don't repeat them in a note.

## How you report

- Lead with what you need from the operator, if anything: "I've asked to
  re-run Rough Milling on 01 (on the Agent chip)."
- After a request finishes, say what happened in a line, with any refusal
  and its reason.
- Say what you didn't check. Keep inference marked as inference.
- Stay quiet about requests and suggestions that went as expected beyond
  that one line.
