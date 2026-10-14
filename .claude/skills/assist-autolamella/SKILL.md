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
Read `get_capabilities` first:

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
| `request_actions` | `POST /app/agent/requests` |
| `get_request` | `GET /app/agent/requests/{request_id}` |
| `suggest_answer` | `POST /app/workflow/prompt/suggest` |
| `add_note` | `POST /app/agent/notes` |
| `stop_workflow`, `stop_milling` | `POST /app/workflow/stop`, `POST /microscope/services/milling/commands/stop`, each with `{"reason": ...}` |

In a request, `tool` is the tool's name without the `mcp__fibsem__`
prefix (`set_task_supervision`, `move_stage_relative`, `requeue_task`), and
`arguments` are what that tool takes. Any tool that acts can go in a
request; the server says so when it can't (`not_an_action`).

## Watch the session

Run the watcher as a background monitor, as the supervise skill says,
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

The seq is in `get_events` (`latest_seq`). Read it just before you file.

Besides the run's lines (`PROMPT`, `QUESTION`, `ANSWERED`, `EVENT`,
`RUN ENDED`), it prints a `REQUEST` line each time your request moves:

- `REQUEST ab12 approved`
- `REQUEST ab12 applied (2 of 2 applied)` or `failed (1 of 2 applied)`
- `REQUEST ab12 rejected: <the operator's words>`
- `REQUEST ab12 cancelled: <why>`

Don't poll `get_request` on a timer. Wait for the line, then read the
request once.

Keep watching until the run ends or the operator tells you to stop. A
question nobody has answered is not a reason to leave; run questions have
no timeout.

## Asking: `request_actions`

```json
{"actions": [
   {"tool": "update_item_task_config",
    "arguments": {"item_name": "01-alive-dory", "task_name": "Rough Milling",
                  "patch": {"milling.mill_rough.stages.0.pattern.depth": 8e-07},
                  "version": "<from get_item_task_config>"}},
   {"tool": "requeue_task",
    "arguments": {"item_name": "01-alive-dory", "task_name": "Rough Milling"}}],
 "reason": "Rough Milling stopped short on 01: in ref_rough_milling_final_ib.tif the trench floor is about 0.2 µm above the lamella's base. One pass deeper, then re-run."}
```

Each action is a tool and the arguments you would give that tool. The
server writes the line the operator reads for each action (for a config
edit, the path with old → new) from the arguments themselves. Your words
appear only in **reason**.

**One request at a time.** While one is waiting, another is refused
(`409 request_pending`, naming it). Put things that belong together in one
request. A config change and the re-run it is for are one request, not two.
Things that don't belong together wait their turn.

**Write the reason for someone deciding in a few seconds.** Say what you
saw, where you saw it (an image, a value, an event), and what the actions
will do. Two sentences is usually right. The reason is kept on the record
with the request, so you don't need a separate note for it.

**Read just before you ask.** Read the config (`get_item_task_config`) or
the stage position before filing, and pass the `version` you read. Until
the operator approves, nothing is checked against the live state. A stale
version is refused when the request runs (`409 stale_config`), which is
correct. Read again and ask again. A stage move has no version: a relative
move runs from wherever the stage is when the operator approves it, so
say in the reason where you read it from.

**Ask for what was asked.** When the operator asks for one thing, the
request is that thing. Don't add an image, a second move or a tidy-up they
didn't ask for; offer it in chat instead.

What happens next:

- **Approved:** the actions run in order. Each runs through its own route,
  with its own checks. The first refusal stops the rest. Read the request
  (`get_request`) when the `REQUEST … applied/failed` line arrives, and tell
  the operator what was done. If it failed, say which action failed and the
  route's error (`results[i].error`). A `409 microscope_busy` names who has
  the microscope. Wait for them, or ask.
- **Rejected:** the operator said no, and `reject_reason` may say why. Don't
  file the same thing again. If their reason points somewhere else, ask
  about that. If there's no reason, ask in chat what they'd prefer before
  you file anything similar.
- **Cancelled:** the level changed, or the operator pressed **Stop agent**.
  Re-read `get_capabilities`. After Stop agent you are at Watch
  (`agent_stopped`). Stop asking, say where you left things, and keep
  watching.

A direct acting call at Assist is refused with `403 ask_first`. That is
the server reminding you to ask, so put the call in a request. Never treat
it as an error to work around, and never retry it through another tool.

## The run's questions: suggest, don't answer

When the watcher prints `PROMPT`, read `get_pending_prompt` and judge it as
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
is dropped. That's their call, not an error. For `EditAlignmentArea` and
`PickPOI` a suggestion may carry a `value`, as `answer_prompt` would. It is
placed on screen only if the operator takes the suggestion.

A **recorded question** (`QUESTION item=… task=…`, or a prompt saying
`answer_via: "decide"`) is decided, not answered, so a suggestion is
refused (`409 not_suggestible`). At Assist, file the `decide_review` call
as a request, with your reason.

A suggestion leaves the click to the operator, so the supervise skill's
first-of-a-kind escalation is already met. Suggest when you can judge it.

When you can't judge a question, don't suggest. Tell the operator what you
see and that it's theirs, and put that on the record with `add_note`
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
`stop_workflow` or `stop_milling` as a request at once, with what you saw as
the reason, and say it in chat: the operator approves it or presses Stop.

A mill you started yourself you may stop directly, at any level. Every stop
takes a `reason`, which the operator reads. Never use a stop to "pause and
think". Ask in chat instead.

## Using the microscope

Hardware actions (stage moves, images, autocontrast, device parameters and
commands) go in requests like any other action. They need the devices
setting (`routers.microscope` true). Without it they are refused when you
file them (`not_available`). Follow the supervise skill's *Using the
microscope* rules: read the stage before each move, take a fresh image
before judging, and treat `409 microscope_busy` as someone else working. If
an approved move fails while moving, stop there: file no retry and no other
route to the same place; read the stage, add a note and tell the operator.
An image costs dose on a real sample. If you need one to check a move,
say so in chat first, and when the operator agrees put the move and the
image in one request.

## Notes

`add_note` works at Assist as at Watch. Use it for a judgement worth
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
