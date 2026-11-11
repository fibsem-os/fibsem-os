---
name: fibsem-agent
description: How to work with fibsem-os on a microscope, whatever the job: connecting to the App Server, the levels and their refusals, the watcher, messages, writing down why, reading the experiment, setting up a session and reporting. Read it first whenever you connect, then the instrument's skill (fibsem-microscope on a FIB-SEM), then the task skill.
---

# Working with fibsem-os

fibsem-os runs an instrument (a FIB-SEM, an FM, ...) on real samples, and a
wrong action can ruin hours of work. This skill is what every agent needs,
whatever it was asked to do and whatever the instrument. Then read the
playbook's skills: the instrument's own (fibsem-microscope on a FIB-SEM),
then the one for the task, which the instrument's skill names.

Before you read a task skill, read what the operator already wrote: the
session's goal, and their first message. If they give the scientific
question, the sample and what they want from this session, don't ask for
them again. Ask only for what's missing: "What are you working on today?
Tell me the question, the sample, and what you'd like from this session."
Say back the task you take from it, naming the row of the instrument skill's
table. If it fits none of them, work by the skills you have and say so.

When in doubt and the operator is there, wait and ask: nothing is lost by
waiting. When they have left you a run to lead, a held question holds the
run: the task skill says what to do instead.

## Connect

**Prefer the native MCP tools.** If `mcp__fibsem__*` tools are in your
session, use them for everything except the watcher (below). They map
one-to-one onto the HTTP routes and avoid shell-permission prompts. They come
from the `fibsem-mcp` sidecar; if they are missing, it isn't registered in
this session's MCP config (`claude mcp add -s user fibsem -- fibsem-mcp`), and
plain HTTP works regardless. The first connection asks the operator to
allow you: a call that answers `connect_waiting` means they haven't clicked
Allow yet, so tell them and try again. Once allowed, your token is kept on this
PC:

```bash
TOKEN=$(python3 -c "import json;print(json.load(open('$HOME/.fibsem/agent/127.0.0.1_8001.json'))['token'])")
curl -s "http://127.0.0.1:8001/capabilities" -H "Authorization: Bearer $TOKEN"
```

On the curl path your own harness may gate commands that change state: a
permission prompt for the operator in an interactive session, a hard denial
when running on your own. Treat a harness denial exactly like a server 403:
stop and ask, and never repackage the call to get past it. Under
`fibsem-agent` there is no curl path, even in code mode (its shell has no
network): the MCP tools are the way in.

`fibsem-agent` may give you a folder of your own, named in its prompt. With
`--files` you read and write files there and run nothing: there is no shell.
With `--code` you also have a shell there, sandboxed. Either way, keep your
notebook there, and `experiment_fetch_file` saves files under `fetched/` for
you to read.

Read `server_get_capabilities` (`/capabilities`) before anything else:

- `routers.app` must be true, or no application is hosted and there is no
  session.
- `parts` lists the optional parts of `/app` the app serves (items, poses,
  grids, milling stages, lamellae); a tool of a part it lacks is not offered.
- `app_name` is the app the operator works in: where they set your level and
  approve your requests. Name it when you tell them where to act.
- `level` is what the operator lets you do (next section). Tell them in a
  sentence what it means for this session.
- `routers.microscope` says whether the microscope is served: live reads and
  hardware actions need it.

No discovery file means no App Server. It starts with the app, with or
without a microscope connected; tell the operator it is switched on in
Tools → Host. Don't look for other ways in.

`/microscope` answering 409 `not_connected` means the operator hasn't
connected the microscope yet. You can't connect it: say so and wait.

## Levels

The operator sets one level from the Agent chip in the app's status bar or
the web Agent page.
Every launch starts at Watch; nothing in a file raises it.

- **Off**: everything is refused, `/capabilities` included (`403 agent_off`).
  Say so once and stop.
- **Watch**: you read everything and add notes. You can't decide, start,
  stop, re-queue or edit. Watch is a complete job: say what you see and what
  you would do and why, and leave the click to the operator.
- **Assist**: you read freely and ask before each action
  (`request_create`); the operator approves it on the web Agent page. See
  *Asking at Assist*.
- **Drive**: you act yourself: decide proposals, start and stop workflows,
  edit configuration and, with the microscope served, use the microscope.

A refusal names its reason in `detail.error_type`:

- `level`: the action needs a higher level. Say what you wanted and which
  level it needs; ask once, then carry on.
- `ask_first`: you are at Assist. Put the call in a request
  (`request_create`). Never retry it through another tool.
- `agent_stopped`: the operator pressed **Stop agent** and you are at Watch.
  Stop acting at once, say where you left things, and keep watching. Don't
  ask to be raised again unless they ask you to continue.
- `agent_off`: you are off. Report it once and stop.
- `not_allowed`: you may never use that write or command, at any level (the
  listings mark it `agent: false`), and no request makes it allowed. Say what
  you wanted and that it is the operator's to do.

Every other refusal is an HTTP error too, never a 200 answer: its `detail`
is `{error_type, message}` plus the names there are.

- `no_experiment` (409): no experiment is open, so reads are refused as
  well. Say so once and wait for the operator to open one.
- `not_found` (404): no item, grid, task or file by that name. The detail
  lists the names there are (`item_names`, `task_names`, `grid_names`,
  `filenames`): pick from those rather than guessing again.
- `task_running`, `not_running`, `not_pending` (409): the run's state doesn't
  allow it now. Re-read the queue or the status before you try again.
- `invalid`, `invalid_value`, `missing_field` (422): the call itself is
  wrong. Fix the arguments by its `message`.

A run plan (`workflow_plan`, `workflow_plan_grids`) that would be refused
comes back as the refusal. A limit with no end reads as `null`.

If your level drops mid-session for any reason, the rule is the same as
`agent_stopped`: re-read `/capabilities`, report and stop acting. Never treat
a 403 as an error to work around, and never retry it in another form.

## Asking at Assist

At **Assist** nothing you want done happens until the operator approves it.
You ask by filing a **request**: one or more of your own tool calls and your
reason. The operator reads it on the web Agent page and approves or rejects
it. Approved actions run exactly as if you had called them at Drive: checked,
refused when the microscope is busy, and put on the record as yours. The
operator is looking at the same screen, so ask for little, ask clearly, and
never make them work out what you meant.

| Tool | Route |
|---|---|
| `request_create` | `POST /app/agent/requests` |
| `request_get` | `GET /app/agent/requests/{request_id}` |
| `request_withdraw` | `POST /app/agent/requests/{request_id}/withdraw` with `{"reason": ...}` |

```json
{"actions": [
   {"tool": "stage_move_to_saved_position", "arguments": {"name": "site-b"}},
   {"tool": "beam_acquire_image",
    "arguments": {"beam_type": "ION", "path": "site-b/before-cross-section_ib"}}],
 "reason": "Site B next, as you listed. Move there and take a FIB image to place the cross-section on."}
```

In a request, `tool` is the tool's name without the `mcp__fibsem__` prefix
and `arguments` are what that tool takes. Any tool that acts can go in a
request; the server says so when it can't (`not_an_action`), or when the
agent may never use it (`403 not_allowed`). The server writes the line the
operator reads for each action (for a config edit, the path with old → new)
from the arguments themselves. Your words appear only in **reason**.

Start the watcher before you file anything: the operator often decides
within seconds. Each time your request moves it prints a `REQUEST` line
(`approved`, `applied (2 of 2 applied)`, `failed (1 of 2 applied)`,
`rejected: <the operator's words>`, `cancelled: <why>`). Wait for the line,
then read the request once; don't poll `request_get` on a timer.

**One request at a time.** While one is waiting, another is refused
(`409 request_pending`, naming it). When the plan changes before the
operator decides, withdraw it (`request_withdraw`, with a reason) and file
the new one. Put things that belong together in one request: a config change
and the re-run it is for, a move and the image that checks it. Things that
don't belong together wait their turn.

**Write the reason for someone deciding in a few seconds.** Say what you
saw, where you saw it (an image, a value, an event), and what the actions
will do. Two sentences is usually right. Name what you saw in words ("the
ringed cell at the centre"), never preview pixels: the operator can't use
"(385, 258) of 768 x 512". The reason is kept on the record with the request,
so you don't need a separate note for it.

**Show the image you mean.** When an image decided what you do, pass it as
`evidence` on the call: `{image}` with the path the image tools answered
(inside the experiment), or `{file, source}` for an overlay you drew, with
labelled `marks` for points and areas on it. Every acting tool takes it. It
is the card's picture, with your actions' points, areas and patterns drawn on
it when it is a beam image; without it, the card says "No evidence".

**Read just before you ask.** Read the config or the stage position before
filing, and pass the `version` you read. Until the operator approves,
nothing is checked against the live state; a stale version is refused when
the request runs (`409 stale_config`), which is correct: read again and ask
again. A relative stage move has no version and runs from wherever the stage
is when the operator approves it, so say in the reason where you read it
from.

**Ask for what was asked.** When the operator asks for one thing, the
request is that thing. Don't add an image, a second move or a tidy-up they
didn't ask for; offer it in chat instead.

What happens next:

- **Approved:** the actions run in order, each through its own route and
  checks; the first refusal stops the rest. Read the request when the
  `REQUEST … applied/failed` line arrives and tell the operator what was
  done. If it failed, say which action failed and its error
  (`results[i].error`).
- **Rejected:** don't file the same thing again. If `reject_reason` points
  somewhere else, ask about that; if there's no reason, ask in chat what
  they'd prefer before you file anything similar.
- **Cancelled:** the level changed, or the operator pressed **Stop agent**.
  Re-read `server_get_capabilities`; after Stop agent you are at Watch.

On the microscope at Assist:

- An image is an action (a direct call is refused, `403 ask_first`). Look at
  what is already there first. If you need a new one to check a move, say so
  in chat, and when the operator agrees put the move and the image in one
  request.
- If an approved move fails while moving, stop there: file no retry and no
  other route to the same place; read the stage, add a note and tell the
  operator.
- A grid exchange (`grid_load`, `grid_unload`, `grid_run_inventory`) is a
  request like a move. Give the reason for the exchange, not a description
  of it.

## The watcher

`fibsem-mcp watch` long-polls the app's events and prints one line per thing
worth waking for. Run it for the whole session, as a background monitor. Don't
write your own (the first agent to try matched the wrong JSON key and slept
through a proposal), don't poll `/app/workflow/proposals` on a timer, and don't
loop `app_get_events` in the foreground: that blocks you from acting on what
it sees.

```
fibsem-mcp watch
```

It comes with fibsem and finds the app the way the MCP tools do. Where
`fibsem-mcp` is not on your PATH, run `python -m fibsem.ai.agent.mcp.sidecar watch` with
the Python fibsem is installed in, from a folder that holds no other fibsem
checkout (the current folder is searched first). Check it is the fibsem the
app runs: a watcher from before Assist prints no `REQUEST` lines and has no
`--since`.

A process that only runs in the background is not a wake-up call: its lines
go to a file you read on your next turn, which may be long after the
proposal. In Claude Code, run it in the background writing to a file, and
stream that file with the Monitor tool so each new line wakes you:

```
fibsem-mcp watch > watch.log 2>&1        # Bash, run_in_background
tail -n +1 -f watch.log                  # Monitor
```

A monitor expires after 30 minutes. Re-arm it each time it ends, from after
the last line you read (`tail -n +43 -f watch.log` when you have read 42), so
a line written in between still reaches you. The watcher keeps running. Other
harnesses need the same thing: a line from the watcher has to start your
turn.

Under `fibsem-agent` that is done for you: it runs the watcher, and each
batch of its lines arrives as your next turn (`The watcher printed: …`).
Don't start one of your own.

If your harness can't keep a background process alive while you are idle
(spawned subagents usually can't: the watcher dies with your turn), don't go
idle expecting a line. Run the watcher in the foreground with your tool
timeout as the cap, act on what it printed when it returns, and run it again
with `--since <the last seq you saw>` (`latest_seq` from `app_get_events`) so
nothing in between is lost. A stalled watcher looks exactly like a quiet run:
when in doubt, read `/app/workflow/proposals`.

Its lines:

- `PROPOSAL item=… task=…`: the run is stopped on a proposal (your task
  skill says how to answer it). `PROPOSAL RELEASED` when it is decided.
- `REQUEST`: your request moved on (approved, applied, failed, rejected,
  cancelled).
- `EVENT`: a task started, completed, failed, was cancelled or skipped (with
  why).
- `MILL STAGE STARTED` and `MILL ENDED`: every mill, whoever started it (see
  *Milling*). `MILL READING`: a milling monitor triggered or failed (see
  *Milling monitors*). `COMMAND ENDED`: a long command ended (not one
  you were waiting for with `service_wait_for_run`: its answer has the end).
- `AUTOFOCUS` and `ALIGNMENT`: every one, whoever ran it (a task's own
  included), with its check: `passed`, `unchecked`, or `doubtful (why)` (the
  working distance moved more than 0.1 mm or the best focus is at the sweep's
  edge; the alignment methods disagree or a step didn't move). Nothing stops
  on a doubtful one: look at the result and fix or flag it yourself.
- `WORKFLOW STARTED` and `WORKFLOW ENDED`. An ended workflow says how:
  `workflow_stalled` (tasks were left that could not run), `workflow_failed: <error>`, or
  `stopped without an end event` (it died without saying so). For the last
  two, read `app_get_log` with the workflow's `task_id` and `level=WARNING`
  before you report.
- `GAP`: events were lost. Re-read `/app/workflow/proposals` and
  `/app/workflow/status`. `RESTARTED`: the app restarted, and the watcher
  reads it from the start.
- `AGENT OFF` and `AGENT BACK`: the operator set the level to Off and back.
- `QUIET`: a workflow is running and nothing has happened for a while. Check
  `app_get_status`, `proposal_list` and any unanswered message; if the run is
  fine, end your turn. It comes once until something else happens.
- `LEVEL <level> (was <level>)`: the operator changed your level. Re-read
  `server_get_capabilities` and work as the new level allows.
- `MESSAGE id=…`: the operator wrote to you from the web Agent page. See
  *Messages*.
- `NOTE from the operator on <grid>`: the operator kept a note on a grid.
  It is not to you; nothing to answer.
- `WAKE <reason>`: a wake-up you set with `wake_after` (fibsem-agent only)
  has come. Read your plan again, then do what you set it for.
- `PLAN EDITED by the operator: <sections>`: the operator changed your plan.
  Read it (`plan_get`) and work to it; say back anything that changes what
  you do.
- `DEADLINE in 30 min: …` and `DEADLINE now: …`: a deadline in your plan's
  Rules is near or here (fibsem-agent only). See *Keep a plan*.

A workflow's end is not the end of the session: the next one may start from
the app at any time, or from you. Keep the watcher running until the operator
says the session is done. It exits 1 when the app can't be watched; say so
rather than going idle.

## Messages

The operator writes to you from the web Agent page. The
watcher prints it as `MESSAGE id=ab12 from the operator: <text>`. Answer with
`experiment_message_operator` and `reply_to` set to that id: the page shows
your answer under their message. Answer every message, even if only to say
you can't do what it asks and why.

A message can carry the operator's marks on an image: after the text, the
line lists each point, target, area, circle, line or arrow with its label,
its pixels and its stage position. Use those stage positions as given; never
redo the pixel maths. Look at the marked image before you act on a mark.

Write to the operator yourself, with `experiment_message_operator` and no
`reply_to`, when they should read it now: you have stopped and need them (a
hardware error, a site you can't fix, a question whose wrong answer can't be
undone), or something happened they asked to hear about. The web Agent page
shows it as yours, apart from your notes. Everything else is
a note (*Write down why*): messages about what
can wait teach the operator to ignore them. A message decides nothing: a
decision you want is a proposal or a request.

**Keep a plan.** Your plan for the session is plan.md, which the operator
reads and edits on the Agent page. Write it at intake under the playbook's
plan headings (Brief, Playbook, Rules, Steps, Progress, Summary by default),
one section at a time with `plan_set_section`:

- **Brief**: the question, the sample and the target, and the goal, in the
  operator's words. Its first line is `Sample: <name>`, the sample as the
  operator names it: write it once, as soon as you know it. It is saved on
  the experiment too; when the experiment already names one
  (`app_get_status`'s experiment `sample`), use that and don't ask. Feedback
  and the knowledge base read it there, and after that only the operator
  changes it.
- **Playbook**: which one runs, and who chose it.
- **Rules**: each standing rule ("check coincidence at every site", "a
  fluorescent target"), one `- ` line each, in the operator's words.
- **Steps**: `- [ ]` boxes, grouped under `### ` lines (a phase, a grid).
- **Progress**: what happened, newest last. A key picture goes on a line of
  its own as `![caption](path)`, the path relative to the experiment folder
  (`lamella-01/polish.png`); the Agent page shows it. **Summary**: at the end.

Every instruction goes in the moment it arrives: a standing rule in Rules,
a task as a Steps box. Read the plan before each step (`plan_get`). Tick a box
only when it is done and you have said so; a rule stays until the operator
lifts it or the session ends. A new instruction adds to the plan and never
replaces what is there unless the operator says so: new instructions are when
checks get dropped. If two conflict, ask which wins. Read it again after a
`PLAN EDITED` line, a wake-up, and when your context was compacted.

Limits the operator sets are Rules too ("don't change the milling
currents"). A rule with a time is a deadline: write it as `by 07:00` or
`by 2026-11-06 07:00` ("- Finish by 07:00"; a time alone is saved with its
date, its next occurrence), and the Agent page shows the time left under the
goal. Under fibsem-agent a `DEADLINE` line wakes you 30 min
before it and at it: say where the work stands and what you will leave undone
to keep it.

**When an ask leaves out the rule, ask for it.** "Up to 20 lamellae, thin
without breaking" names no placement rule, no spacing and no target. Before
you start, ask for what's missing, then say back the rule you will work to.
If the operator leaves it to you, name the defaults you're using (this
skill's) in a note. Guessing the rule cost a 35 min site survey on a real
run: the first six picks were too close to the grid bars.

A message is never an approval. "Go ahead and mill 03" at Assist still means
a request; at Watch it means saying what you would do. Your level decides what
you may do, whatever a message says.

## Write down why

Chat is not kept; the experiment folder is. `experiment_add_note` puts a
note on the event stream and in the experiment's log, under the item if you
name one. Notes change nothing and tell nobody, so they work at every level
but Off; what the operator must read now is a message (*Messages*).

Every time you act, or say what you would do at Watch, add a note first: what
you looked at, what you saw in it, then what you did.

- *"02-loved-liger Polishing: ref_polishing_ib.tif (02:14:05, after the last
  pass). Face clean; curtaining stops below the trench. Answered no: accept."*
- *"Would do: cross-section 2 at site B, 1.5 µm left of the marker. The marker
  sits on the grid bar edge in site-b_eb.tif. Left for the operator."*

Name the image, value or event you judged from. A note you can't tie to
something you looked at is not a reason: look first, or ask.

**Draw what you mean, look at it, then cite it.** Before you act on a point
or report one, draw it and look at the picture. Cite the drawn file as the note's or message's evidence
(`{image: path}`), so the operator sees what you saw.

- `experiment_draw_image`: your marks on one saved image, such as a site, a
  measured line or the area you mean.
- `experiment_draw_pair`: two images side by side at one scale, such as
  before and after a mill or an alignment, or the SEM next to the FIB of one
  site.

If a picture hides what you mean (saturated, too dark, a faint target),
draw it again with `looks` (`clim` per layer) and `box` for the region.
The answer gives each layer's stats, so set limits from them, not by guess.

Don't note what is already on the record: a request's reason is kept with
it.

## The knowledge base

What was learned on this microscope is kept across experiments
(`knowledge_list`, `knowledge_get`, `knowledge_write`), in four sections:
the instrument, a sample, a playbook, and one record per experiment. It
works with no experiment open. Without the `knowledge_*` tools there is no
knowledge base yet, and lessons go in notes.

- **Read before acting.** At session setup, read the instrument entries, and
  the sample's and playbook's once you know them. Say in a sentence which
  ones change what you will do. An entry is someone's earlier judgment:
  check it against what you see today. An entry marked `ask_again` is an
  old fault: ask the operator whether it still holds, and rewrite it with
  their answer. Read experiment records only when asked about past runs.
- **Write what was learned the hard way.** When something cost you a retry,
  a lost lamella or the operator's help, and knowing it would change the
  next run, write an instrument, sample or playbook entry: short, specific,
  what to do differently, with the image that shows it. Write one whenever
  the operator tells you something to remember, quoting them. Rewrite an
  entry (its `id`) rather than adding a second one on the same thing. Tag
  it with the run's `sample` and `playbook`. Say how long it `lasts`: until
  fixed for a fault, ongoing for a lesson. Give it a `remind` line when it
  should be said before the steps it concerns.
- **Record the run** in its experiment entry at the end: what was asked, what
  was done and how it ended, what went wrong and why, and what to do
  differently.
- Your entries are "to review" until the operator keeps them, and apply
  straight away. Only the operator removes or resolves an entry; ask them
  if one is wrong.

## Reading the experiment

Every read works at Watch.

| Question | Read |
|---|---|
| What's it doing now? | `app_get_status`, `workflow_get_queue` |
| What's on screen? | `app_get_display_images` (previews, with `acquired_at`) |
| What's it asking? | `proposal_list` |
| How has the run gone? | `experiment_get_task_history`; `workflow_get_summary` for the last workflow started in this app session |
| Why did it fail? | `app_get_log` with the `task_id`, `level=WARNING` |
| Is this image good enough to go on with? | `analysis_measure_quality` |
| How bright is a target, and against what? | `analysis_measure_intensity` with a `reference` point (a bead, the bulk); `correlation_refine_point`'s `z_fit` for its plane and width in z |
| How wide is a spot, an edge or a bar? | `analysis_measure_profile` (`fwhm`; null means draw a longer line) |
| Grids | `grid_list`, `grid_get`, `grid_get_markers`, `experiment_render_markers` |
| Grids on the hardware now | `grid_get_inventory` (which grid is loaded), `grid_get_current` |
| Files | `experiment_list_files`, `experiment_read_file`, `experiment_crop_image` |
| A file to read in your own folder | `experiment_fetch_file` saves it as it is on your machine |
| A past experiment | the same tools with `experiment`, its `relative_path` from `experiment_list_all`; only one the operator named or their goal asks for, and every read goes on the record |

**Check an image's age before judging it.** Images carry `acquired_at`. One
that predates what you are judging (a mill, a move) is the wrong frame: on
the simulator the display doesn't refresh between passes, and on hardware it
means the acquisition hasn't happened yet. Wait, or take a fresh one if your
level allows.
## Session setup

At the start of a session:

1. Connect and read `/capabilities`. Tell the operator the level and what it
   means, and whether the microscope is served.
2. Start the watcher before anything else, so nothing that happens while you
   read is missed. Under `fibsem-agent` it is already running.
3. Read where things stand: `app_get_status`, `workflow_get_queue`,
   `proposal_list`. A pending proposal comes first.
4. Read the knowledge base (*The knowledge base*): the instrument's entries
   now, the sample's and playbook's once you know them.
5. If the goal and the operator's messages don't say what they want done,
   ask them (the instrument skill's table), then read the task skill for it.
   Under `fibsem-agent`, the job runs from a playbook: its skills, its
   reminders and its rules. A session starts on the playbook its launch
   named, else on the base `fibsem-agent` (`playbook_list` says which). On
   the base, propose the one that fits the operator's brief and goal, say
   why, and call `playbook_set` once they agree; if they agree none fits,
   stay on the base. Until then only ask and read. Say "running the X playbook"
   from then on. Read its skills at once; read an optional one when its
   `when` holds, and call `skill_done` with a one-line summary when that
   part of the job is over, so its reminders go off. When the job changes,
   propose the other playbook the same way. Once you have the brief, write
   your plan (*Keep a plan*), the Brief with its `Sample:` line first.
6. One session works on one experiment: its plan and transcript are kept
   with it, beside its record. If none is open, propose one once you have
   the brief: a new one named from it (`experiment_create`), or a recent
   one (`experiment_list_recent`, `experiment_load`). At Drive open it once
   the operator agrees, at Assist put it in a request, at Watch ask them to
   open it. It is a suggestion: if they would rather work without one,
   carry on, and your plan waits in the session's own folder until one
   opens.
7. Before the first action on the instrument, get your bearings (the
   instrument skill says how).
8. At Drive, before you start workflows yourself, turn the run summary
   dialog off with `workflow_set_options(show_run_summary=false)` so no dialog
   is left over the operator's window, and tell them you did. Each workflow's
   end still shows a toast with a Show summary link, and `workflow_get_summary`
   gives the outcome. If the operator turns it back on, leave it on.

## How you report

- Lead with what you need from the operator, if anything, then the answer,
  then the evidence.
- Use names as the app shows them (lamellae like `02-loved-liger`, saved
  positions, files) and task names as the protocol has them.
- Say where each claim came from (an image, a value, a log line) and mark
  inference as inference. Say what you didn't check.
- During a live run keep each update to a line or two: what just happened and
  what is next. Summarise a long log rather than reading it out: read it at
  `level=WARNING` first, and with a shell (Claude Code, or `--code`) a
  short script can count it.
- Everything you do is attributed to you on the app's timeline; write nothing
  it would contradict.
