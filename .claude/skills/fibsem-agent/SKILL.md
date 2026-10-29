---
name: fibsem-agent
description: How to work with fibsem-os on a microscope, whatever the job: connecting to the App Server, the levels and their refusals, the watcher, messages, writing down why, reading the experiment and using the microscope. Read it first whenever you connect, then the task skill for what the operator wants done.
---

# Working with fibsem-os

fibsem-os runs a FIB-SEM (and an FM, where there is one) on real samples, and
a wrong action can ruin hours of work. This skill is what every agent needs,
whatever it was asked to do. Then read the skill for the task:

| The operator wants to | Read |
|---|---|
| Make lamellae with AutoLamella, or explain a run | autolamella-agent; to run a whole session left to you at Drive, autolamella-lead too |
| Screen grids, take grid overviews or pick sites | screen-grids |
| Do plain FIB work: images, milling patterns, spot burns, cross-sections, moving between sites | fib-experiment |

If you don't know what the operator wants done, ask before you read a task
skill: "What would you like to get done today?", naming the tasks above. If
it fits none of them, work by this skill alone and say so.

When in doubt, wait and ask the operator. Nothing is lost by waiting.

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
  has come. Do what you set it for.
- `DEADLINE in 30 min: …` and `DEADLINE now: …`: a constraint's time is near
  or here (fibsem-agent only). See *Keep a checklist*.

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

**Keep a checklist of the operator's instructions.** Every instruction goes
on your checklist (`server_set_checklist`, shown on the Agent page) in the
operator's words, the moment it arrives: each MESSAGE that asks for something,
and each standing rule ("check coincidence at every site", "a fluorescent
target") as `always`. Read it before each step. Mark an item `done` only when
it is done and you have said so; a standing rule stays until the operator
lifts it or the session ends. A new instruction adds to the list and never
replaces what is there unless the operator says so: new instructions are when
checks get dropped. If two items conflict, ask which wins.

Limits the operator sets go in `server_set_constraints`, beside the goal:
a time ("finish by 07:00", with `by`) or a rule for the whole session ("don't
change the milling currents"). Under fibsem-agent a `DEADLINE` line wakes you
30 min before a time and at it: say where the work stands and what you will
leave undone to keep it.

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

## Using the microscope

Moving the stage, acquiring images, autofocus and auto contrast, device
parameters and commands, milling. At Drive you call these yourself; at Assist
each one is a request (`request_create`); at Watch you only read. Use them
when the operator asks for something on the microscope, or for the
coincidence check before a position. Say what you are about to do first.

**Check before you start.** In `/capabilities`, `routers.microscope` must be
true. If it isn't, or your level doesn't allow the action, say plainly which,
and that the operator changes it (the Agent chip for the level; Tools → Host,
*Serve the microscope*, for the microscope). A hardware tool missing
from your tool list means the same thing. Don't look for another way in.

**An image costs dose.** On a real sample a fresh image is an action, not a
read. To look without one, read what is already there: the app's views
(`app_get_display_images`, blank until someone has acquired) and the images
already saved in the experiment (`experiment_read_file`).

**Get your bearings once, before the first move.** On an instrument you
haven't used this session (every read here works at Watch):

- `microscope_get_info`: which microscope it is (manufacturer, model,
  software version, the optional hardware fitted).
- `device_list`: what it has (beams, detectors, stage, an FM).
- `device_get_parameter` `on` for the electron and ion beams. If one is off,
  turn it on with `beam_turn_on` (at Assist, ask first) and say so. Turn a
  beam off (`beam_turn_off`) only when the work is done or the operator asks.
  Note it if a beam reads `blanked`. Some instruments have no `blanked`.
- `stage_get_geometry`: the poses, tilts and what a stable move does in
  each beam now. If a view says `scan_rotation_handled: false`, its image
  axes are not the stage's: don't `stage_stable_move` from it.
- `grid_get_holder_slots`: each holder slot's stage position and its
  calibration (this matters whenever you move between grids, workflow or
  not; with no holder slots set up, say so and work from stage positions). A slot whose `position` is null is not calibrated (or was
  calibrated against a pre-tilt or rotation reference the system no longer
  has). Say which: the operator calibrates them with the app's wizard, never
  you, and nothing moves to an uncalibrated slot.
- `grid_get_inventory` and `grid_get_current`: which grid is on the stage and
  whether the stage is at a slot. With no grids loaded in the app, ask the
  operator which sample is under the beam if the goal doesn't say.
- `microscope_get_state`: each beam's voltage, current, working distance and
  scan rotation, the detectors' type and mode, the stage position. Say what
  you read, and anything that looks wrong for the task (a milling current on
  the imaging beam, a detector the operator doesn't use).
- Take an initial image with each beam (`beam_acquire_image`) and look at it
  before moving: is the sample in view and in focus?
- Then confirm the geometry with one small `stage_stable_move` and a fresh
  image, and check the feature went where it says. If it didn't, stop and
  tell the operator what you saw rather than correcting by hand.

**Check the beams are on before imaging or milling.** The ion beam switches
itself off after a while idle (30 minutes on some instruments), so a beam on
at the start of a run can be off an hour later. Read `on` for the beams the
step uses; if one switched off, turn it back on with `beam_turn_on` and note
it, then carry on. Switching a beam on or off can take a while (a minute or
more), so don't switch one unless you are sure it is needed: read `on`
first, and let a beam you switched on settle before you image.

**Read the microscope before each step, not once.** The operator, a run or a
script may have moved it since you looked. Before a move, read
`stage_get_position`; before judging what is under the beam, take a fresh
image rather than trusting the last one. Say what you read when you act on
it.

**A blank or black image.** Work through it in order, and stop at the step
that explains it:

1. Read the beam's `on`. If it is off, turn it on with `beam_turn_on` and
   say so.
2. Read `blanked`. If the beam is blanked, set `blanked` to false with
   `device_set_parameter`.
3. Run `beam_auto_brightness_contrast` for that beam. The detector levels
   often stay as another pose or current left them: on the Arctis, a FIB
   image at stage tilt 0° came back black after milling until the levels
   were reset.
4. Take the image again, once. If it is still blank, stop and tell the
   operator what you read at each step.

**Check every result yourself.** An auto function's answer is a claim, not a
fact: alignment, autofocus, brightness and contrast, coincidence, thickness.
After each one, look at the image it left and read its numbers (score,
agreement, converged, at_edge, contrast), and check that they match what you
see. Never apply a correction a tool hands you (a thickness `centre_offset`,
a coincidence `dy`) without finding it on the image first. A task's own
alignment and autofocus are no different: read what they did before you
decide the next proposal. When one went wrong, you own it: fix it, or stop
and tell the operator, and say which result you didn't trust and why.

**Don't judge a poor image.** If an image is out of focus, too dark, saturated
or too noisy to read, fix it before deciding anything from it: refocus, run
`beam_auto_brightness_contrast` or `beam_auto_contrast`, or set
`line_integration` / `frame_integration` in `image_settings` for a cleaner
frame without a long dwell. Then take it again.

**Check what an autofocus did.** Read the working distance before
`beam_auto_focus` and after it. A change of more than about 0.1 mm is worth a
fresh image and a sentence to the operator before you go on. On a coated or
tilted surface that one focus can't hold, use a focus stack. At the 15° milling angle the FIB's
focus changes across the frame: autofocus with a `reduced_area` on the site
and judge focus only near the POI.

**Move with `stage_stable_move`.** Every move to something you see in an
image is a `stage_stable_move`. After a stage rotation or a scan-rotation
change, take a new image and measure again. `stage_move_relative` moves the
raw stage axes: on a tilted stage a y move there also changes the sample's
height, so the feature goes out of focus and drifts out of the other beam.
Use it only when the operator gives you stage coordinates. Before a large
stable move, `service_call_query` `project_stable_move` with the same offset
answers where the stage would go without moving it.

Measure on a preview with its own scale: `hfw / width` per preview pixel. The
`pixelsize` in an image answer is per pixel of the full image, which is
larger than the preview.

The other moves, by when to use them:

- `stage_vertical_move`: the feature is centred in one beam's image but off
  in the other.
- `stage_fm_stable_move`: a move from the fluorescence image.
  `service_call_query` `project_fm_stable_move` answers where it would go
  without moving.
- `stage_move_to_device`: travel between the FIB-SEM and an FM beside it.
- `stage_move_to_orientation` (SEM, FIB, MILLING) and
  `stage_move_to_milling_angle`: change the stage's rotation and tilt.
- `stage_move_absolute`: go back to a position you read. It tilts flat
  before a large rotation.
- Named positions the operator keeps in the Saved Positions panel:
  `stage_get_saved_positions`, `stage_move_to_saved_position`, and
  `stage_save_position` and `stage_remove_saved_position` to change them.

Insert the FM objective with `fm_insert_objective` and its `tilt` (0 or
-π), so it tilts the stage flat with the objective out first, as the FM tab
does. Retract the FM objective
(`fm_retract_objective`) before any move that changes the pose
(`item_move_to_pose`, `stage_move_to_orientation`,
`stage_move_to_milling_angle`, `stage_move_absolute`). In a request, put the
retract first. The one exception: on an offset FM mount, `item_move_to_pose`
or `stage_move_absolute` to a position at the other device crosses the
chamber with the objective out by itself.

If `fm_insert_objective` or `fm_retract_objective` is refused at a tilt,
xT's objective restrictions are on: ask the operator to turn them off in the
xT GUI, and don't retry.

After every move, read the position back and compare it with what you asked
for, then take a fresh image and check where the feature landed before the
next one. A move under about 1 µm may not happen at all; the position read
back says so. Move z in small steps and tilt in steps, with an image after
each.

**Coincidence: do it yourself.** To bring the beams onto the same feature,
centre it in one beam with `stage_stable_move`, find it in the other beam's
image, and `stage_vertical_move` by its offset there (`beam_type` = the beam
you measured in). Cap each step to a few µm, take both images again, and
repeat until it is centred in both. Work from whichever beam has it centred
already. Do it before any overview, and again at each site on a bent
grid: on one grid the coincident height ran from +20.8 to +33.6 µm between
sites. Don't use the coincidence commands: this loop worked better on a
real instrument. A coincidence error shows only as a vertical offset in the
image, never side to side.

To try (experimental): `alignment_check_coincidence`. Start from the SEM
when the error may be large. Look at its overlay, and use the `dy` only if
the `+` sits on your feature; it refuses nothing. Give it a `path` so the
pair can be replayed, and say in a note how it compared with your own
measurement.

**A refused move and a failed move are different.** A move refused before it
starts (outside the stage limits, `microscope_busy`, a bad argument, the
level or the allowlist) changed nothing: fix the request and go again. A move
that fails while moving (a hardware error, the touch alarm, stopped partway)
may have hit something. Then stop moving: don't retry it, and don't reach the
same place another way. Read the stage position, add a note saying what you
asked for and what happened, and hand back to the operator.

**A hardware error is the operator's.** When a command, mill, spot burn, image
or run fails with an error from the instrument, stop. That is a `500` (its
`error_type` names the exception) or a run that ends `failed` in the
instrument's words: "Module is Unavailable (hardware is degraded)", "No
connection to CAPP hardware", a device that timed out. Don't run it again, not
even once, and don't reach the same thing another way (another current,
another tool, requeueing the task): three spot burns into a degraded module
only added errors. Read what you can without imaging (`microscope_get_state`,
`stage_get_position`), then send the operator a message
(`experiment_message_operator`) quoting the error word for word, what you were
doing and where things were left, and wait until they say it is fixed. Our
own refusals (`403`, `409`, `422`) are not hardware errors: they keep their
rules above.

**Milling.** Mill a protocol's stage with `milling_run_stage` and a
multi-stage job with `milling_run_task`. Before milling, look at
`milling_preview_patterns` on a saved ion image. It takes the stages as
`milling_run_stage` does, in metres, so never convert µm to pixels by hand.

Only where neither fits, `prepare` the settings and patterns, mill with `run`
(not `start`, which reports no progress and records nothing), then `clear`
and `restore`, and check the ion beam current is back where it was. `run`
answers at once with its `run_id` (`state: running`) and mills on, in control
of the microscope: call `service_wait_for_run` milling with that id, again
while it says running, before you `clear`. It ends `done` (`result:
completed`), `cancelled` or `failed` with its error. `service_get_parameter`
milling `progress` shows how far it has got. A spot burn
(`milling_run_spot_burn`, `milling_estimate_spot_burn` first) works the same
way, and so do a GIS deposition (`coating_deposit`) and a sputter coat
(`coating_sputter`), followed with `service_wait_for_run` coating.

Follow every mill the same way, whoever started it (you, a workflow task, the
operator): the watcher prints `MILL STAGE STARTED` for each stage and `MILL
ENDED` with each stage's outcome. Check the `current` and `depth` on each
stage's line against what the protocol or you meant to mill; if they are
wrong, stop the mill as your level allows (see *Stopping*) and tell the
operator. The beam mills only at currents it lists: ask for another and it
mills at the nearest one, and `MILL ENDED` says so in brackets after that
stage (each stage's record has `milling_current_used`). Ask only for listed
currents, and when a stage ran at another than you meant, say so before
you judge what it milled (a 60 pA polish ran at 30 pA on a real run).
Collect the result with `service_wait_for_run` milling, with the line's
`run` as `run_id`, or without one for the mill going on.

**Milling monitors.** A stage can list monitors that watch it while it
mills, each `{name, config}` in its `monitors`. `milling_list_options` lists
them with their config fields; you pick one by name and set its config, never
supply code. FM Intensity watches the fluorescence under the mill and triggers
on a drop. It needs an Arctis whose software restrictions on running the FM
while milling an operator has disabled: if they haven't, don't use it. It is
experimental: never turn it on yourself, at any level. Propose it, and use
it only after the operator says yes for this run; then help them set it up
(which stage, the area to watch, the drop settings).
SEM Peek pauses the mill every `interval` seconds of milled time for an SEM image.
RTM, listed only where the instrument has it, reads what the ion beam reads
back from the patterns as they mill; each reading carries every pattern's
mean value, by its draw index. It only watches: it never ends a stage.
A monitor's `on_trigger` decides what a trigger does: `end` ends the stage
itself, `suggest` only says so and the mill goes on. A trigger prints a
`MILL READING` line; for a suggestion, look (`milling_follow_readings` gives
every reading since a `seq`) and end the stage with `milling_end` if you
agree, as your level allows (see *Stopping*). Ending keeps what the stage
milled and a milling task goes on to its next stage; `milling_stop` stops the
whole task. A monitor set to `end` keeps working when you are slow or gone.
A protocol stage's monitor config is patched as
`stages.<stage>.monitors.<i>.config.<field>`.

**Align to the image you placed on.** A task's own alignment before milling
corrects only the shift from changing to the milling current, against an
image it takes there and then; the item's stored reference may be older than
your placement. When you place or move patterns on a new ion image, save that
image (`beam_acquire_image` with a path) and `alignment_align_to_reference` to
it right before the mill, so the patterns land where you put them.

**Aligning to a reference.** `alignment_measure_shift` first to see how far
off you are, then `alignment_align_to_reference`: beam shift for a small
offset, `stage` when it is beyond the beam shift's range. If the methods
don't agree on what is left, or the score stays low, the view has changed
(milled, drifted out of frame): stop and tell the operator rather than
aligning again.

**Reading an FM stack.** Judge a target from its planes, not the max
projection: `experiment_read_file` with `z` (and `channels`) shows one plane,
and the metadata gives each channel's `sharpest_plane`. The projection hides
which depth a spot is at and whether a neighbour sits above it. Look for small
targets (centriole doublets) plane by plane in full-resolution crops
(`experiment_read_file` with `z`, `experiment_crop_image`): a downsampled
view loses them. Check each candidate against the lamella window on a FIB or
SEM image before you call it a target: on a real run a stress-relief cut was
taken for the lamella. Until the emission-filter read-back is fixed, an FM image's
metadata can say Reflection for a fluorescence channel: go by the channel's
name and settings.

**Correlating FM to FIB.** Refine each fiducial pick with
`correlation_refine_point` before `correlation_correlate`, pass
`lamella_path`, and look at the overlay before trusting the answer. Always
give a surface point, `fm_surface_point` or `fib_surface_point`, so the
refractive-index correction is applied, and check the answer's
`ri_correction`. Without one, the targets sit deeper than the FM stack reads:
on a real run 4 of 7 targets were lost this way. Apply the result
(`item_apply_correlation`, as the tab's Continue) only on a `good` verdict,
or when the operator agrees.

**Overviews and focus stacks.** Ask `beam_estimate_overview` before
`beam_acquire_overview`. The stage moves between tiles often take longer than
the scanning, so a 3 × 3 that scans for seconds still takes minutes. Use
`beam_acquire_focus_stack` for a tilted view one focus can't hold. For
`fm_acquire_overview`, or any FM image, the objective must be in and near its
focus: inserted is not enough. Go to the item's stored objective position
(`item_move_to_objective_position`) or run `fm_auto_focus` first. Focus in
reflection, not in a fluorescence channel. Once it is in focus,
`fm_save_focus_position` saves it for an overview's `objective_start`
`focus`.

Set an overview up as the app does. Give the tile size in `image_settings`
(`hfw`, `resolution`, `dwell_time` belong there, not at the top level of the
settings) and use 3:2 tiles (e.g. 1536 × 1024). The instrument snaps a size
it doesn't list to one it does, so a square 2048 × 2048 came back as
2048 × 1768 and the stitch failed. Keep at least 15 % overlap.

**Overview or tile set.** The rule for every overview, beam or FM:

- "Take an overview" means the whole grid, from the grid's centre. Go there
  first (`grid_move_to`) and plan tiles that cover the whole grid: it is about
  2 mm across (1 mm radius), but the stage can't always reach all of it (on
  the Arctis only part of a grid is in travel from its centre). Ask the
  estimate (`beam_estimate_overview`, `fm_estimate_overview`): it judges reach
  from where the stage is, and an acquire with any tile it lists as
  unreachable is refused, so turn those off in `tile_mask` and acquire the
  rest. Reach is judged by each tile's centre, so use an odd number of rows
  and columns: a tile then sits on the grid's centre line. An even count can
  lose a row the stage half reaches (on the Arctis, 4 rows at 500 µm keep 2,
  5 rows keep 3). Say in your note which part of the grid the overview could
  not cover.
- "Take a tile set" (a 3 × 3 at 500 µm, say) means where the stage is now, at
  the size and field of view the operator gave.
- Any other image, yours or the operator's ("image the lamella", a look
  after a move, a check before milling), is one image: `beam_acquire_image`
  or `fm_acquire_image`, never a 1 × 1 overview. An overview moves the stage
  and comes back, takes longer and saves a stitch you don't need. For a wider
  view, take one image at a larger field of view; reach for a tile set only
  when one field can't cover what you need at the detail you need.

A point picked on an overview is only as good as the stitch. The tiles are
placed where the stage was told to go, and a stage that overshoots puts
features a few µm off (about 4 % in x and 7 % in y per step on the Arctis).
After moving to a point picked on an overview, take a fresh image and centre
on the feature itself before acting, as an operator would.

**Grids.** Loading, unloading, screening and picking sites on grids are the
screen-grids skill's.

**`409 microscope_busy` is someone else working.** Its `detail.in_control`
names who (`workflow`, `script`, or another command) and what. Say who has
the microscope and what you wanted to do, then wait for it to finish or ask
the operator. Never retry it in a loop, and never try the same thing through
another tool. While a run waits on a proposal it lends control, so your
command can go through; when the proposal is decided the run takes it back.

**Stop agent stops you mid-action.** If the operator presses it, your mill,
spot burn, auto focus or live imaging is stopped and you drop to Watch
(`agent_stopped`). Report where the microscope was left (read the stage
position; take no new image) and stop acting.

The operator's own buttons don't take control yet, so nothing refuses your
command while they work at the microscope by hand. If the events show them
acting (a `stage_position_changed` or acquisition event with `actor:
operator`), stop and ask before your next command.

## Stopping

A wrong stop isn't free: a mill stopped partway can spoil a lamella, and a
stopped run loses hours. Every stop takes a `reason`, which the operator
reads. Never use a stop to "pause and think": `milling_pause` holds a mill
where it is and `milling_resume` lets it go on, by the same levels as a stop.

- **Watch**: you can't stop anything. If the run looks like it is doing
  damage (milling the wrong place, a stage move that makes no sense, a task
  repeating), tell the operator at once, plainly, and say they should press
  Stop.
- **Assist**: a mill you started yourself you may stop or end directly
  (`milling_stop`, `milling_end`). Anything else (the workflow, the
  operator's mill) is a request, filed at once with what you saw as the
  reason.
- **Drive**: stop first when you believe the run is doing damage
  (`workflow_stop`, `milling_stop`), then say why. End a stage
  (`milling_end`) when it has milled enough, with why.

## Session setup

At the start of a session:

1. Connect and read `/capabilities`. Tell the operator the level and what it
   means, and whether the microscope is served.
2. Start the watcher before anything else, so nothing that happens while you
   read is missed. Under `fibsem-agent` it is already running.
3. Read where things stand: `app_get_status`, `workflow_get_queue`,
   `proposal_list`. A pending proposal comes first.
4. If you don't know what the operator wants done, ask them (the table at
   the top), then read the task skill for it.
5. Before the first microscope action, get your bearings (above).
6. At Drive, before you start workflows yourself, turn the run summary
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
  what is next. Summarise a long log with a short script rather than reading
  it out.
- Everything you do is attributed to you on the app's timeline; write nothing
  it would contradict.
