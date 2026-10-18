---
name: autolamella-agent
description: The ground every agent working with a running AutoLamella stands on — connecting to the App Server, the levels and their refusals, the watcher, writing down why, reading the experiment and using the microscope. Read it first whenever you connect to AutoLamella, then the role skill for your level (observe, assist or supervise).
---

# Working with AutoLamella

AutoLamella mills real samples, and a wrong action can ruin hours of work.
This skill is what every agent needs, whatever it was asked to do. The role
skills build on it:

- **observe-autolamella**: explain a session or a finished experiment, and
  never act. Any level, and offline from a folder.
- **assist-autolamella**: at **Assist**, suggest answers and ask the
  operator to approve each action.
- **supervise-autolamella**: at **Drive**, answer the workflow's questions
  and act.

When in doubt, wait and ask the operator. A workflow's questions have no
timeout, so nothing is lost by waiting.

## Connect

**Prefer the native MCP tools.** If `mcp__fibsem__*` tools are in your
session, use them for everything except the watcher (below). They map
one-to-one onto the HTTP routes and avoid shell-permission prompts. They come
from the `fibsem-mcp` sidecar; if they are missing, it isn't registered in
this session's MCP config (`claude mcp add -s user fibsem -- fibsem-mcp`), and
plain HTTP works regardless. The running app advertises itself in a
discovery file:

```bash
TOKEN=$(python3 -c "import json;print(json.load(open('$HOME/.fibsem/app-server.json'))['token'])")
curl -s "http://127.0.0.1:8001/capabilities" -H "Authorization: Bearer $TOKEN"
```

On the curl path your own harness may gate commands that change state: a
permission prompt for the operator in an interactive session, a hard denial
when running on your own. Treat a harness denial exactly like a server 403:
stop and ask, and never repackage the call to get past it.

Read `server_get_capabilities` (`/capabilities`) before anything else:

- `routers.app` must be true, or no application is hosted and there is no
  session.
- `level` is what the operator lets you do (next section). Tell them in a
  sentence what it means for this session.
- `routers.microscope` says whether the microscope is served: live reads and
  hardware actions need it.

No discovery file means no App Server. Tell the operator it is switched on in
Tools → App Server and starts when a microscope connects. Don't look for
other ways in.

## Levels

The operator sets one level from the Agent chip in AutoLamella's status bar.
Every launch starts at Watch; nothing in a file raises it.

- **Off**: everything is refused, `/capabilities` included (`403 agent_off`).
  Say so once and stop.
- **Watch**: you read everything and add notes. You can't answer, start,
  stop, re-queue or edit. Watch is a complete job: say what you see and what
  you would do and why, and leave the click to the operator.
- **Assist**: you read freely and ask before each action; the operator
  approves it on the Agent chip. You suggest answers rather than answering.
  See assist-autolamella.
- **Drive**: you answer questions, start and stop workflows, change
  supervision, re-queue, edit configuration and, with the microscope served,
  use the microscope. See supervise-autolamella.

A refusal names its reason in `detail.error_type`:

- `level`: the action needs a higher level. Say what you wanted and which
  level it needs; ask once, then carry on.
- `ask_first`: you are at Assist. Put the call in a request
  (assist-autolamella). Never retry it through another tool.
- `agent_stopped`: the operator pressed **Stop agent** and you are at Watch.
  Stop acting at once, say where you left things, and keep watching. Don't
  ask to be raised again unless they ask you to continue.
- `agent_off`: you are off. Report it once and stop.
- `not_allowed`: you may never use that write or command, at any level (the
  listings mark it `agent: false`), and no request makes it allowed. Say what
  you wanted and that it is the operator's to do.

If your level drops mid-session for any reason, the rule is the same as
`agent_stopped`: re-read `/capabilities`, report and stop acting. Never treat
a 403 as an error to work around, and never retry it in another form.

## The watcher

`fibsem-mcp watch` long-polls the app's events and prints one line per thing
worth waking for. Run it for the whole session, as a background monitor. Don't
write your own (the first agent to try matched the wrong JSON key and slept
through a question), don't poll `/app/workflow/prompt` on a timer, and don't
loop `app_get_events` in the foreground: that blocks you from acting on what
it sees.

```
fibsem-mcp watch
```

It comes with fibsem and finds the app the way the MCP tools do. Where
`fibsem-mcp` is not on your PATH, run `python -m fibsem.mcp.sidecar watch` with
the Python fibsem is installed in, from a folder that holds no other fibsem
checkout (the current folder is searched first). Check it is the fibsem the
app runs: a watcher from before Assist prints no `REQUEST` lines and has no
`--since`.

A process that only runs in the background is not a wake-up call: its lines
go to a file you read on your next turn, which may be long after the
question. In Claude Code, run it in the background writing to a file, and
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

If your harness can't keep a background process alive while you are idle
(spawned subagents usually can't: the watcher dies with your turn), don't go
idle expecting a line. Run the watcher in the foreground with your tool
timeout as the cap, act on what it printed when it returns, and run it again
with `--since <the last seq you saw>` (`latest_seq` from `app_get_events`) so
nothing in between is lost. A stalled watcher looks exactly like a quiet run:
when in doubt, read `/app/workflow/prompt`.

Its lines:

- `PROMPT`: a question is standing, with its `question_id`.
- `QUESTION item=… task=…`: the run is stopped on a recorded question (see
  *The workflow's questions*). `QUESTION RELEASED` when it is decided.
- `ANSWERED`: who answered and how; `suggested_by=agent` when the operator
  took your suggestion. If the operator answered, your read of that question
  is stale. `PROMPT WITHDRAWN` when the question went away unanswered.
- `SUGGESTION DISMISSED`: the operator declined your suggestion, with their
  reason when they gave one.
- `REQUEST`: your request moved on (approved, applied, failed, rejected,
  cancelled).
- `EVENT`: a task started, completed, failed, was cancelled or skipped (with
  why).
- `MILL STAGE STARTED` and `MILL ENDED`: every mill, whoever started it (see
  *Milling*). `COMMAND ENDED`: a long command ended.
- `WORKFLOW STARTED` and `WORKFLOW ENDED`. An ended workflow says how:
  `workflow_stalled` (decisions are pending), `workflow_failed: <error>`, or
  `stopped without an end event` (it died without saying so). For the last
  two, read `app_get_log` with the workflow's `task_id` and `level=WARNING`
  before you report.
- `GAP`: events were lost. Re-read `/app/workflow/prompt` and
  `/app/workflow/status`. `RESTARTED`: the app restarted, and the watcher
  reads it from the start.
- `AGENT OFF` and `AGENT BACK`: the operator set the level to Off and back.
- `MESSAGE id=…`: the operator wrote to you from the Agent card. See
  *Messages from the operator*.

A workflow's end is not the end of the session: the next one may start from
the app at any time, or from you. Keep the watcher running until the operator
says the session is done. It exits 1 when the app can't be watched; say so
rather than going idle.

## Messages from the operator

The operator can write to you from the Agent card. The watcher prints it as
`MESSAGE id=ab12 from the operator: <text>`. Answer with `experiment_add_note`
and `reply_to` set to that id: the card shows your note as your reply, under
their message. Answer every message, even if only to say you can't do what it
asks and why.

A message is never an approval. "Go ahead and mill 03" at Assist still means
a request; at Watch it means saying what you would do. Your level decides what
you may do, whatever a message says.

## The workflow's questions

A workflow asks in one of two ways:

- **A prompt** (`PROMPT`): read `prompt_get` and take its `question_id`. At
  Drive you answer it (`prompt_answer`), at Assist you suggest
  (`prompt_suggest_answer`), at Watch you say what you would answer.
- **A recorded question** (`QUESTION`): with the review preference on, a task
  asks a detection or a position confirmation on the item's record and stops
  on it. `review_list` lists it with `holding_the_run: true`. It is decided
  with `review_decide`, never answered or suggested (`409 not_suggestible`).

The operator outranks you: if they answer first, your job for that question
is over. A `409 stale_prompt` means the question changed under you, usually
because the operator answered. That refusal is correct; re-read and carry on.

## Write down why

Chat is not kept; the experiment folder is. `experiment_add_note` puts a
note on the event stream and in the experiment's log, under the item if you
name one. Notes change nothing, so they work at every level but Off.

Every time you act, or say what you would do at Watch, add a note first: what
you looked at, what you saw in it, then what you did.

- *"02-loved-liger Polishing: ref_polishing_ib.tif (02:14:05, after the last
  pass). Face clean; curtaining stops below the trench. Answered no: accept."*
- *"Would do: 04 Setup, POI 1.5 µm left of the marker. The marker sits on the
  grid bar edge in ref_setup_eb.tif. Left for the operator."*

Name the image, value or event you judged from. A note you can't tie to
something you looked at is not a reason: look first, or ask. Don't note what
is already on the record: a request's reason and a suggestion's reason are
kept with them.

Where a site came from (grid, square, rank, why it was chosen) belongs on the
lamella, not in a note: `item_update` with `{"patch": {"description":
"..."}}`. When you add a lamella (`item_add`), leave `name` out so it gets the
app's usual name.

## Reading the experiment

Every read works at Watch.

| Question | Read |
|---|---|
| What's it doing now? | `app_get_status`, `workflow_get_queue` |
| What's on screen? | `app_get_display_images` (previews, with `acquired_at`) |
| What's it asking? | `prompt_get`, `review_list` |
| How has the run gone? | `experiment_get_task_history`; `workflow_get_summary` for the last workflow started in this app session |
| Why did it fail? | `app_get_log` with the `task_id`, `level=WARNING` |
| One lamella | `item_get`, `item_list_outputs`, `item_get_output_image` |
| How thick is it? | `item_measure_thickness` on a FIB image from `item_list_outputs` |
| The plan | `protocol_get`, `item_get_task_config`, `protocol_get_task_config` |
| Grids | `grid_list`, `grid_get`, `grid_get_markers` |
| Grids on the hardware now | `grid_get_inventory` (which grid is loaded), `grid_get_current` |
| Any other file | `experiment_list_files`, `experiment_read_file`, `experiment_crop_image` |

**Check an image's age before judging it.** Images carry `acquired_at`. One
that predates what you are judging (a mill, a move) is the wrong frame: on
the simulator the display doesn't refresh between passes, and on hardware it
means the acquisition hasn't happened yet. Wait, or take a fresh one if your
level allows.

**A lamella's saved images.** `item_get_output_image` with a basename from
`item_list_outputs` gives a 768 px JPEG preview. Without MCP, fetch it over
HTTP; basenames have spaces in them, so URL-encode them:

```bash
curl -s -o out.jpg -H "Authorization: Bearer $TOKEN" \
  "http://127.0.0.1:8001/app/experiment/items/01-tough-mako/outputs/ref_Mill%20Fiducial_start_ib.tif"
```

`app_get_status` gives the experiment's folder (`experiment.path`). On the
microscope PC, the observe skill's offline reader reads the record from there
while the run is going.

## Using the microscope

Moving the stage, acquiring images, autofocus and auto contrast, device
parameters and commands, milling. At Drive you call these yourself; at Assist
each one is a request (assist-autolamella); at Watch you only read. Use them
when the operator asks for something on the microscope, or for the
coincidence check before a position. Say what you are about to do first.

**Check before you start.** In `/capabilities`, `routers.microscope` must be
true. If it isn't, or your level doesn't allow the action, say plainly which,
and that the operator changes it (the Agent chip for the level; Tools → App
Server, *Serve the microscope*, for the microscope). A hardware tool missing
from your tool list means the same thing. Don't look for another way in.

**An image costs dose.** On a real sample a fresh image is an action, not a
read. To look without one, read what is already there: the app's views
(`app_get_display_images`, blank until someone has acquired) and the images
a task saved (`item_get_output_image`).

**Get your bearings once, before the first move.** On an instrument you
haven't used this session (every read here works at Watch):

- `microscope_get_info`: which microscope it is (manufacturer, model,
  software version, the optional hardware fitted).
- `device_list`: what it has (beams, detectors, stage, an FM).
- `device_get_parameter` `on` for the electron and ion beams. If one is off,
  stop and tell the operator: switching a beam on is theirs. Note it if a
  beam reads `blanked`. Some instruments have no `blanked`.
- `stage_get_geometry`: the stage's poses, column tilts, pre-tilt, device
  origins and the milling angle; and, for each beam at the pose and scan
  rotation now, the stage move a +1 µm `stage_stable_move` along each image
  axis makes, and which way raw +z goes. If a view says
  `scan_rotation_handled: false`, its image axes are not the stage's: don't
  `stage_stable_move` from it.
- `grid_get_holder_slots`: each holder slot's stage position and its
  calibration. A slot whose `position` is null is not calibrated (or was
  calibrated against a pre-tilt or rotation reference the system no longer
  has). Say which: the operator calibrates them with the app's wizard, never
  you, and nothing moves to an uncalibrated slot.
- `grid_get_inventory` and `grid_get_current`: which grid is on the stage and
  whether the stage is at a slot.
- `microscope_get_state`: each beam's voltage, current, working distance and
  scan rotation, the detectors' type and mode, the stage position. Say what
  you read, and anything that looks wrong for the task (a milling current on
  the imaging beam, a detector the operator doesn't use).
- Take an initial image with each beam (`beam_acquire_image`) and look at it
  before moving: is the sample in view and in focus?
- Then confirm the geometry with one small `stage_stable_move` and a fresh
  image, and check the feature went where it says. If it didn't, stop and
  tell the operator what you saw rather than correcting by hand.

**Read the microscope before each step, not once.** The operator, a run or a
script may have moved it since you looked. Before a move, read
`stage_get_position`; before judging what is under the beam, take a fresh
image rather than trusting the last one. Say what you read when you act on
it.

**A blank or black image.** Work through it in order, and stop at the step
that explains it:

1. Read the beam's `on`. If it is off, tell the operator: switching a beam on
   is theirs.
2. Read `blanked`. If the beam is blanked, set `blanked` to false with
   `device_set_parameter`.
3. Run `beam_auto_brightness_contrast` for that beam. The detector levels
   often stay as another pose or current left them: on the Arctis, a FIB
   image at stage tilt 0° came back black after milling until the levels
   were reset.
4. Take the image again, once. If it is still blank, stop and tell the
   operator what you read at each step.

**Check what an autofocus did.** Read the working distance before
`beam_auto_focus` and after it. A change of more than about 0.1 mm is worth a
fresh image and a sentence to the operator before you go on. On a coated or
tilted surface that one focus can't hold, use a focus stack.

**Move with `stage_stable_move`.** To bring something you see to the centre
of an image, call `stage_stable_move` with its offset from the image centre in
metres, x to the right and y up, and the beam the image came from. From a
preview: `dx = (px - width/2) * hfw / width`, `dy = (height/2 - py) * hfw /
width` (pixel rows count down, so y flips). The server turns that into the
stage x, y and z for that beam's viewing angle and the stage tilt, rotation
and scan rotation, so the sample slides along its own surface at the same
height: the feature stays in focus, and at the coincidence point it stays in
view of the other beam too. `stage_move_relative` moves the raw stage axes:
on a tilted stage a y move there also changes the sample's height, so the
feature goes out of focus and drifts out of the other beam. Use it only when
the operator gives you stage coordinates.

`stage_stable_move` uses the stage and scan rotation as they are now. An
offset measured on an older image is only right if neither has changed since;
after a rotation or a scan-rotation change, take a new image and measure
again. Before a large stable move, `service_call_query` `project_stable_move`
with the same offset answers where the stage would go without moving it.

Measure on a preview with its own scale: `hfw / width` per preview pixel. The
`pixelsize` in an image answer is per pixel of the full image, which is
larger than the preview.

The other moves, by when to use them:

- `stage_vertical_move`: the feature is centred in one beam's image but off
  in the other. Pass its offset in the image where it is off, and that beam
  as `beam_type` (ION, the default, or ELECTRON); it moves the stage along the
  chamber vertical, which the beam where the feature is centred doesn't see.
- `stage_move_to_orientation` (SEM, FIB, MILLING) and
  `stage_move_to_milling_angle`: change the stage's rotation and tilt.
- `stage_move_absolute`: go back to a saved position. It tilts flat before a
  large rotation.

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
already. Don't use the coincidence commands: this loop worked better on a
real instrument. A coincidence error shows only as a vertical offset in the
image, never side to side.

To try (experimental): `alignment_check_coincidence` finds the feature you
centred in one beam (`beam`, ELECTRON by default) in the other beam's view.
It answers `dy` for `stage_vertical_move` with the other beam's `beam_type`,
and `dx`: a height error can't cause it, so a steady `dx` is a beam shift,
never a stage move. Start from the SEM when the error may be large: from the
FIB it shows about 3x larger in the SEM and can leave the frame. It answers an
overlay too. Look at it, and use the `dy` only if the `+` sits on your
feature; it refuses nothing. Give it a `path` so the pair can be replayed,
and say in a note how it compared with your own measurement.

**A refused move and a failed move are different.** A move refused before it
starts (outside the stage limits, `microscope_busy`, a bad argument, the
level or the allowlist) changed nothing: fix the request and go again. A move
that fails while moving (a hardware error, the touch alarm, stopped partway)
may have hit something. Then stop moving: don't retry it, and don't reach the
same place another way. Read the stage position, add a note saying what you
asked for and what happened, and hand back to the operator.

**Milling.** Mill a protocol's stage with `milling_run_stage` (a
`FibsemMillingStage` dict, e.g. one from `item_get_task_config`) and a
multi-stage job with `milling_run_task` (a `FibsemMillingTaskConfig` dict).
They mill as the app does: each stage's strategy, alignment between stages,
the beam put back. Leave `path` out or give one inside the experiment. Before
milling, `milling_preview_patterns` with a saved ion image and the stages
shows where they will mill; look at it. Both are long commands:
`service_wait_for_run` milling, and the result says the status, the folder
and how each stage ended.

Only where neither fits, `prepare` the settings and patterns, mill with `run`
(not `start`, which reports no progress and records nothing), then `clear`
and `restore`, and check the ion beam current is back where it was. `run`
answers at once with its `run_id` (`state: running`) and mills on, in control
of the microscope: call `service_wait_for_run` milling with that id, again
while it says running, before you `clear`. It ends `done` (`result:
completed`), `cancelled` or `failed` with its error. `service_get_parameter`
milling `progress` shows how far it has got. A spot burn `run` works the same
way.

Follow every mill the same way, whoever started it (you, a workflow task, the
operator): the watcher prints `MILL STAGE STARTED` for each stage and `MILL
ENDED` with each stage's outcome. Check the `current` and `depth` on each
stage's line against what the protocol or you meant to mill; if they are
wrong, stop the mill as your level allows (see *Stopping*) and tell the
operator. Collect the result with `service_wait_for_run` milling, with the
line's `run` as `run_id`, or without one for the mill going on.

**Polishing by hand.** When the operator asks you to thin a lamella outside
the task, one `milling_run_stage` per step:

- Preview every step's patterns on a fresh ion image first
  (`milling_preview_patterns`), and look at where they land.
- Re-align to the reference before every step. Below about 300 nm, if the
  alignment methods disagree, align again before milling. For polish passes,
  set the Polishing task's `alignment.full_frame_below` (80 pA, say) rather
  than re-aligning by hand.
- Measure the band, don't trust the gap. Track the pattern gap you asked for
  and the band you measure on a FIB image, and stop on whichever reaches the
  target first: on the Arctis the band ran at 0.53-0.73× the gap. Measure on
  enough pixels: once the band is under about 400 nm, image at a narrower FIB
  field of view (15 µm, not 30 µm).
- After rough milling, measure the band's tilt and rotate the polishing
  patterns to match. A 0.6° mismatch left a 55 nm wedge along the lamella.
- Check the SEM view of the face after each step as a second opinion:
  contrast changes, cracks, bending.
- Report each step: current, gap, time, the alignment residual, the band
  measured.

**Aligning to a reference.** To bring the view back onto a saved image (a
task's reference from `item_list_outputs`, or one you saved),
`alignment_measure_shift` the reference against a fresh saved image to see
how far off you are, then `alignment_align_to_reference`: beam shift for a
small offset, `stage` when it is beyond the beam shift's range. It is a long
command; `service_wait_for_run` alignment gives each step's shift and whether
the methods agree on what is left. If they don't agree, or the score stays
low, the view has changed (milled, drifted out of frame): stop and tell the
operator rather than aligning again.

**How thick is the lamella?** `item_measure_thickness` with a FIB image from
`item_list_outputs` measures the band between the trenches on the
full-resolution file, around the item's POI: thickness, wedge, the band's
angle, where it sits against the POI, and any holes. Use it instead of
measuring a preview yourself. When `resolution.under_resolved` is true the
band is under 8 px: take the next image at the suggested field width before
trusting the number.

**Correlating FM to FIB.** `correlation_correlate` fits a saved FM stack to
a saved FIB image from fiducial pairs you give (FIB pixels; FM pixels and
slices, same order, at least four) and maps the targets into the FIB image.
Refine each pick first with `correlation_refine_point`, and look at the
overlay before trusting the answer. Pass `lamella_path` so the fit starts from
earlier runs and is saved where the correlation tab finds it. A verdict of
`check` or `bad` names the fiducial to look at. The answer does not move the
lamella's POI: `item_apply_correlation` with the run's `run_path` does, as the
tab's Continue (Drive; at Assist, ask). Only move it on a `good` verdict, or
when the operator agrees.

**Overviews and focus stacks.** `beam_acquire_overview` images a grid of
tiles around where the stage is and stitches them; the stage moves to each
tile and comes back. Ask `beam_estimate_overview` first: it gives the tile
count, the scan time, the resolution the beam will scan and any tiles out of
the stage's reach, but not the stage moves, which often take longer than the
scanning, so a 3 × 3 that scans for seconds still takes minutes.
`beam_acquire_focus_stack` focuses and images a frame in strips, for a tilted
view one focus can't hold. Both save into a folder you name inside the
experiment and are long commands: `service_wait_for_run` imaging gives the
image and its path, and `service_call_command` imaging `stop`, with a reason,
ends either one. `fm_acquire_overview` is the fluorescence one, from the FM
pose with the objective in; `fm_estimate_overview` gives its whole run time,
stage moves included, and the same `stop` ends it. `register_tiles`
(experimental, off by default) places each tile where its picture says it is
rather than where the stage step put it, when no tile moves more than 15 µm.
The answer's `overview` says the resolution used, which tiles were skipped,
how far the steps were off, whether it moved the tiles (and the largest move)
and why not when it didn't; its `paths.record` file has every tile's planned
and read-back stage position.

Set an overview up as the app does. Give the tile size in `image_settings`
(`hfw`, `resolution`, `dwell_time` belong there, not at the top level of the
settings) and use 3:2 tiles (e.g. 1536 × 1024). The instrument snaps a size
it doesn't list to one it does, so a square 2048 × 2048 came back as
2048 × 1768 and the stitch failed. Keep at least 15 % overlap. A grid that
runs past the stage's travel is refused only once it starts, so keep it near
the middle of the travel, and keep it small: a 3 × 3 worked on the Arctis
where a 6 × 6 failed twice.

A point picked on an overview is only as good as the stitch. The tiles are
placed where the stage was told to go, and a stage that overshoots puts
features a few µm off (about 4 % in x and 7 % in y per step on the Arctis).
After moving to a point picked on an overview, take a fresh image and centre
on the feature itself before acting, as an operator would.

**Finding sites.** When you pick sites (cells, squares) on an overview:

- Start with what worked on a real grid: GrabCut on the holey film, tracing
  the grid bars by hand where shading defeats a threshold, and a piece of
  debris as an anchor to find the same place again. Say how sure you are of
  each detection, not only where it is.
- Look at the tile seams. A feature on a seam can be cut in two, doubled or
  hidden under the next tile; check it in the tiles on both sides before you
  count it.
- A square you haven't looked at is unassessed, never "empty" or "bad".
- Confirm shorthand before acting on it. "3:3" or "C2" can name a square, a
  cell or a rank; ask which the operator means.
- Record each site as *Write down why* says: `item_add` without `name`, and
  how to find it again in its description.

**Grids by hand.** `grid_list` is the experiment's records; `grid_get_inventory`
is the hardware now. `grid_load` brings a grid onto the stage by name. With an
autoloader that is an exchange: the loaded grid goes back to the magazine
first, so say which grid comes out before you load. It doesn't move the
stage; follow it with `grid_move_to`. `grid_unload` returns the grid;
`grid_run_inventory` scans the magazine again (minutes). All three are long:
wait with `service_wait_for_run sample_handling`. A grid run loads its own
grids, so never load ahead of one. `grid_add` records the inventory's grids
the experiment doesn't have yet. `grid_rename` renames a grid on its slot and
in the experiment together; a grid that has run keeps its name, so ask the
operator before renaming anything, and say the old and new names.

**`409 microscope_busy` is someone else working.** Its `detail.in_control`
names who (`workflow`, `script`, or another command) and what. Say who has
the microscope and what you wanted to do, then wait for it to finish or ask
the operator. Never retry it in a loop, and never try the same thing through
another tool. While a run waits on a question it lends control, so your
command can go through; when the question is answered the run takes it back.

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
reads. Never use a stop to "pause and think"; ask instead.

- **Watch**: you can't stop anything. If the run looks like it is doing
  damage (milling the wrong place, a stage move that makes no sense, a task
  repeating), tell the operator at once, plainly, and say they should press
  Stop.
- **Assist**: a mill you started yourself you may stop directly
  (`milling_stop`). Anything else (the workflow, the operator's mill) is a
  request, filed at once with what you saw as the reason.
- **Drive**: stop first when you believe the run is doing damage
  (`workflow_stop`, `milling_stop`), then say why.

## Session setup

At the start of a session:

1. Connect and read `/capabilities`. Tell the operator the level and what it
   means, and whether the microscope is served.
2. Start the watcher before anything else, so nothing that happens while you
   read is missed.
3. Read where things stand: `app_get_status`, `workflow_get_queue`,
   `prompt_get`, `review_list`. A standing question or a `holding_the_run`
   review comes first.
4. Before the first microscope action, get your bearings (above).
5. At Drive, before you start workflows yourself, turn the run summary
   dialog off with `workflow_set_options(show_run_summary=false)` so no dialog
   is left over the operator's window, and tell them you did. Each workflow's
   end still shows a toast with a Show summary link, and `workflow_get_summary`
   gives the outcome. If the operator turns it back on, leave it on.

## How you report

- Lead with what you need from the operator, if anything, then the answer,
  then the evidence.
- Use lamella names as the app shows them (`02-loved-liger`) and task names
  as the protocol has them.
- Say where each claim came from (an image, a value, a log line) and mark
  inference as inference. Say what you didn't check.
- During a live run keep each update to a line or two: what just happened and
  what is next. Summarise a long log with a short script rather than reading
  it out.
- Everything you do is attributed to you on the app's timeline; write nothing
  it would contradict.
