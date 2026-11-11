---
name: fibsem-microscope
description: Using a FIB-SEM (and its FM) through fibsem-os at any level - which task skill to read, the stage, images, autofocus and contrast, coincidence, alignment, milling, hardware errors and stopping. Read fibsem-agent first, then this before the first action on a FIB-SEM, then the task skill it names.
---

# Using a FIB-SEM with fibsem-os

This skill is what an agent needs on a FIB-SEM (and its FM, where there is
one), after fibsem-agent, whatever it was asked to do. Then read the skill
for the task:

| The operator wants to | Read |
|---|---|
| Make lamellae with AutoLamella, or explain a run | autolamella, then autolamella-lead if nobody will be at the microscope, or autolamella-assistant if the operator will |
| Polish a lamella (before the first polishing pass) | autolamella-polishing |
| Make lamellae through a fluorescent target | autolamella-correlative too, and autolamella-coincidence-milling where the FM and FIB are coincident |
| Screen grids, take grid overviews or pick sites | screen-grids |
| Do plain FIB work: images, milling patterns, spot burns, cross-sections, moving between sites | fibsem-experiment |

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
sites. Use this loop, not the coincidence commands: it worked better on a
real instrument. A coincidence error shows only as a vertical offset in the
image, never side to side.

The one exception is `alignment_check_coincidence`, experimental, and only
when the operator asks you to try it. Start from the SEM when the error may
be large. Look at its overlay, and use the `dy` only if
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
only added errors. This holds when you lead a run on your own too: don't
carry on with other lamellae or sites, stop the run (`workflow_stop`) if one
is going, and wait. Read what you can without imaging (`microscope_get_state`,
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
supply code. FM Intensity is experimental: never turn it on unless the
operator asks for it.
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

**FM targets.** Reading an FM stack for a target and correlating FM to FIB
are the autolamella-correlative skill's: read it before you pick a
fluorescent target or correlate.

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
