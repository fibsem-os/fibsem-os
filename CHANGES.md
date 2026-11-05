# Changes

## v0.6.0 (unreleased)

0.6.0 is everything on main since v0.5.3 except the agent and the gateway, which are not
part of this release. Its headlines:

- **Python 3.10 or newer**, and the old `microscope.get`/`set` scripting calls are gone
  (see Requirements and Scripting API below).
- **A web app.** The App Server (Tools → Host, off until you turn it on) serves a web app
  in the browser: the microscope, the open experiment and its files, lamellae, grids and
  Slack notifications.
- **METEOR and other offset fluorescence microscopes** join through a Device Server on
  the METEOR PC, with SEM to FM calibration and guarded moves to and from the FM.
- **Grid screening for everyone**: a Grids tab, grid tasks and a screening report, no
  longer behind a preference.
- **A record of every run** (`events.jsonl`), an experiment replay and a new HTML report.
- **A new image file format** (0.60), smaller and quicker to preview. 0.5.x opens its
  FIB/SEM images without their metadata.
- **Settings and experiments move to `~/.fibsem`**, out of the installed package.

The agent and the gateway stay in the code, and a git checkout of fibsem-os runs them
as before. An install from PyPI shows neither: there is no agent chip or Agent page,
`fibsem-agent`, `fibsem-mcp` and `fibsem-gateway` say they are not part of this release,
and a `FibsemClient` script acts as the operator. `FIBSEM_UNRELEASED=1` turns them on.

Fixes for v0.5 are cut from `release/v0.5.x`, the branch v0.5.2 and v0.5.3 were tagged
from, and are listed under their own version below; a fix that belongs in both lands on
main first and is cherry-picked there.

### Requirements

- **Python 3.10 or newer is required.** 0.5.3 is the last release that supports
  Python 3.8 and 3.9, and pip on those versions keeps installing it. Windows 7
  cannot run Python 3.10: a Windows 7 support PC can stay on 0.5.3, or fibsemOS can
  run on a Windows 10 or newer computer on the microscope network.
- **AutoScript 4.9 or later is required**, up from 4.7. With an older AutoScript the
  ThermoFisher microscope is unavailable, and the connection error names the version
  found.
- `imagecodecs` is now installed with fibsemOS, so compressed TIFFs (Zeiss ZEN exports,
  METEOR ImageJ stacks) open on a fresh install. The reports no longer need the
  `reporting` extra.

### Files and folders

- **fibsemOS keeps its settings, session state, logs, new experiments, and your
  protocols and scripts in `~/.fibsem`** (or `FIBSEM_HOME`) instead of inside the
  installed package. The first start copies or moves them there and logs what it did.
- **Images are saved in the 0.60 format.** FIB/SEM images are OME-TIFF with fibsemOS's
  metadata inside, like the FM stacks already were. Pages are zlib-compressed, and a
  large image or a z-stack carries 1/4 and 1/8 size copies so previews open quickly.
  Every image saved by an earlier version opens as before and is never rewritten.
  **fibsemOS 0.5.x opens a FIB/SEM image saved by 0.6 without its metadata** (no pixel
  size, stage position or beam settings), so update every computer that opens your
  experiments. FM stacks open fully in either version.
- **Each run of a task saves into its own folder**, `<lamella>/<task name>/<date-time>/`,
  so running a task again keeps the earlier run's images. Experiments from earlier
  versions keep their files where they are.
- Correlation runs are saved as `correlation.json` version 2. Runs saved by earlier
  versions open and are converted the next time they are saved; earlier versions cannot
  fully open runs saved by 0.6.
- An overview's record beside its tiles is now `overview-metadata.json` (was
  `tile-layout.json`).
- The Legacy protocol buttons are gone: an old AutoLamella protocol opens with Select
  Protocol and is saved as a task protocol, the original kept beside it as
  `<file>.before-v1`.
- AutoLamella refuses to open an experiment written by another application, instead of
  dropping its data on the next save. Experiments saved with `app: volume` open as
  volume milling experiments.
- Create Experiment shows the free space on the chosen drive beside the expected size of
  a 20-lamella experiment, and Load Experiment shows how much space the experiment uses.

### Scripting API: the old string-key calls are removed

Scripts and plugins that drive the microscope directly need these changes. 0.5.3 still
has every call below and does not warn about any of them; 0.6.0 removes them without a
warning release in between. The [device guide](docs/developers/devices.md#from-getset-to-devices)
lists where each key went.

- **`microscope.get("key", beam_type)` and `microscope.set("key", value, beam_type)`
  are removed.** Read and write the device parameter instead, for example
  `microscope.beams[BeamType.ION].current.get_value()` and `.set_value(v)` for
  `"current"`, `microscope.stage.position.get_value()` for `"stage_position"`, and
  `microscope.stage.home()` for `set("stage_home", True)`. The
  named methods (`get_beam_current`, `set_field_of_view`, `move_stage_absolute`,
  `get_microscope_state`, ...) are unchanged and read and write the devices.
- **`get_available_values`, `get_available_values_cached` and
  `clear_available_values_cache` are removed.** A parameter's `choices` is the same
  list: `microscope.beams[BeamType.ION].parameters["current"].choices`. Milling
  settings' choices are `microscope.milling.supported_settings()`.
- **`microscope.home()` is removed.** Call `microscope.stage.home()`, which returns
  whether the stage is homed afterwards.
- **`microscope.milling_progress_signal` and `microscope.spot_burn_progress_signal`
  are removed.** Listen on the services instead:
  `microscope.milling.progress.changed.connect(slot)` and
  `microscope.spot_burn.progress.changed.connect(slot)`. They carry the same
  `MillingProgress` and `SpotBurnProgress` reports, on the same thread.
- **`microscope.tiled_acquisition_signal` is removed.** Listen on the imaging
  service instead: `microscope.imaging.tiled_progress.connect(slot)`. It carries the
  same `TiledProgress` reports from beam and fluorescence overviews, on the same
  thread. A microscope with no beam has no imaging service, so nothing reports there.
- **`microscope.holder_changed` is removed.** Listen on the sample handling service
  instead: `microscope.sample_handling.holder_changed.connect(slot)`. Applying a
  configuration that names a holder now loads it into the same service
  (`sample_handling.load_holder()`) rather than building a new one, so subscribers
  stay connected.
- **`microscope.sem_acquisition_signal` and `microscope.fib_acquisition_signal` are
  removed.** Show an image in the app's SEM or FIB view with
  `microscope.imaging.show(image)`; its own beam type picks the view. Listen on
  `microscope.imaging.image_shown`. As before, only images sent there are shown,
  never every acquisition, and the App Server event kinds are still
  `sem_acquisition` and `fib_acquisition`.
- **`microscope.stage_position_changed` is removed.** Listen on the stage device
  instead: `microscope.stage.position.changed.connect(slot)`. It fires on any read or
  move of the stage, not only from `get_stage_position()`. The last position is
  `microscope.stage.position.last_known` (no hardware call; None before the first
  read); the private `microscope._stage_position` cache is gone.
- **A stage or manipulator position's `changed` signal ignores moves under 1e-6** on
  every axis (metres and radians), compared with the position it last signalled, so
  readback jitter no longer signals a move. The overviews, the stage readout, the
  holder calibration dialog and the App Server feed now follow
  `microscope.stage.position.changed`, so a move made through the stage device is
  seen at once. The App Server event kind is still `stage_position_changed`.
- **`microscope.move_to_microscope(target)` is removed.** Call
  `microscope.move_to_device(target)`; pass `orientation="SEM"` to land at SEM when
  returning to the beams, as the old method did on a compustage.
- **The `fibsem.microscopes.*` driver module names are removed.** Import from
  `fibsem.drivers`: `fibsem.drivers.autoscript.microscope`,
  `fibsem.drivers.tescan.microscope`, `fibsem.drivers.odemis.microscope`
  (`OdemisTescanMicroscope`), `fibsem.drivers.demo.microscope`,
  `fibsem.drivers.demo.simulator`, `fibsem.drivers.demo.sim_scene` and
  `fibsem.drivers.registry`.
- A backend no longer implements `_get`/`_set`; its devices are the whole contract.
  The key tables (`fibsem.devices.BEAM_ROUTES`, `STAGE_ROUTES`, `CHAMBER_ROUTES`,
  `MANIPULATOR_ROUTES` and their command tables) are gone with them.
- **`AutoLamellaTaskStatus` is now `TaskStatus`, in `fibsem.workflows.status`.** The
  values are unchanged, and saved experiments load as before. A plugin task that
  imports the old name from `fibsem.applications.autolamella.structures` imports
  `from fibsem.workflows.status import TaskStatus` instead.
- **`microscope.move_to_milling_angle` takes the angle and the rotation in degrees**,
  like the other milling-angle methods, not radians.
- **`microscope.last_image` and the beam's `last_image` command are removed.** Acquire a
  new image with `microscope.acquire_image` or `beam.acquire`.
- **`fibsem.manufacturers`, `config.DEFAULT_MANUFACTURER` and
  `config.get_default_user_config` are removed.** Driver names come from
  `fibsem.drivers.registry` (`normalize_name`), and `microscope.manufacturer` reports
  each driver's own name (Demo reports "Demo"). Import `ThermoMicroscope` from
  `fibsem.drivers.autoscript.microscope`, not `fibsem.microscope`.
- **`fibsem.calibration` is removed.** The EasyLift needle calibration runs as
  `python -m fibsem.drivers.autoscript.manipulator_calibration`, and
  `insert_manipulator()` with no name inserts to the instrument's own insert position
  (PARK on ThermoFisher and Demo). `run_multi_position_autofocus` is removed from
  `fibsem.fm.calibration`.
- **Plugin entry points moved.** Milling patterns and strategies register under
  `fibsem.milling.patterns` and `fibsem.milling.strategies` (were `fibsem.patterns` and
  `fibsem.strategies`), AutoLamella tasks under `fibsem.autolamella.tasks` (was
  `fibsem.tasks`), and grid tasks under `fibsem.grid.tasks`. Grid task plugins import
  from `fibsem.workflows.grid`, and `Attention` from `fibsem.workflows.attention`.
- `FibsemClient.wait_for_run` raises the run's own exception
  (`OperationCancelledError` for a stopped run, `DeviceFault` for a hardware refusal)
  instead of `RuntimeError`.
- A new image-based brightness and contrast routine,
  `microscope.imaging.auto_brightness_contrast`, limits how many images it takes and
  raises rather than leave a mostly black or white image.
- A script or headless run on a METEOR no longer hangs at exit after moving the
  objective or changing the filter. The example scripts and demos run again.
- The two FM notebooks (`fibsem/fm/fm-notebook.ipynb`, `fm-example-notebook.ipynb`) are
  removed.
- **For task and plugin authors:** a task type lists the questions it asks in `asks`
  (was `questions`) and asks them with `ask`; every decision lands through
  `Experiment.decide`. Milling stages can list monitors (plugins that run during each
  mill), and a strategy can return a `MillingStrategyResult` for the record. Dataclass
  fields declare their metadata as device parameters do (an SI unit, SI limits,
  choices, a doc); plugins written with the v0.5 field metadata still load, and log
  what replaced it.

### Web app and App Server

The App Server lets a browser on this PC, or one you pair, see and run the microscope.
It is off by default: turn it on, and choose whether it serves the microscope and its
devices, in Tools → Host. Both choices are remembered.

- **The web app** is at `/ui/` on the App Server. Its pages:
  - **Microscope**: connect and disconnect the microscope (refused while a run is on),
    and the latest SEM, FIB and FM images with a scalebar and the same metadata bar as
    the window.
  - **Experiment**: what the open experiment is and a file explorer for its files.
    TIFF images open in a full image viewer.
  - **Lamellae** and **Grids**: each lamella's History and each grid's overview,
    read-only. Grid overviews open in the image viewer.
  - Volume experiments show their positions with a slice scrubber.
  - **Settings**, including Slack notifications (below).
- A browser pairs with a code shown in the window. Decisions made from the web app are
  recorded as the operator's.
- `FibsemClient` and the App Server's routes can insert the FM objective at a safe tilt,
  plan a lamella run or a queue addition with its time estimate, save and return to
  named stage positions, set and move to Volume positions, and export an image with its
  scalebar or a lamella's animation as the window does.
- The microscope routes are under `/microscope` (`/microscope/state`,
  `/microscope/devices`, `/microscope/services`); read-only queries are at
  `/services/{service}/queries/{query}`. A refused move or bad value answers 4xx with
  `error_type` and `message`, a cancelled command 409, a bad argument 422 naming the
  parameter, and a key the command does not read is refused naming where it belongs.
- `fibsem-autolamella-server` and `fibsem-volume` with `--config` connect to that
  configuration's microscope instead of the Demo one.

### Slack notifications

- **The instrument can post to a Slack channel**: what needs an answer, what failed and
  when a run finishes. Set it up in the web app's Settings → Notifications, or with
  `fibsem-cli notify setup '#channel'`, which sends a test.
- **Store the Slack token once** with `fibsem-cli key set FIBSEM_SLACK_TOKEN`. It is kept
  in Windows Credential Manager or the macOS Keychain instead of an environment
  variable.

### METEOR and offset fluorescence microscopes

- **A Device Server on the METEOR PC serves its FM.** Name it as an `fm` entry in
  `~/.fibsem/device-server.yaml` and start the server with `--config`. Pair the
  microscope PC with it once with `fibsem-cli pair-device <address>:<port> <code>`;
  remote entries no longer take `code:` or `token:`.
- **SEM to FM calibration.** Microscope Configuration → Calibration has "Calibrate SEM
  to FM…", and the Movement tab has "Move to FM", off until the fitted holder is
  calibrated. The way back to the beams is always allowed, and a move that travels
  between the beams and the FM asks first, listing the route.
- **Moves at the FM are guarded.** A stage move that would leave the FM, or rotate there,
  with the objective inserted is refused: retract it first or use move to device.
- Moving to the SEM, FIB or milling orientation by name keeps the sample under the
  beam, rather than turning the stage and leaving it off-centre.
- Lamellae marked at the beams get a fluorescence pose, and lamellae can be marked from
  the FM view. Update position keeps a lamella's stored FM objective position.
- Light power in watts matches the selected source, and a relative focus move stops at
  the objective's user-defined limit.
- On Odemis systems, drawing the fiducial pattern and any move or setting computed with
  numpy no longer fail with "No module named 'numpy._core'" or hang.
- The microscope configuration can turn the FM off with `fm.enabled: false` on every
  system, including compustage and Odemis systems.

### Devices

- **Arctis**: the gas injection system (GIS port or multichem) is the `gis` device, with
  its needle, heater, valve and temperature, and the sputter coater is the
  `sputter_coater` device. The autoloader's working slot can be calibrated from the
  Sample view with the pencil, and is kept across connects, so grid overviews centre on
  the loaded grid.
- **Hydra Bio**: the GIS and the MicroSputter coater are the `gis` and `sputter_coater`
  devices; a sputter run names the grids to coat.
- **Thermo cryo systems**: the chamber and sample loader report the touch alarm,
  nitrogen flow, temperatures and dewar levels (read-only). The electron beam's source
  tilt and lens alignment can be read and set as advanced beam settings.
- **Tescan**: Link stage leaves the stage as it is instead of failing, the manipulator
  panel offers the Parking preset, and stage moves (also on Odemis) return the position
  they ended at.
- A manipulator driver can give per-axis limits, and the needle refuses moves outside
  them. A manipulator position no longer has to name a coordinate system.
- A driver can ask for its vendor library's folder in the guided setup and in
  Preferences, kept per computer.
- A stage move outside the stage limits is refused with an error instead of being sent
  to the microscope. A rotation of 359.9° counts as the same place as 0.1°.
- On a stage with a built-in shuttle, its slot is "Shuttle" and its holder "Built-in
  holder" (a slot calibrated under the old name is kept).
- Turn the microscope's devices on or off for the session in Tools → Host; Preferences
  sets what a new session starts with. Disabling a beam in an applied configuration
  now takes effect.

### Workflow

- **What a task asks is on the record.** Each question a task asks while it runs (a
  position, a point of interest, an alignment area, a detection, a mill's result) stays
  in the experiment record with who decided it: your answer when supervised; nobody
  when automated, in which case the value is used as proposed and recorded as
  unreviewed. The Review tab is always on, and the "Review workflow (early access)"
  preference is gone.
- **One `attention` per task in the protocol**: `automated` or `supervised`, in place
  of the `supervise` and `review` flags. Protocols saved with the flags still load and
  are written back in the new form. A supervised question always waits for an answer.
- **The milling question reads Run Milling / Skip, then Mill Again / Accept**, and the
  spot burn question works the same way, with a fresh FIB image after each burn. Skip
  records the task as Skipped, so later tasks don't run on a lamella that was never
  milled.
- **Setup's confirmations each have a switch.** The tilt to the milling angle and the
  move to the milling position are asked as before, and each answer is recorded with
  the stage position. `confirm_tilt` and `confirm_position` (both on) turn either
  prompt off. Setup and Mill Fiducial ask you to check the alignment area, with a
  Confirm Alignment Area setting; the fluorescence tasks' position and autofocus
  confirmations and Mill Undercut's four detections are recorded the same way.
- A question the app closed on is withdrawn when the experiment loads, instead of
  reappearing as a decision to make.
- On the Workflow tab, a task can't be dragged above a task it requires. Run again no
  longer queues a task twice. Mill Fiducial requires Setup Lamella Position in the
  default protocol.
- Global Edit with "Also update existing lamella configurations" changes only each
  lamella's reference imaging and milling field of view, not its own pattern depths,
  currents and task settings.
- With no lamella card selected, the Lamella tab's Protocol page is empty instead of
  showing (and editing) the last lamella.
- A lamella card's menu has **Export GIF**: the images its tasks took as an animated GIF
  or WebP, one labelled frame per task, or its fluorescence z-stacks by channel.
- A workflow that fails before its first task, or dies with an error, now says it
  failed. A save that fails during a run (a read-only or locked file, a full disk) fails
  the run instead of closing the app, and a task whose result can't be saved is
  recorded as failed.
- "Turn beams off after completion" applies to headless runs too. Stopping a run from
  a script stops the mill at once, as the window's Stop does.
- A finished run shows one toast with a Show summary link; the run summary no longer
  blocks the window and can be turned off ("Show summary after completion").
- Decisions and verdicts are saved with an actor (`operator:`, `task:`) instead of an
  author; experiments saved by older versions still load.

### Grids

- **The grid workflow is available to everyone**: the Grids tab (a card per grid with
  its overviews and history), the Workflow tab's Grids view, the Protocol tab's Grid
  page and the grid screening report no longer need a preference.
- New experiments' default protocol includes grid screening: an SEM overview (3 × 3 at
  500 µm) and a FIB overview (5 × 5 at 250 µm), sized to stay within the Arctis stage's
  reach.
- A grid overview larger than the stage can reach acquires the tiles in reach and names
  the ones it skipped, and the Grid page warns about it beforehand.
- A grid can be marked Rework or Failed as well as Poor, a verdict records who set it,
  and a grid's card has Edit note; the note shows in the screening report.
- **The grid screening report** opens as a page in the browser, with a PDF beside it
  where Edge or Chrome is installed.
- A grid can't be renamed once a task has run on it, and naming a grid on the Sample
  view renames the experiment's grid instead of adding a second one.
- Starting a grid run clears the ticked grids, so Add to Queue doesn't queue them twice;
  a grid task removed during a run is skipped on the grids still to run.
- The sample holder and magazine controls are a Sample tab under Microscope. The Arctis
  simulator's autoloader takes five seconds per load and each grid shows its own
  content.

### Overview tab and correlation

- **An image from another microscope can be laid over the Overview tab**: a fluorescence
  image, or any TIFF, PNG or JPEG. An import dialog confirms its axes, pixel size,
  channels and mirroring; place it from its metadata, by dragging and turning, by
  clicking three matching points, or by resizing a corner. It is saved with the grid,
  and each channel can be recoloured, hidden or re-contrasted. Large images open and
  respond quickly.
- The grid bars can be dragged and turned into place with Align bars, are drawn
  foreshortened in a tilted view, and are saved with the grid. These controls are under
  a new Align button on the canvas toolbar.
- After each overview the tiles are checked against each other, with a warning when
  they don't match the stage moves; an experimental "Align tiles" option places tiles
  where they match their neighbours. Tiles are laid out at the resolution the
  instrument actually scans, and the estimate names tiles out of the stage's reach.
- The stage limits box, grid boundary and holder slots are drawn at the stage's height,
  so a click inside the box is no longer refused in a tilted view. Opening the tab and
  "reset view" frame the planned tiles and the stage, and "Centre on stage" works.
- The overview pre-flight dialog says how far the run moves in z.
- Correlation fits no longer use random restarts: the same points give the same answer.
  Fit from points says when the points fit much better mirrored, and reopens with the
  points last picked. Selection between the point list and the image stays in step.
- Fitting a reflection hole works on the lower planes of a z-stack and near the image
  edges. The refractive-index correction works offline.

### Milling

- **Milling monitors** watch a stage while it mills and can pause it, report or end it.
  Two are built in: FM Intensity (ends or flags the stage on a fluorescence drop) and
  SEM Peek (an SEM image every N seconds). The History tab shows what each one said.
- Alignment can use the whole frame for stages at or below a chosen current ("Full
  Frame At Or Below", off by default). Alignment scores now say how good each match
  was.
- Prepare and the time estimate say which ion current the patterns mill at, and warn
  when the recipe asks for one the instrument doesn't list. A pattern the instrument
  can't draw (a bitmap on Tescan) stops with an error before anything is drawn.
- The record names the strategy that milled each stage, how each stage ended and why a
  coincidence mill stopped.
- The trench bitmap's image paths can be chosen in the form; the coincidence strategy's
  counts are whole numbers; the overtilt strategy offers the beam's own resolutions.
- Fixes a stage that could hang after a failed pause, and the Arctis FM live view ending
  during FM-monitored milling. Double-clicking a beam view during milling is refused in
  AutoLamella, as in the standalone UI.

### Fluorescence

- A channel's emission is its emission filter, shown by name with its bands drawn; the
  Arctis and iFLM filter shows as "Quad-band". Saved channels and images load as before.
- OME-TIFFs from other software, ImageJ hyperstacks and RGB TIFFs open, and stacks with
  as many channels as z-slices load the right way round.
- Interpolating an FM volume along z is over ten times faster.
- The fluorescence autofocus records how it ended and where it saved, and the History
  tab shows it.
- Acquire Fluorescence Image with no channels configured fails up front saying so.
- FM settings are kept in the session state; the Development menu's Load and Save
  Fluorescence Configuration items are removed.

### The experiment's record and reports

- **Each experiment records `events.jsonl`** beside `logfile.log`: task starts and ends,
  each question and decision, stage moves and beam shifts, images and FM stacks with
  their files, milling stages and their results, alignments, autofocus, spot burns,
  grid loads, every edit to a lamella's plan with its value before and after, and who
  did each (the operator or the task). Runs from scripts record it too.
- **The experiment replay** reads it: stage moves, alignment, milling and FM activity,
  each question and its decision, each plan edit, and who took each action.
- **Tools → Reporting → Generate Report** writes an HTML report from the record, and its
  PDF where Edge, Chrome or Chromium is installed: where the session's time went, a card
  per lamella with its final images, runs and decisions, and what is worth a look. The
  older report built from the log file is gone.
- **The History tab** shows each milling run and how its stages ended, drift correction
  under each mill, autofocus and alignment done outside a task, and folds a run's steps
  into one line when they all completed. It can show newest first.
- The event log opens on the experiment's whole record, so an overnight run can be read
  the next morning. The dashboard shows when a run stopped, stalled or failed.
- The fibsem revision recorded in experiments and shown in About is the commit alone.

### The app

- AutoLamella and the fibsemOS UI show a splash screen while they start, and have their
  own icons in the window, the taskbar and desktop shortcuts. Tools → Create Desktop
  Shortcut offers both; shortcuts from a venv or uv environment start without an
  activated terminal.
- A chip in the tab bar says when a newer release is available; turn the check on at
  start-up in Preferences → Display.
- Preferences and the View menu have "Toasts for Problems Only".
- The Imaging defaults offer the resolutions the instrument reports.
- The Overview tab's Lamella Positions list comes back after a reconnect, the milling
  settings form no longer comes up empty on an unrecognised manufacturer, and figures no
  longer come out in the wrong colours.

### Image export

- **File › Export Image saves an image with its scalebar and a bar of its acquisition
  details**, to a PNG or TIFF or the clipboard, for a slide, a message or a figure. It
  opens on the experiment's folder and takes an SEM/FIB image or a fluorescence stack.
  The bar shows the detector or objective and up to five values (HFW, pixel size,
  voltage, current, working distance, dwell time, z-stack); a second row, off by
  default, says where the image came from: experiment, lamella or grid, task, date,
  instrument, user, version. A value the file does not record is left out rather than
  guessed. SEM/FIB images have contrast and gamma; a fluorescence stack shows a channel
  legend, and any channel can be left out. The export is drawn from the image file, not
  from the canvas, so canvas overlays (milling patterns, points, the ruler) are not in
  it.

### Microscope tab

- **The quad view's fourth cell has pages**, in place of the empty "No Data" panel. The
  arrows in its title step from one page to the next.
- **Chamber: a drawing of the stage and the beams.** It is drawn from the stage position
  the Microscope tab already reads and from the configuration, not from a camera, so it
  works on every microscope. A side view shows the stage tilt and the two columns, turned
  round when the stage is a half turn from its reference. A map of the stage from above
  shows the travel limits, the calibrated holder slots with their grid boundaries, the
  holder as a square round them, a separate station such as an offset fluorescence
  microscope, the lamellae and the stage. One fills the cell and the other sits in the
  corner; click the corner to swap them. The map has three zoom steps (travel, holder,
  grid), on the − and + buttons or the wheel. At holder zoom it shows the whole holder
  while the stage is on it, and anything off the map gets an arrow at its edge. The
  holder's square is drawn from the slots, not measured: a holder file has no outline.
- **Overview: the newest overviews of the grid the stage is on**, with that grid's
  lamellae, the stage and the grid's edge marked. It follows the stage from grid to grid,
  and the grid selector holds it on one. Chips switch between the grid's views. Overviews
  from grid tasks and from the Overview tab both appear, as soon as they are saved; one
  from the Overview tab is placed on the grid it was taken over, or listed under "Not on
  a grid" when no grid can be found for it. Clicking a lamella selects it. Nothing on the
  page moves the stage. Overviews from the fluorescence Overview tab are not shown yet.

### fibsem-cli

- **`fibsem-cli` acts through the App Server, as the operator.** It no longer connects
  to the microscope itself: it reaches the App Server running on this machine (or
  `--url`) with the operator token, so its moves and writes take the same routes and
  checks as the window, and go on the experiment's record. `--manufacturer`,
  `--ip-address` and `--config` are gone from it (the headless servers keep them).
- Commands are in groups: `microscope get-info` and `get-state`; `stage get-position`,
  `move-absolute`, `move-relative`, `move-to-orientation`, `move-to-device`,
  `move-to-milling-angle`; `beam acquire-image`, `auto-focus`, `auto-contrast`, `blank`,
  `unblank`, `set-parameters`, `turn-on`, `turn-off`; `hardware-check`;
  `experiment list-all`; `device` and `service` with `list`, `describe`,
  `get-parameter`, `set-parameter`, `call-command`, `call-query`. Beams are chosen with
  `--beam-type ELECTRON|ION`, stage axes with `--x --y --z --r --t`.
- `stage move-absolute` is the safe move, and crosses to another device by the
  traverse. `beam set-parameters` refuses a read-only or invalid value and says when a
  value was clipped to its limits.
- Every length is in µm and dwell times in µs (`acquire --hfw` was in metres). `beam
  acquire-image --out DIR` saves the images on this machine.
- `device`/`service call-command` waits for a long command; Ctrl-C stops it.
- `experiment list-all` lists the app's experiments folder, filtered by `--instrument`,
  `--operator`, `--days`, `--since` and `--until`. `--root` and `--recent` are gone.
- `hardware-check` checks the devices the App Server serves, including every
  fluorescence setting; it never moves the objective or the stage.
- New commands that need no App Server: `pair-device` (pair with a Device Server),
  `notify` (Slack notifications) and `key` (store a key or token in this PC's
  credential store).

### Changes to what v0.5.2 shipped

- **Grid boundaries and holder slot markers are drawn when the holder has a calibrated
  slot**, rather than never. v0.5.2 turned them off for everyone, which also took them
  from an Arctis, whose working slot is always calibrated. Either way they toggle under
  the overlays button on the canvas.
- **The "Report an Issue" bug reporter and user scripts are on for everyone**, and their
  flags are gone. In v0.5.2 both were opt-in. A user script still has the application's
  access to the microscope and none of its checks, so read one before running it.

## v0.5.3 (30/09/2026)

A patch release, cut from the v0.5.2 release branch: fixes, and one addition to
installation. Nothing here changes how a workflow is set up or run.

### Installation

- **Tools → Create Desktop Shortcut...** makes a shortcut that starts AutoLamella from
  the environment it is installed in, in a folder you choose (the Desktop is offered): a
  `.lnk` on Windows, a `.desktop` file on Linux and a `.command` file on macOS. It asks
  before replacing an existing shortcut. The script-and-shortcut steps in the install
  guide remain for anyone who prefers them.

### Bug fixes

- **An overview or a lamella could be sent back to the height and tilt the Overview tab
  was opened at.** Setting coincidence, or tilting to a different milling angle, after the
  tab had been opened was not carried into a dragged tile grid or into a position marked
  by right-clicking the overview: the run, or the lamella's first task, drove the stage
  back to the old height or tilt. Reported as an overview and its lamellae landing at
  Z 31.88 mm after coincidence had been set at 32.37 mm, and as an ion tileset acquired at
  the tilt the application was launched at rather than the milling angle the stage had
  been moved to. The dragged grid and marked positions now take their height and pose
  from the stage at the moment they are used, and Acquire reads the stage once before
  planning, so a move made in the microscope vendor's software is seen as well.
- **A dragged tile grid is dropped when the view it was dragged in is left** — a change of
  beam or of stage orientation — and the plan returns to the stage position, rather than
  being re-read in a view it was never placed in.
- **"Centre on stage" works on the FIB/SEM Overview tab.** The button was never enabled
  there, so a dragged grid could not be returned to the stage without restarting.

### Known issues

- Ion overviews acquired at different stage heights or milling tilts are drawn on one
  canvas plane, so an ion overview taken *before* a coincidence correction or a tilt change
  appears offset from one taken after it. Acquisitions go to the right place regardless.
  Setting coincidence before the first ion overview avoids it. Electron overviews are not
  affected.

## v0.5.2 (16/09/2026)

A running workflow became something you can edit, the image display moved onto a
purpose-built canvas with overview acquisition rebuilt around it, and a large batch of
correctness work landed in correlation and in what the experiment record remembers.
360 commits since v0.5.1.

### Highlights

- **Edit the queue while it runs** — reorder, remove, re-run and "run next" from the
  Workflow Timeline, and add work to a run already in progress. Stop Task abandons the
  task now running without ending the workflow.
- **Scheduled tasks are on for everyone.** A task can wait until a scheduled time before
  running. This shipped behind a flag in v0.5.1; the flag is gone, and a protocol
  carrying a future `scheduled_at` now actually waits where it was previously ignored.
- **The image displays were rebuilt.** The Lamella Editor, Microscope, Overview, FM Image
  Viewer and both correlation canvases render on a purpose-built canvas instead of napari.
  Contrast is built into the canvas, and zoom and pan persist across acquisitions rather
  than resetting on every frame. The old napari overview is no longer shown by default
  and is removed in the next release.
- **One Overview tab** on a real-space canvas, with the imaging modality chosen on the
  canvas chrome instead of by switching tabs.
- **Overviews can be sparse** — a tile grid carries a per-tile enable mask, so a run
  acquires only the tiles that were asked for. For fluorescence the tiles can be chosen
  by drawing regions on a beam overview and letting them project onto the FM grid.
- **First-run guided setup writes your microscope configuration.** On a fresh install a
  wizard asks which computer this is, which instrument, and the few things no shipped
  file can imply, then writes one configuration and registers it. It starts from a
  shipped configuration for that model, so everything it does not ask about keeps the
  value the project already ships.
- **Tescan support caught up.** Stage movement geometry corrected against the instrument
  and hardware-verified, native spot burn, preset changes that keep your scan rotation
  and field of view, and milling time estimates from the dose model.
- **Every task failure and cancellation is now recorded** in the experiment. Previously
  only successes reached `task_history`, so a failed run left no trace outside the log.
- **Correlation correctness** — a family of bugs where the correlation quietly gave a
  wrong answer or quietly lost points.

### Workflow

- **The running queue can be edited while it runs.** Reorder, remove, re-run and
  "run next" from the Workflow Timeline's row menu, and add work from its header.
  Edits are anchored to the piece of work rather than to its position, so they do
  what you meant while the queue moves underneath them.
- **Stop Task** abandons the task now running and continues with the rest of the
  queue, as distinct from Stop Workflow, which ends the run.
- Removed and cancelled tasks stay in the experiment record and appear in the run
  summary rather than being deleted.
- **Task lifecycle hooks.** A run announces what it is doing — task started,
  completed, failed, cancelled, skipped; item and experiment completion — and a hook
  can write to the log, raise a toast, POST to a webhook, or post to Slack. Hooks
  subscribe to groups (`any_failure`, `any_terminal`, …) rather than event lists.
  **There is no interface for them yet**: the configuration dialog is built but is not
  reachable from any menu, so hooks are configured by editing the `hooks:` section of the
  user preferences file by hand.
- **Duration estimates**, measured from per-operation costs, shown ahead of a run and
  on the workflow timeline, including what adding to the queue costs.
- **First-run guided setup**, offered where a first-time user will meet it rather than
  buried in a menu.
- A task's dependency gets its own column; the experiment completion predicate was
  corrected.

### Overview acquisition

- **One Overview tab**, with the imaging modality chosen on the canvas chrome. The
  FIB/SEM and fluorescence overviews were two top-level tabs driving the same stage
  through two widgets sharing fifty-five method names; every fix had to be made twice,
  and twice it was not.
- **Overviews are drawn on a real-space canvas** — tiles placed where they were
  acquired rather than stitched first — with click-to-move and drag-to-aim stage
  control, a per-overview record, and saved overviews loadable back onto it.
- **Any overview can be sparse.** A tile grid carries a per-tile enable mask, toggled by
  clicking a tile on the canvas or edited in the mask beside the grid settings, so a run
  acquires only the tiles that were asked for.
- **Sparse FM overview selection.** Draw regions on a beam overview and the FM tiles that
  selection resolves to are previewed beside it, then acquired. Selecting this way is
  specific to fluorescence: the regions are drawn on the beam view and projected onto the
  FM grid. Opened from the Overview tab.
- Tiles the stage cannot reach are flagged while the grid is dragged, and an unreachable
  grid is refused in the pre-flight dialog.
- A marked position is boxed with the field of view it stands for.
- **Grid boundaries and holder slot markers are no longer drawn by default.** Both
  describe a cryo sample holder, so on a system without one they drew a holder that was
  not there. Turn them back on under the overlays button on the canvas. Stage travel
  limits are unchanged and still shown — those are a property of the stage itself.
- Cancelled overviews no longer report "Done"; one overview can no longer drive the
  stage while the other acquires; stitching and saving now say so.
- A tab-page key no longer shadows the imaging modality carried on the progress payload,
  which had let one tab's updates be read as the other's.

### Image display

The viewers are moving off napari onto a purpose-built canvas. This release is partway
through that migration; napari is still a dependency.

- The **Lamella Editor**, **Microscope**, **Overview** and **FM Image Viewer** surfaces
  and both correlation canvases now render on the new canvas. The segmentation,
  model-training and FM-import widgets do not.
- On a migrated surface: contrast is built into the canvas, modified-scroll no longer
  zooms, and zoom/pan persist across image updates at the same resolution.
- Images are reduced for display by averaging rather than sampling.
- The standalone application's minimap moved onto the Overview canvas too, so the two
  applications no longer show different overview implementations.
- **The old napari overview is off by default, and is removed in the next release.**
  The Overview tab replaces it. If you still need the old one for something, turn it back
  on under **Edit → Preferences… → Features**; this is the last release in which that is
  possible, so report anything the Overview tab cannot yet do while there is still
  something to fall back to.

### Fluorescence

- The FM Overview tab, its overviews landing on disk, and lamella positions markable and
  savable from it.
- The shared imaging channel is held across view-dependent operations, beam grabs,
  autofocus sweeps and z-stacks — the FM and the beams share one active view.
- The objective moved off the GUI thread and is guarded from two hands; objective
  displays and the FM overview no longer poll hardware on UI events.
- Z-stack acquisition order (channel-wise vs z-level-wise) is covered and honoured.
- A second FM acquisition is refused, and the overview is gated on stage pose.

### Plugins and scripting

- **`fibsem-cli plugins`** and **Tools → Plugins…** show what resolved: where each
  extension came from, what failed to load and why, what was shadowed, and which install
  this is. Group headers are the literal entry point strings, because a mistyped group
  is otherwise invisible.
- User scripts can be run from the GUI (behind a flag — a script has the application's
  access to the microscope and none of its checks).
- A scripting guide for working with experiment data.
- The plugin entry point contract is covered by tests against an installed fixture.

### Performance

- Lamella cards no longer decode a full-resolution PNG each (FIB-681).
- The experiment is written once when editing settles, not once per field.
- The tile grid is blitted while dragged; an FM overview draws only the part on screen,
  at the screen's size.
- Images are filtered with OpenCV rather than SciPy.

### Packaging and development

- A PyPI publish workflow guarded by a tag/version check, and a documented release
  process — see `RELEASE.md`.
- The running git revision is reported in the version string.
- CI runs on Windows as well as Linux, across Python 3.8–3.13.
- **`tests/ui/` now actually runs in CI.** Every file there opens with
  `pytest.importorskip("PyQt5")`, and the matrix job installs `.[test]` without it, so all
  103 files skipped silently — a green matrix said nothing about that directory. A separate
  job installs `.[test,ui]` and fails loudly if the extra did not resolve.
- Progress signals for tiled acquisition, fluorescence acquisition and spot burn carry a
  typed payload rather than a bare dict, so a consumer reading a key that is not there
  fails at the boundary instead of deep inside a handler.
- The UI tests destroy the top-level widgets they build, instead of leaking them across
  files.
- ruff runs on every pull request, formatting the files a PR touches rather than the
  whole tree.
- `matplotlib_scalebar` became a core dependency; the core stays importable without the
  `ui` extra, guarded repo-wide; `lxml` is capped on Python 3.8, which lost its Windows
  wheel.

### Correlation

Mostly correctness — a family of bugs where the correlation quietly gave a wrong
answer or quietly lost points.

- Swapping the image after a run no longer commits a point of interest at the old
  image's scale.
- Autosave could fail silently for a whole session; fixed.
- Editing a point after a run no longer leaves the stale result on disk, and loading a
  saved state with no result no longer resurrects the previous one.
- A run folder is only created once there is a result to put in it.
- **`correlation_data.json` and `correlation_result.json` are now one file**, so points
  and the result they produced cannot drift apart.
- Seed a new correlation from the lamella's previous run, with a history picker.
- Per-point fit confirmation with rebuilt diagnostic figures; refractive-index
  parameters read from FM metadata on load; fit and RI settings persist on the protocol.
- Lamella setup happens inside the correlation window, with the canvas as its preview.
- Shift+scroll steps through Z on a discrete mouse wheel, which previously did nothing.
- Seven further interface states that misrepresented the data were corrected.

### The experiment record

- **Every task failure and cancellation is now recorded.** `task_history` was appended
  only on the success path, and the live task state that held the outcome is overwritten
  by the lamella's next task — so failures vanished from `experiment.yaml` entirely.
- The acquiring user is recorded correctly on every platform (it was the literal string
  `username` on Linux and macOS).
- Image metadata records the experiment **ID**, not its name, and registration happens
  wherever a run starts rather than only when the GUI adopts an experiment — so scripted
  and headless runs are no longer anonymous.
- Tasks record the files they produced; images record the file they were written to or
  read from, and which item and task they were acquired for.
- The report's "Per-Lamella Workflow Images" checkbox now works — it was read under one
  name and written under another, so the default always won.
- Sessions that worked on an experiment are recorded (instrument, operator, software),
  and experiments can be enumerated by instrument and time window.

### Milling

- Milling time is estimated over the **enabled** stages only.
- The imaging voltage a task captured is restored, not the config default; overtilt
  always restores the stage pose, even on cancel.
- Milling controls are locked while a mill is running.
- Polygon patterns draw the right way up, and pattern plotting goes through one shared
  image-to-microscope conversion in both directions.
- The post-task FIB refresh is opt-out.

### Microscope support

- **Tescan stage movement was corrected against the instrument.** Ion-view stable moves
  overshot by about 1.65× at the milling pose: the y-axis is mounted on the tilt module
  and travels along the tilted plate while z stays chamber-vertical, so the axes are
  non-orthogonal at tilt. At zero tilt the corrected decomposition reduces exactly to
  the old one, which is why nothing at zero tilt showed it. Hardware-verified.
- Coincidence correction from the SEM view, and milling time estimates from the dose
  model on preset-driven backends.
- **One canonical spelling per manufacturer**, normalised where values enter, including
  on read of older experiments. Comparisons previously coped locally or silently took
  the wrong branch on real hardware.
- Vertical moves maintain SEM working distance and only reset FIB to eucentric for large
  moves; SEM-view vertical moves are refused on backends that cannot do them.
- The autofocus sweep is skipped on beams whose working distance cannot be set, and a
  declined sweep no longer crashes on `None`.
- **Spot burn works on Tescan.** The default implementation blanks the beam, parks it and
  unblanks it, which Tescan cannot do — `FIB.Scan` is a strict subset of `SEM.Scan` and no
  FIB blanker exists anywhere in the SDK. It now runs through a polymorphic
  `run_spot_burn` seam, and the Tescan side uses DrawBeam natively: every point goes into
  one layer as a timed exposure. The beam conditions come from a fixed preset, so a
  requested milling current is ignored and logged rather than silently applied.
- **Preset changes preserve scan rotation, field of view and beam shift** on Tescan, and
  wait for the beam to settle. Configurable if you would rather the preset's own stored
  values win. The default FIB imaging current after milling drops from 150 pA to 10 pA.

### Behaviour changes

- **A task protocol carrying a future `scheduled_at` now actually waits.** It was
  ignored while the scheduling flag was off, and the task editor preserved one rather
  than clearing it, so a schedule could outlive any control able to see it. A time in
  the past was, and remains, a no-op.
- **Acquisition-time gamma correction was removed**, and `autogamma` is no longer
  recorded as an acquisition setting — it is post-processing applied after the image is
  taken.
- The embedded `SystemSettings` in image metadata is replaced by one compact hardware
  geometry record.
- The lamella-position live view overlay is gone: the image and the stage position
  update independently, so the crosshairs were always one stage move behind.
- The deprecated streamlit review app was removed.
- Several development feature flags were removed and their features turned on for
  everyone: editing the running queue, the FM Overview tab, scheduled tasks, guided
  setup, and sparse FM selection. A preferences file carrying a removed key still loads;
  unknown keys are skipped.
- `overview_canvas_tab` was replaced by `napari_overview_tab`, which is **off** by
  default: the canvas Overview tab now ships to everyone and holds both modalities, and
  the old napari overview is opt-in until it is removed in the next release. A preferences
  file carrying the old key is unaffected — unknown keys are skipped.

### Known issues

- **The METEOR / Odemis driver work is not yet validated on real hardware.** It is
  covered by tests against a simulated backend only; treat the first METEOR session on
  this release as a commissioning run.
- **FIB autofocus on Tescan reports success without doing anything.** The
  working-distance writes it depends on are no-ops on that backend. Do not rely on it.

## v0.5.1 (21/07/2026)

Fluorescence microscope support, coincidence milling, and a large correlation batch.
111 commits since v0.5.0.

### Fluorescence

- **Fluorescence microscope support for the ThermoFisher Arctis** (#90): channel
  configuration, acquisition, z-stacks, and an objective control with a crash guard,
  threaded insert/retract and move-to-position.
- **Coincidence milling with fluorescence guidance** (#95) — milling and FM acquisition
  at the same time, with histogram controls on the viewer.
- The default FM orientation when adding a lamella is configurable, and
  `default_orientation` persists in the fluorescence configuration.
- FM initialisation is skipped on non-compustage TFS and simulator systems.
- OME metadata is recovered from files carrying `<Channel><Filter/></Channel>`.

### Correlation

- **FM surface point and pre-correlation refractive-index correction.**
- A point-type registry behind the canvas and list plumbing.
- UX polish and a batch of result-state correctness fixes; `Load Coordinates` rejects
  the wrong JSON instead of crashing.
- The napari `CorrelationUI` and the orphaned prototype widgets were removed.

### Workflow and UI

- **Schedule AutoLamella tasks from the GUI.**
- A post-workflow summary dialog, and a user-Cancelled task status with notifications.
- Recent experiments quick-select; experiments load by folder.
- The Preferences dialog was rebuilt on a sidebar and stack, with Restore Defaults.
- napari's `thread_worker` was replaced with a napari-free `FunctionWorker`.
- Icons render offline through qtawesome instead of network Iconify.
- **"Report an Issue"** bug reporter.
- Four widgets migrated off Qt Designer, and the unused `.ui` files deleted.

### Autofocus and alignment

- **An `autofunctions` module** consolidating autofocus: ACB, image autofocus, charge
  neutralisation, and a multi-pass FM UI.
- `AutoFocusStrategy` and an `AutoFocusResult` dataclass, with plotting extracted.
- **The alignment code became a package**, with multi-method comparison and an
  `AlignmentRun` record.

### Spot burn

- Status-bar progress, a unified UI, and a milling-style workflow hook.
- Acquire on finish, and reliable coordinate-mode switching.
- Fixed a crash in unsupervised mode from string parameters and out-of-bounds
  coordinates.

### Milling

- `passes` on `TrenchPattern` and `TrenchBitmapPattern`; maximum passes raised to 100
  million.
- Cooperative cancellation, and the pre-milling imaging current is restored afterwards.
- The compatible application file is set for Rectangle patterns.

### Other

- **`fibsem-cli`** for command-line microscope control.
- The sample holder gained slot/grid separation, a redesigned widget and preferences
  integration.
- Tile grid positions are validated against stage limits before acquisition.
- `move_to_orientation` on `FibsemMicroscope`, replacing `move_flat_to_beam` at the
  movement widget's call sites.
- `lamella.objective_position` moved to `fluorescence_pose.objective_position`.
- Fixed `np.float_` (removed in NumPy 2) and the `pyqt5-qt5` dependency on Windows.
- Installation docs recommend Miniforge and document the `uv` option; AutoScript 4.7+.

## v0.5.0 (01/05/2026)

The largest release in the project's history: AutoLamella and 3DCT were merged into this
repository, the AutoLamella workflow was rebuilt around tasks, and the GUI was rebuilt in
a single window. 578 commits since v0.4.0.

### AutoLamella and 3DCT live here now

- **AutoLamella was migrated into the repository** (#11) and is developed alongside
  `fibsem` rather than against a released version of it.
- **3DCT was migrated in** (#9), including the `pyto` components, so correlation ships
  with the package.

### A task-based workflow

- The AutoLamella workflow was rebuilt around **tasks**: one file per task, a task type
  allowing multiple copies, per-task configuration, and a task manager driven by a queue.
- **Task and protocol editors** in the GUI, with save/load, global parameter edits across
  lamellae, and applying one lamella's configuration to others.
- A **workflow timeline** showing queue status, step status, completion timestamps and
  skip reasons.
- Task status gained **Skipped**, and defect state moved to an enum rather than several
  booleans.
- `None` orientation support for the trench, undercut and grid tasks; the MILLING
  orientation extends to tilts above SEM.
- Plugin-style registration for tasks.

### A rebuilt GUI

- **Single-window interface** with napari-style stylesheets and a consolidated palette.
- **Lamella cards** with thumbnails, a card strip and sub-tabs in the Lamella Editor,
  and per-row actions.
- New beam, detector, image and system settings widgets, defined in code rather than
  Qt Designer.
- A **workflow tab** with supervision controls, a run button, progress bars, a status
  border, a user-attention button and a stop-workflow confirmation.
- Right-click context menus for milling and the minimap, a view menu, and toast
  notifications with a notification sound.
- The minimap was reworked and renamed to **Overview Acquisition**, with a grid boundary
  overlay, TEM stage limits, click-to-select near a point, and stage-limit checks.
- A **MicroscopeConfigWidget** with tabbed UI and YAML import/export.
- Zoom/pan lightbox on review images, an interactive measurement tool, and a milling FOV
  rectangle when the milling FOV is smaller than the image.
- Scroll-wheel guards on spinboxes and combo boxes, so scrolling a panel no longer
  changes values.
- **User preferences** and a preferences dialog.

### Milling

- **Bitmap milling patterns** (rectangle and trench), with interpolation and resizing
  options.
- **Circular array** pattern support, a `fillet` parameter for trenches, and a crop
  threshold for polygons.
- A **plugin-style `BasePattern` system** (#10) and generic type inheritance for patterns
  and strategies.
- The milling **pose** replaced the milling position, with separate poses for milling and
  landing, and automatic milling-angle alignment.
- Stop-milling moved from a flag to a **stop event**.
- Milling time estimates, per-stage enable, alignment-area editing, and an
  eye-icon show/hide for patterns.
- Circle and line pattern plotting, depth display, and scalebar consolidation.

### Microscope support

- **Updated TESCAN support** (#341), including detector initialisation and simulator
  fixes.
- **Odemis** integration reworked, with SEM-based coincidence movement.
- Live acquisition in the UI for SEM and FIB, beam on/off and blank/unblank controls,
  scanning-mode and channel setters.
- A **limits API**, available voltage/current ranges for SEM, compucentric rotation
  support, and beam-shift clipping to limits.
- A sputter coater for TFS and the simulator.
- A **`FibsemMicroscope` HTTP server and client** (#184).
- Simulated autocontrast/autofocus outcomes, simulated acquisition time, and image
  generators that replay a directory of real images.

### Imaging and alignment

- `align_until_converged` with dynamic early exit, and cv2 phase-correlation alignment
  with a response score.
- Subpixel alignment, and alignment diagnostic plots.
- `FibsemImage` gained `extract_region`, `resize`, `apply_gamma` and a `brightness`
  property; display goes through a filtered-data cache.
- Focus stacking moved into the `acquire` module.

### Hooks and reporting

- **Initial hook support** (#87) and a webhook configuration widget.
- Overview images in the report PDF, lamella descriptions on the overview export plot,
  and a task-history widget.

### Packaging and release

- A **PyPI publish workflow** and release documentation.
- Updated GitHub Actions CI (#84).
- Python 3.11 support; py3.8 compatibility fixes throughout.
- A v2 nnU-Net model for inference, with checkpoints read from cache when Hugging Face
  is unavailable.

## v0.4.0 (24/04/2025)

Current Status: Pre-Release

### Installation
- The minimum required python version is now 3.8 (down from 3.9). This should enable installing fibsem on older systems that cannot be updated from Windows 7. 
- Both fibsem and autolamella can now be run in 'headless' mode without the UI (or requiring it's dependencies). This is used for embedding openfibsem into other standalone applications. 
 - The machine learning dependencies (used for more advanced methods) are now optional.
- Instaling packages is now slightly different to reflect these optional dependencies:

```
pip install fibsem          # only install fibsem headless mode
pip install fibsem[ui]      # install fibsem + ui dependencies
pip install fibsem[ui,ml]   # install fibsem + ui and ml dependencies
```

### Milling
- The milling code has been consolidated into the fibsem.milling module.

```

fibsem
    milling
        base.py             # base milling structures 
        core.py             # core milling workflow
        patterning/
            patterns.py     # pattern definitions
            plotting.py     # plotting utilities
        strategy/
            standard.py    # standard (default) milling strategy
            ...            # additional strategy files
```

#### Patterns
- Milling patterns directly store parameters, instead of reading in a protocol dictionary.
- Some patterns have had their parameter names adjusted for clarity and generality:


##### Trench, HorseshoePattern, HorseshoeVertical
- Trench based parameters have been adjusted:
- lamella_width -> width
- lamella_height -> height
- size_ratio -> split into upper_trench_height, lower_trench_height
- Loading an older protocol in autolamella should automatically convert to the new format. If older protocols are not read correctly, consider it a bug and please get in contact.

##### RectangleOffset
- RectangleOffset patterns have been removed, as their purpose was to position rectangle patterns via the protocol. The position of patterns can be directly specified in prootocol, using point: {x, y}

#### Milling Stage

- Estimated milling time can now be calculated independently from the microscope. This will be less accurate than the real duration calculated during milling.
- Milling stages now have additional configuration options:

##### Milling Strategy
-  Previously, openfibsem only supported a basic milling process; Set milling settings, draw patterns, mill patterns, restore imaging settings. Milling strategies enable customising the milling process. 
- Currently only Standard and Overtilt milling are implemented, but more will be added in the future.
- Developers can add additional strategies by implementing the spec in milling.base. Additional strategies can be registered by:
  - Registering them using a plugin-style registry
    ```
    from fibsem.milling.strategy import register_strategy
    from custom_strategy import CustomMillingStrategy

    register_strategy(CustomMillingStrategy)

    ```
  - Creating a `fibsem.strategies` entrypoint that points to the strategy class and installing the package in the same environment as fibsem, e.g. in pyproject.toml:
    ```
    [project.entry-points.'fibsem.strategies']

    # Make the strategy discoverable by fibsem
    # The class CustomMillingStrategy is in my_pkg/strategy.py

    strategy = "my_pkg.strategy:CustomMillingStrategy"

    ```

##### Milling Alignment
- Previously, aligning milling currents was only available via the autolamella option (align_at_milling_current). This was not straightforward to use or easily discoverable. 
- Initial milling alignment is now available for each stage. This will acquire an image after changing to the milling current and re-align to the imaging current. By default it will use the alignment area (fiducial area) defined in autolamella.
- Interval based drift correction will be enabled in the next version (v0.4.1)

##### Milling Acquisition
- You can now specify to acquire an image at the end of each milling stage. The acquisition settings can be adjusted per stage.

### User Interface:
- Parameters now display units directly on the control (rather than the label)
- Tooltips are being added to UI elements to help explain different parameters and options. 
- Acquire Image has been split into individual channels (Acquire SEM/ Acquire FIB)
- You can now show/hide milling patterns in the UI.
- You can now pause and resume milling from the UI.
- You can now select individual stages to mill, rather than having to mill all at once.
- Advanced options have been added to the imaging UI (e.g. line integration)


### Developer Notes:
- New tools are available for debugging milling patterns:

```
import matplotlib.pyplot as plt

from fibsem import utils
from fibsem.milling import get_milling_stages
from fibsem.milling.patterning.plotting import draw_milling_patterns
from fibsem.structures import FibsemImage

# load protocol
PROTOCOL_PATH = "/path/to/protocol/protocol-on-grid.yaml"
protocol =  utils.load_protocol(PROTOCOL_PATH)

# get the milling stages
stages = get_milling_stages("mill_rough", settings.protocol["milling"]) 
stages.extend(get_milling_stages("mill_polishing", settings.protocol["milling"]))
stages.extend(get_milling_stages("microexpansion", settings.protocol["milling"]))

# create a blank image
image = FibsemImage.generate_blank_image(hfw=stages[0].milling.hfw)

# plot the milling stages
fig = draw_milling_patterns(image, stages)
plt.show()
```

- More milling data is now logged at each stage, and can be exported to run analysis. The milling related data is exported in milling.csv (see AutoLamella v0.4.0)


### Experimental Features
- There is now an experimental writer for exporting openfibsem images in OME-TIFF format. This will be enabled as default in the next version (v0.4.1). This should enable other applications  (e.g. ImageJ) to open the images and read the metadata correctly.


## v0.2.2 - 31/07/2023

### Highlights

- OpenFIBSEM is now available on PyPI. Use pip to install: `pip install fibsem`. On ThermoFisher systems, OpenFIBSEM will automatically find your Autoscript installation if it installed. On Tescan, please install into the same environment as the Automation API.
- Minimap: Added a minimap widget for collecting tiled images, selecting positions and correlation. Provides an overview of the current stage position and the positions of the selected locations. Also provides an integrated correlation user interface. You can use the minimap to select locations for other applications, such as AutoLamella.  

### Features

- Added a safe_absolute_stage_movement. This function will tilt flat before performing large movements to prevent collions.
- Added cleaning_cross_section and scan_direction to the milling widget user interface.
- Rectangle Patterns now sputter a 'passes' parameter. This allows you to explicitly set the number of passes the beam will scan.
- Adjusted the milling widget to allow for the selection of multiple milling stages. This allows you to move multiple stages together.
- Added automatic logging for alignment data. All alignment data is now logged to a file in the log/crosscorrelation directory. You can change this log directory in the config.
- Added a cryo sputter widget for automated sputtering in cryo conditions.
- Added two way projection between image and stage coordinates. This allows you to click on an image and move the stage to that location, as well as project a stage coordinate to an image coordinate (currently located in fibsem.imaging._tile).
- To move milling stages in the UI, you now need to 'Shift' + 'Left Click' (Was 'Right Click')
- To move the stage vertically (eucentric_move), you now need to 'Alt' + 'Left Click' (Was previously an option in the UI).

### Fixes / Updates

- Fixed an issue where masks were not calculated for alignment.correct_stage_drift.
- Changed the model checkpoint lookup to search the fibsem/segmentation/models directory instead of expecting an absolute path.
- Fixed an issue where coordinate system was flipped when moving using a detection.
- Fixed an issue where milling protocols were being overwritten when setting the milling stages directly. [USER-INTERFACE]
- The milling widget hfw should now update automatically when changing the imaging settings. [USER-INTERFACE]
- The user interface won't try to draw the cross hair if no image is available. [USER-INTERFACE]
- Explicitly converting the last_image to np.uint8 (was np.uint16) [THERMO]
- Explictly settings the manipulator coordinate system when performing movements [THERMO]
- Post milling current now set to 30keV: 150 pA instead of 30keV: UHR Imaging [TESCAN]
- Fixed milling rate conversions, where the milling rate units were not converted correctly [TESCAN]

## 12/07/2023

- Added Documentation
      - Added documentation for the detection and labelling widget
      - Added Instructions for installation using python v-env

- New features
      - Installation and Running .bat scripts
      - Manipulator positions calibration for TESCAN
      - Microscope positions available in the movement widget
      - Added minimap of microscope positions
      - Added a fibsem version number for development tracking
      - Live chat (experimental)
      - Autoliftout utils
      - GIS Widget for cryo-control of gas injection
      - Embedded detection widget

- Fixed bugs
      - fixed issue where parameters were passed incorrectly for milling
      - fixed Eucentric movement where z-direction was flipped

- Updated Functionality / Improved Processes
      - system/model yaml files can now be modified from the system widget
      - demo log paths now in fibsem base directory
      - scan/image rotation now saved to microscope state
      - An option to click to move multiple milling stages together is now available
      - Added a crosshair to the images
      - movement of milling pattern now emits a pyqt signal (backend)
      - Manufacturer / model /serial no info can now be accessed/saved
      - Manipulator UI adaptive based on if manipulator is retracted or inserted
      - Enabled granular hardware control for stage and manipulator (backend), eg: disable rotation only

## 24/05/2023

- Added new features
      - FIB current alignment
      - Manipulator Controls
      - Measurement tools
      - Segment Anything Labelling
      - Added new milling patterns (Bitmap, Annulus)
      - Separated stage pretilt
- Fixed bugs
      - Autolamella example
      - Set microscope stage
      - HFW
      - Milling widget
      - Application file/Presets set on startup
      - Import TESCAN image files
