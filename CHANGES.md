# Changes

## v0.5.2 (unreleased)

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

### Grids (preview)

- **A grid workflow, behind a preference.** Preferences → Enable Grid Workflow adds a
  Grids tab (a card per grid the inventory found, with its overviews and history), a
  Grid page beside Lamella on the Protocol tab, and a Workflow → Grids view that runs the
  grid tasks over the selected grids, loading each from the magazine as it comes up, or
  screens every grid in the magazine in one go. Off by default; it has run end to end on
  the simulator and is waiting on bench time.
- **The sample holder and magazine controls are a Sample tab** under Microscope, always
  shown, in place of the block on the Movement tab and the preference that gated it.
  Load and unload are icons on the grid's row; inventory is a refresh icon that asks
  first.
- The protocol's name, description and version panel on the Protocol tab is now one
  line with a pencil that opens a dialog, and the settings columns are wider.

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
- **Grid boundaries and holder slot markers are drawn only when the holder has a
  calibrated slot.** Both describe a cryo sample holder, so on a system without one they
  drew a holder that was not there; on an Arctis, whose working slot is always
  calibrated, they stay on. Either way they toggle under the overlays button on the
  canvas. Stage travel limits are unchanged and still shown — those are a property of
  the stage itself.
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
