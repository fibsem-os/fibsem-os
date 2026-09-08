# Extending fibsemOS

fibsemOS has four supported extension points, listed here in increasing
order of integration: a script, a plugin, a workflow task, and a microscope
backend. Each section describes the interface, the reference implementation,
and the constraints that are not obvious from the code. Other approaches are
possible; open an issue before relying on internals that are not listed
here.

## Choosing a route

| Goal | Mechanism | Where it runs |
| -- | -- | -- |
| Read or modify experiment data, offline | A Python script using `Experiment.load()` | Anywhere; no application |
| Drive the microscope for an acquisition sweep or a one-off procedure | A Python script using `utils.setup_session()` | Anywhere; no application |
| Run a procedure against the experiment the application has open | A user script in the scripts folder, run from **Tools → Scripts** | Inside the application |
| Add a milling pattern, a milling strategy or a workflow task that others can install | A plugin package with entry points | Inside the application, discovered at start |
| Add a step to the AutoLamella workflow with supervision and history | A workflow task (as a plugin, or in the repository) | Inside the workflow queue |
| Support an instrument fibsemOS does not drive yet | A `FibsemMicroscope` implementation | The library; everything else is built on it |

## Scripts

### Against the library

```python
from fibsem import utils, acquire
from fibsem.structures import BeamType, ImageSettings

microscope, settings = utils.setup_session(manufacturer="Demo")
image = acquire.acquire_image(microscope, ImageSettings(hfw=80e-6, beam_type=BeamType.ION))
```

`setup_session()` connects using a configuration file (the shipped Demo
configuration by default) and returns the microscope and its settings. The
`fibsem.acquire`, `fibsem.imaging` and `fibsem.milling` modules provide the
higher-level operations, and every `FibsemMicroscope` method is available
directly. Use `acquire.acquire_image(microscope, settings)` rather than the
microscope's own `acquire_image`: the method returns the raw frame and
ignores `save=True`; the module function applies autocontrast and gamma and
writes the file.

`Experiment.load()` gives access to an experiment's lamellae, their history
and the summary dataframes. [SCRIPTING.md](../../SCRIPTING.md) covers this
in full, with examples that the test suite executes against real data.

### Inside the application

AutoLamella can run a `.py` file from the scripts folder against the open
experiment, from **Tools → Scripts → Manage scripts…**. The feature is off
by default; enable it under **File → Preferences… → Features**. A script
is one module-level function:

```python
"""Export the experiment summary to CSV."""   # the first line becomes the tooltip

def run(ctx):
    out = ctx.path / "summary.csv"
    ctx.experiment.experiment_summary_dataframe().to_csv(out, index=False)
    return out
```

`ctx` is a `ScriptContext` (`fibsem/applications/autolamella/scripting.py`)
carrying the experiment, its path, a logger and, if the script declares
`uses_microscope = True`, the microscope. Declaring `writes = True` saves
the experiment afterwards. Return values are displayed: a string as a
notification, a DataFrame as a table, a path by opening its folder. Use
`ctx` rather than the UI object; the UI's internals change between
versions, and its `experiment` attribute is replaced on load, so a cached
reference becomes stale without error.

Microscope scripts run on a background thread, must not modify
`ctx.experiment`, and must call `ctx.raise_if_cancelled()` between steps for
Stop to take effect. Nothing validates what a script does to the hardware.
The [Running scripts from the GUI](../../SCRIPTING.md#running-scripts-from-the-gui)
and [Microscope scripts](../../SCRIPTING.md#microscope-scripts) sections give
the full contract; `examples/scripts/` contains three working examples.

## Plugins

fibsemOS loads three entry-point groups at start: `fibsem.patterns`,
`fibsem.strategies` and `fibsem.tasks`. A plugin is a Python package that
declares one or more of these in its `pyproject.toml` and is installed into
the same environment as fibsemOS. No registration calls or files in the
fibsem tree are needed; the application discovers it on the next start.

The [fibsem-plugin-example](https://github.com/fibsem-os/fibsem-plugin-example)
repository is a template covering all three groups. It includes tests that
check the contract, a CI workflow that installs against fibsemOS `main` so
breaking changes are detected early, and a README describing how to rename
it. Its "When nothing shows up" section covers the diagnostic steps.

Each group contributes one kind of object:

- **A pattern** (`fibsem.patterns`): a subclass of `BasePattern` from
  `fibsem.milling.patterning.patterns2` whose `define()` returns the shapes
  to mill. It appears in the pattern list of every milling stage.
- **A strategy** (`fibsem.strategies`): how a pattern is milled, for example
  in N passes. It appears in the strategy list of every stage.
- **A task** (`fibsem.tasks`): an `AutoLamellaTask` with its configuration
  class, as described in the next section. It appears in the Add Task
  dialog.

Three constraints on loading. Each produces no error when violated, so the
example's tests check them:

1. **Pattern modules are imported while `fibsem.milling.base` is only
   partly initialised.** Import `BasePattern` from `patterns2`, not from the
   `fibsem.milling.patterning` package, and import nothing that depends on
   `fibsem.milling.base`, which includes everything under
   `fibsem.applications`. A pattern and a task defined in the same module
   breaks the pattern. Strategies and tasks have no such restriction.
2. **A plugin that fails to import is absent.** The application starts
   without it and the class does not appear in the list. The reason is
   recorded in the log and in the Plugins panel (**Tools → Plugins…**, or
   `fibsem-cli plugins`), which lists every declared entry point and its
   outcome, including entry points shadowed by a built-in of the same name.
   `fibsem/plugins/loader.py` is the loader and documents the records it
   keeps.
3. **Changes to `pyproject.toml` take effect only after reinstalling.**
   Entry points are read from the installed metadata, so run
   `pip install -e .` again after editing them. Changes to Python files need
   no reinstall.

Distances in configuration forms are stored in metres and scaled for
display. A distance field without `scale` metadata renders as `0.000 m`.
Spread `DEFAULT_DISTANCE_METADATA` from `fibsem.milling.properties` into
distance fields, as the built-in patterns do; the example's tests check this
as well.

## Workflow tasks

A task is one step applied to one lamella. Subclass `AutoLamellaTask`
(`fibsem/applications/autolamella/workflows/tasks/base.py`) with a matching
`AutoLamellaTaskConfig`, and register both with `register_task()` in the
repository or through the `fibsem.tasks` entry point from a plugin.
`fiducial.py` is a representative task module of moderate complexity.

Four contracts:

- **Override `_run()`, not `run()`.** `run()` is the lifecycle wrapper: it
  calls `pre_task()`, the hooks, `_run()` and `post_task()`. Overriding
  `run()` loses the task's state, its history entry and every hook, without
  an error.
- **Questions go through `ask()`** (`workflows/interaction.py`). Build a
  `Request` carrying everything needed to answer it and block on the
  responder. Do not access widgets from the workflow thread. This interface
  is what allows the GUI, the operator and a remote agent to answer the same
  question, and it is why supervised and unsupervised runs share one code
  path.
- **Record outputs** on the task's history entry (`task_state.outputs`,
  mapping role to files), and write images under `lamella.path`. The Review
  panel and the reports read from there; unrecorded files are not shown,
  and files written elsewhere are lost when the experiment is copied off
  the microscope.
- **Cancellation is cooperative.** Stop sets an event. A task that does not
  call `self._check_for_abort()`, or a strategy that does not check
  `stop_event`, cannot be stopped.

`task_type` is written into protocol files and is therefore permanent:
renaming it orphans every protocol that names it. Configuration classes are
serialised into the protocol as well, so they should contain only plain
types that round-trip through YAML.

## Supporting a microscope

The interface is `FibsemMicroscope` (`fibsem/microscope.py`), an abstract
class covering acquisition, movement, milling and state. Once a backend
implements it and is registered, the workflows, UI and server work
unchanged.

- **Reference implementations.** `DemoMicroscope` in
  `fibsem/microscopes/simulator.py` is the complete, hardware-free
  reference. `fibsem/microscopes/tescan.py` shows a vendor SDK behind the
  same interface. `microscopes/zeiss.py` is an empty placeholder awaiting a
  SerialFIB migration; a Zeiss backend should be built there rather than in
  a new file.
- **Registration.** Implementing the class is not sufficient. The
  manufacturer dispatch is hard-coded in four places, and a missing entry
  raises `NotImplementedError` at connection time, not at import:
  `fibsem/manufacturers.py` (the constant and alias), `setup_session()` in
  `fibsem/utils.py` (the branch that constructs the class),
  `fibsem/configuration.py` (the accepted manufacturers), and
  `fibsem/guided_setup.py` (`MANUFACTURERS`, and a `MicroscopeModel` per
  instrument, which is what the first-run wizard offers).
- **Configuration.** Instruments are described by a YAML file in
  `fibsem/config/`; `fibsem-generate-config` scaffolds one. For a
  manufacturer it does not know, scaffold a Demo configuration and edit it.
- **Verification.** Connect with `utils.setup_session()` and run the tests
  that exercise the Demo through the same interface (`tests/test_acquire.py`,
  `tests/test_movement.py`, `tests/test_microscope.py`), then connect through
  the application. Not every abstract method needs to work at first:
  acquisition and stage movement are the prerequisites for the rest, and
  unimplemented methods can raise until their subsystem is addressed.

### Fluorescence microscopes

Subclass `FluorescenceMicroscope` (`fibsem/fm/microscope.py`), which covers
the objective, the filter set, the camera and acquisition.
`SimulatedFluorescenceMicroscope` in `fibsem/microscopes/simulator.py` is
the hardware-free reference, and the Thermo Fisher and Odemis backends in
`fibsem/fm/` are the two hardware implementations. This interface is under
active development and is expected to change; open an issue before building
on it.

## Segmentation models

Models live under `fibsem/segmentation/`. `SegmentationModelHuggingFace`
loads checkpoints from the Hugging Face Hub, and the local-checkpoint paths
are beside it. A general model library (sidecar metadata, local and Hub
resolution) is planned but not implemented. For now, match the loading
interface of the existing models; this area is expected to change.
