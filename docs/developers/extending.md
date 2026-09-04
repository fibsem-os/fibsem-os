# Extending fibsemOS

The extension points we recommend and keep stable, in order of how much
they ask of you: a script, a plugin, a workflow task, a microscope backend.
Each section says what the seam is, where the reference implementation is,
and the mistakes that have cost people time. Other routes exist in the code;
if you need one, open an issue first.

## Choosing a route

| You want to | Use | Where it runs |
| -- | -- | -- |
| Read or change experiment data, offline | A plain Python script with `Experiment.load()` | Anywhere, no app |
| Drive the microscope for an acquisition sweep or a one-off procedure | A plain script with `utils.setup_session()` | Anywhere, no app |
| Run something against the experiment the app has open | A user script in the scripts folder, run from **Tools → Scripts** | Inside the app |
| Add a milling pattern, a milling strategy or a workflow task that others can install | A plugin package with entry points | Inside the app, discovered at start |
| Add a step to the AutoLamella workflow with supervision and history | A workflow task (as a plugin, or in the tree) | Inside the workflow queue |
| Support an instrument fibsemOS does not drive yet | A `FibsemMicroscope` implementation | The library, under everything else |

## Scripts

### Against the library, with no app

```python
from fibsem import utils, acquire
from fibsem.structures import BeamType, ImageSettings

microscope, settings = utils.setup_session(manufacturer="Demo")
image = acquire.acquire_image(microscope, ImageSettings(hfw=80e-6, beam_type=BeamType.ION))
```

`setup_session()` connects using a configuration file (the shipped Demo one
by default) and returns the microscope and its settings. The `fibsem.acquire`,
`fibsem.imaging` and `fibsem.milling` modules do the rest, and every
`FibsemMicroscope` method is available directly. Use
`acquire.acquire_image(microscope, settings)` rather than the microscope's
own `acquire_image`: the method is the raw grab and ignores `save=True`; the
module function applies autocontrast and gamma and writes the file.

To read or change an experiment's data, `Experiment.load()` gives you the
lamellae, their history and the ready-made dataframes.
[SCRIPTING.md](../../SCRIPTING.md) is the guide, with examples that the test
suite executes against real data.

### Inside the app

AutoLamella runs a `.py` file from the scripts folder against the experiment
it has open, from **Tools → Scripts → Manage scripts…** (off by default;
enable it under **File → Preferences… → Features**). The contract is one
module-level function:

```python
"""Export the experiment summary to CSV."""   # first line becomes the tooltip

def run(ctx):
    out = ctx.path / "summary.csv"
    ctx.experiment.experiment_summary_dataframe().to_csv(out, index=False)
    return out
```

`ctx` is a `ScriptContext` (`fibsem/applications/autolamella/scripting.py`):
the experiment, its path, a logger, and, if the script declares
`uses_microscope = True`, the microscope. Declare `writes = True` to have the
experiment saved afterwards. Return the result rather than printing it: a
string becomes a toast, a DataFrame opens in a table, a path opens its
folder. Use `ctx` rather than reaching into the UI object; the GUI's
internals move, and its `experiment` attribute is rebound on load, so a
cached reference goes stale silently.

Microscope scripts run on a background thread, must not mutate
`ctx.experiment`, and must call `ctx.raise_if_cancelled()` between steps or
Stop cannot stop them. Nothing validates what they do to the hardware. The
[Running scripts from the GUI](../../SCRIPTING.md#running-scripts-from-the-gui)
and [Microscope scripts](../../SCRIPTING.md#microscope-scripts) sections have
the full contract; three working examples are in `examples/scripts/`.

## Plugins

fibsemOS loads three entry-point groups at start: `fibsem.patterns`,
`fibsem.strategies` and `fibsem.tasks`. A plugin is an ordinary Python
package that declares one or more of them in its `pyproject.toml` and is
installed into the same environment as fibsemOS. No registration calls, no
files in the fibsem tree; the app finds it on the next start.

**Start from the example.** The
[fibsem-plugin-example](https://github.com/fibsem-os/fibsem-plugin-example)
repository is a template covering all three groups, with tests that check
the contract, a CI workflow that installs against fibsemOS `main` so you
learn about breaking changes before a collaborator does, and a README that
walks through renaming it. Read its "When nothing shows up" section before
you need it.

What each group contributes:

- **A pattern** (`fibsem.patterns`): a subclass of `BasePattern` from
  `fibsem.milling.patterning.patterns2` whose `define()` returns the shapes
  to mill. Appears in the pattern list of every milling stage.
- **A strategy** (`fibsem.strategies`): how a pattern is milled, such as in
  N passes. Appears in the strategy list of every stage.
- **A task** (`fibsem.tasks`): an `AutoLamellaTask` with its config class,
  as described in the next section. Appears in the Add Task dialog.

Three facts about loading that the example's tests enforce, because getting
them wrong is silent:

1. **Pattern modules are imported while `fibsem.milling.base` is only
   partly initialised.** Import `BasePattern` from `patterns2`, never from
   the `fibsem.milling.patterning` package, and import nothing that reaches
   back into `fibsem.milling.base`, which includes everything under
   `fibsem.applications`. A pattern and a task in the same file is enough to
   break the pattern. Strategies and tasks have no such restriction.
2. **A plugin that fails to import is simply absent.** The app starts, the
   class is not in the list, and the reason is in the log and in the
   Plugins panel (**Tools → Plugins…**, or `fibsem-cli plugins`), which
   lists every declared entry point with what became of it, including ones
   shadowed by a built-in of the same name. `fibsem/plugins/loader.py` is
   the loader and documents the records it keeps.
3. **Editing `pyproject.toml` does nothing until you reinstall.** Entry
   points live in the installed metadata, so `pip install -e .` again after
   changing them. Editing Python files needs no reinstall.

One trap in the forms: distances are stored in metres and scaled for
display, and a distance field without `scale` metadata renders as
`0.000 m`. Spread `DEFAULT_DISTANCE_METADATA` from `fibsem.milling.properties`
into distance fields, as the built-in patterns do; the example's tests check
this too.

## Workflow tasks

A task is one step done to one lamella. Subclass `AutoLamellaTask`
(`fibsem/applications/autolamella/workflows/tasks/base.py`) with a matching
`AutoLamellaTaskConfig`, and register both with `register_task()` in the
tree or with the `fibsem.tasks` entry point from a plugin. Read one existing
task module end to end first; `fiducial.py` is a good mid-complexity
example.

Four contracts:

- **Override `_run()`, not `run()`.** `run()` is the lifecycle wrapper:
  `pre_task()`, the hooks, `_run()`, `post_task()`. Overriding it loses the
  task's state, its history entry and every hook, with no error.
- **Questions go through `ask()`** (`workflows/interaction.py`). Build a
  `Request` carrying everything needed to answer it and block on the
  responder. Never reach into widgets from the workflow thread. This seam is
  what lets the GUI, the operator and a remote agent all answer the same
  question, and what makes supervised and unsupervised runs the same code.
- **Record what you produce** on the task's history entry
  (`task_state.outputs`, role → files), and write images under
  `lamella.path`. That is what the Review panel and the reports read;
  unrecorded files are invisible, and files elsewhere are lost when the
  experiment is copied off the microscope.
- **Cancellation is cooperative.** Stop sets an event. A task that never
  calls `self._check_for_abort()`, or a strategy that never checks
  `stop_event`, cannot be stopped.

`task_type` is written into protocol files, so it is permanent: renaming it
orphans every protocol that names it. Config classes are serialised into the
protocol as well, so keep them to plain types that round-trip through YAML.

## Supporting a microscope

The seam is `FibsemMicroscope` (`fibsem/microscope.py`), an abstract class
covering acquisition, movement, milling and state. Once a backend implements
it and is registered, everything above, workflows, UI and server, works
unchanged.

- **Template**: `DemoMicroscope` in `fibsem/microscopes/simulator.py` is the
  reference implementation, complete and hardware-free.
  `fibsem/microscopes/tescan.py` shows a vendor SDK behind the same
  interface. `microscopes/zeiss.py` is an empty placeholder awaiting a
  SerialFIB migration; if Zeiss is your goal, fill it in rather than starting
  a new file.
- **Register it.** Implementing the class is not enough: the manufacturer
  dispatch is hardcoded in a few places, and missing one gives
  `NotImplementedError` at connect, not at import. `fibsem/manufacturers.py`
  (the constant and alias), `setup_session()` in `fibsem/utils.py` (the
  branch that constructs your class), `fibsem/configuration.py` (the
  accepted manufacturers), and the manufacturer gates in
  `fibsem/ui/widgets/microscope_config_widget.py`.
- **Configuration.** Instruments are described by a YAML file in
  `fibsem/config/`; `fibsem-generate-config` scaffolds one. For a
  manufacturer it does not know, scaffold a Demo configuration and edit it.
- **Verify.** Connect with `utils.setup_session()` and run the tests that
  exercise the Demo through the same interface (`tests/test_acquire.py`,
  `tests/test_movement.py`, `tests/test_microscope.py`), then connect through
  the app. You do not need all of the abstract methods working to start: get
  acquisition and stage movement right and let the rest raise until their
  subsystem's turn.

### A fluorescence microscope

Subclass `FluorescenceMicroscope` (`fibsem/fm/microscope.py`), which covers
the objective, the filter set, the camera and acquisition; the
`SimulatedFluorescenceMicroscope` in `fibsem/microscopes/simulator.py` is
the hardware-free reference, and the Thermo Fisher and Odemis backends in
`fibsem/fm/` are the two real ones. This interface is still in active
development: expect it to change, and open an issue before building on it
so the change can go the right way.

## Your own segmentation model

`fibsem/segmentation/` is the home: `SegmentationModelHuggingFace` loads
checkpoints from the Hub, and the local-checkpoint paths sit beside it. A
general model library (sidecar metadata, local and Hub resolution) is
planned but not built; for now match the loading shape the existing models
use, and expect this area to firm up.
