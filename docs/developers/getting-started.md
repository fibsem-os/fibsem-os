# Getting started as a developer

This page covers the repository layout, how to run the application and the
tests, and which extension point applies to a given goal. Contribution rules
(pull request size, the Python 3.8 floor, formatting, tests) are in
[CONTRIBUTING.md](../../CONTRIBUTING.md) and [AGENTS.md](../../AGENTS.md).
The supported extension points are described on
[Extending fibsemOS](extending.md). Other approaches are possible; open an
issue before relying on internals that are not listed there.

## Repository layout

```
fibsem/                     the instrument library (no application logic)
  microscope.py             FibsemMicroscope, the abstract class every backend
                            implements; also holds ThermoMicroscope, the
                            Thermo Fisher implementation
  microscopes/              tescan, odemis, simulator (DemoMicroscope, the
                            reference implementation); autoscript.py holds
                            Thermo Fisher helper subsystems, not the class
  structures.py             shared types: FibsemImage, Point, FibsemRectangle,
                            stage positions, settings
  milling/, imaging/        beam operations built on the abstract class
  segmentation/             segmentation models
  ui/                       shared Qt widgets, the palette (tokens.py), canvases
  server/                   the agent/bench HTTP server (build_server)
  mcp/                      the fibsem-mcp sidecar (MCP to HTTP)
  plugins/                  entry-point loading: fibsem.patterns,
                            fibsem.strategies, fibsem.tasks

fibsem/applications/autolamella/
  structures.py             Experiment, Lamella, AutoLamellaTaskProtocol
  workflows/tasks/          the task system: base.py (AutoLamellaTask),
                            registration, one module per task
  workflows/interaction.py  how tasks ask questions (ask(), Request types)
  ui/                       the application windows
  server/                   AgentContext, the view remote agents get
  scripting.py              ScriptContext, the view user scripts get

tests/                      mirrors the layout; tests/ui requires
                            QT_QPA_PLATFORM=offscreen
```

## Running the application

```bash
pip install -e ".[ui,test,dev]"
fibsem-autolamella-ui
```

Connect with the **Demo** configuration. No hardware is needed; every
workflow runs against the simulator, which images a synthetic cryo-grid
with cells, film, defects and a holder of grids, so navigation, alignment
and milling can be exercised end to end. See [the simulator](../simulator.md).

Running the application is the primary check for any change to wiring or
user interface. CI does not run the Qt test suite, and a passing test run
is not sufficient evidence that a widget works; see AGENTS.md.

## Running tests

Run the test files for the code you changed rather than the whole suite:

```bash
QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_something.py -q
```

## Which extension point

**Supporting a microscope.** Implement `FibsemMicroscope`, using the Demo
implementation as the reference, and register the manufacturer in the four
places listed under [Supporting a microscope](extending.md#supporting-a-microscope).
Implementing the class without registering it fails at connection time.

**Automating a procedure.** Three options, in increasing order of
integration: a plain script against the library with no application; a user
script that the application runs against the open experiment, receiving a
`ScriptContext`; or a workflow task, when the procedure should run in the
queue with supervision and history. The [choosing a route](extending.md#choosing-a-route)
table compares them, and [SCRIPTING.md](../../SCRIPTING.md) covers scripts in
full.

**Adding or extending a workflow task.** Subclass `AutoLamellaTask` with a
matching configuration class, override `_run()`, ask questions through
`ask()`, and record outputs on the task's history entry. The contracts, the
failure modes, and shipping a task as a plugin are under
[Workflow tasks](extending.md#workflow-tasks) and [Plugins](extending.md#plugins).

**Using a segmentation model.** Models live under `fibsem/segmentation/`;
see [Segmentation models](extending.md#segmentation-models).

**Building against the agent server.** The agent server is internal for
now. `docs/agent-server.md` describes it; the tool catalogue in
`fibsem/server/catalog.py` is the contract, and the supervision skill under
`.claude/skills/` is a reference client.

**Working on the user interface.** Use the palette tokens in
`fibsem/ui/tokens.py` and the shared stylesheets rather than literal
colours. Two rules: UI event handlers do not read from the microscope
(updates are pushed, or come from cached state), and cross-thread
communication goes through signals or the responder seam, never through
direct widget access. Check layout with an offscreen screenshot
(`widget.grab().save(...)`) and by running the application.
