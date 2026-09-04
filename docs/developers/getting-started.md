# Getting started as a developer

This is a router, not a manual: find your goal below and follow its path.
The rules of the road live in [CONTRIBUTING.md](../../CONTRIBUTING.md) (PR
size, py3.8 floor, format-on-touch, tests) and [AGENTS.md](../../AGENTS.md);
read those once before your first change. The paved roads themselves, the
extension points we recommend and keep stable, are on
[Extending fibsemOS](extending.md); this page says which one fits your goal.
Other routes exist in the code; if you need one, open an issue first.

## The lay of the land

```
fibsem/                     the instrument library (no app logic)
  microscope.py             FibsemMicroscope — the ABC every microscope
                            implements (plus, historically, the TFS
                            implementation itself — ThermoMicroscope lives
                            in this file, not in microscopes/)
  microscopes/              tescan, odemis, simulator (DemoMicroscope — the
                            reference); autoscript.py is TFS helper
                            subsystems, not the microscope class
  structures.py             the shared vocabulary: FibsemImage, Point,
                            FibsemRectangle, stage positions, settings
  milling/, imaging/        beam operations built on the ABC
  segmentation/             detection models (see "your own model")
  ui/                       shared Qt widgets, tokens.py palette, canvases
  server/                   the agent/bench HTTP server (build_server)
  mcp/                      the fibsem-mcp sidecar (MCP → HTTP)
  plugins/                  entry-point loading: fibsem.patterns,
                            fibsem.strategies, fibsem.tasks

fibsem/applications/autolamella/
  structures.py             Experiment, Lamella, AutoLamellaTaskProtocol
  workflows/tasks/          the task system: base.py (AutoLamellaTask),
                            registration, one module per task
  workflows/interaction.py  how tasks ask questions (ask(), Request types)
  ui/                       the application windows
  server/                   AgentContext — what remote agents see
  scripting.py              ScriptContext — what user scripts see

tests/                      mirrors the layout; tests/ui needs
                            QT_QPA_PLATFORM=offscreen
```

## First: get it running

```bash
pip install -e .[ui,test,dev]
fibsem-autolamella-ui
```

In the app, connect with the **Demo** (simulator) configuration — no
hardware needed; every workflow runs against it. That launch-and-click
loop is the ground truth here: green tests are weaker evidence than usual
(see AGENTS.md), so run the app for anything about wiring.

The simulator can image a synthetic cryo-grid, with cells, film, rips and a
holder full of grids, so navigation, alignment and milling can be exercised for
real; see [the simulator page](../simulator.md).

Run tests per affected file, never the whole suite by default:

```bash
QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_something.py -q
```

## "I want to support my microscope"

The seam is `FibsemMicroscope`, the reference implementation is the Demo,
and implementing the class is only half of it: the manufacturer has to be
registered in four places or connecting fails at runtime. All of it, with
the tests to run, is under [Supporting a microscope](extending.md#supporting-a-microscope).

## "I want to automate something"

Three tiers, from least to most involved: a plain script against the
library with no app; a user script that the app runs against the experiment
it has open, receiving a `ScriptContext`; or a workflow task, when the
automation should live in the queue with supervision and history. The
[choosing a route](extending.md#choosing-a-route) table says which, and
[SCRIPTING.md](../../SCRIPTING.md) covers scripts in full.

## "I want to add or extend a workflow task"

Subclass `AutoLamellaTask` with a matching config, override `_run()`, ask
questions through `ask()`, record what you produce. The contracts, the
mistakes that lose state silently, and shipping a task as a plugin are under
[Workflow tasks](extending.md#workflow-tasks) and [Plugins](extending.md#plugins).

## "I want to use my own detection model"

`fibsem/segmentation/` is the home; the caveats are under
[Your own detection model](extending.md#your-own-detection-model).

## "I want to build against the agent server"

The agent server is internal for now. `docs/agent-server.md` in this
repository describes it; the tool catalog in `fibsem/server/catalog.py` is
the contract, and the supervision skill under `.claude/skills/` shows what a
well-behaved client looks like.

## "I want to work on the UI"

Use the role-named palette tokens (`fibsem/ui/tokens.py`) and prebuilt
stylesheets — never pasted hex. Two rules with history behind them: UI
event handlers never read the microscope (push updates or cached state
only — an observer must not make the app poll hardware), and anything
cross-thread goes through signals or the responder seam, never direct
widget access. Check layout with an offscreen screenshot
(`widget.grab().save(...)`) before calling a widget done, and remember CI
does not run the Qt tests — launch the app.
