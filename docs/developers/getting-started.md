# Getting started as a developer

This is a router, not a manual: find your goal below and follow its path.
The rules of the road live in [CONTRIBUTING.md](../../CONTRIBUTING.md) (PR
size, py3.8 floor, format-on-touch, tests) and [AGENTS.md](../../AGENTS.md);
read those once before your first change. This document covers the paved
roads — the extension points we recommend and keep stable. Other routes
exist in the code; if you need one, open an issue first.

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

Run tests per affected file, never the whole suite by default:

```bash
QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_something.py -q
```

## "I want to support my microscope"

The seam is `FibsemMicroscope` (`fibsem/microscope.py`) — an ABC covering
acquisition, movement, milling, and state. Once implemented **and
registered** (below), everything above — workflows, UI, server — works
unchanged.

- **Template**: `DemoMicroscope` in `fibsem/microscopes/simulator.py` is
  the reference implementation — complete, hardware-free, and the shape to
  copy. `fibsem/microscopes/tescan.py` shows a real vendor SDK behind the
  same interface. (`microscopes/zeiss.py` is an empty placeholder awaiting
  a SerialFIB migration — if Zeiss is your goal, fill it in rather than
  starting a new file.)
- **Register it** — implementing the ABC alone is not enough; the
  manufacturer dispatch is hardcoded in a few places, and missing one
  gives `NotImplementedError` at connect, not at import:
  `fibsem/manufacturers.py` (the constant and alias — Zeiss already has
  one), `fibsem/utils.py` `setup_session()` (the if/elif that constructs
  your class), `fibsem/configuration.py` (the generator's accepted
  manufacturers), and the manufacturer gates in
  `fibsem/ui/widgets/microscope_config_widget.py`.
- **Configuration**: microscopes are described by a configuration YAML
  (see `fibsem/config/`); `fibsem-generate-config` scaffolds one — for a
  manufacturer it does not know yet, scaffold a Demo config and hand-edit.
- **Verify**: connect via `utils.setup_session()` and run the tests that
  exercise Demo through the same interface (`tests/test_acquire.py`,
  `tests/test_movement.py`, `tests/test_microscope.py`); then connect
  through the app. You do not need all ~45 abstract methods working to
  start — get acquisition and stage movement right first and let the rest
  raise until their subsystem's turn.

## "I want to automate something" (three tiers)

1. **A plain script against the library** — no app at all:
   `utils.setup_session()` gives you a connected microscope; the
   `fibsem.milling` and `fibsem.imaging` modules do the rest. Right for
   acquisition sweeps and offline analysis.
2. **A user script inside AutoLamella** — a Python file in the scripts
   folder (`fibsem/applications/autolamella/scripting.py`:
   `discover_scripts` finds them, the app lists them). Your script
   receives a **`ScriptContext`** — the experiment, microscope, and
   protocol as a stable, documented surface. Use it rather than reaching
   into the UI object: the GUI's internals are still moving and its
   `experiment` attribute is rebound on load, so a cached `ui` reference
   goes stale silently.
3. **A real workflow task** — when the automation should live in the
   queue, with supervision and history. Next section.

## "I want to add or extend a workflow task"

Subclass `AutoLamellaTask` (`workflows/tasks/base.py`) with a matching
`AutoLamellaTaskConfig`, and register both with `register_task()`
(`workflows/tasks/__init__.py` — docstring shows the pattern). Read one
existing task module end to end first; `fiducial.py` is a good
mid-complexity example.

Two contracts to respect:

- **Questions go through `ask()`** (`workflows/interaction.py`): build a
  `Request` carrying everything needed to answer it, and block on the
  responder. Never reach into widgets from the workflow thread — the
  request/responder seam is what keeps the GUI, the operator, and remote
  agents all able to answer the same question.
- **Record what you produce** on the task's history entry
  (`task_state.outputs`, role → files): that is what the review panel,
  `task_outputs`, and the dashboard read. Unrecorded files are invisible.

Third parties can ship tasks as plugins via the `fibsem.tasks` entry-point
group (`fibsem/plugins/loader.py` documents loading and its failure
reporting); patterns and milling strategies have their own groups.

## "I want to use my own detection model"

`fibsem/segmentation/` is the home: `SegmentationModelHuggingFace` loads
checkpoints from the Hub, and the local-checkpoint paths sit beside it.
The honest caveat: a general model-library design (sidecar metadata,
local/HF resolution) is planned but not built — for now, match the loading
shape the existing models use, and expect this area to firm up.

## "I want to build against the agent server"

Start with [docs/agent-server.md](../agent-server.md) — what the server
exposes, the security model, and how agents connect (MCP sidecar or plain
HTTP). The tool catalog (`fibsem/server/catalog.py`) is the contract: a
catalog entry without a sidecar implementation fails loudly at startup, so
extend both together. The supervision skill in
`.claude/skills/supervise-autolamella/` shows what a well-behaved agent
client looks like.

## "I want to work on the UI"

Use the role-named palette tokens (`fibsem/ui/tokens.py`) and prebuilt
stylesheets — never pasted hex. Two rules with history behind them: UI
event handlers never read the microscope (push updates or cached state
only — an observer must not make the app poll hardware), and anything
cross-thread goes through signals or the responder seam, never direct
widget access. Check layout with an offscreen screenshot
(`widget.grab().save(...)`) before calling a widget done, and remember CI
does not run the Qt tests — launch the app.
