# Developer documentation

For people, and coding agents, changing or extending fibsemOS. The user guide
for operating the application is at [fibsemos.org/docs](https://www.fibsemos.org/docs).

| Page | Read it when |
| -- | -- |
| [Getting started as a developer](getting-started.md) | You are new here: the lay of the land, how to run it, and which path fits your goal. |
| [Contributing](../../CONTRIBUTING.md) | Before your first pull request: PR size, the Python floor, formatting, tests, network rules. |
| [Extending fibsemOS](extending.md) | You want to add a pattern, strategy or task, drive the microscope from a script, or support an instrument. |
| [Scripting experiments](../../SCRIPTING.md) | You want to read or change an experiment's data from Python, in a notebook or from the app. |
| [The simulator](../simulator.md) | What the Demo microscope images, every scene key, and how the figures are generated. |
| [AGENTS.md](../../AGENTS.md) | You are a coding agent. Everything in Contributing applies; this adds what is specific to agents. |

Developers and agents read the same pages. There is no separate agent
documentation beyond `AGENTS.md` and the skills under `.claude/skills/`,
which are entry points into these pages rather than rewrites of them.

## Keeping these true

Two of these pages are partly generated, and the rest are checked by tests
where a claim can be executed:

- `simulator.md`'s figures and key table are written by
  [`render_simulator_examples.py`](render_simulator_examples.py) from the
  scene's own defaults. Re-run it after changing the simulator.
- The user guide's screenshots are written by
  [`render_user_guide.py`](render_user_guide.py) from the running
  application; a page state that names a widget that no longer exists fails
  the run rather than leaving a stale image.
- `SCRIPTING.md`'s example scripts run in the test suite against a real
  experiment and the simulator.

If you change something one of these pages describes, change the page in
the same pull request.
