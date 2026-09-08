# Developer documentation

Documentation for changing or extending fibsemOS. The user guide for
operating the application is at [fibsemos.org/docs](https://www.fibsemos.org/docs).

| Page | Contents |
| -- | -- |
| [Getting started as a developer](getting-started.md) | Repository layout, running the application against the simulator, running tests, and which extension point applies to a given goal. |
| [Contributing](../../CONTRIBUTING.md) | Pull request size, the Python version floor, formatting, tests, and network rules. |
| [Extending fibsemOS](extending.md) | Scripts, plugins (patterns, strategies, tasks), workflow tasks, and microscope backends. |
| [Scripting experiments](../../SCRIPTING.md) | Reading and modifying experiment data from Python, in a notebook or from the application. |
| [The simulator](../simulator.md) | What the Demo microscope images, the scene configuration keys, and how the figures are generated. |
| [The screenshot harness](screenshot-harness.md) | How the user guide's screenshots are rendered from the application, and how to add a page. |
| [AGENTS.md](../../AGENTS.md) | Conventions for coding agents. Contributing applies in full; this file adds what is specific to agents. |

Developers and coding agents use the same pages. `AGENTS.md` and the skills
under `.claude/skills/` are entry points into them, not separate
documentation.

## Generated content

Two of these pages are partly generated, and the code examples in a third
are executed by the test suite:

- The figures and the key table in `simulator.md` are written by
  [`render_simulator_examples.py`](render_simulator_examples.py) from the
  scene's defaults. Re-run it after changing the simulator.
- The user guide's screenshots are written by
  [`render_user_guide.py`](render_user_guide.py) from the running
  application. A page state that names a widget that no longer exists fails
  the run.
- The example scripts in `SCRIPTING.md` run in the test suite against a real
  experiment and the simulator.

A change to something one of these pages describes should update the page in
the same pull request.
