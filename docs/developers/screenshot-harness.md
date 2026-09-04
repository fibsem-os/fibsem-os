# The user guide's screenshot harness

Every screenshot in the user guide on fibsemos.org is rendered by
[`render_user_guide.py`](render_user_guide.py) from the real AutoLamella
window, driven against the Demo microscope with the sample scene on. Nothing
is captured by hand. The point is that the guide can be re-shot after any
change to the interface instead of rotting, and that a control the guide
names must still exist for the render to succeed.

## Running it

From the repository root, with the docs site checked out beside this
repository (the script also looks a few directories up, so a git worktree
works):

```bash
python docs/developers/render_user_guide.py            # every page
python docs/developers/render_user_guide.py milling    # one page
python docs/developers/render_user_guide.py --list     # the page names
python docs/developers/render_user_guide.py --site ../fibsem-os.github.io
```

It runs on Qt's offscreen platform, so it needs no display; set
`QT_QPA_PLATFORM` to override. A full run takes twenty to thirty minutes,
most of it the two supervised workflow runs on the Workflows page; a single
page is usually under two minutes.

Images land in the site checkout under `public/doc/img/<page>/<name>.png`,
at 1x, with `public/doc/img/manifest.json` listing what each page
produced. After a run, rebuild the site and commit the images together with
the page that uses them.

## What it isolates

The harness never touches the machine it runs on:

- The configuration registry, preferences, saved positions and the sample
  holder files are redirected to a temporary directory. The stage module
  imports the holder paths by name, so those are rebound in that module too,
  or a calibration would land in the real file.
- The working directory is moved to the temporary directory, because
  acquisitions made with no experiment open save into it.
- The shipped `sim-arctis` and `sim-iflm` configurations are registered in
  memory under those names; the worked example's experiment is created in
  the temporary directory and adopted the way Create Experiment does it.
- Everything a screenshot could show of the machine is replaced by the
  worked example: `C:\fibsemOS\config`, `D:\fibsemOS\experiments`,
  the experiment `yeast-grid-a`, a configuration named `Arctis Bay 2`. A
  path to this checkout or the home directory in an image is a bug.

## Adding a page

A page is one function, registered by name in the order the guide lists
them:

```python
@page("milling")
def render_milling(h: Harness) -> None:
    h.first_run(False)
    h.show_tab(0)
    h.connect("sim-arctis")
    ...
    h.shot("milling-tab", target=mv, callouts=[Box(cw.core_panel), runner.pushButton_run_milling],
           numbered=True, crop=True)
```

`Harness` owns one application and one main window for the whole run and
carries the state between pages, so a page may find a previous page's
connection or experiment already in place. The helpers are written to be
idempotent for that reason: `connect()` keeps a connection to the same
configuration and would otherwise toggle the button and disconnect,
`ensure_experiment()` and `ensure_lamellae()` create only what is missing.

The helpers a page uses:

| Helper | What it does |
| -- | -- |
| `connect(name)` | Connect through the Connection tab to `sim-arctis` or `sim-iflm`. |
| `first_run(on)` | Show or hide the first-run offer on the Connection tab. |
| `show_tab(i)`, `show_main_tab("Protocol")` | Select a main-window tab. |
| `ensure_experiment()`, `ensure_lamellae(n)` | The worked example's experiment and lamellae. |
| `cell_positions(n)` | Stage positions of `n` cells on plain film, from the scene's own feature list and support masks. Every marked position comes from here, so none sits on a grid bar. |
| `wait_acquisition(iw)`, `wait_move(ctrl, iw)`, `wait_fm(fmc)` | Pump the event loop until the worker behind an acquisition, a stage move or a fluorescence acquisition has finished. |
| `pump(ms)` | Run the event loop for a while: paints, timers, queued signals. |
| `shot(name, ...)` | Grab and annotate; below. |

### Photographing

`shot()` grabs the main window by default, or the widget passed as
`target`, and writes `<page>/<name>.png`. Its options:

- `callouts`: widgets to mark, numbered in order. A bare widget gets a
  numbered badge at its corner; `Box(widget)` gets a light box round it as
  well. The rule is badges for controls and boxes for regions (a panel, a
  tab bar, the quad view), because a box round a single button only repeats
  its edge. A callout that is not visible raises: the guide must not
  describe a control the reader cannot see.
- `callout_rects`: rectangles in the target's coordinates, for things that
  are not widgets, such as a menu item from `menu.actionGeometry(action)`.
- `clicks`: `(rect, "Alt + Double click")` pairs. Drawn as a box, a
  crosshair at the click itself, and a pill naming the gesture.
  `image_point_rect(canvas, panel, x, y)` gives the rect for an image
  pixel.
- `crop=True`: trim a panel to the area its visible children occupy, with
  a margin for the badges; `height=` caps a list that stretches to fill its
  tab.

Menus are their own top-level windows, so a menu is grabbed by popping it
up and passing it as the target. Dialogs are shown with `show()`, never
`exec_()`, so the harness keeps the event loop.

## Getting to a state

The harness drives the same handlers the buttons do, in preference to
reaching into state, so a screenshot shows what a user's click produces.
Some things in the application stand in the way of that, and the pages have
had to work round them. Knowing these saves an afternoon:

- **Modal dialogs block the harness.** A `dialog.exec_()` inside a handler
  runs its own event loop and the harness never regains control. Confirm
  steps are answered by replacing the function that shows them
  (`ow._confirm = lambda *_: True` on the overview widget,
  `confirm_run_workflow_dialog` in the main-window module), and the
  workflow's completion summary is suppressed and then shown by hand with
  `show()` for its own screenshot.
- **Supervised prompts live on the Experiment tab** under the Microscope
  tab. Poll `ui.WAITING_FOR_USER_INTERACTION`, switch there, photograph,
  then answer with `pushButton_yes` or `pushButton_no`. A milling prompt
  comes back after the run (Yes is Run Milling again, No is Continue), so
  the Workflows page answers Yes once per task name and Continue after.
- **Lists rebuild after a run.** The Workflow tab's lamella and task
  checkboxes are cleared when a run finishes; re-tick both before the next
  run, and check `ui.is_workflow_running` right after pressing Run or a run
  that silently did not start looks like one that finished instantly.
- **Acquire Image on the Fluorescence tab takes the selected channel
  only**, and the Microscope tab's fluorescence view shows one frame. A
  multi-channel composite is a two-plane z-stack opened in the standalone
  Fluorescence Image Viewer, whose canvas has the channel controls. A
  loaded stack opens in max projection; turn it off to see the slice
  controls. The fluorescence toolbar exists only on the selected view.
- **The ion beam's overview tiles step further across the stage than
  their width**, and a 3 × 3 at 400 µm runs past the simulated stage
  limits. FIB overviews are taken at 250 µm, and at the MILLING orientation
  as a single row.
- **The development environment's example plugin** appears in the task and
  pattern lists. It is removed from a combo before the combo is
  photographed; it is not part of the product.
- **Panels inside a scroll area** are grabbed from the scroll area's content
  widget, which lays out at full height, not from the tab, which shows only
  the visible part.

## Determinism

Runs are deterministic apart from one pixel column at the seam between the
quad view and the control panel in full-window shots, a layout-rounding
artefact at the vispy canvas edge. Every panel and dialog shot is
byte-identical across runs. The scene seed, window size (1600 × 1000, no
screen to clamp it) and stage positions are fixed; the run is offscreen so
fonts and metrics do not depend on the machine.

## When a render fails

The most common failure is a callout naming a widget that has been renamed
or removed: the run raises with the callout's index and type. That is the
mechanism working. Fix the page function, or the guide's prose if the
control really has gone, and re-render that page. The second most common is
a wait helper timing out because a modal dialog appeared; find it and
replace the function that shows it, as above.
