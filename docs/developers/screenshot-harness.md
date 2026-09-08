# The user guide's screenshot harness

Every screenshot in the user guide on fibsemos.org is rendered by
[`render_user_guide.py`](render_user_guide.py) from the AutoLamella window,
running against the Demo microscope with the sample scene enabled. None are
captured by hand. This allows the guide to be re-rendered after any change
to the interface, and it means a control the guide names must still exist
for the render to succeed.

## Running it

From the repository root, with the docs site checked out beside this
repository (the script also searches a few directories up, so a git
worktree works):

```bash
python docs/developers/render_user_guide.py            # every page
python docs/developers/render_user_guide.py milling    # one page
python docs/developers/render_user_guide.py --list     # the page names
python docs/developers/render_user_guide.py --site ../fibsem-os.github.io
```

It runs on Qt's offscreen platform and needs no display; set
`QT_QPA_PLATFORM` to override. A full run takes twenty to thirty minutes,
most of it the two supervised workflow runs on the Workflows page. A single
page usually takes under two minutes.

Images are written to the site checkout under
`public/doc/img/<page>/<name>.png`, at 1x, and `public/doc/img/manifest.json`
lists what each page produced. After a run, rebuild the site and commit the
images together with the page that uses them.

## Isolation

The harness does not modify the machine it runs on:

- The configuration registry, preferences, saved positions and the sample
  holder files are redirected to a temporary directory. The stage module
  imports the holder paths by name, so those are rebound in that module as
  well; otherwise a calibration would be written to the real file.
- The working directory is changed to the temporary directory, because
  acquisitions made with no experiment open are saved there.
- The shipped `sim-arctis` and `sim-iflm` configurations are registered in
  memory under those names. The worked example's experiment is created in
  the temporary directory the same way Create Experiment does it.
- Everything a screenshot could show of the machine is replaced by the
  worked example: `C:\fibsemOS\config`, `D:\fibsemOS\experiments`, the
  experiment `yeast-grid-a`, and a configuration named `Arctis Bay 2`. A
  path to the checkout or the home directory appearing in an image is a
  bug.

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
carries state between pages, so a page may find a previous page's connection
or experiment already in place. The helpers are idempotent for that reason:
`connect()` keeps an existing connection to the same configuration rather
than toggling the button and disconnecting; `ensure_experiment()` and
`ensure_lamellae()` create only what is missing.

| Helper | Purpose |
| -- | -- |
| `connect(name)` | Connect through the Connection tab to `sim-arctis` or `sim-iflm`. |
| `first_run(on)` | Show or hide the first-run offer on the Connection tab. |
| `show_tab(i)`, `show_main_tab("Protocol")` | Select a main-window tab. |
| `ensure_experiment()`, `ensure_lamellae(n)` | Create the worked example's experiment and lamellae if absent. |
| `cell_positions(n)` | Stage positions of `n` cells on plain film, from the scene's feature list and support masks. Every marked position comes from here, so none sits on a grid bar. |
| `wait_acquisition(iw)`, `wait_move(ctrl, iw)`, `wait_fm(fmc)` | Pump the event loop until the worker behind an acquisition, a stage move or a fluorescence acquisition has finished. |
| `pump(ms)` | Run the event loop for a period: paints, timers, queued signals. |
| `shot(name, ...)` | Capture and annotate; see below. |

### Capturing

`shot()` captures the main window by default, or the widget passed as
`target`, and writes `<page>/<name>.png`. Options:

- `callouts`: widgets to mark, numbered in order. A bare widget gets a
  numbered badge at its corner; `Box(widget)` adds a light box around it.
  The convention is badges for controls and boxes for regions (a panel, a
  tab bar, the quad view), since a box around a single button only repeats
  its edge. A callout that is not visible raises an error, so the guide
  cannot describe a control the reader cannot see.
- `callout_rects`: rectangles in the target's coordinates, for things that
  are not widgets, such as a menu item from `menu.actionGeometry(action)`.
- `clicks`: `(rect, "Alt + Double click")` pairs, drawn as a box, a
  crosshair at the click point, and a label naming the gesture.
  `image_point_rect(canvas, panel, x, y)` gives the rectangle for an image
  pixel.
- `crop=True`: trim a panel to the area its visible children occupy, with
  a margin for the badges. `height=` caps a list that stretches to fill its
  tab.

A menu is its own top-level window, so it is captured by popping it up and
passing it as the target. Dialogs are shown with `show()`, not `exec_()`,
so the harness keeps control of the event loop.

## Reaching a state

The harness drives the same handlers the buttons do, in preference to
setting state directly, so a screenshot shows what a user's action produces.
Several parts of the application require workarounds:

- **Modal dialogs block the harness.** A `dialog.exec_()` inside a handler
  runs its own event loop and the harness does not regain control.
  Confirmation steps are answered by replacing the function that shows them
  (`ow._confirm = lambda *_: True` on the overview widget,
  `confirm_run_workflow_dialog` in the main-window module). The workflow's
  completion summary is suppressed and then shown with `show()` for its own
  screenshot.
- **Supervised prompts appear on the Experiment tab** under the Microscope
  tab. Poll `ui.WAITING_FOR_USER_INTERACTION`, switch there, capture, then
  answer with `pushButton_yes` or `pushButton_no`. A milling prompt is
  repeated after the run (Yes runs milling again, No continues), so the
  Workflows page answers Yes once per task name and Continue afterwards.
- **Lists are rebuilt after a run.** The Workflow tab's lamella and task
  checkboxes are cleared when a run finishes. Re-tick both before the next
  run, and check `ui.is_workflow_running` immediately after pressing Run;
  otherwise a run that did not start is indistinguishable from one that
  finished instantly.
- **Acquire Image on the Fluorescence tab takes every channel in the list**,
  and the Microscope tab's fluorescence view shows one frame. A
  multi-channel composite is composed in the standalone Fluorescence Image
  Viewer (View menu), whose canvas has the channel controls. A loaded stack
  opens in maximum projection; turn it off to reach the slice controls. The
  fluorescence toolbar exists only on the selected view.
- **The ion beam's overview tiles step further across the stage than their
  width**, and a 3 × 3 at 400 µm exceeds the simulated stage limits. FIB
  overviews are taken at 250 µm, and at the MILLING orientation as a single
  row.
- **The development environment's example plugin** appears in the task and
  pattern lists. It is removed from a combo box before the combo box is
  captured; it is not part of the product.
- **Panels inside a scroll area** are captured from the scroll area's
  content widget, which lays out at full height, rather than from the tab,
  which shows only the visible part.

## Determinism

Runs are deterministic apart from one pixel column at the seam between the
quad view and the control panel in full-window captures, a layout-rounding
artefact at the vispy canvas edge. Every panel and dialog capture is
byte-identical across runs. The scene seed, window size (1600 × 1000, with
no screen to clamp it) and stage positions are fixed, and the run is
offscreen, so fonts and metrics do not depend on the machine.

## When a render fails

The most common failure is a callout naming a widget that has been renamed
or removed: the run raises an error giving the callout's index and type.
This is the intended behaviour. Fix the page function, or the guide's prose
if the control has been removed, and re-render that page. The second most
common failure is a wait helper timing out because a modal dialog appeared;
find the dialog and replace the function that shows it, as described above.
