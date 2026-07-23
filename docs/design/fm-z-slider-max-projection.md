# FM Display — z-slider + max projection

Deferred follow-up from `lamella-editor-cutover.md`. The FM composite
(`FMCanvasWidget`, `fm_canvas.py`) currently **always shows the max-intensity projection** of each
channel's z-stack. Add:

- a **z-slider** to scrub individual z-planes, and
- a **"Max projection" checkbox** (in the layers popover) to toggle MIP vs the current plane.

`FMCanvasWidget` is shared by the main microscope tab and the lamella editor (both via
`controller.set_fm_image`), so this lands in one place and both hosts get it.

## Current behaviour

`FluorescenceImage.data` is **CZYX**. `set_fm_image` computes a 2-D MIP per channel
(`image.max_intensity_projection(channel=ci, return_2d=True)`), upserts it via `set_channel`, and the
composite is 2-D. The z-dimension is discarded on display. Each `FMLayer` holds 2-D `data` + display
props (`color` / `opacity` / `gamma` / `clim` / `autocontrast`), and the composite
(`composite_fm_layers`) blends the 2-D planes.

## Design

- **Keep the stacks.** `self._stacks: Dict[str, np.ndarray]` (channel name → ZYX), populated by
  `set_fm_image`. `set_channel` (the live 2-D path) drops any stack for its channel — live streaming
  stays 2-D and non-scrubbable.
- **State.** `self._z_index: int = 0`, `self._max_projection: bool = True` (default MIP =
  behaviour-preserving).
- **Plane selection.** A `_apply_z_mode()` helper sets, for each channel that has a stack,
  `layer.data = stack.max(0) if max_projection else stack[z_index]`, then `_recomposite()`. It updates
  `layer.data` **directly** (not via `set_channel`) so per-channel display props are preserved. Called
  on slider move + checkbox toggle.
- **z-slider.** A horizontal `QSlider` + "z i/n" label below the canvas (in the widget's `QVBoxLayout`).
  Range `[0, nz-1]`. Visible only when a scrubbable stack exists (`nz > 1`) and MIP is off; hidden/
  disabled otherwise.
- **Checkbox.** A global "Max projection" `QCheckBox` (default checked) in `FMLayersPanel`, emitting
  `max_projection_changed(bool)` → the widget flips `_max_projection`, shows/hides the slider, and
  re-planes.
- **nz** from `image.data.shape[1]`; assume channels share nz (acquired together), clamp `z_index` per
  channel.

## The one decision: contrast while scrubbing

`FMLayer.autocontrast` defaults to `True` and its clim cache is keyed on the data-array identity — so
swapping in a new z-plane **recomputes clim per plane**, making brightness jump as you scrub.

- **A — fixed clim (recommended):** when MIP is off, compute each channel's clim once (from the stack
  MIP or global percentiles), set `autocontrast=False` + that `clim`, and hold it across all planes.
  Scrubbing then shows *true relative intensity* between planes; dim planes read as dim. The user can
  still override contrast in the panel.
- **B — per-plane autocontrast (current default):** each plane is auto-contrasted independently.
  Every plane is individually visible, but brightness "pops" between planes — usually reads as jarring
  for a z-scrub.

Recommend **A**. Restore autocontrast when MIP is turned back on.

> **Decided: A** — fixed clim across planes while scrubbing (computed once off the stack MIP);
> autocontrast restored when Max projection is re-enabled.

## Interaction

| Max projection | z-slider | Display |
|---|---|---|
| on (default) | hidden | MIP (`stack.max(0)`), autocontrast |
| off | active | plane `stack[z]`, fixed clim (decision A) |
| `nz == 1` or live 2-D | hidden | the single plane (MIP == plane) |

## Slices

- **F0 — stacks + mode (no new UI).** Store per-channel stacks in `set_fm_image`; add `_z_index` /
  `_max_projection` + `_apply_z_mode`; default MIP. Behaviour-preserving.
- **F1 — z-slider.** Slider + label below the canvas, wired to `_z_index`, with the visibility rules.
- **F2 — checkbox + clim.** "Max projection" checkbox in `FMLayersPanel` + signal; wire to mode +
  slider; apply the contrast decision (A: fixed clim off-MIP, restore autocontrast on-MIP).
- **F3 — verify.** Headless: inject a synthetic CZYX `FluorescenceImage`; assert (a) default = MIP,
  (b) unchecking + scrubbing changes the composite to the selected plane, (c) `nz==1`/live-2-D hides
  the slider. Build in both hosts (`test_fm_canvas` / the viewer-less harness).

## Risks / watch-items

- **MIP parity:** `stack.max(axis=0)` must match `max_intensity_projection(return_2d=True)` (it's a
  plain max over z; verify dtype/orientation).
- **Don't break live streaming:** `set_channel` (in-place data-only update) must keep preserving
  per-channel controls; it just also clears the stack for that channel.
- **Memory:** holding full CZYX stacks (vs a single 2-D MIP) raises memory use — fine for typical
  stacks; note it.
- **Non-uniform nz across channels:** assume uniform; clamp per channel.
- **Panel already floats** (`FMLayersPanel` is a top-level popover) — the checkbox is just one more row
  at its top; keep it a *global* control (not per-channel).
