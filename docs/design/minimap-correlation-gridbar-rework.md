# Minimap Correlation + Gridbar — Rework (future work)

Status: **disabled** on the matplotlib minimap (2026-07-02). The controls (Load
Correlation Image, correlation layer combobox, Enable Correlation Mode, grid-overlay
checkbox + spacing/width) are hidden and their handlers are no-op stubs. This doc
captures why, and what a proper implementation needs.

Context: the Overview (minimap) tab was migrated off napari onto the matplotlib
`FibsemImageCanvas` (see `overview-minimap-cutover.md`). Correlation + gridbar were
ported as a first cut (composited `FMLayer`s), then reverted — the model is wrong, and
doing it right is a sizeable piece of work best done deliberately.

## Why the ported approach was reverted

The M4 cut composited correlation images as additive, colour-tinted `FMLayer`s over the
overview (reusing the FM canvas compositor), and resized (**stretch-to-fit**) any image
that didn't match the overview resolution. Two fundamental problems:

1. **No alignment.** A correlation image (an FM map, a prior EM overview, a screenshot)
   has its own scale, rotation, and position relative to the overview. Stretch-to-fit
   forces it onto the overview's exact extent — so the overlay is *geometrically wrong*
   except in the degenerate same-FOV case. Correlation is *by definition* about aligning
   two coordinate frames; without a transform it isn't correlation.
2. **napari's inline "transform mode" was never the answer either.** The old napari
   minimap let you drag a layer (`layer.mode = 'transform'`) to eyeball alignment, but
   the resulting affine was **never saved or read back** — purely visual, lost on close.
   So even the original feature didn't produce anything usable downstream.

The gridbar (a synthetic grid image generated at the overview shape) *did* composite
correctly, but its model deserves the same reconsideration (below), so it's bundled into
this rework rather than shipped alone.

## What a proper implementation needs

### Correlation
- **A real, persisted alignment transform** per correlation image — translate + rotate +
  scale (affine), stored with the experiment/overview so it survives reload and is
  available to downstream steps, not just visual.
- **Reuse the existing correlation machinery.** The repo already has a 3D correlation
  tool (`fibsem/correlation/`, `CorrelationTabWidget` / `CorrelationResult`). The minimap
  should consume/produce the same result type rather than inventing a second, weaker
  alignment path. Open question: does the minimap *launch* that tool for a pair, or embed
  a lightweight 2D affine editor and store a compatible result?
- **Interaction on the matplotlib canvas.** If we keep inline alignment, it needs a
  proper affine-drag overlay (handles for translate/rotate/scale) — the deferred
  "correlation drag-align" from the cutover doc. Rendering the aligned image = apply the
  affine to the layer before compositing (matplotlib supports an image transform, or
  resample into the overview frame via the affine).
- **Multiple correlation images**, each with its own transform + display props
  (colormap / opacity / visibility), likely surfaced in the layer popover.

### Gridbar
- Reconsider **vector overlay vs composited image.** As a `MinimapShapesOverlay` (lines)
  it would be crisp at any zoom, toggle instantly, and not fight the composite contrast —
  vs. the current synthetic-image-in-the-composite approach.
- **Tie to real grid geometry** where possible (grid type / bar pitch from the
  holder/config) instead of free spacing/width spinboxes, or at least seed the defaults
  from it.
- Decide whether bar placement should be **interactive** (drag to line the grid up with
  the physical bars in the overview) — which overlaps with the correlation-alignment work.

## Open questions
- Launch the existing `CorrelationTabWidget` vs. an embedded 2D affine editor on the minimap?
- Where is the transform persisted — on the `Lamella`/`Experiment`, or a per-overview sidecar?
- Is aligned correlation on the minimap actually needed, or does the dedicated correlation
  tool already cover the workflow (i.e. is this display-only convenience worth the cost)?
- Gridbar: vector overlay on the canvas, or keep it in the image composite?

## Current state (what's in the code)
- `add_correlation_image`, `toggle_gridbar_display`, `update_gridbar_layer`,
  `update_correlation_ui`, `_toggle_correlation_mode` — **no-op stubs**.
- Controls hidden in `setup_connections` (`correlation_panel`,
  `pushButton_load_correlation_image`, `label/checkBox/comboBox_pattern_overlay`).
- `generate_gridbar_image`, `COLOURS`, `CORRELATION_IMAGE_LAYER_PROPERTIES`,
  `GRIDBAR_IMAGE_LAYER_PROPERTIES`, and the empty `self._correlation_layers` /
  `_recomposite` correlation path remain as scaffolding for the rework.
- The milling-pattern overlay is separately deferred (see `overview-minimap-cutover.md`).
