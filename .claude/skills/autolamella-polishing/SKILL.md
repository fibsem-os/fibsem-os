---
name: autolamella-polishing
description: Polishing a lamella to its target thickness — the target and the stop rules, the fields of view, measuring the band, and Iterative Polishing one pass at a time with an SEM image. Read fibsem-agent and autolamella first, then this before the first polishing pass, in the Polishing task or by hand.
---

# Polishing a lamella

Read fibsem-agent and autolamella first. This skill is how to thin a
lamella to the operator's target: in the Polishing task's milling proposal,
by hand with one `milling_run_stage` per step, or with Iterative Polishing.
The same rules hold in all three. A lamella targeted with the FM adds
autolamella-correlative's *Polishing to a target*, and with a coincident FM,
autolamella-coincidence-milling.

## The target and the stop rules

The target is a ceiling: done means the band is under the
operator's target, 200 nm unless they gave another. How far below it to go
is the operator's answer to "stop thicker, or try thinner?":

- **Stop thicker** (the default, and whenever you can't ask): accept at the
  first pass that measures under the target. A lost lamella is to be
  avoided.
- **Try thinner**: once under the target, keep polishing while nothing says
  stop, accepting that a lamella may be lost.

Either way, stop at once on a stop sign below. For every pass:

- Start gently and step down. Begin with a wide gap and narrow it a step at
  a time, lowering the current as it narrows; never jump to the final gap.
  Drop to 30 pA once the slab is under about 500 nm. On the
  Arctis the band ran at 0.53-0.73× the gap, so expect the target near a
  250-300 nm gap there; other instruments differ, so measure, don't assume.
- Preview every step's patterns on a fresh ion image first
  (`milling_preview_patterns`), and look at where they land.
- Polish in passes at least 1 µm deep, each placed from a fresh ion image
  (the field of view is in *Fields of view*). 0.4 µm passes at 10 pA removed
  nothing seven times out of seven.
- Align before every step. In the Polishing task, the task aligns before
  each pass: set its `alignment.full_frame_below` (80 pA, say) so the
  low-current passes align on the full frame, and don't re-align by hand as
  well. By hand (`milling_run_stage`), re-align to the reference yourself
  before every step (fibsem-agent, *Align to the image you placed on*).
- After every step, a FIB image and an SEM image, both saved as files, not
  previews, so the operator can review them later. Read the faces only on
  FIB images 3072 px wide: on a real run, faces read on 1536 px images led
  to three passes that milled nothing.
- Measure the band, don't trust the gap, and stop on the first of: the band
  under the target (with stop thicker); the band faint or low in contrast, even if it still looks
  wide (a faint band is already thin: a lamella milled through on the next
  pass when this was read as width); the SEM shows a crack, a break, a hole,
  bending or the face deteriorating.
- On a correlative lamella, the faces follow the target:
  autolamella-correlative's *Polishing to a target*.
- The polishing patterns can't be rotated to follow a tilted band. If the
  band is tilted enough to leave a wedge, say so before thinning further.
- Report each step: current, gap, time, the alignment residual, the band
  measured, what the SEM showed.

## Fields of view

| Job | Field of view |
|---|---|
| Placing a polishing pass, and checking coincidence | 50 µm |
| Imaging the band to judge a pass, until it is under about 400 nm | 30 µm |
| Imaging the band once it is under about 400 nm | 15 µm |
| Comparing thickness readings (`item_measure_thickness`) | 20 µm, at similar contrast |

In a lead run, the recipe card records the fields of view the first lamella
used; use the card's where it differs from this table.

## Iterative Polishing

Polishing one pass at a time, judging each pass on
its images and readings before the next. Use it only when the operator asks
for it, and only at Drive. You choose every pass's settings: the pattern and
where it sits, the current and depth, the images and stacks.

Each pass is one `milling_run_stage` with the strategy set:

```json
{"name": "Polish top 3",
 "milling": {"milling_current": 3e-11, "hfw": 1.5e-05},
 "pattern": {"name": "Rectangle", "point": {"x": 0.0, "y": 4.5e-07},
             "width": 1e-05, "height": 1e-06, "depth": 1e-06},
 "strategy": {"name": "Iterative Polishing",
              "config": {"face": "top", "acquire_sem_image": true,
                         "acquire_reflection_stack": false}}}
```

The pass mills the stage as given (a Rectangle for one face, a Trench for
both, `face` `both`), then takes its images. `service_wait_for_run` milling
answers with its facts and a figure (pass.png): the FIB band's top and
bottom edge (µm, +y up, the frame a pattern is placed in), its FWHM and
centre offset, and the file paths of each image. Look at the figure and the
facts before the next pass. With a coincident FM, the pass also takes FM
stacks and follows a target: autolamella-coincidence-milling's *Iterative
Polishing with the FM* adds its settings and stop points.

- **Before the first pass.** Take a FIB image and measure the band.
- **Each pass needs an SEM image.** Without a coincident FM, set
  `acquire_reflection_stack` false and `acquire_sem_image` true: stacks are
  refused there.
- **Both faces.** It isn't done until the top and the bottom face have each
  reached their stop point, in either order; a `both` pass counts for both.
  If you stop a face early, tell the operator. With a trench, narrow its
  `spacing` by twice the step each pass and keep its centre where it is;
  move the centre only to correct a band centre offset, or to thin one face
  on purpose.
- **Stop and ask the operator** (guidance, not hard limits) when: about
  0.7 µm has come off a face; the FWHM nears its floor; the FIB image shows
  damage; the SEM check finds a problem.
- **The SEM check**, on every pass with an SEM image (on a system without a
  coincident FM it is the main one): the protective layer still covers the
  front; no cracks, breaks or holes; no curtaining, bending or bowing; no
  charging; an even thickness. Any of these: stop and ask.
- **The final thickness** is the operator's to measure.

## How thick is the lamella?

Use `item_measure_thickness` instead of
measuring a preview yourself, on an image taken at the item's milling
position so the POI falls on the lamella. Read its `warnings` first. On
`under_resolved`, take the next image at the suggested field width. On
`low_contrast`, take a cleaner image: a ratio of 1.5 gave misleading numbers
on a real run. On `several_bands`, look for residue in a trench or a stepped
wall. Check the answer on the image before you use it:
the band it finds may be the wrong one, and then its `centre_offset` is wrong
too (a polish window moved by it landed on trench residue). If the band you
checked sits away from the POI, pass `centre` [x, y] at it rather than
moving the item. Compare readings
only across images of similar contrast, at a 20 µm field width: on a real
run one lamella read 226 nm on a high-contrast image and 305-341 nm on
low-contrast ones. A number that moves with contrast is not a measurement;
image again before acting on it.
