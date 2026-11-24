---
name: autolamella-correlative
description: Lamellae targeted with the FM — reading FM stacks, correlating FM to FIB, and rough milling and polishing so the target stays in the lamella. Read fibsem-agent, fibsem-microscope and autolamella first, then this when the lamellae must go through a fluorescent target.
---

# Correlative lamellae

Read the fibsem-agent, fibsem-microscope and autolamella skills first. This skill adds what a
lamella through a fluorescent target needs: finding the target in the FM,
carrying it into the FIB, and milling so it stays in the slab. FM device use
(inserting and retracting the objective, focusing, FM overviews) is
fibsem-microscope's. Milling while the FM watches, on an instrument where the FM
and FIB are coincident, is the autolamella-coincidence-milling skill's.

## Reading an FM stack

Judge a target from its planes, not the max projection: `experiment_read_file`
with `z` (and `channels`) shows one plane, and the metadata gives each
channel's `sharpest_plane`. The projection hides which depth a spot is at and
whether a neighbour sits above it. Look for small targets (centriole
doublets) plane by plane in full-resolution crops (`experiment_read_file`
with `z`, `analysis_crop_image`): a downsampled view loses them. Check each
candidate against the lamella window on a FIB or SEM image before you call it
a target: on a real run a stress-relief cut was taken for the lamella. Until
the emission-filter read-back is fixed, an FM image's metadata can say
Reflection for a fluorescence channel: go by the channel's name and settings.

A dim target can hide under auto contrast (the 1st to 99th percentile),
which a bright bead or neighbour sets. `experiment_read_file` and
`analysis_crop_image` only auto-contrast; to look for a dim target, draw
its plane with `analysis_draw_image` and `looks`, setting each channel's
`clim` from the stats the answer gives: p1 to p99 first, then tighter around
the target's own values. Check a candidate at two settings before you call
it a target.

Once the FM is focused on a lamella, store the objective's position on its
fluorescence pose (`item_set_objective_position`), so its next stack starts
there. `item_set_objective_position_on_all` copies it to every lamella and
overwrites theirs: use it only when the operator asks.

## Correlating FM to FIB

Where the lamella has a spot-burn task, seed the FIB fiducials from its burns
(`correlation_find_spot_burns`, with `fib_image_path`) instead of picking
them by eye. Take each FM fiducial's z from its own fit, not one z for all:
the burns sit on the cell surface and follow its topography, which is rarely
flat in the FM (a few degrees of tilt is several slices across the field),
and one slice of z moves the target across the lamella. A burn far off the
surface the others trace is a wrong pick. If a burn's fit fails, fit it again
starting from the z its neighbours give there (`z`) before leaving it out.
Refine each fiducial pick with
`correlation_refine_point` before `correlation_correlate`, pass `lamella_path`, and look at the overlay before
trusting the answer. Always give a surface point, `fm_surface_point` or
`fib_surface_point`, so the refractive-index correction is applied, and check
the answer's `ri_correction`. Without one, the targets sit deeper than the FM
stack reads: on a real run 4 of 7 targets were lost this way. Apply the
result (`item_apply_correlation`, as the tab's Continue) only on a `good`
verdict, or when the operator agrees. To place another FM point (a second
target, a refitted one) through a correlation you already have, use
`correlation_predict` with its run, rather than correlating again.

## Rough milling to a target

Rough mill to a 4-6 µm slab first, by autolamella's *Rough milling*. Then
take an FM stack and correlate again, with a surface point, before
polishing.

## Polishing to a target

The polishing rules are autolamella-polishing's; the target decides where
the faces go:

- Estimate the target's depth in the slab from the FM stack, and mill the
  face away from it first.
- Start the near face 150 nm outside the target's half-maximum edge and step
  it in 50 nm at a time.

With a coincident FM, follow the target pass by pass with Iterative
Polishing (autolamella-coincidence-milling).

## Workers

When the lead hands jobs to `fibsem-worker` (autolamella-lead, *Workers*),
each lamella's target and its correlation are two more jobs, one worker
each, so the FM planes and crops stay out of the lead's context. They are
separate because finding the target can mean imaging: trying channels,
exposures and focus until the target shows.

- **Target**, one worker per lamella, before its first correlation. Brief
  it with: the task ("find 04's target"); the limits (no milling; the stage
  at this lamella; FM stacks and channels as needed, the objective
  retracted when it is done); the skill (fibsem-microscope for the FM objective, focus and images, and this
  one's *Reading an FM stack*); the lamella, any stack already saved by
  path, the intake's target (what it looks like, the channels it may show
  in) and the plan's What worked; and what to return: the channel and
  settings it shows best in, the stack's path, the target and a control
  punctum (FM pixel and z), crops at two contrast settings, and how sure it
  is. It stores the objective's position on the lamella
  (`item_set_objective_position`). With no target it is sure of, it says
  so and why.
- **Correlation**, one worker per lamella, once the target is found and a
  FIB image is saved. Brief it with: the task ("correlate 04's target to
  the FIB"); the limits (no milling; no stage moves; apply only on a `good`
  verdict); the skill (this one, *Correlating FM to FIB*); the stack's and
  the FIB image's paths and the target worker's answer; and what to return:
  the verdict, `ri_correction`, the overlay and whether it applied it.
- **After rough milling**, the lead takes a new stack (or the Target worker
  does, briefed with the last channel and target to refit) and a
  Correlation worker correlates again with a surface point.

The best channel and settings for the target go in the plan's What worked,
so the next lamella's Target worker starts there. A doubtful target or a
correlation it didn't apply comes back to the lead, which holds the
lamella or asks the operator; a worker never asks them.
