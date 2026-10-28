---
name: fib-experiment
description: Plain FIB work with no app workflow — images, milling patterns you place, spot burns, cross-sections, an imaging series, moving between saved sites. Read fibsem-agent first, then this when the operator wants FIB work that isn't making lamellae with AutoLamella.
---

# A plain FIB experiment

Read the fibsem-agent skill first. Here there is no workflow and no proposals:
each step is one you take yourself (Drive) or ask for (Assist), as the
operator asks for it. Nothing waits on you, so nothing is lost by stopping to
ask.

## Before you start

**Agree the plan.** Ask what the operator wants done, at which sites, and
what counts as done: the patterns and their size and depth, the milling
currents, which images to take after each step and where to save them. Say
the plan back as a short numbered list and put each item on your checklist
(`server_set_checklist`). Ask for anything missing rather than guessing
(fibsem-agent, *When an ask leaves out the rule*).

**An experiment to save into.** Your images and mills are saved in the open
experiment, and only its files can be read back, drawn and measured. If none
is open, ask the operator to open or create one. At Drive you may create it
yourself once they agree: `experiment_create` with a name for the day's work
(`2026-10-29/fib-sites`). Without one, what you do is still on the session's
record, but you can't look at your own images again.

**Get your bearings** before the first move (fibsem-agent, *Using the
microscope*).

## Sites

- Save each site the operator names as a position (`stage_save_position`,
  named for the site: `site-a`), and go back with
  `stage_move_to_saved_position`. Saved positions show in the app's Saved
  Positions panel, so the operator can go to them too.
- At each site, before any step: an image in each beam, centre the feature,
  and check coincidence (fibsem-agent, *Coincidence*). A site saved earlier
  is a starting point, not the place: centre on the feature itself.
- Show the sites on an overview with `experiment_draw_image`, a labelled
  point each, and cite that drawing in your note.

## Images

- Save every image you judge from, with `beam_acquire_image` and a `path`
  inside the experiment: one folder per site (`site-a/`) and a filename that
  says the step (`after-cross-section-2_eb`). A preview isn't kept.
- An imaging series takes the same `image_settings` every time (field width,
  resolution, dwell, detector), so the images compare. Say the settings once
  in a note, then only what changed.

## Milling

`milling_run_stage` mills one stage. A stage is a dict:

```json
{
  "name": "Cross-section A",
  "milling": {"milling_current": 3e-10, "milling_voltage": 30000.0, "hfw": 5e-05},
  "pattern": {"name": "Rectangle", "point": {"x": 0.0, "y": 0.0},
              "width": 1e-05, "height": 5e-06, "depth": 2e-06,
              "cross_section": "RegularCrossSection"}
}
```

Units are SI: metres, amps, volts, seconds. `point` is from the image
centre, y up. Fields you leave out take their defaults.

- `milling_list_options` lists the pattern names and each one's fields, the
  strategies, and the currents and settings this instrument mills with. Ask
  for a listed current only.
- `milling_estimate_time` first, and tell the operator how long.
- Preview on a fresh ion image at the site (`milling_preview_patterns`), look
  at where it lands against the feature, and draw it for the operator.
- Save that ion image and align to it before the mill (fibsem-agent, *Align
  to the image you placed on*).
- Mill, follow it (`service_wait_for_run` milling), and check the `MILL
  ENDED` line: the current it ran at, and whether it completed.
- Then image in both beams and look before the next step.

**A cross-section.** Mill a rectangle with `cross_section`
`RegularCrossSection` at a high current, then a thin cleaning pass on the
face (`CleaningCrossSection`, a lower current, a smaller height). Look at the
face in the SEM. Check in the preview that the cleaning pass sits on the
face you mean to look at.

**A spot burn.** `milling_estimate_spot_burn`, then `milling_run_spot_burn`.

**A trench, a line, a circle, an array.** Pick the pattern from
`milling_list_options` and set its fields there. Don't build one from many
rectangles when a pattern exists.

## At Assist

Each microscope step is a request (fibsem-agent, *Asking at Assist*). One
request waits at a time, so put one site's steps in one request (move, image,
mill, image) with the preview drawing named in the reason.

## When you're done

Message the operator a short summary: for each site, what was milled and
imaged and the files, then anything left undone and why. Add the same as a
note, citing the last image of each site.
