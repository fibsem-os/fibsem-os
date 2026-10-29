---
name: screen-grids
description: Grids — loading and unloading them, screening runs, whole-grid overviews, reviewing grids after a screen, and picking sites (cells, squares) for lamellae. Read fibsem-agent first, then this when the operator wants grids screened or sites picked, before or during AutoLamella work.
---

# Grids and sites

Read the fibsem-agent skill first: it covers connecting, the levels, the
watcher, and overviews (*Overview or tile set*). This skill adds the grids
themselves and how to pick sites on them. Making lamellae at those sites is
the autolamella-agent skill's.

## Grids by hand

`grid_list` is the experiment's records; `grid_get_inventory` is the hardware
now (which grid is loaded), and `grid_get_current` the grid under the beam.
With an autoloader `grid_load` is an exchange, so say which grid comes out
before you load. A grid run loads its own grids, so never load ahead of one.
Ask the operator before renaming a grid (`grid_rename`), and say the old and
new names. At Assist an exchange is a request like a move; put `grid_load`
and `grid_move_to` in one request when you want to look at the grid.

## A screening run

Starting one is a control action (Drive). Plan it first with
`workflow_plan_grids` and show the operator the plan: every step, one load
per grid. An exchange is the most physical thing the server commands. Then
start it with `workflow_start_grids` and the same body; `screen_all: true` inventories and runs every
present grid. One run at a time: it is refused while a lamella run is live,
and a lamella start is refused while a grid run is.

## Reviewing grids

After a screening run, `grid_list` lists each grid with its latest overview
per role. `grid_get_output_image` is the preview, and `grid_get_markers`
places the experiment's items on it in source pixels with their status. Use
it for "which items sit near the edge or off the grid", and never work out
the stage geometry yourself. `experiment_render_markers` draws the items on
any beam image in the experiment.

`quality` is the operator's verdict, not yours: say what you see, propose,
and leave the verdict to them.

## Finding sites

When you pick sites (cells, squares):

- Ask for the operator's placement rule first (fibsem-agent, *When an ask
  leaves out the rule*). Without one, use these rules and say so.
- Survey before GIS deposition and sputter coating when you can: cells and
  grid bars stand out far better. Positions are stage coordinates, so a site
  picked before coating holds after it while the grid stays loaded. Grids
  often arrive coated, so you will also have to pick on coated grids.
- Read a large overview in parts: `experiment_read_file` with `cells` cuts it
  into named cells (A1, B3) with their stage positions, and
  `experiment_crop_image` with `cells` gives up to 8 of them at full
  detail. A point read off a crop goes to
  `experiment_get_image_stage_position` with that `crop`.
- Prefer the FIB image for picking cells when you have it (at the milling
  angle): good cells are often easier to tell there than on the SEM
  overview. A FIB image doses the sample, so keep the current low and the
  frames few.
- One lamella per grid square. A square usually has room for one good one
  without overlapping another lamella's trenches, fiducial or stress-relief
  cuts, or a grid bar. If two must share a square, the film between their
  cuts can tear (two lamellae about 15 µm apart tore during rough milling):
  ask the operator first.
- Where in the square, best first: (1) the clear middle, more than about
  10 µm from any bar, with room for the trenches, fiducial and stress relief;
  (2) within 10 µm of a bar, only if the square has no cell in the middle, and
  ranked below every square that has; (3) on a bar: never mill it.
- The film under the site has to be intact. A broken, sheeted or covered
  square is not a site. Check each site in the FIB and SEM (bars, torn or
  curled film) before spending FM time on it.
- Methods that worked on an uncoated grid: GrabCut on the holey film, tracing
  the grid bars by hand where shading defeats a threshold, and a piece of
  debris as an anchor to find the same place again. GrabCut needs code you
  may not be able to run; then judge by eye and say so. Say how sure you are
  of each detection, not only where it is.
- Look at the tile seams. A feature on a seam can be cut in two, doubled or
  hidden under the next tile; check it in the tiles on both sides before you
  count it.
- A square you haven't looked at is unassessed, never "empty" or "bad".
- Confirm shorthand before acting on it. "3:3" or "C2" can name a square, a
  cell or a rank; ask which the operator means.

**Recording sites.** For AutoLamella, each site is a lamella: `item_add`
without `name`, so it gets the app's usual name, and how to find it again
(grid, square, rank, why it was chosen) in its description, as
autolamella-agent's *A lamella's record* says. Without AutoLamella, save each
as a position (`stage_save_position`). Show the operator what you picked:
`experiment_render_markers` draws the items on any beam image in the
experiment, your own overview included, and its `path` is the evidence of
the note that reports them (`experiment_add_note` with `evidence: {image:
path}`), shown in the card.
