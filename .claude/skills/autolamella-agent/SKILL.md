---
name: autolamella-agent
description: Working with a running AutoLamella to make lamellae — the workflow's proposals, a lamella's record and images, and the rules for rough milling, polishing, thickness and finding sites. Read fibsem-agent first, then this, then the skill for your level (observe at Watch, assist at Assist, supervise at Drive).
---

# Working with AutoLamella

Read the fibsem-agent skill first: it covers connecting, the levels, the
watcher, messages, writing down why, reading the experiment and using the
microscope. This skill adds what is particular to making lamellae with
AutoLamella. The role skills build on both:

- **observe-autolamella**: explain a session or a finished experiment, and
  never act. Any level, and offline from a folder.
- **assist-autolamella**: at **Assist**, ask the operator to approve each
  action.
- **supervise-autolamella**: at **Drive**, decide the workflow's proposals
  and act. Its lead mode runs a whole session with three `autolamella-*`
  agents: a Verifier, an Investigator and an Auditor.

A workflow's proposals have no timeout, so nothing is lost by waiting.

## The workflow's proposals

Before you start a lamella run (`workflow_start`), plan it with
`workflow_plan` and show the operator what it would queue and how long it
would take (`workflow_plan_grids` before a grid run).

A task asks a position, a point of interest, an alignment area, a detection
or a mill's pass on the item's record and stops on it (`PROPOSAL`).
`proposal_list` lists it, and `proposal_decide` decides it: at Drive you
decide it, at Assist you suggest the decision with `proposal_suggest` for the
operator to approve or decline, at Watch you say what you would decide.

The operator outranks you: if they decide first, your job for that proposal
is over. A `409 stale_proposal` means the proposal changed under you, usually
because the operator decided it. That refusal is correct; re-read and carry on.

## A lamella's record

Draw an item's milling as well as your own marks (fibsem-agent, *Draw what
you mean*):

- `item_draw_milling`: an item's patterns as its task configs have them, on
  its latest ion image. Draw it before a mill, and before you decide or
  suggest a milling proposal.

Where a site came from (grid, square, rank, why it was chosen) belongs on the
lamella, not in a note: `item_update` with `{"patch": {"description":
"..."}, "version": "..."}`, the version from `item_get`. When you add a lamella (`item_add`), leave `name` out so it gets the
app's usual name.

## Reading a lamella

As well as the reads in fibsem-agent's *Reading the experiment*:

| Question | Read |
|---|---|
| One lamella | `item_get`, `item_list_outputs`, `item_get_output_image` |
| How thick is it? | `item_measure_thickness` on a FIB image of the item |
| The plan | `protocol_get`, `item_get_task_config`, `protocol_get_task_config` |

**A lamella's saved images.** `item_get_output_image` with a basename from
`item_list_outputs` gives a 768 px JPEG preview. Without MCP, fetch it over
HTTP; basenames have spaces in them, so URL-encode them:

```bash
curl -s -o out.jpg -H "Authorization: Bearer $TOKEN" \
  "http://127.0.0.1:8001/app/experiment/items/01-tough-mako/outputs/ref_Mill%20Fiducial_start_ib.tif"
```

`app_get_status` gives the experiment's folder (`experiment.path`). On the
microscope PC, the observe skill's offline reader reads the record from there
while the run is going.

## Milling lamellae

These add to fibsem-agent's *Using the microscope*.

**Setup's own autofocus** (`use_autofocus`, on by default) applies its best
working distance without the check fibsem-agent asks for: on dark FIB views
it has picked a wrong one, so read the working distance after Setup and
compare.

**What tasks move for you.** Some tasks edit other configs as they run, so
check the next one before you rely on an edit you made by hand. Setup moves
the fiducial to the centre of the decided alignment area
(`sync_fiducial_position`): a wide area can push it into a neighbour.
Rough Milling moves the Polishing patterns to its own position
(`sync_polishing_position`). Setup can move the fluorescence pose to follow
the lamella (`sync_fluorescence_pose`, off by default). Turn a sync off in
that task's config to keep your edit.

**Polishing.** The target is a lamella under 200 nm, thinner if it holds.
The same rules apply in the Polishing task's milling proposal and by hand
(one `milling_run_stage` per step):

- Start gently and step down. Begin with a wide gap and narrow it a step at
  a time, lowering the current as it narrows; never jump to the final gap.
  Drop to 30 pA once the slab is under about 500 nm. On the
  Arctis the band ran at 0.53-0.73× the gap, so expect the target near a
  250-300 nm gap there; other instruments differ, so measure, don't assume.
- Preview every step's patterns on a fresh ion image first
  (`milling_preview_patterns`), and look at where they land.
- Polish in passes at least 1 µm deep, each placed from a fresh 50 µm ion
  image. 0.4 µm passes at 10 pA removed nothing seven times out of seven.
- Re-align to the reference before every step. Below about 300 nm, if the
  alignment methods disagree, align again before milling. For polish passes,
  set the Polishing task's `alignment.full_frame_below` (80 pA, say) rather
  than re-aligning by hand.
- After every step, a FIB image and an SEM image, both saved as files, not
  previews, so the operator can review them later. Once the band is under
  about 400 nm, image the FIB at a narrow field of view (15 µm, not 30 µm).
  Read the faces only on FIB images 3072 px wide: on a real run, faces read
  on 1536 px images led to three passes that milled nothing.
- Measure the band, don't trust the gap, and stop on the first of: the band
  under 200 nm; the band faint or low in contrast, even if it still looks
  wide (a faint band is already thin: a lamella milled through on the next
  pass when this was read as width); the SEM shows a crack, a break, a hole,
  bending or the face deteriorating.
- On a correlative lamella, estimate the target's depth in the slab from the
  FM stack and mill the face away from it first. Start the near face 150 nm
  outside the target's half-maximum edge and step it in 50 nm at a time.
- The polishing patterns can't be rotated to follow a tilted band. If the
  band is tilted enough to leave a wedge, say so before thinning further.
- Report each step: current, gap, time, the alignment residual, the band
  measured, what the SEM showed.

**Rough milling.** Run the rough stages in the protocol's order (Rough Mill
01 before Rough Mill 02), and tell the operator before you turn one off.
Once the slab is under about 1.5 µm, don't add rough rounds: thin with the
polishing stages. An extra rough round on a real run cut through the slab
and left a double lamella.

On a correlative run, rough mill to a 4-6 µm slab first. Then take an FM stack
and correlate again, with a surface point, before polishing.

**How thick is the lamella?** Use `item_measure_thickness` instead of
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

**Finding sites.** When you pick sites (cells, squares):

- Ask for the operator's placement rule first (fibsem-agent's *When an ask leaves out the
  rule*). Without one, use these rules and say so.
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
- Record each site as *A lamella's record* says: `item_add` without `name`, and
  how to find it again in its description.
- Show the operator what you picked: `experiment_render_markers` draws the
  items on any beam image in the experiment, your own overview included, and
  its `path` is the evidence of the note that reports them
  (`experiment_add_note` with `evidence: {image: path}`), shown in the card.
