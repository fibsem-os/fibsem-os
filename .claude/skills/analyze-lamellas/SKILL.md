---
name: analyze-lamellas
description: Visually analyze autolamella experiment lamellas — load key stage images, inspect each one, and produce a quality summary
argument-hint: /path/to/experiment  [lamella-selector ...]  [--json]
---

Analyze FIB-SEM cryo-lamella preparation results from an autolamella experiment directory.

## Step 1 — Discover directories

Parse `$ARGUMENTS` as follows:
- The first token that starts with `/` is the **experiment path** (may be an experiment root or a single lamella dir).
- Any remaining tokens are **lamella selectors** — used to filter which lamellas to analyse. Each selector can be:
  - A numeric prefix / index: `1`, `01`, `3` → match lamella dirs whose name starts with `01-`, `03-`, etc.
  - A range: `1-5` → expand to indices 1, 2, 3, 4, 5 and match as above.
  - A name substring: `vocal-walrus` → match any lamella dir containing that string.
  - A full path: `/path/to/lamella-dir` → use directly.
- If no selectors are given, analyse **all** lamellas (or just the one dir if it's a single lamella directory).
- `--json` is a flag (not a selector); strip it before processing selectors.

Run `ls $EXPERIMENT_PATH` to see what's there.

- If it contains `experiment.yaml` → experiment root. Run `ls -d $EXPERIMENT_PATH/*/` to list lamella subdirectories (exclude `GRID_SCREEN_RESULTS*`). Then filter by selectors if any were provided.
- If it contains `ref_*.tif` files → single lamella directory. Analyse just this one (ignore selectors).

## Step 2 — Load and analyse in batches of 3

Process lamellas **3 at a time**: load all 4 images for 3 lamellas in parallel (up to 12 `mcp__fibsem__load_image` calls in one batch), write up the assessments and verdicts for those 3, then move on to the next batch. This keeps context manageable without being too slow.

For a single lamella directory, just load its 4 images.

For each lamella, load these images using `mcp__fibsem__load_image`. Skip gracefully (note "not found") if the file doesn't exist — it means that stage was not reached.

| Label | Filename |
|---|---|
| Setup (IB) | `ref_Setup Lamella Position_start_ib.tif` |
| Rough Mill (IB) | `ref_Rough Milling_final_res_02_ib.tif` |
| Polish (IB) | `ref_Polishing_final_res_01_ib.tif` |
| Polish (EB) | `ref_Polishing_final_res_01_eb.tif` |

For each image that loads successfully, write **1–2 sentences** of visual assessment. Cover:
- **Setup IB**: surface state, ice contamination, any pre-existing damage
- **Rough Mill IB**: trench geometry (symmetric/asymmetric), lamella thickness estimate, curtaining, redeposition
- **Polish IB**: lamella thinness, edge straightness, any remaining bulk material
- **Polish EB**: distinguish carefully — a *thin but intact* lamella shows a moderately brighter uniform band with visible contrast variations (organelles, membranes); a *completely over-milled* lamella shows an extremely/uniformly bright void with no internal contrast (beam passing through nothing); also note holes, collapse, or fringes at edges

## Step 3 — Per-lamella verdict

After the image assessments, give a single-line verdict:
- **Success** — lamella is thin, transparent, intact
- **Incomplete** — workflow stopped before polishing (note last stage reached)  
- **Failed** — over-milled, collapsed, or no material remaining
- **Poor** — polished but quality is marginal (thick, asymmetric, heavy curtaining)

Include a one-sentence reason.

## Step 4 — Summary table

After all lamellas, output a markdown table:

| # | Name | Status | Last stage | Key finding | STATUS |
|---|---|---|---|---|---|
| 01 | vocal-walrus | Failed | Polish | Over-milled, EB shows no material | failure |
| 05 | game-horse | Success | Polish | Clean lamella, organelles visible in EB | completed |
...

The **STATUS** column reflects the actionable workflow state of the lamella:

| STATUS | When to use |
|---|---|
| **completed** | All stages done and lamella is successful (thin, intact, electron-transparent) |
| **continue** | Workflow stopped early but lamella material is intact and undamaged — can resume from the last completed stage |
| **rework** | Polishing completed but quality is marginal (thick, asymmetric, heavy curtaining) — lamella exists but the polishing stage needs to be redone |
| **failure** | Lamella is over-milled, collapsed, or otherwise unrecoverably lost — cannot be salvaged |

Mapping from verdict:
- Success → **completed**
- Incomplete (setup only, or rough mill only) → **continue**
- Poor → **rework**
- Failed → **failure**

## Step 5 — JSON output (optional)

If `--json` was in `$ARGUMENTS`, after the summary table output a fenced JSON block with the following structure:

```json
{
  "lamellas": [
    {
      "name": "01-vocal-walrus",
      "status": "Failed",
      "workflow_status": "failure",
      "last_stage": "Polish",
      "key_finding": "Over-milled, EB shows no material",
      "assessments": {
        "setup_ib": "One or two sentence assessment, or null if image not found.",
        "rough_mill_ib": "...",
        "polish_ib": "...",
        "polish_eb": "..."
      }
    }
  ],
  "summary": {
    "total": 5,
    "success": 2,
    "failed": 1,
    "incomplete": 1,
    "poor": 1
  }
}
```

Rules:
- `status` must be exactly one of: `"Success"`, `"Failed"`, `"Incomplete"`, `"Poor"`
- `workflow_status` must be exactly one of: `"completed"`, `"continue"`, `"rework"`, `"failure"`
- `last_stage` must be exactly one of: `"Setup"`, `"Rough Mill"`, `"Polish"`, `"Unknown"`
- `assessments` values are strings (1–2 sentence assessment) or `null` if the image was not found
- The JSON block must be the **last thing in your output** so it can be reliably extracted

## Notes on FIB-SEM image interpretation

- **IB images** (ion beam): high contrast, shows surface topology and milling geometry
- **EB images** (electron beam): lower contrast, electron transparency indicates thin areas (appears bright/uniform); organelles appear as dark puncta or membrane structures
- **Curtaining**: vertical stripes in IB images from uneven milling through varying density material
- **Redeposition**: bright haze around trench edges from re-deposited material
- **Over-milling**: holes through the lamella, or a uniformly/extremely bright EB with zero internal contrast — this is a void, not transparency. Do NOT confuse a complete void with a thin lamella. A thin lamella still shows some contrast variation; a void looks like blank white with no structure at all. Also check the Polish IB: a missing lamella leaves an empty gap between the trenches with no distinct thin-line signature.
- **Carbon contamination**: dark patchy areas, common in cryo samples
