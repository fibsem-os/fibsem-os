# Embedded TIFF thumbnails (deferred)

**Status:** planned, not started. Deferred as too large for now (2026-06-22).

**Goal:** speed up the per-grid Results tab (and the lamella task image widget)
when there are many images, and remove the "only the latest task's thumbnail is
kept" limitation — by storing a reduced-resolution thumbnail *inside* each image
TIFF and reading only that for display.

## Motivation / current state

- `FibsemImage.save` (`fibsem/structures.py`) writes a **single-page** TIFF via
  `tff.imwrite(path, self.data, metadata=metadata_dict)` (metadata is JSON in the
  page-0 `ImageDescription` tag).
- `FibsemImage.load` reads `TiffFile.asarray()` (full resolution) plus page-0
  metadata.
- The grid results gallery (`fibsem/ui/widgets/grid_results_widget.py`) shows
  ~150 px cells but reads the **full-resolution** artifact TIFF for each one
  (`_load_and_resize` → `FibsemImage.load` → `skimage.transform.resize`). With
  overview stitches and 1536²/4096² images this is the dominant cost.
- `GridTask._save_grid_thumbnail` writes one `<grid_dir>/thumbnail.png` that
  **every task overwrites**, so only the latest task's thumbnail persists; the
  grid card uses that single file.

Concurrency + non-blocking cancel were already added to the loader
(`_ImageLoader` in `fibsem/ui/widgets/lamella_task_image_widget.py`), which hides
the latency but does not remove the full-res decode. This change removes it.

## Feasibility — TIFF SubIFDs (confirmed)

`tifffile` supports attaching a reduced-resolution thumbnail to a page as a
**SubIFD** (the standard pyramid/thumbnail mechanism):

- `TiffFile.asarray()` / `imread()` still return the **main** image (SubIFDs are
  ignored by default), and `pages[0].tags["ImageDescription"]` is unchanged — so
  **`FibsemImage.load` keeps working untouched**.
- The thumbnail is read on its own via `series[0].levels[1].asarray()`, decoding
  only that small page — not the full image. That is the speed win.

## Plan

### 1. Save side — embed a thumbnail (`FibsemImage.save`)

```python
with tff.TiffWriter(path) as tif:
    tif.write(self.data, metadata=metadata_dict, subifds=1)
    tif.write(thumbnail, subfiletype=1)   # reduced-resolution SubIFD
```

- `thumbnail` = `self.data` downsampled to a fixed width (suggest **768 px** —
  covers both the 520 px hero and 150 px gallery without upscaling).
- Adds a small cost to every save.

### 2. Load side — a thumbnail reader

Add `FibsemImage.load_thumbnail(path) -> (array, pixel_size_x)`:

- Reads the SubIFD level if present (fast); scales `pixel_size_x` by the
  downsample factor.
- **Falls back to full-res `load()`** when there is no embedded thumbnail — this
  is the back-compat path for every TIFF saved before this change. No migration
  needed.

### 3. Display integration (`lamella_task_image_widget.py`)

- `_load_and_resize`: when the requested width ≤ embedded thumbnail size, use
  `load_thumbnail` (cheap); otherwise full `load`. Gallery (150) and hero (520)
  both hit the thumbnail path.
- The **expanded / zoom dialog keeps loading full-res** (`FibsemImage.load`) so
  zoom stays crisp.

### 4. Grid integration

- Card + results gallery read the embedded thumbnail per artifact → every
  artifact carries its own thumbnail and the "latest-only" limitation goes away.
- `_save_grid_thumbnail` / `thumbnail.png` can be dropped (or kept only as the
  card's single representative overview thumbnail).

## Decisions to make at implementation time

- **Scope:** embed in core `FibsemImage.save` (benefits lamella images too, but
  touches a hot, widely-used method) vs. a grid-only save helper. Leaning
  **core** — right home, low risk since `load` is unaffected — but bigger blast
  radius.
- **Thumbnail width:** 768 px default (one size covering hero + gallery).
- **Keep `thumbnail.png`?** Suggest dropping it for the gallery; let the card
  read the overview's embedded thumbnail (one source of truth).
- **Back-compat:** handled by the `load_thumbnail` fallback.

## Risks / notes

- Verify when implementing: `tff.imwrite(..., subifds=...)` vs the explicit
  `TiffWriter` form, and that `asarray()` on the resulting file returns **only**
  the main image in our `tifffile` version (a quick 2-line check). The
  experimental OME-TIFF save path in `FibsemImage` is separate and untouched.
- External TIFF readers that iterate pages will see the SubIFD, but most ignore
  SubIFDs; fibsem's own readers are unaffected.
- This **supersedes** the earlier "on-disk thumbnail cache" idea — embedded
  thumbnails travel with the file and need no separate cache dir / invalidation.

## Related code

- `fibsem/structures.py` — `FibsemImage.save` / `load`
- `fibsem/ui/widgets/lamella_task_image_widget.py` — `_load_and_resize`,
  `_ImageLoader`, `ExpandedImageDialog`
- `fibsem/ui/widgets/grid_results_widget.py` — gallery + hero
- `fibsem/applications/autolamella/workflows/tasks/grid/base.py` —
  `_save_grid_thumbnail`, `acquire_grid_reference_image`
