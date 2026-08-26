"""The document a grid travels to the TEM with.

The person who loads the grid into the TEM is usually not the person who milled it, and
never has fibsemOS installed. What they get is whatever was exported before the operator
went home, so this document is the entire downstream interface -- which is why it is a
PDF rather than anything that needs software to read.

Three pages, in the order the questions get asked:

1. **The map.** Where the lamellae are on the grid, marked and named, with a scalebar and
   a provenance line. One page per *view*: a fluorescence overview and a beam overview
   are pictures taken from different directions -- different stage tilt, different
   instrument -- so they cannot be composited, and putting them on one page would imply
   they could. See `Experiment.find_overview_images`.
2. **The table.** Every lamella with the facts that decide whether to spend beam time on
   it: how thick, how wide, at what angle, when it finished, whether anyone flagged it.
3. **The cards.** One per lamella, with its final ion-beam image, so it can be recognised
   in an atlas before a hole is committed to it.

Everything here is read from the experiment record. Nothing is inferred: in particular a
lamella is *not* called ready, finished or good anywhere, because the only judgement in
the record is `defect.state`, which a human sets by hand. A lamella whose polishing task
never ran is unfinished, which is a different claim from defective, and this document
makes neither on anyone's behalf -- it reports the last task that completed and lets the
reader draw the conclusion.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from fibsem.applications.autolamella.structures import (
    DefectType,
    Experiment,
    Lamella,
)
from fibsem.constants import DATETIME_DISPLAY_SHORT, SI_TO_MICRO, SI_TO_NANO

logger = logging.getLogger(__name__)

# What the defect states are called on the page. The record's own names (NONE, FAILURE,
# REWORK) are the wrong register for a document someone reads once, at a microscope.
DEFECT_LABELS = {
    DefectType.NONE: "-",
    DefectType.REWORK: "rework",
    DefectType.FAILURE: "failed",
}


@dataclass
class HandoffOptions:
    """What goes in the document, and what it is called.

    `grid` and `slot` are free text and deliberately so. Nothing in the codebase records
    which physical grid a lamella is on or which cassette slot it travels in -- there is
    no grid record yet -- and for a single-grid experiment the experiment *is* the grid,
    so a typed-in string carries the whole of what is known. They are stored on
    `Experiment.metadata` so they survive the dialog closing, and a real grid record can
    later fill the same two header fields without the page changing.
    """

    title: str = ""
    note: str = ""
    grid: str = ""
    slot: str = ""
    include_map: bool = True
    include_table: bool = True
    include_cards: bool = True
    # None means every lamella. A list of names means only those, in the experiment's
    # own order -- so a map of the four that survived is a matter of unticking the rest.
    lamella_names: Optional[List[str]] = None
    marker_color: str = "cyan"
    show_descriptions: bool = True
    dpi: int = 200
    extra: Dict[str, Any] = field(default_factory=dict)

    def selected(self, experiment: Experiment) -> List[Lamella]:
        """The lamellae this document covers."""
        if self.lamella_names is None:
            return list(experiment.positions)
        wanted = set(self.lamella_names)
        return [lam for lam in experiment.positions if lam.name in wanted]


def _final_trench_pattern(lamella: Lamella):
    """The pattern whose gap *is* the lamella, or None.

    Walks the completed tasks backwards and returns the last stage of the last one whose
    final pattern has a `spacing` -- that gap being the lamella itself, left between the
    two trenches cut either side of it.

    The `spacing` requirement is the whole of the discrimination, and it is load-bearing.
    "The last completed milling task" is not good enough: on a real experiment that is
    routinely `Mill Fiducial`, and a fiducial pattern has a `width` like a trench does.
    Reading it produced a straight-faced "lamella width: 1.0 um" for a lamella whose
    milling had barely started. A pattern with no spacing describes something that is not
    a lamella, so it is skipped rather than read.
    """
    completed = [t.name for t in lamella.task_history if t.name in lamella.task_config]
    for name in reversed(completed):
        config = lamella.task_config.get(name)
        if config is None or not config.milling:
            continue

        stages = []
        for milling_task in config.milling.values():
            stages.extend(
                [s for s in milling_task.stages if getattr(s, "enabled", True)]
            )
        if not stages:
            continue

        pattern = getattr(stages[-1], "pattern", None)
        if pattern is not None and getattr(pattern, "spacing", None):
            return pattern
    return None


def final_geometry(lamella: Lamella) -> Dict[str, Optional[float]]:
    """The lamella's final thickness and width, in metres, or Nones.

    Read off the last trench milled into it: the pattern's `spacing` is the gap left
    between its two trenches -- the lamella -- and its `width` is how wide that gap was
    cut.

    Nones rather than guesses when no trench has been milled yet. A dash on the page is
    honest; a number lifted from a fiducial is not.
    """
    pattern = _final_trench_pattern(lamella)
    if pattern is None:
        return {"thickness": None, "width": None}
    return {
        "thickness": getattr(pattern, "spacing", None),
        "width": getattr(pattern, "width", None),
    }


def lamella_row(lamella: Lamella) -> Dict[str, str]:
    """One lamella as the strings that go in the table.

    Formatting lives here rather than in the PDF layer so the same row can back a CSV or
    a card without the numbers being rendered twice, differently.
    """
    geometry = final_geometry(lamella)
    last = lamella.last_completed_task

    thickness = geometry["thickness"]
    width = geometry["width"]
    angle = lamella.milling_angle

    finished = "-"
    if last is not None and getattr(last, "end_timestamp", None):
        try:
            finished = datetime.fromtimestamp(last.end_timestamp).strftime(
                DATETIME_DISPLAY_SHORT
            )
        except (ValueError, OSError, TypeError):
            finished = "-"

    fm_count = 0
    for task in lamella.task_history:
        fm_count += len((task.outputs or {}).get("fluorescence", []))

    return {
        "Lamella": lamella.name,
        "Defect": DEFECT_LABELS.get(lamella.defect.state, "-"),
        "Last task": last.name if last is not None else "-",
        "Thickness": f"{thickness * SI_TO_NANO:.0f} nm" if thickness else "-",
        "Width": f"{width * SI_TO_MICRO:.1f} um" if width else "-",
        "Angle": f"{angle:.1f} deg" if angle is not None else "-",
        "Finished": finished,
        "FM": str(fm_count) if fm_count else "-",
        "Note": lamella.description or lamella.defect.description or "",
    }


def provenance_line(experiment: Experiment) -> str:
    """Instrument, operator, version and date, for the header of every page.

    Empty when no session has adopted the experiment -- which is every experiment written
    before the session record existed. A header that admits it knows nothing beats one
    filled with "unknown", which reads as a measurement.
    """
    session = getattr(experiment, "session", None)
    parts: List[str] = []
    if session is not None:
        system = getattr(session, "system", None)
        user = getattr(session, "user", None)
        model = getattr(system, "model", "") or ""
        serial = getattr(system, "serial_number", "") or ""
        if model:
            parts.append(f"{model} ({serial})" if serial else model)
        operator = getattr(user, "name", "") or ""
        if operator:
            parts.append(operator)
        version = getattr(system, "fibsem_version", "") or ""
        if version:
            parts.append(f"fibsemOS {version}")

    created = getattr(experiment, "created_at", None)
    if created:
        try:
            parts.append(
                "milled "
                + datetime.fromtimestamp(created).strftime(DATETIME_DISPLAY_SHORT)
            )
        except (ValueError, OSError, TypeError):
            pass

    if not parts:
        return ""
    parts.append("map " + datetime.now().strftime(DATETIME_DISPLAY_SHORT))
    return "  |  ".join(parts)


def summary_line(experiment: Experiment, options: HandoffOptions) -> str:
    """The one line under the title: what is on this grid, and where it is.

    Counts only what a human flagged. There is deliberately no "N ready" here: readiness
    is not recorded anywhere, and inventing it from task history would put a judgement
    nobody made onto the page that travels with the sample.
    """
    lamellae = options.selected(experiment)
    failed = sum(1 for lam in lamellae if lam.defect.state is DefectType.FAILURE)
    rework = sum(1 for lam in lamellae if lam.defect.state is DefectType.REWORK)

    parts: List[str] = []
    if options.grid:
        parts.append(f"Grid {options.grid}")
    if options.slot:
        parts.append(f"slot {options.slot}")
    parts.append(f"{len(lamellae)} lamellae")
    if rework:
        parts.append(f"{rework} rework")
    if failed:
        parts.append(f"{failed} failed")
    return "  |  ".join(parts)


def generate_handoff_map(
    experiment: Experiment,
    output_filename: str,
    options: Optional[HandoffOptions] = None,
) -> str:
    """Write the handoff document for *experiment* to *output_filename*.

    No Qt anywhere in here, deliberately: the artifact is most wanted at the end of a run
    that nobody stayed for, so it has to be reachable from a workflow hook and not only
    from a dialog someone remembered to open.

    Returns:
        The path written.
    """
    # Imported here rather than at module scope so that reading a lamella's geometry --
    # which the dialog does on every keystroke -- does not cost a reportlab import.
    from reportlab.lib.units import inch

    from fibsem.applications.autolamella.tools.reporting import PDFReportGenerator

    options = options or HandoffOptions()
    lamellae = options.selected(experiment)

    pdf = PDFReportGenerator(output_filename=output_filename)
    pdf.add_title(
        options.title or f"Handoff Map: {experiment.name}",
        summary_line(experiment, options),
    )

    provenance = provenance_line(experiment)
    if provenance:
        pdf.add_paragraph(provenance)
    if options.note:
        pdf.add_paragraph(options.note)

    if options.include_map:
        _add_map_pages(pdf, experiment, lamellae, options, inch)

    if options.include_table:
        _add_table_page(pdf, lamellae)

    if options.include_cards:
        _add_card_pages(pdf, lamellae, inch)

    pdf.generate()
    logger.info(f"Wrote handoff map for {experiment.name} to {output_filename}")
    return output_filename


def _add_map_pages(pdf, experiment, lamellae, options: HandoffOptions, inch) -> None:
    """A page per beam overview, each marked with the selected lamellae.

    A page *per overview* rather than one page with everything on it. Two overviews of
    the same grid at different times are two pictures, and a fluorescence overview is a
    picture from a different direction entirely -- so there is no single set of axes they
    all belong on.
    """
    from fibsem.imaging.tiled import (
        DEFECT_FAILURE_COLOUR,
        DEFECT_REWORK_COLOUR,
        plot_minimap,
    )
    from fibsem.structures import FibsemImage

    paths = experiment.find_overview_images()
    if not paths:
        pdf.add_page_break()
        pdf.add_heading("Map")
        pdf.add_paragraph(
            "No overview image was saved with this experiment, so there is no map. "
            "The table below still locates each lamella by stage position."
        )
        return

    selected = {lam.name for lam in lamellae}
    positions = [
        pos for pos in experiment.get_milling_positions() if pos.name in selected
    ]
    descriptions = {lam.name: lam.description for lam in lamellae}
    colours = {}
    for lam in lamellae:
        if lam.defect.state is DefectType.FAILURE:
            colours[lam.name] = DEFECT_FAILURE_COLOUR
        elif lam.defect.state is DefectType.REWORK:
            colours[lam.name] = DEFECT_REWORK_COLOUR

    for path in paths:
        try:
            image = FibsemImage.load(path)
        except Exception as e:
            logger.warning(f"Could not load the overview {path}: {e}")
            continue
        try:
            fig = plot_minimap(
                image,
                positions,
                color=options.marker_color,
                colors=colours,
                show_scalebar=True,
                show_names=True,
                show_descriptions=options.show_descriptions,
                descriptions=descriptions,
                figsize=None,
            )
        except Exception as e:
            logger.warning(f"Could not draw the map for {path}: {e}")
            continue

        pdf.add_page_break()
        pdf.add_heading(f"Map - {os.path.basename(path)}")
        pdf.add_mpl_figure(fig, width=6.5 * inch, height=4.0 * inch)

    fm_paths = experiment.find_fluorescence_overview_images()
    if fm_paths:
        # Named rather than drawn. They belong to a different view and cannot be marked
        # on these axes, but a reader who knows a fluorescence overview was acquired
        # should not have to wonder where it went.
        pdf.add_paragraph(
            f"{len(fm_paths)} fluorescence overview(s) were also acquired, and are not "
            "shown here: they are a different view of the sample and do not register "
            "with a beam overview. Files: "
            + ", ".join(os.path.basename(p) for p in fm_paths)
        )


def _add_table_page(pdf, lamellae) -> None:
    """Every lamella, and the numbers that decide whether to spend beam time on it."""
    import pandas as pd

    pdf.add_page_break()
    pdf.add_heading("Lamellae")
    if not lamellae:
        pdf.add_paragraph("No lamellae were selected for this map.")
        return
    rows = [lamella_row(lam) for lam in lamellae]
    pdf.add_dataframe(pd.DataFrame(rows))


def _add_card_pages(pdf, lamellae, inch) -> None:
    """One card per lamella: its final ion-beam image, so it can be recognised.

    The ion beam rather than the electron beam: the ion image is the view the lamella was
    milled in, and so the one that shows the trenches, the tabs and the curtaining a
    reader is checking for.
    """
    from fibsem.structures import FibsemImage

    if not lamellae:
        return

    pdf.add_page_break()
    pdf.add_heading("Lamella detail")

    for lam in lamellae:
        row = lamella_row(lam)
        pdf.add_heading(lam.name, level=3)
        pdf.add_paragraph(
            " | ".join(
                f"{key}: {row[key]}"
                for key in ("Defect", "Last task", "Thickness", "Width", "Angle", "FM")
            )
        )
        if row["Note"]:
            pdf.add_paragraph(row["Note"])

        path = _final_ion_image(lam)
        if path is None:
            pdf.add_paragraph("No final ion-beam image was recorded for this lamella.")
            continue
        try:
            image = FibsemImage.load(path)
        except Exception as e:
            logger.warning(f"Could not load {path} for {lam.name}: {e}")
            continue
        fig = _plot_bare_image(image)
        pdf.add_mpl_figure(fig, width=4.0 * inch, height=3.0 * inch)


def _final_ion_image(lamella: Lamella) -> Optional[str]:
    """The last final ion-beam reference image the lamella recorded, or None.

    From `task_history[].outputs`, which records paths relative to the lamella directory
    -- which is where they are, so they resolve against it as-is.
    """
    for task in reversed(lamella.task_history):
        names = (task.outputs or {}).get("final_fib", [])
        for name in reversed(names):
            path = os.path.join(str(lamella.path), name)
            if os.path.exists(path):
                return path
    return None


def _plot_bare_image(image) -> "Any":
    """The image, a scalebar, and nothing else."""
    import matplotlib.pyplot as plt

    from fibsem.imaging.tiling.plotting import figsize_for_image

    fig, ax = plt.subplots(figsize=figsize_for_image(image.data.shape, width_in=6.0))
    ax.imshow(image.data, cmap="gray")
    ax.axis("off")
    try:
        from matplotlib_scalebar.scalebar import ScaleBar

        ax.add_artist(
            ScaleBar(
                dx=image.metadata.pixel_size.x,
                color="black",
                box_color="white",
                box_alpha=0.5,
                location="lower right",
            )
        )
    except Exception as e:
        logger.debug(f"Could not add a scalebar: {e}")
    fig.tight_layout()
    return fig
