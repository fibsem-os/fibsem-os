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
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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

# What a geometry column says when no trench has been cut yet. Spelled out rather than
# left as a dash: a dash in a numeric column reads as a value that failed to render,
# and this document is read by someone with no way to check.
NOT_MILLED = "not milled"


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
    # None means every overview the experiment holds. A list of paths means only those.
    overview_paths: Optional[List[str]] = None
    # Which images go on each lamella's card. All three by default: they answer
    # different questions -- the ion view shows how it was cut, the electron view shows
    # it from above, and the fluorescence projection is the only one that says whether
    # the thing worth imaging is actually inside it.
    include_ion_image: bool = True
    include_electron_image: bool = True
    include_fluorescence_image: bool = True
    marker_color: str = "cyan"
    show_descriptions: bool = True
    dpi: int = 200
    extra: Dict[str, Any] = field(default_factory=dict)

    def selected_overviews(self, experiment: Experiment) -> List[str]:
        """The overview files this document draws."""
        available = experiment.find_overview_images()
        if self.overview_paths is None:
            return available
        wanted = set(self.overview_paths)
        return [p for p in available if p in wanted]

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

    # Column order is the order the questions get asked: which lamella, how far did it
    # get, when, then its geometry, then the one judgement anyone recorded, then the
    # free text. Fluorescence is deliberately not a column any more -- a count of
    # stacks told the reader nothing they could act on, and the projection itself is
    # on the lamella's card, which is where "is my target in it" is actually answered.
    return {
        "Lamella": lamella.name,
        "Last task": last.name if last is not None else "-",
        "Finished": finished,
        "Angle": f"{angle:.1f} deg" if angle is not None else "-",
        "Thickness": f"{thickness * SI_TO_NANO:.0f} nm" if thickness else NOT_MILLED,
        "Width": f"{width * SI_TO_MICRO:.1f} um" if width else NOT_MILLED,
        "Defect": DEFECT_LABELS.get(lamella.defect.state, "-"),
        "Note": lamella.description or lamella.defect.description or "",
    }


# The order the table draws, and which of those read as numbers.
TABLE_COLUMNS = (
    "Lamella",
    "Last task",
    "Finished",
    "Angle",
    "Thickness",
    "Width",
    "Defect",
    "Note",
)
NUMERIC_COLUMNS = ("Finished", "Angle", "Thickness", "Width")


def fluorescence_stacks(lamella: Lamella) -> List[str]:
    """Every fluorescence stack this lamella recorded, newest last."""
    out: List[str] = []
    for task in lamella.task_history:
        out.extend((task.outputs or {}).get("fluorescence", []))
    return out


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
    nobody stayed for, so it has to be reachable from a workflow hook and not only from a
    dialog someone remembered to open.

    Returns:
        The path written.
    """
    from fibsem.applications.autolamella.tools.handoff_document import HandoffDocument

    options = options or HandoffOptions()
    lamellae = options.selected(experiment)

    doc = HandoffDocument(output_filename)
    doc.masthead(
        name=options.title or experiment.name,
        summary=summary_line(experiment, options),
        provenance=provenance_line(experiment),
    )
    doc.note(options.note)

    if options.include_map:
        _add_map_pages(doc, experiment, lamellae, options)

    if options.include_table:
        doc.page_break()
        doc.heading("Lamellae")
        doc.table(
            TABLE_COLUMNS,
            [lamella_row(lam) for lam in lamellae],
            numeric=NUMERIC_COLUMNS,
        )

    if options.include_cards:
        _add_card_pages(doc, lamellae, options)

    doc.build()
    logger.info(f"Wrote handoff map for {experiment.name} to {output_filename}")
    return output_filename


def view_key(image) -> Tuple[str, int, int]:
    """What decides whether two overviews may share a page.

    The beam, and the stage orientation to the nearest degree. Two images register with
    each other only if they were taken through the same beam with the stage in the same
    place; anything else is a picture from a different direction, and compositing them
    would put real pixels at coordinates they were never acquired at -- which would look
    exactly as authoritative as a correct page.

    Rounded degrees rather than the orientation's *name* (SEM / FIB / MILLING), because
    naming it needs `FibsemMicroscope.get_stage_orientation` and this runs from a hook
    with no microscope. FIB-811 makes the name derivable; this becomes it when it lands.
    """
    import math

    beam = getattr(getattr(image.metadata, "image_settings", None), "beam_type", None)
    state = getattr(image.metadata, "microscope_state", None)
    position = getattr(state, "stage_position", None)
    r = getattr(position, "r", None) or 0.0
    t = getattr(position, "t", None) or 0.0
    name = getattr(beam, "name", None) or str(beam)
    return (name, int(round(math.degrees(r))), int(round(math.degrees(t))))


def view_label(key: Tuple[str, int, int]) -> str:
    """How a view is named on the page, given that its real name is unavailable."""
    beam, r, t = key
    beam_name = {"ELECTRON": "Electron beam", "ION": "Ion beam"}.get(beam, str(beam))
    return f"{beam_name}  -  stage r {r} deg, t {t} deg"


def _add_map_pages(doc, experiment, lamellae, options: "HandoffOptions") -> None:
    """One page per *view*, with every selected overview of that view composited on it.

    Per view rather than per file, because several overviews of one view are several
    pictures of one thing: a re-acquired overview, or two covering different parts of the
    same grid. Given one page each, the reader has to stitch them mentally and has
    nothing telling them which is current. Composited, a lamella that falls off the edge
    of one overview but inside another is simply on the page.
    """
    from fibsem.imaging.tiled import (
        DEFECT_FAILURE_COLOUR,
        DEFECT_REWORK_COLOUR,
        plot_overview_composite,
    )
    from fibsem.structures import FibsemImage

    paths = options.selected_overviews(experiment)
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

    if not paths:
        doc.heading("Map")
        doc.note(
            "No overview image was saved with this experiment, so there is no map. "
            "The table below still locates each lamella by stage position."
        )
        return

    # Grouped in the order the files were found, so the groups themselves come out in
    # acquisition order rather than in whatever order a dict happened to build.
    groups: "OrderedDict[Tuple[str, int, int], List[Any]]" = OrderedDict()
    for path in paths:
        try:
            image = FibsemImage.load(path)
        except Exception as e:
            logger.warning(f"Could not load the overview {path}: {e}")
            continue
        groups.setdefault(view_key(image), []).append((path, image))

    first = True
    for key, entries in groups.items():
        images = [image for _, image in entries]
        try:
            fig = plot_overview_composite(
                images,
                positions,
                color=options.marker_color,
                colors=colours,
                descriptions=descriptions,
                show_names=True,
                show_descriptions=options.show_descriptions,
                show_scalebar=True,
                figsize=None,
            )
        except Exception as e:
            logger.warning(f"Could not draw the map for {view_label(key)}: {e}")
            continue

        if not first:
            doc.page_break()
        first = False

        doc.heading(f"Map  -  {view_label(key)}")
        doc.figure(fig)

        marked = count_marked(images, positions)
        detail = f"{len(images)} overview(s) composited"
        if positions:
            # Said explicitly, because a page can legitimately come out with no markers
            # on it -- an overview of ground where no lamella was placed. Without this
            # an empty map reads as a rendering failure.
            detail += f"  -  {marked} of {len(positions)} selected lamellae fall here"
        detail += "  -  " + ", ".join(os.path.basename(p) for p, _ in entries)
        doc.caption(detail)


def count_marked(images, positions) -> int:
    """How many of *positions* land inside any of *images*.

    Computed against the images rather than read off the figure, so the caption cannot
    disagree with the picture by being derived from something else.
    """
    from fibsem.conversions import is_inside_image_bounds
    from fibsem.imaging.tiled import reproject_stage_positions_onto_image2

    seen = set()
    for image in images:
        try:
            points = reproject_stage_positions_onto_image2(image, positions)
        except Exception:
            continue
        shape = (image.data.shape[0], image.data.shape[1])
        for pt in points:
            if is_inside_image_bounds((pt.y, pt.x), shape):
                seen.add(pt.name)
    return len(seen)


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


def _add_card_pages(doc, lamellae, options: "HandoffOptions") -> None:
    """One card per lamella: a fact line, then whichever images were asked for."""
    if not lamellae:
        return

    doc.page_break()
    doc.heading("Lamella detail")

    for i, lam in enumerate(lamellae):
        row = lamella_row(lam)
        facts = [
            ("Defect", row["Defect"] if row["Defect"] != "-" else "none flagged"),
            ("Last task", row["Last task"]),
            ("Thickness", row["Thickness"]),
            ("Width", row["Width"]),
            ("Angle", row["Angle"]),
        ]
        doc.lamella_card(
            name=lam.name,
            facts=facts,
            note=row["Note"],
            shots=_card_shots(lam, options),
            first=(i == 0),
        )


def _card_shots(lamella: Lamella, options: "HandoffOptions"):
    """The images for one card, as (label, figure or None, why-not) triples.

    A modality that was never acquired keeps its slot and says so. Closing up would make
    a two-image card look like a short one, and "not acquired" would be indistinguishable
    from "did not render".
    """
    from fibsem.applications.autolamella.tools.handoff_document import (
        bare_image_figure,
        fluorescence_figure,
    )

    shots = []

    if options.include_ion_image:
        # The ion view first: it is the one the lamella was milled in, so it shows the
        # trenches, the tabs and the curtaining a reader is checking for.
        shots.append(_beam_shot(lamella, "final_fib", "Ion beam", bare_image_figure))
    if options.include_electron_image:
        shots.append(
            _beam_shot(lamella, "final_sem", "Electron beam", bare_image_figure)
        )
    if options.include_fluorescence_image:
        shots.append(_fluorescence_shot(lamella, fluorescence_figure))

    return shots


def _beam_shot(lamella: Lamella, key: str, label: str, make_figure):
    from fibsem.structures import FibsemImage

    path = latest_output(lamella, key)
    if path is None:
        return (label, None, "Not acquired")
    try:
        return (label, make_figure(FibsemImage.load(path)), "")
    except Exception as e:
        logger.warning(f"Could not render {path} for {lamella.name}: {e}")
        return (label, None, "Could not be read")


def _fluorescence_shot(lamella: Lamella, make_figure):
    """The newest fluorescence stack, blended the way the application blends it."""
    paths = fluorescence_stacks(lamella)
    if not paths:
        return ("Fluorescence, max proj.", None, "No fluorescence acquired")
    path = os.path.join(str(lamella.path), paths[-1])
    if not os.path.exists(path):
        return ("Fluorescence, max proj.", None, "Not acquired")
    try:
        from fibsem.fm.structures import FluorescenceImage

        return (
            "Fluorescence, max proj.",
            make_figure(FluorescenceImage.load(path)),
            "",
        )
    except Exception as e:
        logger.warning(f"Could not render {path} for {lamella.name}: {e}")
        return ("Fluorescence, max proj.", None, "Could not be read")


def latest_output(lamella: Lamella, key: str) -> Optional[str]:
    """The last file recorded under *key* that is actually on disk, or None."""
    for task in reversed(lamella.task_history):
        for name in reversed((task.outputs or {}).get(key, [])):
            path = os.path.join(str(lamella.path), name)
            if os.path.exists(path):
                return path
    return None
