"""The knowledge base's tools: what was learned on this microscope, kept across
experiments (`fibsem.knowledge`)."""

from typing import Any, Dict, List, Tuple

from fibsem.ai.agent.catalog.spec import ToolSpec, arg
from fibsem.util import wording

_SECTION = (
    "instrument (the state of this microscope), sample (what was learned about "
    "a sample type here) or experiment (a completed experiment's summary)"
)
_SAMPLE = "the sample, a short slug (yeast, hela-gfp); a sample entry needs it"


def _get_call(a: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    query: Dict[str, Any] = {"pictures": True}
    if a.get("sample") is not None:
        query["sample"] = a["sample"]
    return {"section": a["section"], "id": a["id"]}, query


def _write_line(a: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    verb = "Update" if a.get("id") else "Write"
    return {
        "what": "Knowledge",
        "change": f"{verb} the {a.get('section')} entry: {a.get('title')}",
    }


TOOLS: Tuple[ToolSpec, ...] = (
    ToolSpec(
        name="knowledge_list",
        returns="KnowledgeIndex",
        description="The knowledge base of this microscope, kept across experiments: one row per entry (section, sample, id, title, when written, by the agent or the operator, image count), newest first, and the sample slugs it has. Read the instrument entries and the run's sample entries before acting; open one with knowledge_get. Works with no experiment open.",
        method="GET",
        path="/app/knowledge",
        level="watch",
        router="app",
        params={
            "section": f"optional: only {_SECTION}",
            "sample": "optional: only this sample's entries",
        },
        optional=("section", "sample"),
    ),
    ToolSpec(
        name="knowledge_get",
        returns="KnowledgeEntry",
        description="One knowledge entry: its text and front matter, with its images as pictures you can look at. Ids are from knowledge_list. Works with no experiment open.",
        method="GET",
        path="/app/knowledge/{section}/{id}",
        level="watch",
        router="app",
        params={
            "section": _SECTION,
            "id": "the entry's id, from knowledge_list; an experiment summary's is the experiment's name",
            "sample": f"optional: {_SAMPLE}",
        },
        optional=("sample",),
        call=_get_call,
    ),
    ToolSpec(
        name="knowledge_write",
        returns="KnowledgeEntry",
        description="Write an entry in the microscope's knowledge base, kept across experiments: an instrument or sample entry when something learned the hard way would change the next run, or whenever the operator asks; the experiment's summary at the end of a run. Short and specific, with what to do differently. Giving an id rewrites that entry: its title and text replaced, images added, marked edited. A new entry's id is the date and the title's slug. An experiment summary's id is the experiment's name, the open one's when left out. Answers the entry as written.",
        method="POST",
        path="/app/knowledge",
        level="watch",  # Watch writes knowledge: it changes nothing on the microscope
        router="app",
        params={
            "section": _SECTION,
            "title": "a one-line title, at most 200 characters",
            "text": "the entry, Markdown, at most 20000 characters",
            "sample": f"optional: {_SAMPLE}",
            "images": arg(
                "optional: up to 8 image paths in the open experiment (a figure you drew, an image the image tools answered), copied in at most 1024 px; refused with no experiment open",
                List[str],
            ),
            "id": "optional: the entry to rewrite, from knowledge_list",
        },
        optional=("sample", "images", "id"),
        line=_write_line,
    ),
)
