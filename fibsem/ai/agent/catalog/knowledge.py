"""The knowledge base's tools: what was learned on this microscope, kept across
experiments (`fibsem.knowledge`)."""

from typing import Any, Dict, List, Tuple

from fibsem.ai.agent.catalog.spec import ToolSpec, arg
from fibsem.util import wording

_SECTION = (
    "instrument (the state of this microscope), sample (what was learned about "
    "a sample type here), playbook (what was learned running a playbook) or "
    "experiment (a past run's record)"
)
_SAMPLE = "the sample, a short slug (yeast, hela-gfp); a sample entry needs it"
_PLAYBOOK = "the playbook's name; a playbook entry needs it"
_LASTS = (
    "until fixed (a fault, asked about again after 14 days until the operator "
    "resolves it), ongoing (true until shown otherwise) or record (a past run's, "
    "an experiment entry only)"
)


def _get_call(a: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    query: Dict[str, Any] = {"pictures": True}
    for key in ("sample", "playbook"):
        if a.get(key) is not None:
            query[key] = a[key]
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
        description="The knowledge base of this microscope, kept across experiments: one row per entry (section, id, title, when written, by the agent or the operator, status, how long it lasts, tags, its reminder, image count), newest first, and the samples and playbooks it has entries for. 'to review' entries are yours the operator hasn't kept yet: use them as you would kept ones. An entry with ask_again is an old fault: ask the operator whether it still holds, and rewrite it with what they say. Resolved entries are left out. Read the instrument entries and the run's sample and playbook entries before acting, record entries only when asked about past runs; open one with knowledge_get. Works with no experiment open.",
        method="GET",
        path="/app/knowledge",
        level="watch",
        router="app",
        params={
            "section": f"optional: only {_SECTION}",
            "sample": "optional: only entries tagged with this sample",
            "playbook": "optional: only entries tagged with this playbook",
            "status": "optional: only 'to review' or 'kept' entries",
            "lasts": f"optional: only entries that last {_LASTS}",
        },
        optional=("section", "sample", "playbook", "status", "lasts"),
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
            "id": "the entry's id, from knowledge_list; an experiment entry's is the experiment's name",
            "sample": f"optional: {_SAMPLE}",
            "playbook": f"optional: {_PLAYBOOK}",
        },
        optional=("sample", "playbook"),
        call=_get_call,
    ),
    ToolSpec(
        name="knowledge_write",
        returns="KnowledgeEntry",
        description="Write an entry in the microscope's knowledge base, kept across experiments: an instrument, sample or playbook entry when something learned the hard way would change the next run, or when the operator tells you something to remember (quote them); the run's record at the end of a run. Short and specific, with what to do differently and the evidence. Your entries are 'to review' until the operator keeps them, and apply straight away. Giving an id rewrites that entry: its title and text replaced, images added, marked edited, 'to review' again. A new entry's id is the date and the title's slug. An experiment entry's id is the experiment's name, the open one's when left out. Images are copied in, so the entry still reads once the experiment is gone. Answers the entry as written.",
        method="POST",
        path="/app/knowledge",
        level="watch",  # Watch writes knowledge: it changes nothing on the microscope
        router="app",
        params={
            "section": _SECTION,
            "title": "a one-line title, at most 200 characters",
            "text": "the entry, Markdown, at most 20000 characters",
            "sample": f"optional: {_SAMPLE}; any entry may be tagged with the run's sample",
            "playbook": f"optional: {_PLAYBOOK}; any entry may be tagged with the run's playbook",
            "lasts": f"optional: {_LASTS}; ongoing when left out",
            "remind": "optional: one line said before the steps it concerns (e.g. 'the 10 pA aperture is down: use 30 pA'); an empty one removes it",
            "images": arg(
                "optional: up to 8 image paths in the open experiment (a figure you drew, an image the image tools answered), copied in at most 1024 px; refused with no experiment open",
                List[str],
            ),
            "item": "optional: the lamella or grid it is about, by name",
            "id": "optional: the entry to rewrite, from knowledge_list",
        },
        optional=("sample", "playbook", "lasts", "remind", "images", "item", "id"),
        line=_write_line,
    ),
)
