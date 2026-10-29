"""The knowledge base's store (`fibsem.knowledge`), under a state root of its own."""

from pathlib import Path

import numpy as np
import pytest
import tifffile
import yaml
from PIL import Image

from fibsem import knowledge


@pytest.fixture(autouse=True)
def state_root(tmp_path, monkeypatch):
    monkeypatch.setenv("FIBSEM_HOME", str(tmp_path / "state"))
    return tmp_path / "state"


def _front(path):
    _, front, text = path.read_text(encoding="utf-8").split("---\n", 2)
    return yaml.safe_load(front), text


def test_an_entry_is_written_under_the_state_root_and_read_back(state_root):
    entry = knowledge.write(
        "instrument", "10 pA aperture worn", "Use 30 pA for polishing.", "agent"
    )
    assert entry.id.endswith("-10-pa-aperture-worn")
    path = state_root / "knowledge" / "instrument" / entry.id / "entry.md"
    assert entry.path == str(path)
    front, text = _front(path)
    assert front["title"] == "10 pA aperture worn"
    assert front["by"] == "agent" and front["images"] == []
    assert "edited" not in front
    assert text == "Use 30 pA for polishing.\n"

    read = knowledge.read("instrument", entry.id)
    assert read == entry
    assert read.written == front["written"]


def test_a_sample_entry_is_kept_under_its_sample_and_filtered_by_it(state_root):
    yeast = knowledge.write("sample", "Mills fast", "1.4x", "agent", sample="Yeast")
    knowledge.write("sample", "Thin", "x", "operator", sample="hela-gfp")
    knowledge.write("instrument", "Stage drifts", "x", "agent")
    assert yeast.sample == "yeast"
    assert (state_root / "knowledge" / "sample" / "yeast" / yeast.id).is_dir()
    assert knowledge.samples() == ["hela-gfp", "yeast"]

    assert [r.title for r in knowledge.entries("sample", "yeast")] == ["Mills fast"]
    assert len(knowledge.entries()) == 3
    assert [r.section for r in knowledge.entries("instrument")] == ["instrument"]
    with pytest.raises(knowledge.KnowledgeError, match="names its sample"):
        knowledge.write("sample", "No sample", "x", "agent")
    with pytest.raises(knowledge.KnowledgeError, match="Only a sample entry"):
        knowledge.write("instrument", "x", "x", "agent", sample="yeast")


def test_the_index_is_newest_first():
    first = knowledge.write("instrument", "First", "x", "agent")
    second = knowledge.write("instrument", "Second", "x", "agent")
    path = Path(knowledge.read("instrument", first.id).path)
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(first.written, "2020-01-01T00:00:00+00:00"))
    assert [r.id for r in knowledge.entries()] == [second.id, first.id]


def test_the_same_title_on_the_same_day_gets_its_own_id():
    one = knowledge.write("instrument", "Drift", "a", "agent")
    two = knowledge.write("instrument", "Drift", "b", "agent")
    assert two.id == f"{one.id}-2"


def test_a_rewrite_keeps_its_author_and_is_marked_edited_by_who_rewrote_it():
    entry = knowledge.write("instrument", "Drift", "first", "agent")
    edited = knowledge.write(
        "instrument", "Drift, again", "second", "operator", id=entry.id
    )
    assert edited.id == entry.id
    assert (edited.title, edited.text) == ("Drift, again", "second")
    assert edited.by == "agent" and edited.written == entry.written
    assert edited.edited and edited.edited_by == "operator"
    front, _ = _front(Path(knowledge.read("instrument", entry.id).path))
    assert front["edited"] is True and front["edited_by"] == "operator"

    with pytest.raises(knowledge.KnowledgeError) as missing:
        knowledge.write("instrument", "x", "x", "agent", id="2026-01-01-nothing")
    assert missing.value.error_type == "not_found"


def test_an_experiment_summary_takes_the_experiment_as_its_id(state_root):
    entry = knowledge.write(
        "experiment", "Grid-03, 27 Oct", "3 of 7 polished.", "agent", id="Grid-03"
    )
    assert (
        state_root / "knowledge" / "experiment" / "Grid-03" / "summary.md"
    ).is_file()
    assert entry.experiment == "Grid-03"
    again = knowledge.write("experiment", "Grid-03", "4 of 7.", "agent", id="Grid-03")
    assert again.text == "4 of 7." and again.edited
    with pytest.raises(knowledge.KnowledgeError, match="experiment's name"):
        knowledge.write("experiment", "No id", "x", "agent")


@pytest.mark.parametrize("bad", ["../up", "a/b", "..", ".hidden", "", "x" * 200])
def test_an_id_is_one_name_never_a_path(bad):
    with pytest.raises(knowledge.KnowledgeError) as refused:
        knowledge.read("instrument", bad)
    assert refused.value.error_type == "invalid"


def test_images_are_copied_in_downsized_to_1024_px(tmp_path):
    png = tmp_path / "Drawn Figure.png"
    Image.fromarray(np.zeros((1500, 3000, 3), np.uint8)).save(png)
    tif = tmp_path / "beam.tif"
    tifffile.imwrite(tif, (np.random.rand(2048, 1536) * 60000).astype(np.uint16))

    entry = knowledge.write("instrument", "Spot", "x", "agent", images=[png, tif])
    assert entry.images == ["drawn-figure.png", "beam.png"]
    sizes = []
    for name in entry.images:
        with Image.open(knowledge.image("instrument", entry.id, name)) as kept:
            sizes.append(kept.size)
    assert sizes == [(1024, 512), (768, 1024)]
    # A rewrite adds images to those it has.
    again = knowledge.write(
        "instrument", "Spot", "y", "agent", id=entry.id, images=[png]
    )
    assert again.images == ["drawn-figure.png", "beam.png", "drawn-figure-2.png"]
    assert knowledge.entries()[0].images == 3

    with pytest.raises(knowledge.KnowledgeError, match="no image"):
        knowledge.image("instrument", entry.id, "other.png")


def test_a_file_that_is_not_an_image_is_refused_and_nothing_is_written(tmp_path):
    bad = tmp_path / "notes.png"
    bad.write_text("not a picture")
    with pytest.raises(knowledge.KnowledgeError, match="can't be read"):
        knowledge.write("instrument", "Spot", "x", "agent", images=[bad])
    assert knowledge.entries() == []


def test_delete_removes_the_entry_and_an_emptied_sample(state_root):
    entry = knowledge.write("sample", "Fast", "x", "agent", sample="yeast")
    removed = knowledge.delete("sample", entry.id, "yeast")
    assert removed.title == "Fast"
    assert not (state_root / "knowledge" / "sample" / "yeast").exists()
    assert knowledge.entries() == []
    with pytest.raises(knowledge.KnowledgeError) as missing:
        knowledge.delete("sample", entry.id, "yeast")
    assert missing.value.error_type == "not_found"


def test_a_hand_written_entry_reads_with_what_it_has(state_root):
    folder = state_root / "knowledge" / "instrument" / "2026-10-01-by-hand"
    folder.mkdir(parents=True)
    (folder / "entry.md").write_text(
        "---\ntitle: By hand\nwritten: 2026-10-01\nby: operator\n---\nWritten in an editor.\n"
    )
    (state_root / "knowledge" / "instrument" / "no-front").mkdir()
    (state_root / "knowledge" / "instrument" / "no-front" / "entry.md").write_text(
        "Just text."
    )
    by_hand = knowledge.read("instrument", "2026-10-01-by-hand")
    assert (by_hand.title, by_hand.written, by_hand.text) == (
        "By hand",
        "2026-10-01",
        "Written in an editor.",
    )
    plain = knowledge.read("instrument", "no-front")
    assert (plain.title, plain.text, plain.by) == ("no-front", "Just text.", "")
    assert len(knowledge.entries()) == 2
