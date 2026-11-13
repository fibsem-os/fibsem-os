"""What an event record says (fibsem.workflows.event_text), and the App Server
sending it with each record."""

from fibsem.workflows.event_text import describe, note_asked, with_text


def test_a_failed_task_says_so_with_its_error():
    text = describe(
        {"kind": "task_failed", "payload": {"task_name": "Polishing", "error": "boom"}}
    )
    assert text == {
        "label": "failed",
        "detail": "Polishing",
        "error": "boom",
        "failed": True,
    }


def test_a_stable_move_says_how_far():
    text = describe(
        {
            "kind": "stage_moved",
            "payload": {
                "move": "stable_move",
                "request": {"dx": 2e-6, "dy": -1e-6, "beam_type": "ION"},
            },
        }
    )
    assert text == {
        "label": "moved stage",
        "detail": "stable move · ION",
        "note": "+2.00 µm · −1.00 µm",
    }


def test_an_answer_says_what_was_asked_and_which_button():
    asked = {}
    note_asked(
        asked,
        {
            "kind": "prompt_raised",
            "session": "s",
            "payload": {"question_id": 3, "message": "Mill?", "positive": "Go"},
        },
    )
    answer = {
        "kind": "prompt_answered",
        "session": "s",
        "payload": {"question_id": 3, "response": True},
    }
    assert describe(answer, asked) == {"label": "answered", "detail": "Mill? · Go"}


def test_an_unknown_kind_shows_its_plain_fields():
    text = describe({"kind": "made_up", "payload": {"a": 1, "b": [1], "c": None}})
    assert text == {"label": "made up", "note": "a 1 · c null"}


def test_with_text_adds_the_group_and_leaves_the_record_alone():
    record = {"kind": "milling_progress", "payload": {}}
    described = with_text(record)
    assert described["group"] == "live"
    assert described["text"]["label"] == "milling progress"
    assert "text" not in record
