"""The prompt bar: a question and its buttons, and an answer said once."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.ui.prompt_bar import PromptBar  # noqa: E402


def test_an_answer_is_said_once_until_the_bar_reopens(qapp):
    bar = PromptBar()
    answers = []
    bar.answered.connect(answers.append)
    bar.ask("Skip the tilt?", "Continue", "Skip")

    bar.no.click()
    bar.no.click()
    bar.yes.click()
    assert answers == [False]

    bar.reopen()
    bar.yes.click()
    assert answers == [False, True]


def test_reopening_enables_only_the_buttons_asked_for(qapp):
    bar = PromptBar()
    bar.ask("Press Continue when ready.", "Continue")
    bar.yes.click()
    bar.reopen()
    assert bar.yes.isEnabled()
    assert not bar.no.isEnabled()


def test_no_question_shows_nothing(qapp):
    bar = PromptBar()
    bar.ask("Run the pass?", "Run Milling", "Skip")
    bar.clear()
    assert not bar.message.isVisibleTo(bar)
    assert not bar.yes.isVisibleTo(bar)
    assert not bar.no.isVisibleTo(bar)
