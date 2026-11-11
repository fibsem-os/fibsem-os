"""The review page: disagreements with the key, settling them, and its routes."""

import json

import yaml
from fastapi.testclient import TestClient

from fibsem.ai.evals import review, visual
from fibsem.ai.evals.cases import load_case
from fibsem.ai.evals.serve import build_app
from tests.evals.test_visual import run  # noqa: F401  (the fixture)

EXTRA = {"x_px": 400, "y_px": 300, "reason": "two cells, clear of the bars"}


def _with_extra_pick(out, repeats=(1, 2)):
    for repeat in repeats:
        path = out / "site" / f"r{repeat}" / "work" / "answer.json"
        answer = json.loads(path.read_text())
        answer["picks"].append(dict(EXTRA, x_px=400 + repeat))
        path.write_text(json.dumps(answer))


def _runs(out, name):
    return [r for _, r in visual.load_runs(out)[name]]


def test_a_site_not_in_the_key_is_asked_once_for_both_repeats(run):  # noqa: F811
    cases_root, out = run
    _with_extra_pick(out)
    case = load_case(cases_root / "site")
    (item,) = review.disagreements(case, _runs(out, "site"))
    assert item["kind"] == "site" and item["repeats"] == [1, 2]
    assert item["said"][0] == EXTRA["reason"]


def test_adding_a_site_as_good_changes_the_key_and_logs_the_old_one(run):  # noqa: F811
    cases_root, out = run
    _with_extra_pick(out)
    case = load_case(cases_root / "site")
    runs = _runs(out, "site")
    (item,) = review.disagreements(case, runs)
    after = review.settle(case, runs, item["id"], "good", run="out")
    assert after.key["good"][-1]["x_px"] == item["x_px"]
    assert review.disagreements(after, runs) == []
    (entry,) = review.read_log(after)
    assert entry["action"] == "good" and entry["run"] == "out"
    assert len(entry["key_before"]["good"]) == 1
    # The rest of the case is as it was.
    data = yaml.safe_load((cases_root / "site" / "case.yaml").read_text())
    assert data["brief"] == "site brief" and data["task"] == "site_selection"


def test_keeping_the_key_changes_nothing_and_is_not_asked_again(run):  # noqa: F811
    cases_root, out = run
    _with_extra_pick(out)
    case = load_case(cases_root / "site")
    runs = _runs(out, "site")
    before = (cases_root / "site" / "case.yaml").read_text()
    (item,) = review.disagreements(case, runs)
    after = review.settle(case, runs, item["id"], "keep")
    assert (cases_root / "site" / "case.yaml").read_text() == before
    assert review.disagreements(after, runs) == []


def test_a_square_label_can_be_accepted_as_well(run):  # noqa: F811
    cases_root, out = run
    case = load_case(cases_root / "quality")
    runs = _runs(out, "quality")
    (item,) = review.disagreements(case, runs)
    assert item["id"] == "square:2:good" and item["key"] == ["damaged"]
    after = review.settle(case, runs, item["id"], "accept")
    assert after.key["labels"]["2"] == ["damaged", "good"]
    assert review.disagreements(after, runs) == []


def test_a_late_stop_widens_the_span(run):  # noqa: F811
    cases_root, out = run
    case = load_case(cases_root / "polish")
    runs = _runs(out, "polish")
    (item,) = review.disagreements(case, runs)
    assert item["pass"] == "p3" and "late" in item["title"]
    after = review.settle(case, runs, item["id"], "accept")
    assert after.key["stop_after"] == ["p2", "p3"]


def test_a_case_with_parts_has_nothing_to_settle(run):  # noqa: F811
    cases_root, out = run
    case = load_case(cases_root / "parts")
    assert review.disagreements(case, _runs(out, "parts")) == []


def test_an_unknown_choice_is_refused(run):  # noqa: F811
    cases_root, out = run
    case = load_case(cases_root / "polish")
    runs = _runs(out, "polish")
    for item_id, action in (("stop:p1", "accept"), ("stop:p3", "good")):
        try:
            review.settle(case, runs, item_id, action)
        except review.SettleError:
            continue
        raise AssertionError(f"{item_id} {action} was taken")


def test_the_routes(run):  # noqa: F811
    cases_root, out = run
    _with_extra_pick(out)
    client = TestClient(build_app(out, cases_root))
    (listed,) = client.get("/evals/runs").json()
    assert listed["name"] == "out" and listed["cases"] == 7
    summary = client.get("/evals/runs/out").json()
    site = next(c for c in summary["cases"] if c["id"] == "site")
    assert site["open"] == 1 and site["now"] == [["hit"], ["hit"]]
    detail = client.get("/evals/runs/out/cases/site").json()
    assert detail["can_settle"] and len(detail["repeats"]) == 2
    (item,) = detail["items"]
    figure = client.get("/evals/runs/out/cases/site/figures/0.png")
    assert figure.status_code == 200 and figure.content[:4] == b"\x89PNG"
    close = client.get(f"/evals/runs/out/cases/site/items/{item['id']}.png")
    assert close.status_code == 200 and close.content[:4] == b"\x89PNG"
    settled = client.post(
        "/evals/runs/out/cases/site/settle", json={"item": item["id"], "action": "bad"}
    )
    assert settled.status_code == 200
    assert (
        settled.json()["items"] == []
        and settled.json()["settled"][0]["action"] == "bad"
    )
    again = client.post(
        "/evals/runs/out/cases/site/settle", json={"item": item["id"], "action": "bad"}
    )
    assert again.status_code == 409
    assert client.get("/evals/runs/out/cases/nope").status_code == 404
    assert client.get("/ui/evals").status_code == 200
