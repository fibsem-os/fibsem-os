"""``python -m fibsem.ai.evals``: check cases, score one answer, run the agent
on cases and report the results."""

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from fibsem.ai.evals import report as report_mod
from fibsem.ai.evals import visual
from fibsem.ai.evals.cases import check_case, find_cases, load_case
from fibsem.ai.evals.runner import RunConfig, RunnerError, run_cases
from fibsem.ai.evals.scoring import score_case


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m fibsem.ai.evals")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="check every case under a folder")
    check.add_argument("root", type=Path)
    score = sub.add_parser("score", help="score one answer against one case")
    score.add_argument("case", type=Path)
    score.add_argument("answer", type=Path)
    run = sub.add_parser(
        "run", help="run the agent on every keyed case under a folder, and score it"
    )
    run.add_argument("root", type=Path)
    run.add_argument(
        "--out", type=Path, required=True, help="a new folder for the runs"
    )
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument(
        "--only", default=None, help="only cases whose path holds this text"
    )
    run.add_argument("--playbook", default=None)
    run.add_argument("--model", default=None)
    run.add_argument("--effort", default=None)
    run.add_argument(
        "--budget-usd",
        type=float,
        default=1.0,
        help="each run's spending cap, unless the case sets budget_usd (default 1)",
    )
    run.add_argument(
        "--timeout-min",
        type=float,
        default=20.0,
        help="each run's time limit (default 20)",
    )
    run.add_argument(
        "--knowledge",
        type=Path,
        default=None,
        help="the knowledge folder each run starts from (default: the lab's)",
    )
    run.add_argument(
        "--agent",
        default=None,
        help="the command run in place of fibsem-agent, as one string",
    )
    report = sub.add_parser("report", help="the results of a run folder, as a table")
    report.add_argument("out", type=Path)
    report.add_argument(
        "--html",
        action="store_true",
        help="also write report.html: the key and every answer on each case's images",
    )
    report.add_argument(
        "--cases",
        type=Path,
        default=None,
        help="the cases folder, for runs made before results kept their case's path",
    )
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "check":
        failed = 0
        for case in find_cases(args.root):
            result = check_case(case)
            rel = case.path.relative_to(args.root)
            print(f"{case.status:8} {'ok' if result.ok else 'PROBLEMS'}  {rel}")
            for problem in result.problems:
                print(f"           - {problem}")
            failed += not result.ok
        return 1 if failed else 0

    if args.command == "score":
        answer = json.loads(args.answer.read_text())
        for s in score_case(load_case(args.case), answer):
            print(json.dumps(s.to_dict()))
        return 0

    if args.command == "report":
        print(report_mod.table(report_mod.summarise(report_mod.load_results(args.out))))
        if args.html:
            print(f"Wrote {visual.write_html(args.out, args.cases)}")
        return 0

    cases = [
        c
        for c in find_cases(args.root)
        if args.only is None or args.only in c.path.as_posix()
    ]
    broken = [c for c in cases if not check_case(c).ok]
    if broken:
        names = ", ".join(c.id for c in broken)
        print(f"Cases with problems (python -m fibsem.ai.evals check): {names}")
        return 1
    config = RunConfig(
        out=args.out,
        repeats=args.repeats,
        playbook=args.playbook,
        model=args.model,
        effort=args.effort,
        budget_usd=args.budget_usd,
        timeout_s=args.timeout_min * 60,
        knowledge=args.knowledge,
    )
    if args.agent:
        # Windows paths keep their backslashes.
        config.agent_command = shlex.split(args.agent, posix=os.name != "nt")
    keyed = [c for c in cases if c.status == "keyed"]
    print(
        f"{len(keyed)} cases x {args.repeats} repeats, at most "
        f"${args.budget_usd:.2f} each, into {args.out}"
    )

    def said(result):
        results = ", ".join(s["result"] for s in result["scores"])
        print(f"{result['case']} r{result['repeat']}: {result['status']}, {results}")

    try:
        run_cases(keyed, config, on_result=said)
    except RunnerError as error:
        print(f"error: {error}")
        return 1
    rows = report_mod.summarise(report_mod.load_results(args.out))
    text = report_mod.table(rows)
    (args.out / "report.md").write_text(text)
    print(text)
    print(f"Wrote {visual.write_html(args.out, args.root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
