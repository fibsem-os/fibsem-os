"""``python -m fibsem.ai.evals``: check cases, score one answer, run the agent
on cases and report the results."""

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from fibsem import paths
from fibsem.ai.agent import playbooks, sources
from fibsem.ai.evals import matrix as matrix_mod
from fibsem.ai.evals import report as report_mod
from fibsem.ai.evals import trace as trace_mod
from fibsem.ai.evals import visual
from fibsem.ai.evals.cases import check_case, find_cases, load_case
from fibsem.ai.evals.runner import (
    CASE_PLAYBOOK,
    RunConfig,
    RunnerError,
    code_version,
    run_cases,
)
from fibsem.ai.evals.scoring import score_case


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m fibsem.ai.evals")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="check every case under a folder")
    check.add_argument("root", type=Path)
    score = sub.add_parser("score", help="score one answer against one case")
    score.add_argument("case", type=Path)
    score.add_argument("answer", type=Path)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("root", type=Path)
    common.add_argument(
        "--out", type=Path, required=True, help="a new folder for the runs"
    )
    common.add_argument("--repeats", type=int, default=3)
    common.add_argument("--jobs", type=int, default=1, help="runs at once (default 1)")
    common.add_argument(
        "--only", default=None, help="only cases whose path holds this text"
    )
    common.add_argument("--effort", default=None)
    common.add_argument(
        "--budget-usd",
        type=float,
        default=1.0,
        help="each run's spending cap, unless the case sets budget_usd (default 1)",
    )
    common.add_argument(
        "--timeout-min",
        type=float,
        default=20.0,
        help="each run's time limit (default 20)",
    )
    common.add_argument(
        "--knowledge",
        type=Path,
        default=None,
        help="the knowledge folder each run starts from (default: the lab's)",
    )
    common.add_argument(
        "--agent",
        default=None,
        help="the command run in place of fibsem-agent, as one string",
    )
    run = sub.add_parser(
        "run",
        parents=[common],
        help="run the agent on every keyed case under a folder, and score it",
    )
    run.add_argument(
        "--code",
        action="store_true",
        help="code mode: the agent may also write and run code (sandboxed, "
        "macOS and Linux only); files only when left out",
    )
    run.add_argument(
        "--playbook",
        default=None,
        help=f"the playbook each run starts on; '{CASE_PLAYBOOK}' for each case's own",
    )
    run.add_argument("--model", default=None)
    matrix = sub.add_parser(
        "matrix",
        parents=[common],
        help="run the cases under every model x mode x playbook, each in its "
        "own folder, on one commit",
    )
    matrix.add_argument("--models", default="opus,sonnet")
    matrix.add_argument("--modes", default=",".join(matrix_mod.MODES))
    matrix.add_argument(
        "--playbooks",
        default=",".join(matrix_mod.PLAYBOOKS),
        help=f"base (the base playbook) and/or {CASE_PLAYBOOK} (each case's own)",
    )
    matrix.add_argument(
        "--allow-changes",
        action="store_true",
        help="run although the checkout has uncommitted changes",
    )
    matrix_report = sub.add_parser(
        "matrix-report", help="a matrix folder's results as tables"
    )
    matrix_report.add_argument("out", type=Path)
    matrix_report.add_argument("--cases", type=Path, default=None)
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
    compare = sub.add_parser(
        "compare",
        help="run folders side by side (such as one per model), scored with "
        "today's keys",
    )
    compare.add_argument("outs", type=Path, nargs="+")
    compare.add_argument("--cases", type=Path, default=None)
    trace = sub.add_parser(
        "trace",
        help="what the agent did in each repeat: skills read, images, tools, "
        "its words and answer",
    )
    trace.add_argument("out", type=Path)
    trace.add_argument("--only", default=None, help="cases whose id contains this")
    trace.add_argument("--write", type=Path, default=None, help="a markdown file")
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

    if args.command == "compare":
        print(visual.compare_table(args.outs, args.cases))
        return 0

    if args.command == "trace":
        text = trace_mod.trace_markdown(args.out, args.only)
        if args.write:
            args.write.write_text(text, encoding="utf-8")
            print(f"Wrote {args.write}")
        else:
            print(text)
        return 0

    if args.command == "matrix-report":
        print(matrix_mod.matrix_markdown(args.out, args.cases))
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
        jobs=args.jobs,
        effort=args.effort,
        budget_usd=args.budget_usd,
        timeout_s=args.timeout_min * 60,
        knowledge=args.knowledge,
    )
    if args.agent:
        # Windows paths keep their backslashes.
        config.agent_command = shlex.split(args.agent, posix=os.name != "nt")
    keyed = [c for c in cases if c.status == "keyed"]

    def said(result):
        results = ", ".join(s["result"] for s in result["scores"])
        where = ""
        if args.command == "matrix":
            mode = "code" if result.get("code") else "files"
            where = f"[{result.get('model')} {mode} {result.get('playbook')}] "
        print(
            f"{where}{result['case']} r{result['repeat']}: {result['status']}, {results}"
        )

    if args.command == "matrix":
        return _matrix(args, config, keyed, said)
    config.code = args.code
    config.playbook = args.playbook
    config.model = args.model
    missing = _missing_playbooks(keyed, args.playbook)
    if missing:
        print(f"error: no playbook {', '.join(missing)}")
        return 1
    print(
        f"{len(keyed)} cases x {args.repeats} repeats, at most "
        f"${args.budget_usd:.2f} each, into {args.out}"
    )
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


def _missing_playbooks(cases, playbook) -> list:
    """The playbooks the runs would start on that this checkout lacks."""
    names = (
        {c.data.get("playbook") for c in cases} - {None}
        if playbook == CASE_PLAYBOOK
        else {playbook} - {None}
    )
    root = sources.find_root()
    return sorted(n for n in names if playbooks.find(n, root) is None)


def _matrix(args, config, keyed, said) -> int:
    grid = matrix_mod.cells(
        args.models.split(","), args.modes.split(","), args.playbooks.split(",")
    )
    bad = [c.mode for c in grid if c.mode not in matrix_mod.MODES] + [
        c.playbook for c in grid if c.playbook not in matrix_mod.PLAYBOOKS
    ]
    if bad:
        print(f"error: unknown {', '.join(sorted(set(bad)))}")
        return 1
    if CASE_PLAYBOOK in args.playbooks.split(","):
        missing = _missing_playbooks(keyed, CASE_PLAYBOOK)
        if missing:
            print(f"error: no playbook {', '.join(missing)}")
            return 1
    knowledge = config.knowledge or paths.state_root() / "knowledge"
    version = code_version(knowledge)
    if version.get("changed") and not args.allow_changes:
        print(
            "error: the checkout has uncommitted changes, so the cells might "
            "not run the same code: commit or stash them, or add "
            "--allow-changes.\n  " + "\n  ".join(version["changed"])
        )
        return 1
    try:
        matrix_mod.start(args.out, grid, version)
    except matrix_mod.MatrixError as error:
        print(f"error: {error}")
        return 1
    print(
        f"{len(keyed)} cases x {len(grid)} cells x {args.repeats} repeats = "
        f"{len(keyed) * len(grid) * args.repeats} runs, at most "
        f"${args.budget_usd:.2f} each, into {args.out}"
    )
    try:
        matrix_mod.run_matrix(keyed, args.out, grid, config, on_result=said)
    except RunnerError as error:
        print(f"error: {error}")
        return 1
    text = matrix_mod.matrix_markdown(args.out, args.root)
    (args.out / "matrix.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
