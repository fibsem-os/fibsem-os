"""``python -m fibsem.ai.evals``: check cases, score one answer, run the agent
on cases and report the results, and make cases from the simulator."""

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

import yaml

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
    run.add_argument(
        "--tier",
        choices=("real", "sim", "all"),
        default="all",
        help="only real cases, or only sim cases (default: both, scored apart)",
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
    make_sim = sub.add_parser(
        "make-sim",
        help="make grid cases (squares, labels, sites) from the simulator, one per seed",
    )
    make_sim.add_argument("--out", type=Path, required=True, help="the cases folder")
    make_sim.add_argument("--seeds", type=int, nargs="+", required=True)
    make_sim.add_argument(
        "--beam",
        nargs="+",
        choices=("SEM", "FIB"),
        default=["SEM"],
        help="whose overview: the SEM face on, the FIB at the milling angle, or both",
    )
    make_sim.add_argument(
        "--milling-angle",
        type=float,
        default=38.0,
        help="the FIB overview's milling angle in degrees (default 38)",
    )
    make_sim.add_argument(
        "--task",
        nargs="+",
        choices=("grid", "polishing"),
        default=["grid"],
        help="grid: squares, labels and sites from an overview; polishing: a "
        "lamella milled pass by pass (default grid)",
    )
    make_sim.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="a scene option for every seed, such as crack_density=0.5",
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

    if args.command == "make-sim":
        from fibsem.ai.evals import sim_cases, sim_polishing

        overrides = {}
        for item in args.set:
            name, _, value = item.partition("=")
            overrides[name.strip()] = yaml.safe_load(value)
        for seed in args.seeds:
            if "polishing" in args.task:
                folder = sim_polishing.write_case(seed, args.out, overrides)
                print(f"Wrote {folder}" if folder else f"Seed {seed}: no case")
            for beam in args.beam if "grid" in args.task else []:
                for folder in sim_cases.write_cases(
                    seed,
                    args.out,
                    overrides,
                    beam=beam,
                    milling_angle=args.milling_angle,
                ):
                    print(f"Wrote {folder}")
        return 0

    if args.command == "report":
        print(report_mod.table(report_mod.summarise(report_mod.load_results(args.out))))
        if args.html:
            print(f"Wrote {visual.write_html(args.out, args.cases)}")
        return 0

    cases = [
        c
        for c in find_cases(args.root)
        if (args.only is None or args.only in c.path.as_posix())
        and args.tier in ("all", c.tier)
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
