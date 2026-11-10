"""``python -m fibsem.ai.evals check <cases>`` and ``... score <case> <answer.json>``."""

import argparse
import json
import sys
from pathlib import Path

from fibsem.ai.evals.cases import check_case, find_cases, load_case
from fibsem.ai.evals.scoring import score_case


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fibsem.ai.evals")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="check every case under a folder")
    check.add_argument("root", type=Path)
    score = sub.add_parser("score", help="score one answer against one case")
    score.add_argument("case", type=Path)
    score.add_argument("answer", type=Path)
    args = parser.parse_args(argv)

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

    answer = json.loads(args.answer.read_text())
    for s in score_case(load_case(args.case), answer):
        print(json.dumps(s.to_dict()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
