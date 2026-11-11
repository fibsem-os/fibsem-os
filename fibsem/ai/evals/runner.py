"""Running the agent on eval cases, one case and repeat at a time.

Each run gets its own state root (``FIBSEM_HOME``) holding a frozen copy of
the lab's knowledge and a new experiment with the case's inputs in it, an App
Server on the Demo microscope serving that experiment, and fibsem-agent at
Watch with the case's task as its goal. The run ends when the agent writes
``answer.json``, stops, or runs out of time; the answer is then scored and the
session's usage kept beside it.

The App Server is one per PC, so the runner refuses to start while another is
serving. Nothing here moves a microscope: the Demo microscope is the only one
it connects, and Watch is the level.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import yaml

from fibsem import paths
from fibsem.ai.agent import manager
from fibsem.ai.evals.cases import Case, input_paths
from fibsem.ai.evals.scoring import ANSWER_SHAPES, score_case
from fibsem.server import discovery

log = logging.getLogger(__name__)

ANSWER = "answer.json"
TASK = "task.md"
RESULT = "result.json"
INPUTS = "experiment"
# The session files kept with each run.
SESSION_FILES = (manager.RECORD, manager.LOG, "usage.jsonl")
POLL_S = 2.0
STOP_WAIT_S = 60.0


class RunnerError(RuntimeError):
    pass


def agent_command() -> List[str]:
    return manager.agent_command()


def server_command() -> List[str]:
    return [
        sys.executable,
        "-m",
        "fibsem.applications.autolamella.launch",
        "--manufacturer",
        "Demo",
    ]


@dataclass
class RunConfig:
    out: Path
    repeats: int = 3
    playbook: Optional[str] = None
    model: Optional[str] = None
    effort: Optional[str] = None
    budget_usd: float = 1.0
    timeout_s: float = 20 * 60
    server_wait_s: float = 120.0
    # The knowledge each run starts from; the lab's own when None.
    knowledge: Optional[Path] = None
    agent_command: List[str] = field(default_factory=lambda: agent_command())
    server_command: List[str] = field(default_factory=lambda: server_command())


def run_dir(config: RunConfig, case: Case, repeat: int) -> Path:
    return Path(config.out) / case.id / f"r{repeat}"


def task_text(case: Case, inputs: Iterable[str]) -> str:
    """What the agent is told: the case's brief, its files and the answer's
    shape, never its key or what the run did."""
    data = case.data
    lines = [f"# Eval task: {case.id}", "", str(data.get("brief", "")).strip(), ""]
    for name in ("sample", "instrument", "mode"):
        if data.get(name):
            lines.append(f"- {name.capitalize()}: {data[name]}")
    stack = data.get("stack")
    if stack:
        about = {k: v for k, v in stack.items() if k not in ("file", "axes")}
        lines.append(f"- Stack {_shown(stack.get('file'))}: {json.dumps(about)}")
    for name, value in (data.get("inputs") or {}).items():
        if isinstance(value, (int, float)) or name.endswith("_um"):
            lines.append(f"- {name}: {value}")
    passes = data.get("passes") or []
    if passes:
        lines += ["", "## Passes, in order", ""]
        for entry in passes:
            files = {
                k: _shown(v)
                for k, v in entry.items()
                if isinstance(v, str) and k != "name"
            }
            measured = entry.get("measured")
            extra = f" measured {json.dumps(measured)}" if measured else ""
            lines.append(f"- {entry['name']}: {json.dumps(files)}{extra}")
    lines += [
        "",
        "## Files",
        "",
        "The open experiment holds the case's files; read them with the "
        "experiment tools by these paths:",
        "",
    ]
    lines += [f"- {path}" for path in sorted(inputs)]
    lines += ["", "## Answer", ""]
    if case.parts:
        lines.append(f'Write {ANSWER} in your files folder as {{"parts": {{...}}}}:')
        for name, part in case.parts.items():
            shape = json.dumps(ANSWER_SHAPES[part["task"]])
            lines += ["", f"- Part {name}: {part.get('question', '')}", f"  {shape}"]
    else:
        lines.append(f"Write {ANSWER} in your files folder, shaped like this:")
        lines += ["", json.dumps(ANSWER_SHAPES[case.task], indent=2)]
    lines += [
        "",
        "Look and read only. Once the answer is written you are done; say so and stop.",
    ]
    return "\n".join(lines) + "\n"


def _shown(path: Optional[str]) -> Optional[str]:
    """A case input's path as the experiment holds it."""
    if path is None:
        return None
    prefix = INPUTS + "/"
    if path.startswith("../"):
        return Path(path).name
    return path[len(prefix) :] if path.startswith(prefix) else path


def goal_text(case: Case) -> str:
    return (
        f"Eval case {case.id}: read {TASK} in your files folder, do what it "
        f"asks, and write {ANSWER} there."
    )


def link_inputs(case: Case, experiment: Path) -> List[str]:
    """The case's inputs, linked (copied where links can't be made) into the
    experiment's folder: everything under its experiment/, and a file it
    shares from another case (``../``) at the top. The paths the experiment
    holds them at."""
    source = case.path / INPUTS
    files = {}
    for path in sorted(source.rglob("*")) if source.is_dir() else []:
        if path.is_file() and path.name != "experiment.yaml":
            files[path.relative_to(source).as_posix()] = path
    for rel in input_paths(case):
        if rel.startswith("../"):
            files[Path(rel).name] = case.path / rel
    for rel, path in files.items():
        target = experiment / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)
    return sorted(files)


def prepare_home(home: Path, knowledge: Optional[Path]) -> None:
    """A state root that lets a local agent connect without a click and
    starts from a copy of *knowledge*."""
    prefs = home / "config" / "user-preferences.yaml"
    prefs.parent.mkdir(parents=True, exist_ok=True)
    install = {"app_server": {"enabled": True, "allow_local_agents": True}}
    prefs.write_text(yaml.safe_dump({"install": install}))
    if knowledge is not None and knowledge.is_dir():
        shutil.copytree(knowledge, home / "knowledge")


def _wait_for(check: Callable[[], bool], timeout_s: float, proc=None) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if check():
            return True
        if proc is not None and proc.poll() is not None:
            return False
        time.sleep(0.5)
    return check()


def _end(proc: subprocess.Popen, wait_s: float) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(wait_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _read_answer(path: Path) -> Optional[Dict[str, Any]]:
    try:
        answer = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return answer if isinstance(answer, dict) else None


def run_case(case: Case, repeat: int, config: RunConfig) -> Dict[str, Any]:
    """One run of the agent on *case*; its result, also saved in the run's
    folder."""
    folder = run_dir(config, case, repeat)
    if folder.exists():
        raise RunnerError(f"{folder} exists: give another --out, or remove it.")
    if discovery.read_discovery_file() is not None:
        raise RunnerError(
            "An App Server is already running on this PC; close it first "
            "(one App Server per PC)."
        )
    home, work = folder / "home", folder / "work"
    experiments = home / "experiments"
    experiment = experiments / case.id
    work.mkdir(parents=True)
    experiments.mkdir(parents=True)
    knowledge = config.knowledge or paths.state_root() / "knowledge"
    prepare_home(home, knowledge)
    env = {**os.environ, paths.STATE_ROOT_ENV: str(home)}

    started = time.monotonic()
    result: Dict[str, Any] = {"case": case.id, "repeat": repeat, "task": case.task}
    server_log = open(folder / "server.log", "w")
    server = subprocess.Popen(
        config.server_command
        + ["--create", str(experiment), "--experiments", str(experiments)],
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )
    agent = None
    try:

        def serving() -> bool:
            found = discovery.read_discovery_file()
            return found is not None and found.get("pid") == server.pid

        if not _wait_for(serving, config.server_wait_s, server):
            raise RunnerError(
                f"The App Server did not start; see {folder / 'server.log'}."
            )
        inputs = link_inputs(case, experiment)
        (work / TASK).write_text(task_text(case, inputs))

        session_id = str(uuid.uuid4())
        command = config.agent_command + [
            "--session-id",
            session_id,
            "--files",
            str(work),
            "--goal",
            goal_text(case),
            "--max-budget-usd",
            str(config.budget_usd),
        ]
        for flag, value in (
            ("--playbook", config.playbook),
            ("--model", config.model),
            ("--effort", config.effort),
        ):
            if value:
                command += [flag, value]
        result["session_id"] = session_id
        with open(folder / "agent.log", "w") as agent_log:
            agent = subprocess.Popen(
                command, env=env, stdout=agent_log, stderr=subprocess.STDOUT
            )
            answered = _wait_for(
                lambda: _read_answer(work / ANSWER) is not None,
                config.timeout_s,
                agent,
            )
            timed_out = not answered and agent.poll() is None
            session = manager.session_folder(session_id)
            if agent.poll() is None:
                session.mkdir(parents=True, exist_ok=True)
                (session / manager.STOP).touch()
                try:
                    agent.wait(STOP_WAIT_S)
                except subprocess.TimeoutExpired:
                    _end(agent, 10)
        result["agent_exit"] = agent.returncode
        result["status"] = (
            "answered" if answered else "timeout" if timed_out else "no_answer"
        )
        result.update(_collect_session(session, folder / "session"))
    finally:
        if agent is not None and agent.poll() is None:
            _end(agent, 10)
        _end(server, 30)
        server_log.close()

    result["seconds"] = round(time.monotonic() - started, 1)
    answer = _read_answer(work / ANSWER)
    result["scores"] = [s.to_dict() for s in score_case(case, answer or {})]
    (folder / RESULT).write_text(json.dumps(result, indent=2))
    return result


def _collect_session(session: Path, into: Path) -> Dict[str, Any]:
    """The session's record, log and usage, copied beside the run; what it
    spent."""
    into.mkdir(parents=True, exist_ok=True)
    for name in SESSION_FILES:
        if (session / name).is_file():
            shutil.copy2(session / name, into / name)
    runs = manager.read_json(session / manager.RECORD).get("runs") or [{}]
    last = runs[-1]
    return {
        key: last[key]
        for key in ("spent_usd", "turns", "tokens", "reason")
        if key in last
    }


def run_cases(
    cases: Iterable[Case],
    config: RunConfig,
    on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """Every keyed case, *config.repeats* times each, one after another."""
    results = []
    for case in cases:
        if case.status != "keyed":
            log.info("Skipping %s: %s", case.id, case.status)
            continue
        for repeat in range(1, config.repeats + 1):
            result = run_case(case, repeat, config)
            results.append(result)
            if on_result is not None:
                on_result(result)
    return results
