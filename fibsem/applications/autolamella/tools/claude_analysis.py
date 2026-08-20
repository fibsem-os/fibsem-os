import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).parents[4]

_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)

_CLAUDE_FALLBACKS = [
    Path.home() / ".claude" / "local" / "claude",                                    # Linux/Mac
    Path(os.environ.get("LOCALAPPDATA", "")) / "AnthropicClaude" / "claude.exe",     # Windows desktop
    Path(os.environ.get("APPDATA", "")) / "npm" / "claude.cmd",                      # Windows npm
]


def _claude_bin() -> str:
    path = shutil.which("claude")
    if path:
        return path
    for fallback in _CLAUDE_FALLBACKS:
        if fallback.exists():
            return str(fallback)
    raise FileNotFoundError(
        "claude CLI not found on PATH or at known install locations. "
        "Install Claude Code: https://claude.ai/code"
    )


def analyze_lamellas(
    experiment_path: str | Path,
    selectors: list[str] | None = None,
    json_output: bool = False,
    cwd: str | Path | None = None,
    timeout: int = 300,
) -> str | dict[str, Any]:
    """Run the analyze-lamellas Claude Code skill on an experiment directory.

    Args:
        experiment_path: Path to experiment root (contains experiment.yaml)
                         or a single lamella directory (contains ref_*.tif files).
        selectors:       Optional list of lamella selectors to restrict which lamellas are
                         analysed. Each selector can be:
                           - A numeric index or zero-padded prefix: "1", "01", "3"
                           - A range: "1-5" (expands to indices 1–5)
                           - A name substring: "vocal-walrus"
                           - A full path: "/path/to/lamella-dir"
                         If None or empty, all lamellas are analysed.
        json_output:     If True, passes --json and returns a parsed dict instead of
                         a markdown string. The dict has keys "lamellas" and "summary".
        cwd:             Working directory for the claude subprocess (defaults to the
                         fibsem repo root so .claude/settings.json with MCP config is found).
        timeout:         Seconds before the subprocess is killed (default 300).

    Returns:
        Markdown analysis report as a string, or a parsed dict when json_output=True.

    Raises:
        FileNotFoundError: If the `claude` CLI is not on PATH.
        subprocess.TimeoutExpired: If analysis exceeds `timeout` seconds.
        RuntimeError: If claude exits with a non-zero return code, or if json_output=True
                      and no JSON block is found in the output.
    """
    prompt = f"/analyze-lamellas {experiment_path}"
    if selectors:
        prompt += " " + " ".join(selectors)
    if json_output:
        prompt += " --json"

    result = subprocess.run(
        [_claude_bin(), "-p", "--dangerously-skip-permissions", prompt],
        capture_output=True,
        text=True,
        cwd=cwd or _REPO_ROOT,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(f"claude exited with code {result.returncode}")

    output = result.stdout

    if not json_output:
        return output

    match = _JSON_BLOCK_RE.search(output)
    if not match:
        raise RuntimeError("No JSON block found in analyze-lamellas output")
    return json.loads(match.group(1))


if __name__ == "__main__":
    import sys

    EXPERIMENT = "/home/patrick/github/fibsem/fibsem/applications/test-data/sk-project/20260402_MyrGFP_FCIso"

    def _markdown():
        print("=== All lamellas (markdown) ===")
        print(analyze_lamellas(EXPERIMENT))

    def _selector():
        print("=== Lamellas 1-2 (markdown) ===")
        print(analyze_lamellas(EXPERIMENT, selectors=["1-2"]))

    def _json():
        print("=== Lamella 8 (JSON) ===")
        data = analyze_lamellas(EXPERIMENT, selectors=["8"], json_output=True)
        print(json.dumps(data, indent=2))
        lam = data["lamellas"][0]
        print(f"\nName:   {lam['name']}")
        print(f"Status: {lam['status']} ({lam['workflow_status']})")
        print(f"Last stage: {lam['last_stage']}")
        print(f"Finding: {lam['key_finding']}")
        print(f"Summary: {data['summary']}")

    tests = {"markdown": _markdown, "selector": _selector, "json": _json}
    name = sys.argv[1] if len(sys.argv) > 1 else "json"
    if name not in tests:
        print(f"Usage: python {sys.argv[0]} [{'|'.join(tests)}]")
        sys.exit(1)
    tests[name]()
