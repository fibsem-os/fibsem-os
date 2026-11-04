"""The build step pyproject.toml can't state: the agent's skills, specialists
and profiles live in the checkout's .claude/ (Claude Code reads them there),
and the wheel ships a copy in fibsem/ai/agent/shipped/, so a pip install can
run fibsem-agent (fibsem.ai.agent.sources)."""

import importlib.util
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent


def _sources():
    # By path: importing fibsem here would need its dependencies.
    spec = importlib.util.spec_from_file_location(
        "_fibsem_agent_sources", ROOT / "fibsem" / "ai" / "agent" / "sources.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildWithAgentFiles(build_py):
    def run(self):
        super().run()
        if not self.editable_mode:
            target = Path(self.build_lib) / "fibsem" / "ai" / "agent" / "shipped"
            _sources().ship(ROOT, target)


setup(cmdclass={"build_py": BuildWithAgentFiles})
