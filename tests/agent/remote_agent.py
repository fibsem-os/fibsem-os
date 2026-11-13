"""The agent computer of test_two_computers.py, in a process with its own home
and state root: pair with the code the microscope PC shows, then run the
harness at the tunnel's end with a stand-in model that prints each prompt as a
JSON line. Its log goes to stderr.

    python remote_agent.py URL CODE
"""

import asyncio
import json
import logging
import sys

from fibsem.ai.agent import watch as watch_mod
from fibsem.ai.agent.harness import Harness, TurnEnd
from fibsem.client.pairing import pair


class PrintingAgent:
    session_id = "remote"

    async def start(self):
        pass

    async def turn(self, prompt):
        print(json.dumps({"prompt": prompt}), flush=True)
        return TurnEnd()

    async def interrupt(self):
        pass

    async def close(self):
        pass


def main() -> int:
    url, code = sys.argv[1], sys.argv[2]
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    pair(code, url, name="agent-laptop")
    harness = Harness(
        PrintingAgent(),
        lambda: watch_mod.connect(url),
        url=url,
        poll_s=0.3,
        retry_s=0.05,
        server_wait_s=120,
    )
    return asyncio.run(harness.run())


if __name__ == "__main__":
    sys.exit(main())
