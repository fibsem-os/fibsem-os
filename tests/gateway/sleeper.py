"""Stand-ins for an app's serve function, run by the gateway's child."""

import time


def serve(argv, stop):
    print("serving", " ".join(argv), flush=True)
    stop.wait(60)
    return 0


def stubborn(argv, stop):
    time.sleep(60)
    return 0


def quits(argv, stop):
    return 3
