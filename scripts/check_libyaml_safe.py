#!/usr/bin/env python
"""Is it safe to dump experiment.yaml with libyaml on THIS machine?

Run this on each platform the app actually ships to -- especially the Windows
microscope PCs, which CI (ubuntu-latest only) never covers. libyaml is a separate
C library whose version varies by install method (pip wheel vs conda), so a clean
result on one machine is not a clean result on another. This script answers it
with data rather than inference.

    python check_libyaml_safe.py [EXPERIMENT_DIR ...]

With no arguments it searches the current directory tree for experiment.yaml and
protocol.yaml files. Exit code 0 = safe, 1 = do not use libyaml here.

It compares, for every document it can find plus 3000 generated ones:
    D0 = safe_load(raw)
    safe_dump(D0)   -> reload -> A
    CSafeDumper(D0) -> reload -> B
and requires A == B == D0. Byte differences are reported but are not failures --
the two emitters may legally escape or wrap exotic characters differently. Only a
change of *meaning* is a failure.
"""
import glob
import os
import random
import string
import sys

import yaml

try:
    from yaml import CSafeDumper, CSafeLoader
except ImportError:
    print("libyaml is NOT available in this PyYAML build.")
    print("Nothing to check: the code falls back to safe_dump and behaves exactly as today.")
    sys.exit(0)


def probe_versions() -> None:
    print(f"python      : {sys.version.split()[0]}  ({sys.platform})")
    print(f"pyyaml      : {yaml.__version__}   with_libyaml={yaml.__with_libyaml__}")
    try:
        import yaml._yaml as _y

        print(f"libyaml     : {_y.get_version_string()}")
    except Exception as exc:  # pragma: no cover - probe only
        print(f"libyaml     : version probe failed ({exc})")
    print()


def check_document(label: str, d0):
    """Return (ok, byte_identical). ok=False means the meaning changed."""
    py_text = yaml.safe_dump(d0, indent=4)
    c_text = yaml.dump(d0, Dumper=CSafeDumper, indent=4)
    a = yaml.safe_load(py_text)
    b = yaml.safe_load(c_text)
    c = yaml.load(c_text, Loader=CSafeLoader)  # the C reader must agree too
    ok = (a == d0) and (b == d0) and (c == d0)
    if not ok:
        print(f"  MISMATCH  {label}")
        print(f"      safe_dump reloads equal : {a == d0}")
        print(f"      libyaml   reloads equal : {b == d0}")
        print(f"      C reader  reloads equal : {c == d0}")
    return ok, py_text == c_text


def check_real_files(roots):
    patterns = ("experiment.yaml", "protocol.yaml")
    files = []
    for root in roots:
        for name in patterns:
            files += glob.glob(os.path.join(root, "**", name), recursive=True)
    files = sorted(set(files))

    print(f"[1/3] real files  ({len(files)} found under {', '.join(roots)})")
    if not files:
        print("      none found -- point this at an experiment directory to cover real data")
        return True, 0

    n_ok = n_bytediff = n_fail = 0
    kb = 0.0
    for path in files:
        try:
            raw = open(path, encoding="utf-8").read()
            d0 = yaml.safe_load(raw)
        except Exception as exc:
            print(f"      skipped (unreadable as shipped): {path}: {exc}")
            continue
        if d0 is None:
            continue
        kb += len(raw) / 1024
        ok, identical = check_document(path, d0)
        if ok:
            n_ok += 1
            n_bytediff += not identical
        else:
            n_fail += 1

    print(f"      round-trip identical : {n_ok}/{n_ok + n_fail}   ({kb / 1024:.1f} MB)")
    print(f"      byte-identical out   : {n_ok - n_bytediff}/{n_ok}")
    print(f"      MEANING CHANGED      : {n_fail}")
    return n_fail == 0, n_ok


def check_shared_objects():
    """The one structural difference: SafeDumper has Serializer in its MRO and
    CSafeDumper does not (CEmitter absorbs it). Serializer generates the anchors
    and aliases used when two places reference one object -- which fibsem can
    produce, because AutoLamellaTaskConfig.to_dict assigns parameters by
    reference. Check anchors specifically."""
    print("\n[2/3] shared objects, anchors and aliases")
    shared_list = [1.0, 2.0, 3.0]
    shared_dict = {"a": 1}
    doc = {
        "positions": [
            {"name": f"L-{i:03d}", "milling_angles": shared_list, "cfg": shared_dict}
            for i in range(3)
        ]
    }
    ok, identical = check_document("<shared-container document>", doc)
    b = yaml.safe_load(yaml.dump(doc, Dumper=CSafeDumper, indent=4))
    pos = b["positions"]
    still_shared = pos[0]["milling_angles"] is pos[1]["milling_angles"]
    print(f"      round-trip identical : {ok}")
    print(f"      byte-identical out   : {identical}")
    print(f"      alias reloads shared : {still_shared}  (must match safe_dump's behaviour)")

    cyc = {"k": 1}
    cyc["self"] = cyc
    try:
        yaml.dump(cyc, Dumper=CSafeDumper)
        print("      self-reference       : handled")
    except Exception as exc:
        ok = False
        print(f"      self-reference       : FAILED {type(exc).__name__}: {exc}")
    return ok


ALPHA = string.printable[:95] + "µ°±—Ωåäö  "
NASTY = ["yes", "no", "on", "off", "~", "null", "0123", "1e5", ".inf", "- x", " a ",
         "a\nb", "a\tb", "", "#c", "a: b", '"q"', "\\path\\x", "---", " "]


def _scalar(rng):
    r = rng.random()
    if r < 0.45:
        return "".join(rng.choices(ALPHA, k=rng.choice([0, 1, 2, 5, 20, 80, 200])))
    if r < 0.60:
        return rng.choice([True, False, None])
    if r < 0.78:
        return rng.choice([0, -1, 2 ** 62, -(2 ** 62), rng.randint(-10 ** 9, 10 ** 9)])
    if r < 0.95:
        return rng.choice([0.0, -0.0, 1e-30, 1e30, rng.uniform(-1e6, 1e6)])
    return rng.choice(NASTY)


def _doc(rng, depth=0):
    if depth > 3 or rng.random() < 0.3:
        return _scalar(rng)
    if rng.random() < 0.5:
        return {str(_scalar(rng))[:40] or "k": _doc(rng, depth + 1) for _ in range(rng.randint(0, 6))}
    return [_doc(rng, depth + 1) for _ in range(rng.randint(0, 6))]


def check_fuzz(n=3000):
    print(f"\n[3/3] fuzz  ({n} generated documents over adversarial scalars)")
    rng = random.Random(20260818)  # fixed seed: same corpus on every machine
    n_fail = n_bytediff = 0
    for i in range(n):
        d0 = _doc(rng)
        try:
            ok, identical = check_document(f"<fuzz #{i}>", d0)
        except Exception as exc:
            print(f"  ERROR on fuzz #{i}: {type(exc).__name__}: {exc}")
            n_fail += 1
            continue
        n_fail += not ok
        n_bytediff += not identical
    print(f"      byte-different out   : {n_bytediff}/{n}  (expected, not a failure)")
    print(f"      MEANING CHANGED      : {n_fail}/{n}")
    return n_fail == 0


def main() -> int:
    probe_versions()
    roots = sys.argv[1:] or [os.getcwd()]
    ok_real, n_real = check_real_files(roots)
    ok_shared = check_shared_objects()
    ok_fuzz = check_fuzz()

    print("\n" + "=" * 62)
    if ok_real and ok_shared and ok_fuzz:
        print("VERDICT: safe to dump with libyaml on this machine.")
        if n_real == 0:
            print("         NOTE: no real experiment files were checked. Re-run with a")
            print("         path to an experiment directory to cover your own data.")
        return 0
    print("VERDICT: DO NOT use libyaml here -- a document changed meaning.")
    print("         Keep yaml.safe_dump on this platform and report the mismatch above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
