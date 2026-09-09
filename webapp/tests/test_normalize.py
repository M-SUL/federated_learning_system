"""Normalizer regressions against the real result files.

Run with any Python 3.11 -- no venv, no dependencies. That is itself part of what
is being tested: the dashboard must not need the training environment.

    "C:\Program Files\python311\python.exe" webapp/tests/test_normalize.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from server.normalize import CLASS_COLORS, CLASSES, normalize_run  # noqa: E402

RESULTS = ROOT.parent / "fl_rnaseq" / "results"

failures = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not cond:
        failures.append(name)


def main() -> int:
    runs = {}
    for p in sorted(RESULTS.glob("*/*/metrics.json")):
        raw = json.loads(p.read_text(encoding="utf-8"))
        if "split_fingerprint" not in raw:
            continue
        runs[f"{p.parent.parent.name}/{p.parent.name}"] = normalize_run(
            raw, p.parent.parent.name
        )

    print(f"\n{len(runs)} runs normalized\n")
    check("at least 15 runs found", len(runs) >= 15, str(len(runs)))

    # The property the whole frontend rests on.
    key_sets = {k: frozenset(v) for k, v in runs.items()}
    base = next(iter(key_sets.values()))
    odd = [k for k, v in key_sets.items() if v != base]
    check("every run has an identical top-level key set", not odd, str(odd))

    # One regression per documented inconsistency in the source files.
    check("logreg_l1 has no curve (no history key)",
          runs["centralized/logreg_l1"]["curves"]["kind"] is None)
    check("logreg curve is the CV path on a log axis",
          runs["centralized/logreg"]["curves"]["kind"] == "cv"
          and runs["centralized/logreg"]["curves"]["x"]["scale"] == "log")
    check("mlp curve is epochs, not rounds",
          runs["centralized/mlp_30ep"]["curves"]["kind"] == "epochs")
    check("federated curve marks round 0 as the untrained init",
          runs["federated/iid_seed42"]["curves"]["x"]["zero_is_init"] is True)
    check("dirichlet-legacy_a0.1 has no per_class block",
          runs["local_only/dirichlet-legacy_a0.1_seed42"]["per_class"] is None)
    check("local_only per_class carries no precision",
          all(e["precision"] is None
              for e in (runs["local_only/iid_seed42"]["per_class"] or [])))
    check("no per_class entry has support 0",
          all(e["support"] > 0 for r in runs.values() for e in (r["per_class"] or [])))
    check("local_only runs expose 5 sites",
          len(runs["local_only/dirichlet_a0.1_seed42"]["sites"]) == 5)
    check("logreg_l1 reports no parameter count",
          runs["centralized/logreg_l1"]["model"]["params"] is None)
    check("logreg parameter count is coalesced from num_params",
          runs["centralized/logreg"]["model"]["params"] == 101325)

    # Duplicated constants must not drift from fl_rnaseq.
    ds = (ROOT.parent / "fl_rnaseq" / "fl_rnaseq" / "dataset.py").read_text(encoding="utf-8")
    mt = (ROOT.parent / "fl_rnaseq" / "fl_rnaseq" / "metrics.py").read_text(encoding="utf-8")
    check("CLASSES still matches dataset.py",
          f'CLASSES = {CLASSES!r}'.replace("'", '"') in ds.replace("'", '"'))
    check("CLASS_COLORS still match metrics.py",
          all(f'"{c}": "{h}"' in mt for c, h in CLASS_COLORS.items()))

    print()
    if failures:
        print(f"FAILED ({len(failures)}): {', '.join(failures)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
