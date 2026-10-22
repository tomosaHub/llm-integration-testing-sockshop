"""
restore_invalid_repairs.py  —  undo repairs that were made against the wrong errors

A repair is only valid if the error output it was given came from the backend
services through the test router. If Sock Shop was down, or the edge-router was
the entry point, the model was asked to "fix" connection errors or front-end
refusals instead, and the repaired file must not be used.

For every run whose pytest_pre_repair.txt shows such errors, this script:
  1. moves the invalid repair into runs/<run>/invalid_repair/<timestamp>/
     (the repaired test file, prompt_3.txt, response_3.txt, repair.json and
     the error output), so the attempt and its cost stay on record
  2. puts the original test file back from runs/<run>/pre_repair/
  3. removes pre_repair/, so repair_all.py will repair the run again

Usage (from the folder containing runs/):
    python3 restore_invalid_repairs.py            # show what it would do
    python3 restore_invalid_repairs.py --apply    # do it
"""

import argparse
import shutil
import time
from pathlib import Path

RUNS_DIR = Path("runs")
BAD_SIGNS = ("Connection refused", "Max retries exceeded",
             "Failed to establish a new connection",
             "Cannot POST", "Cannot GET", "Cannot PUT", "Cannot DELETE",
             "User not logged in", '"error":{}')   # front-end error format
ARTEFACTS = ("repair.json", "prompt_3.txt", "response_3.txt",
             "pytest_pre_repair.txt")


def main():
    ap = argparse.ArgumentParser(description="Undo repairs made against the wrong errors.")
    ap.add_argument("--apply", action="store_true", help="Actually make the changes")
    ap.add_argument("--runs", nargs="+", default=[],
                    help="Also restore these runs by name, e.g. --runs full-r1 ablated-r5")
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    bad = []
    for run_dir in sorted(d for d in RUNS_DIR.iterdir() if d.is_dir()):
        err = run_dir / "pytest_pre_repair.txt"
        named = run_dir.name in args.runs
        if named or (err.exists() and any(s in err.read_text(errors="replace") for s in BAD_SIGNS)):
            bad.append(run_dir)
    missing = set(args.runs) - {d.name for d in bad}
    if missing:
        print(f"[RESTORE] WARNING: no such run(s): {', '.join(sorted(missing))}")

    print(f"[RESTORE] {len(bad)} run(s) to restore")
    lost_cost = 0.0
    for run_dir in bad:
        backup = run_dir / "pre_repair"
        originals = sorted(backup.glob("test_*.py")) if backup.exists() else []
        if not (run_dir / "repair.json").exists():
            print(f"  [SKIP] {run_dir.name}: not repaired yet, nothing to restore")
            continue
        if len(originals) != 1:
            print(f"  [SKIP] {run_dir.name}: expected 1 original in pre_repair/, "
                  f"found {len(originals)}; fix by hand")
            continue
        original = originals[0]
        current = run_dir / "tests" / original.name
        log = run_dir / "repair.json"
        if log.exists():
            import json
            lost_cost += sum(e.get("cost_usd") or 0.0 for e in json.loads(log.read_text()))

        print(f"  {run_dir.name}: restore {original.name}")
        if not args.apply:
            continue

        archive = run_dir / "invalid_repair" / stamp
        archive.mkdir(parents=True, exist_ok=True)
        if current.exists():
            shutil.move(str(current), archive / ("repaired_" + original.name))
        for name in ARTEFACTS:
            p = run_dir / name
            if p.exists():
                shutil.move(str(p), archive / name)
        shutil.copy2(original, current)
        shutil.rmtree(backup)

    print(f"[RESTORE] Cost of the invalid repairs: ${lost_cost:.4f}")
    if not args.apply:
        print("[RESTORE] Dry run only. Re-run with --apply to make these changes.")
    else:
        print("[RESTORE] Done. Next: python3 router.py up, then python3 repair_all.py --ir ir.json")


if __name__ == "__main__":
    main()