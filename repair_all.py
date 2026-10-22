"""
repair_all.py  —  apply the repair prompt (Prompt 3) once to every failing test file

Run AFTER compute_metrics.py has produced the pre-repair results, with Sock
Shop up and the test router active (python3 router.py up):

    cp -r results results_pre_repair          # keep the first-attempt numbers
    python3 repair_all.py --ir ir.json        # all modes
    python3 repair_all.py --ir ir.json --mode full

For each run directory in runs/:
  1. skip it if it was already repaired (runs/<run>/pre_repair/ exists), so
     each file is repaired AT MOST ONCE, as the methodology states
  2. run its tests with pytest and save the output to pytest_pre_repair.txt
  3. skip it if every test passed
  4. otherwise call  generate_tests.py --repair <test file> --error <output>
     generate_tests.py backs the original up to pre_repair/, repairs the file
     in place, and records prompt_3.txt, response_3.txt and repair.json

Afterwards, measure again with:
    python3 compute_metrics.py --ir ir.json --mode <mode> --rerun

The error output passed to the model is capped at MAX_ERROR_CHARS so that one
very long traceback cannot dominate the prompt; the cap is the same for every
mode, and the exact text sent is kept in pytest_pre_repair.txt.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

RUNS_DIR        = Path("runs")
MAX_ERROR_CHARS = 12000
PYTEST_TIMEOUT  = 900
MARKERS = ("edge(name): inter-service boundary under test\n"
           "service(name): single service under test")


def run_pytest(tests_dir: Path) -> tuple[int, str]:
    cmd = [sys.executable, "-m", "pytest", str(tests_dir), "-q",
           "--tb=short", "-p", "no:cacheprovider", "-o", f"markers={MARKERS}"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=PYTEST_TIMEOUT)
        return r.returncode, (r.stdout + r.stderr)
    except subprocess.TimeoutExpired:
        return 1, f"pytest timed out after {PYTEST_TIMEOUT}s"


def main():
    ap = argparse.ArgumentParser(description="Repair failing test files once.")
    ap.add_argument("--ir", required=True)
    ap.add_argument("--mode", choices=["full", "ablated", "baseline"])
    ap.add_argument("--mock", action="store_true",
                    help="Pass --mock to generate_tests.py (no API calls)")
    args = ap.parse_args()

    if not Path("generate_tests.py").exists():
        sys.exit("[ERROR] run this from the folder containing generate_tests.py")

    run_dirs = sorted(d for d in RUNS_DIR.iterdir() if d.is_dir())
    if args.mode:
        run_dirs = [d for d in run_dirs if d.name.startswith(args.mode + "-")]

    repaired, skipped, failed = [], [], []
    for run_dir in run_dirs:
        meta_path = run_dir / "meta.json"
        tests_dir = run_dir / "tests"
        if not meta_path.exists() or not tests_dir.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("mock"):
            continue
        # repair.json is written only after a successful repair call, so a
        # run that was backed up but interrupted (e.g. low balance) is retried.
        if (run_dir / "repair.json").exists():
            skipped.append((run_dir.name, "already repaired"))
            continue
        files = sorted(tests_dir.glob("test_*.py"))
        if len(files) != 1:
            skipped.append((run_dir.name, f"{len(files)} test files"))
            continue

        code, output = run_pytest(tests_dir)
        if code == 0:
            skipped.append((run_dir.name, "all tests passed"))
            continue
        if code == 5:   # pytest: no tests collected
            output = output or "pytest collected no tests from this file"

        error_text = output[:MAX_ERROR_CHARS]
        err_path = run_dir / "pytest_pre_repair.txt"
        err_path.write_text(error_text)

        mode = meta.get("mode", run_dir.name.split("-")[0])
        print(f"\n[REPAIR-ALL] {run_dir.name}")
        cmd = [sys.executable, "generate_tests.py", "--ir", args.ir,
               "--mode", mode, "--repair", str(files[0]),
               "--error", str(err_path)]
        if args.mock:
            cmd.append("--mock")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            failed.append(run_dir.name)
            print(f"[REPAIR-ALL] repair failed for {run_dir.name}; "
                  "stopping so you can fix the cause and rerun "
                  "(finished runs are skipped next time).")
            break
        repaired.append(run_dir.name)

    total = 0.0
    for name in repaired:
        log = RUNS_DIR / name / "repair.json"
        if log.exists():
            total += sum(e.get("cost_usd") or 0.0
                         for e in json.loads(log.read_text()))

    print(f"\n[REPAIR-ALL] Repaired {len(repaired)} file(s), "
          f"skipped {len(skipped)}, failed {len(failed)}.")
    for name, why in skipped:
        print(f"             skipped {name}: {why}")
    print(f"[REPAIR-ALL] Estimated repair cost: ${total:.4f}")
    print("[REPAIR-ALL] Next: python3 compute_metrics.py --ir ir.json "
          "--mode <mode> --rerun")


if __name__ == "__main__":
    main()