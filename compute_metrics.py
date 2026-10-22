"""
compute_metrics.py  —  success rate and edge coverage for every generation run

Run AFTER generate_tests.py, with the Sock Shop testbed up:

    python3 compute_metrics.py --ir ir.json

For each run directory in runs/ it:
  1. executes runs/<run>/tests/ with pytest, saving pytest_report.json
     (an existing report is reused unless --rerun is given)
  2. SUCCESS RATE = passed tests / tests the model GENERATED
       The denominator is counted from the source file, not from what pytest
       collected: a file that fails to import collects nothing, and using the
       collected count would silently drop it from the denominator.
  3. COVERAGE = active IR edges targeted by >= 1 PASSING test / active edges
       using the @pytest.mark.edge("a -> b") markers (declared coverage).
       Active edges are those whose confidence is not REJECTED.

Aggregation per mode (mean and variance across repetitions):
  full, ablated  one unit per run
  baseline       one unit per REPETITION: the five single-service suites of a
                 repetition are pooled (success rate) and unioned (coverage),
                 as the methodology describes. Baseline coverage is expected
                 to be 0: its tests are single-service by design.

Truncated runs (meta.json "truncated_any": true) are EXCLUDED from the
aggregates and counted separately: truncation is a harness limit, not a
model failure.

Outputs (in results/):
  per_test.csv    one row per test: outcome, edge, short failure message
  per_run.csv     one row per run
  summary.csv     one row per mode

Requires:  python3 -m pip install pytest pytest-json-report requests
"""

import argparse
import ast
import csv
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

RUNS_DIR     = Path("runs")
RESULTS_DIR  = Path("results")
REPORT_NAME  = "pytest_report.json"
RUN_TIMEOUT  = 900   # seconds for one run's whole test file


# ---------------------------------------------------------------------------
# Reading generated test files
# ---------------------------------------------------------------------------

def normalise_edge(e: str) -> str:
    """'Orders->User' and 'orders -> user' both become 'orders -> user'."""
    parts = [p.strip().lower() for p in e.replace("→", "->").split("->")]
    return " -> ".join(parts) if len(parts) == 2 else e.strip().lower()


def _marker_edge(dec) -> str | None:
    """Return the edge string if dec is @pytest.mark.edge("a -> b")."""
    if not isinstance(dec, ast.Call):
        return None
    f = dec.func
    if (isinstance(f, ast.Attribute) and f.attr == "edge"
            and dec.args and isinstance(dec.args[0], ast.Constant)
            and isinstance(dec.args[0].value, str)):
        return normalise_edge(dec.args[0].value)
    return None


def read_tests(test_file: Path) -> tuple[dict[str, str | None], bool]:
    """
    Map each generated test function -> its declared edge (or None).
    Returns (mapping, parsed_ok). A file with a syntax error still has its
    test functions counted by regex, so they enter the denominator as failures.
    """
    src = test_file.read_text(errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        names = re.findall(r"^\s*(?:async\s+)?def\s+(test_\w+)", src, re.MULTILINE)
        return {n: None for n in names}, False
    out = {}
    for node in ast.walk(tree):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test_")):
            edge = None
            for dec in node.decorator_list:
                edge = _marker_edge(dec) or edge
            out[node.name] = edge
    return out, True


# ---------------------------------------------------------------------------
# Running pytest
# ---------------------------------------------------------------------------

def run_pytest(run_dir: Path, rerun: bool) -> dict | None:
    report_path = run_dir / REPORT_NAME
    if report_path.exists() and not rerun:
        return json.loads(report_path.read_text())

    tests_dir = run_dir / "tests"
    cmd = [
        sys.executable, "-m", "pytest", str(tests_dir),
        "-q", "-p", "no:cacheprovider",
        # Register the markers so pytest does not warn about them.
        "-o", "markers=edge(name): inter-service boundary under test\n"
              "service(name): single service under test",
        "--json-report", f"--json-report-file={report_path}",
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True,
                       timeout=RUN_TIMEOUT, cwd=Path.cwd())
    except subprocess.TimeoutExpired:
        print(f"  [WARN] pytest timed out after {RUN_TIMEOUT}s in {run_dir}")
    if not report_path.exists():
        return None
    return json.loads(report_path.read_text())


def _short(msg: str, limit: int = 160) -> str:
    msg = " ".join((msg or "").split())
    return msg[:limit]


def test_outcomes(report: dict | None) -> tuple[dict[str, tuple[str, str]], str]:
    """
    Map test function name -> (outcome, short message) from a pytest JSON
    report. Also returns a collection-error message if the file failed to load.
    """
    outcomes, collect_err = {}, ""
    if not report:
        return outcomes, "no pytest report produced"
    for c in report.get("collectors", []):
        if c.get("outcome") == "failed":
            collect_err = _short(c.get("longrepr", "collection failed"))
    for t in report.get("tests", []):
        name = t["nodeid"].split("::")[-1].split("[")[0]
        outcome = t.get("outcome", "failed")
        msg = ""
        for phase in ("setup", "call", "teardown"):
            ph = t.get(phase) or {}
            crash = ph.get("crash") or {}
            if crash.get("message"):
                msg = crash["message"]
                break
        outcomes[name] = (outcome, _short(msg))
    return outcomes, collect_err


# ---------------------------------------------------------------------------
# Per-run and per-mode metrics
# ---------------------------------------------------------------------------

def mean_var(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    return statistics.mean(xs), (statistics.variance(xs) if len(xs) > 1 else 0.0)


def evaluate_run(run_dir: Path, active_edges: set[str], rerun: bool,
                 per_test_rows: list[dict]) -> dict | None:
    meta_path = run_dir / "meta.json"
    tests_dir = run_dir / "tests"
    if not meta_path.exists() or not tests_dir.exists():
        print(f"  [SKIP] {run_dir.name}: no meta.json or tests/ (incomplete run)")
        return None
    meta = json.loads(meta_path.read_text())
    if meta.get("mock"):
        print(f"  [SKIP] {run_dir.name}: mock run")
        return None
    files = sorted(tests_dir.glob("test_*.py"))
    if not files:
        print(f"  [SKIP] {run_dir.name}: no test file")
        return None

    generated: dict[str, str | None] = {}
    parsed_ok = True
    for f in files:
        g, ok = read_tests(f)
        generated.update(g)
        parsed_ok = parsed_ok and ok

    print(f"  [RUN ] {run_dir.name}  ({len(generated)} generated tests)")
    report = run_pytest(run_dir, rerun)
    outcomes, collect_err = test_outcomes(report)

    passed = 0
    covered = set()
    for name, edge in generated.items():
        outcome, msg = outcomes.get(
            name, ("not_run", collect_err or "not collected by pytest"))
        if outcome == "passed":
            passed += 1
            if edge and edge in active_edges:
                covered.add(edge)
        per_test_rows.append({
            "run_id": run_dir.name, "mode": meta.get("mode", ""),
            "service": meta.get("service", ""), "test": name,
            "edge": edge or "", "outcome": outcome, "message": msg,
        })

    n = len(generated)
    declared = {e for e in generated.values() if e and e in active_edges}
    return {
        "run_id":         run_dir.name,
        "mode":           meta.get("mode", ""),
        "service":        meta.get("service", ""),
        "repeat":         run_dir.name.rsplit("-r", 1)[-1],
        "truncated":      bool(meta.get("truncated_any")),
        "parsed_ok":      parsed_ok,
        "collect_error":  collect_err,
        "generated":      n,
        "passed":         passed,
        "success_rate":   round(passed / n, 4) if n else 0.0,
        "edges_declared": "; ".join(sorted(declared)),
        "edges_covered":  "; ".join(sorted(covered)),
        "coverage":       round(len(covered) / len(active_edges), 4)
                          if active_edges else 0.0,
        "cost_usd":       meta.get("cost_usd", ""),
    }


def summarise(rows: list[dict], active_edges: set[str]) -> list[dict]:
    summary = []
    for mode in ("full", "ablated", "baseline"):
        mrows  = [r for r in rows if r["mode"] == mode]
        usable = [r for r in mrows if not r["truncated"]]
        if not mrows:
            continue
        if mode == "baseline":
            reps: dict[str, list[dict]] = {}
            for r in usable:
                reps.setdefault(r["repeat"], []).append(r)
            sr, cov = [], []
            for _, rs in sorted(reps.items()):
                gen = sum(r["generated"] for r in rs)
                sr.append(sum(r["passed"] for r in rs) / gen if gen else 0.0)
                edges = set()
                for r in rs:
                    if r["edges_covered"]:
                        edges |= set(r["edges_covered"].split("; "))
                cov.append(len(edges) / len(active_edges) if active_edges else 0.0)
            units = len(reps)
        else:
            sr  = [r["success_rate"] for r in usable]
            cov = [r["coverage"] for r in usable]
            units = len(usable)

        sr_m, sr_v   = mean_var(sr)
        cov_m, cov_v = mean_var(cov)
        summary.append({
            "mode":               mode,
            "repetitions":        units,
            "truncated_excluded": sum(1 for r in mrows if r["truncated"]),
            "tests_generated":    sum(r["generated"] for r in usable),
            "tests_passed":       sum(r["passed"] for r in usable),
            "success_rate_mean":  round(sr_m, 3),
            "success_rate_var":   round(sr_v, 4),
            "coverage_mean":      round(cov_m, 3),
            "coverage_var":       round(cov_v, 4),
        })
    return summary


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def print_table(rows: list[dict], cols: list[str]):
    if not rows:
        print("  (none)")
        return
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def main():
    ap = argparse.ArgumentParser(
        description="Success rate and edge coverage per run and per mode.")
    ap.add_argument("--ir", required=True, help="Path to ir.json")
    ap.add_argument("--mode", choices=["full", "ablated", "baseline"],
                    help="Only evaluate runs of this mode")
    ap.add_argument("--rerun", action="store_true",
                    help="Re-execute tests even if a pytest report exists")
    args = ap.parse_args()

    try:
        import pytest_jsonreport  # noqa: F401
    except ImportError:
        sys.exit("[ERROR] pytest-json-report is not installed.\n"
                 "        Run: python3 -m pip install pytest pytest-json-report requests")

    ir = json.loads(Path(args.ir).read_text())
    active_edges = {
        normalise_edge(f"{e['from_service']} -> {e['to_service']}")
        for e in ir["edges"] if e["confidence"] != "REJECTED"
    }
    print(f"[METRICS] Active IR edges ({len(active_edges)}): "
          f"{', '.join(sorted(active_edges))}")

    run_dirs = sorted(d for d in RUNS_DIR.iterdir() if d.is_dir()) \
        if RUNS_DIR.exists() else []
    if args.mode:
        run_dirs = [d for d in run_dirs if d.name.startswith(args.mode + "-")]
    if not run_dirs:
        sys.exit(f"[ERROR] No run directories found in {RUNS_DIR}/")

    per_test: list[dict] = []
    rows = []
    for d in run_dirs:
        r = evaluate_run(d, active_edges, args.rerun, per_test)
        if r:
            rows.append(r)

    RESULTS_DIR.mkdir(exist_ok=True)
    summary = summarise(rows, active_edges)
    suffix = f"_{args.mode}" if args.mode else ""
    write_csv(RESULTS_DIR / f"per_test{suffix}.csv", per_test)
    write_csv(RESULTS_DIR / f"per_run{suffix}.csv", rows)
    write_csv(RESULTS_DIR / f"summary{suffix}.csv", summary)

    print("\n[METRICS] Per run")
    print_table(rows, ["run_id", "generated", "passed", "success_rate",
                       "coverage", "edges_covered", "truncated"])
    print("\n[METRICS] Per mode")
    print_table(summary, ["mode", "repetitions", "tests_generated",
                          "tests_passed", "success_rate_mean",
                          "success_rate_var", "coverage_mean", "coverage_var",
                          "truncated_excluded"])
    print(f"\n[METRICS] Written to {RESULTS_DIR}/ "
          f"(per_test{suffix}.csv, per_run{suffix}.csv, summary{suffix}.csv)")


if __name__ == "__main__":
    main()
