"""
audit_repairs.py  —  did the repair stage change what tests EXPECT?

For every run that was repaired, compares the original test file
(runs/<run>/pre_repair/) with the repaired one (runs/<run>/tests/) and flags,
per test function:

  RENAMED      a test that disappeared and a new one that appeared
  FLIPPED      the expected status codes changed between failure (>=400)
               and success (2xx), or the reverse: the test now expects the
               opposite outcome
  ASSERTS_CHG  the assert statements changed in some other way

It also reports, for every PASSING test in results/per_test_*.csv, whether its
assertions check anything beyond "an error came back" (WEAK_NEGATIVE).

Writes results/repair_audit.csv and prints a summary. Read-only: changes no
test files.

    python3 audit_repairs.py
"""

import ast
import csv
import re
from pathlib import Path

RUNS_DIR = Path("runs")
RESULTS = Path("results")

ERR_PAT = re.compile(r">=\s*400|>\s*399|in\s*\(\s*4\d\d|==\s*4\d\d|in\s*\(\s*5\d\d|==\s*5\d\d|"
                     r"raise_for_status|status_code\s*>=\s*500")
OK_PAT  = re.compile(r"in\s*\(\s*2\d\d|==\s*2\d\d|<\s*300|<\s*400|\.ok\b")


def functions(path: Path) -> dict[str, tuple[list[str], str]]:
    """test name -> (assert source lines, full source)."""
    src = path.read_text(errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}
    out = {}
    lines = src.splitlines()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            body = "\n".join(lines[node.lineno - 1: node.end_lineno])
            asserts = [ast.get_source_segment(src, n) or "" for n in ast.walk(node)
                       if isinstance(n, ast.Assert)]
            out[node.name] = (asserts, body)
    return out


def helper_asserts(path: Path) -> dict[str, list[str]]:
    """non-test helper name -> its assert lines (to follow _assert_* helpers)."""
    src = path.read_text(errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}
    return {n.name: [ast.get_source_segment(src, a) or "" for a in ast.walk(n) if isinstance(a, ast.Assert)]
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("test_")}


def with_helpers(path: Path) -> dict[str, tuple[list[str], str]]:
    """Like functions(), but each test's asserts include those of the helpers it calls."""
    fns, helpers = functions(path), helper_asserts(path)
    out = {}
    for name, (asserts, body) in fns.items():
        extra = []
        for h, h_asserts in helpers.items():
            if re.search(rf"\b{re.escape(h)}\s*\(", body):
                extra += h_asserts
        out[name] = (asserts + extra, body)
    return out


GENERIC_KEYS = {"error", "errors", "message", "messages", "detail", "details",
                "status", "timestamp", "exception", "path"}


def checks_specific_content(asserts: list[str]) -> bool:
    """True if an assertion checks a specific value or word, not just that an error exists."""
    text = " ".join(asserts).lower()
    for m in re.finditer(r"['\"]([a-z_ ]{3,})['\"]\s+(?:not\s+)?in\b", text):
        if m.group(1).strip() not in GENERIC_KEYS:
            return True
    return bool(re.search(r"==\s*['\"][^'\"]+['\"]|==\s*-?\d+\.?\d*\b(?!\s*\))", text)
             and not re.fullmatch(r"[^=]*status_code\s*==\s*\d+[^=]*", text))


def expectation(asserts: list[str]) -> str:
    text = " ".join(asserts)
    e, o = bool(ERR_PAT.search(text)), bool(OK_PAT.search(text))
    return "error" if e and not o else "success" if o and not e else "mixed" if e and o else "none"


def stem(name: str) -> str:
    """name with outcome words removed, to pair renamed tests."""
    words = ["rejected", "accepted", "rejects", "accepts", "fails", "succeeds", "failure",
             "success", "allowed", "denied", "blocked", "invalid", "valid"]
    s = name
    for w in words:
        s = s.replace("_" + w, "")
    return s


def main():
    rows = []
    for run in sorted(d for d in RUNS_DIR.iterdir() if d.is_dir()):
        pre_dir = run / "pre_repair"
        if not (run / "repair.json").exists() or not pre_dir.exists():
            continue
        pres = list(pre_dir.glob("test_*.py"))
        if len(pres) != 1:
            continue
        before = with_helpers(pres[0])
        after = with_helpers(run / "tests" / pres[0].name)
        gone = [n for n in before if n not in after]
        new = [n for n in after if n not in before]
        pairs = {n: n for n in before if n in after}
        for g in gone:
            match = [n for n in new if stem(n) == stem(g)]
            if match:
                pairs[g] = match[0]
                new.remove(match[0])
        for old, cur in pairs.items():
            a_old, _ = before[old]
            a_new, _ = after[cur]
            e_old, e_new = expectation(a_old), expectation(a_new)
            flags = []
            if old != cur:
                flags.append("RENAMED")
            if {e_old, e_new} == {"error", "success"}:
                flags.append("FLIPPED")
            elif sorted(a_old) != sorted(a_new):
                flags.append("ASSERTS_CHG")
            rows.append({"run_id": run.name, "test_before": old, "test_after": cur,
                         "expects_before": e_old, "expects_after": e_new,
                         "flags": " ".join(flags) or "UNCHANGED"})
        for g in gone:
            if g not in pairs:
                rows.append({"run_id": run.name, "test_before": g, "test_after": "",
                             "expects_before": expectation(before[g][0]), "expects_after": "",
                             "flags": "REMOVED"})
        for n in new:
            rows.append({"run_id": run.name, "test_before": "", "test_after": n,
                         "expects_before": "", "expects_after": expectation(after[n][0]),
                         "flags": "ADDED"})

    # Weak negative tests among the PASSING tests
    weak = []
    for f in sorted(RESULTS.glob("per_test_*.csv")):
        for r in csv.DictReader(open(f)):
            if r["outcome"] != "passed":
                continue
            tf = sorted((RUNS_DIR / r["run_id"] / "tests").glob("test_*.py"))
            if not tf:
                continue
            fns = with_helpers(tf[0])
            if r["test"] not in fns:
                continue
            asserts, _ = fns[r["test"]]
            exp = expectation(asserts)
            specific = checks_specific_content(asserts)
            weak.append({"mode": r["mode"], "run_id": r["run_id"], "test": r["test"],
                         "expects": exp, "checks_specific_content": specific,
                         "verdict": "WEAK_NEGATIVE" if exp == "error" and not specific else "ok"})

    RESULTS.mkdir(exist_ok=True)
    if rows:
        with open(RESULTS / "repair_audit.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    if weak:
        with open(RESULTS / "passing_test_audit.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(weak[0]))
            w.writeheader(); w.writerows(weak)

    print("[AUDIT] Repair changes per mode")
    for mode in ("full", "ablated", "baseline"):
        mr = [r for r in rows if r["run_id"].startswith(mode + "-")]
        count = lambda flag: sum(flag in r["flags"].split() for r in mr)
        print(f"  {mode:<9} tests={len(mr):<4} flipped={count('FLIPPED'):<3} renamed={count('RENAMED'):<3} "
              f"asserts_changed={count('ASSERTS_CHG'):<4} removed={count('REMOVED'):<3} added={count('ADDED'):<3} "
              f"unchanged={count('UNCHANGED')}")
    for r in rows:
        if "FLIPPED" in r["flags"]:
            print(f"    FLIPPED  {r['run_id']}: {r['test_before']} -> {r['test_after']} "
                  f"({r['expects_before']} -> {r['expects_after']})")

    print("\n[AUDIT] Passing tests")
    for mode in ("full", "ablated", "baseline"):
        mw = [w for w in weak if w["mode"] == mode]
        print(f"  {mode:<9} passing={len(mw):<4} weak_negative={sum(w['verdict'] == 'WEAK_NEGATIVE' for w in mw):<4} "
              f"expect_error={sum(w['expects'] == 'error' for w in mw):<4} "
              f"expect_success={sum(w['expects'] == 'success' for w in mw)}")
    print("\n[AUDIT] Written: results/repair_audit.csv, results/passing_test_audit.csv")
    print("        The flags are heuristics: read the flagged tests before quoting a number.")


if __name__ == "__main__":
    main()
