"""
generate_tests.py  —  LLM-driven integration test generation
LLM-Based Integration Testing of Inter-Service Communication
in Microservice Architectures

Implements the three-prompt chained pipeline:
  Prompt 1 — Scenario extraction
  Prompt 2 — Test case generation (executable pytest code)
  Prompt 3 — Test repair

Modes:
  --mode full      IR graph + specs            (RQ1, and RQ3 full condition)
  --mode ablated   specs only, no IR           (RQ3 ablated condition)
  --mode baseline  one spec at a time, no IR   (RQ2 single-service condition)

  --mock           Save prompts to disk instead of calling the API.
  --model NAME     DeepSeek model ID (default: DEFAULT_MODEL below). Use the
                   SAME model for every mode and repetition.
  --list-models    Print the DeepSeek models your API key can use, then exit.

MODEL PROVIDER
--------------
Generation uses the DeepSeek API through its OpenAI-compatible endpoint.
  pip install openai
  export DEEPSEEK_API_KEY='...'    (from https://platform.deepseek.com)

Every call logs input / cache-hit / output / reasoning tokens and an
estimated USD cost (peak or off-peak rate, chosen by the UTC time of the
call). Per-run totals go to meta.json and runs/manifest.csv.

PROMPT SYMMETRY (important for RQ3 internal validity)
-----------------------------------------------------
build_prompt_1_full() and build_prompt_1_ablated() are generated from one
shared template. The ONLY difference between them is the presence of the
dependency-graph block. Instruction count, wording, and the requested JSON
schema are identical. Do not add an instruction to one without adding it to
the other; assert_prompt_symmetry() checks this at startup and refuses to run
if it is violated.

Each run writes a self-contained artefact directory under runs/:
  runs/full-r1/
    prompt_1.txt  response_1.txt  scenarios.json
    prompt_2.txt  response_2.txt
    tests/test_full_r1.py
    timing.json   meta.json
"""

import argparse
import csv
import difflib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "deepseek-v4-pro"    # override with --model
MODEL        = DEFAULT_MODEL          # set from --model in main()
BASE_URL     = "https://api.deepseek.com"
# Thinking mode is DeepSeek's default and is left on for every mode, so the
# condition is identical across full / ablated / baseline. Reasoning tokens
# count against max_tokens, so keep generous headroom for the test file.
MAX_TOKENS   = 65536   # trial run used 29.7k (27.6k reasoning); 32k was too tight
TEMPERATURE  = 1.0     # kept consistent across runs (may be ignored in thinking mode)
REQUEST_TIMEOUT = 1200.0  # seconds; trial test-generation call took 449s
MAX_RETRIES  = 6       # for 429 rate limits and 500/502/504
MAX_BACKOFF  = 90.0    # seconds
# 503 = server overloaded on DeepSeek's side; wait long, try few times.
OVERLOAD_WAITS = (30, 60, 120, 240)   # seconds; 5 attempts over ~7.5 min
MOCK_DIR     = Path("mock_prompts")
RUNS_DIR     = Path("runs")

# USD per 1M tokens, from https://api-docs.deepseek.com/quick_start/pricing
# (read 13 Oct 2026). Off-peak is half of peak. Update if DeepSeek changes it.
PRICES = {
    "deepseek-v4-pro": {
        "peak":     {"hit": 0.044, "miss": 1.32, "out": 3.96},
        "off_peak": {"hit": 0.022, "miss": 0.66, "out": 1.98},
    },
    "deepseek-flash": {
        "peak":     {"hit": 0.006, "miss": 0.30, "out": 1.20},
        "off_peak": {"hit": 0.003, "miss": 0.15, "out": 0.60},
    },
}
# Peak windows in UTC, Monday-Friday: 01:00-04:00 and 06:00-10:00.
# (Chinese public holidays are off-peak too; they are treated as peak here,
# so the logged cost can only over-estimate, never under-estimate.)
PEAK_HOURS_UTC = set(range(1, 4)) | set(range(6, 10))

SESSION_COST_USD = 0.0   # running total for this invocation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences if present."""
    code = text.strip()
    if code.startswith("```"):
        code = "\n".join(code.split("\n")[1:])
    if code.endswith("```"):
        code = "\n".join(code.split("\n")[:-1])
    return code.strip()


def parse_scenarios(raw: str) -> dict:
    """
    Parse the scenario JSON returned by Prompt 1.
    Returns {"scenarios": [...]} on success, or
    {"scenarios": [], "parse_error": "..."} on failure.

    Parse failure is recorded rather than retried: it is itself a result.
    """
    text = strip_fences(raw)
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "scenarios" in data:
            return data
        return {"scenarios": [], "parse_error": "no 'scenarios' key in response"}
    except json.JSONDecodeError as e:
        # last resort: pull the outermost {...} block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                if isinstance(data, dict) and "scenarios" in data:
                    return data
            except json.JSONDecodeError:
                pass
        return {"scenarios": [], "parse_error": f"{type(e).__name__}: {e}"}


def _spec_summary(service: str, spec: dict) -> str:
    """Compact, LLM-readable summary of one service's endpoints."""
    lines = [f"Service: {service}"]
    for ep in spec.get("endpoints", []):
        params = ", ".join(ep["path_params"]) if ep["path_params"] else "none"
        body   = "yes" if ep["body_schema"] else "no"
        resp   = "yes" if ep["response_schema"] else "no"
        lines.append(
            f"  {ep['method']:<7} {ep['path']:<40}"
            f"  path_params=[{params}]  body={body}  response={resp}"
        )
        if ep["summary"]:
            lines.append(f"           description: {ep['summary']}")
    return "\n".join(lines)


def _ir_summary(edges: list[dict]) -> str:
    """
    Compact summary of the IR graph for inclusion in prompts.
    REJECTED edges are excluded: they are candidates the direction check
    discarded, not call relationships.
    """
    active = [e for e in edges if e["confidence"] != "REJECTED"]
    lines  = [
        f"Cross-service dependency graph ({len(active)} edges):",
        "Confidence tiers: CONFIRMED = schema evidence and architecture "
        "annotation agree; KNOWN_CALL = architecture annotation only; "
        "CANDIDATE = schema evidence only, direction unverified.",
        "",
    ]
    for e in active:
        fields = (
            f"  shared fields: {e['shared_fields']}"
            if e.get("shared_fields") else ""
        )
        lines.append(
            f"  [{e['confidence']}]  {e['from_service']} -> {e['to_service']}"
        )
        lines.append(f"  evidence: {e['evidence']}{fields}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt 1 — built from ONE template so full and ablated stay symmetric
# ---------------------------------------------------------------------------

_P1_TEMPLATE = """You are an expert software test engineer specialising in microservice integration testing.

You are given the OpenAPI specifications for a microservice system called Sock Shop — an e-commerce application.

Your task is to identify meaningful INTEGRATION SCENARIOS that test the communication BETWEEN services, not within a single service.

OPENAPI SPECIFICATIONS
======================
{spec_block}
{graph_section}
INSTRUCTIONS
============
1. Identify 5 to 8 integration scenarios that exercise communication between services.
2. For each scenario write:
   - A short descriptive name (e.g. "place_order_triggers_payment")
   - The services involved
   - The inter-service boundary being tested, as a directed pair
   - A Given/When/Then description of the scenario
3. Include at least one negative scenario (e.g. payment authorisation failure).
4. Do NOT generate test code yet — scenarios only.

Respond in JSON with this structure:
{{
  "scenarios": [
    {{
      "name": "scenario_name",
      "services_involved": ["service_a", "service_b"],
      "target_edge": "service_a -> service_b",
      "given": "...",
      "when": "...",
      "then": "..."
    }}
  ]
}}"""

_GRAPH_SECTION = """
CROSS-SERVICE DEPENDENCY GRAPH (INTERMEDIATE REPRESENTATION)
=============================================================
{ir_block}
"""


def build_prompt_1_full(ir: dict) -> str:
    spec_block = "\n\n".join(
        _spec_summary(svc, data) for svc, data in ir["services"].items()
    )
    graph_section = _GRAPH_SECTION.format(ir_block=_ir_summary(ir["edges"]))
    return _P1_TEMPLATE.format(spec_block=spec_block, graph_section=graph_section)


def build_prompt_1_ablated(ir: dict) -> str:
    spec_block = "\n\n".join(
        _spec_summary(svc, data) for svc, data in ir["services"].items()
    )
    return _P1_TEMPLATE.format(spec_block=spec_block, graph_section="")


def assert_prompt_symmetry(ir: dict) -> str:
    """
    Verify that the full and ablated Prompt 1 texts differ ONLY by the
    dependency-graph block. Returns a unified diff for the run record.
    Exits if any line outside the graph block differs.
    """
    full    = build_prompt_1_full(ir).splitlines()
    ablated = build_prompt_1_ablated(ir).splitlines()
    diff    = list(difflib.unified_diff(ablated, full, "ablated", "full", lineterm=""))

    graph_block = _GRAPH_SECTION.format(
        ir_block=_ir_summary(ir["edges"])
    ).splitlines()
    allowed = set(graph_block) | {""}

    for line in diff:
        if line.startswith(("---", "+++", "@@", " ")):
            continue
        if line.startswith("-"):
            sys.exit(
                "[ERROR] Prompt asymmetry: the ablated prompt contains a line "
                f"the full prompt does not:\n  {line}"
            )
        if line.startswith("+") and line[1:] not in allowed:
            sys.exit(
                "[ERROR] Prompt asymmetry: the full prompt contains a line "
                f"outside the dependency-graph block:\n  {line}\n"
                "        Full and ablated must differ only by the IR block."
            )
    return "\n".join(diff)


def build_prompt_1_baseline(service: str, spec: dict) -> str:
    """Single-service scenario extraction. No cross-service context."""
    spec_block = _spec_summary(service, spec)
    return f"""You are an expert software test engineer specialising in REST API testing.

You are given the OpenAPI specification for the {service} service from Sock Shop.

Your task is to identify meaningful test scenarios for this single service.

OPENAPI SPECIFICATION
=====================
{spec_block}

INSTRUCTIONS
============
1. Identify 3 to 5 test scenarios for this service.
2. For each scenario write:
   - A short descriptive name
   - A Given/When/Then description
   - The endpoint(s) under test
3. Include at least one negative scenario.
4. Do NOT generate test code yet — scenarios only.

Respond in JSON with this structure:
{{
  "scenarios": [
    {{
      "name": "scenario_name",
      "endpoints": ["METHOD /path"],
      "given": "...",
      "when": "...",
      "then": "..."
    }}
  ]
}}"""


# ---------------------------------------------------------------------------
# Prompt 2 — mode-specific: baseline gets NO cross-service instructions
# ---------------------------------------------------------------------------

_P2_COMMON_REQUIREMENTS = """- Use pytest and the requests library
- Base URL: http://127.0.0.1 (Sock Shop running via Docker Compose)
- Each scenario becomes one pytest test function
- Function names must match the scenario name (prefix with test_)
- Include setup steps (create user, add items to cart, etc.) before assertions
- Assert specific response fields, not just status codes
- Include both happy-path and the negative scenario
- Add a brief docstring to each test explaining what it tests
- Do NOT use pytest fixtures for now — keep each test self-contained"""


def build_prompt_2_cross_service(scenarios_json: str, ir: dict, mode: str) -> str:
    """Prompt 2 for full and ablated. IR context is included in full only."""
    ir_context = (
        f"\nCross-service dependency graph for reference:\n"
        f"{_ir_summary(ir['edges'])}\n"
        if mode == "full" else ""
    )
    return f"""You are an expert software test engineer.

You have previously identified the following integration test scenarios:

{scenarios_json}
{ir_context}
INSTRUCTIONS
============
Generate executable Python pytest test code for ALL scenarios above.

Requirements:
{_P2_COMMON_REQUIREMENTS}
- For cross-service scenarios, assert that data flows correctly across the boundary
  (e.g. an order response contains the correct cart items and customer details)
- Immediately above each test function, add a marker naming the boundary it
  targets, taken from that scenario's "target_edge" value, in this exact form:
      @pytest.mark.edge("service_a -> service_b")

Respond with ONLY the Python code, no explanation, no markdown fences."""


def build_prompt_2_baseline(scenarios_json: str, service: str) -> str:
    """
    Prompt 2 for the single-service baseline.

    Deliberately contains NO cross-service instruction: the baseline is the
    matched single-service condition for RQ2, and telling it to write
    cross-boundary assertions would contaminate the comparison.
    """
    return f"""You are an expert software test engineer.

You have previously identified the following test scenarios for the {service} service:

{scenarios_json}

INSTRUCTIONS
============
Generate executable Python pytest test code for ALL scenarios above.

Requirements:
{_P2_COMMON_REQUIREMENTS}
- Immediately above each test function, add this marker:
      @pytest.mark.service("{service}")

Respond with ONLY the Python code, no explanation, no markdown fences."""


def build_prompt_3(test_code: str, error_output: str) -> str:
    return f"""You are an expert software test engineer debugging integration tests.

The following pytest test code was generated but produced errors when executed:

TEST CODE
=========
{test_code}

ERROR OUTPUT
============
{error_output}

INSTRUCTIONS
============
1. Analyse the error output carefully.
2. Fix the test code to resolve all errors.
3. Common issues to check:
   - Incorrect endpoint paths (check the OpenAPI spec paths)
   - Missing required request fields
   - Wrong assertion on response structure
   - Authentication or session handling missing
   - Path parameter format errors
4. Preserve every @pytest.mark marker exactly as it appears.
5. Return the complete corrected test code.

Respond with ONLY the corrected Python code, no explanation, no markdown fences."""


# ---------------------------------------------------------------------------
# API caller
# ---------------------------------------------------------------------------

def _get_client():
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("[ERROR] openai package not installed.\n"
                 "        Run: pip install openai")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        sys.exit("[ERROR] DEEPSEEK_API_KEY not set in environment.\n"
                 "        Create a key at https://platform.deepseek.com, add\n"
                 "        export DEEPSEEK_API_KEY='...' to ~/.zshrc, then run\n"
                 "        source ~/.zshrc\n"
                 "        Or use --mock to save prompts without calling the API.")
    # max_retries=0 disables the SDK's own hidden retries so that every
    # request is made, counted and logged by call_llm() alone.
    # Accept-Encoding is pinned to gzip/deflate: some environments (e.g.
    # Anaconda) ship a brotli/zstd package older than the HTTP client expects,
    # and decoding a compressed response then fails with
    # "process() takes no keyword arguments". gzip is always supported.
    return OpenAI(api_key=api_key, base_url=BASE_URL,
                  max_retries=0, timeout=REQUEST_TIMEOUT,
                  default_headers={"Accept-Encoding": "gzip, deflate"})


def _retry_delay(err) -> float | None:
    """Honour a Retry-After header if the server sends one."""
    resp = getattr(err, "response", None)
    if resp is None:
        return None
    value = resp.headers.get("retry-after")
    try:
        return float(value) if value else None
    except ValueError:
        return None


def _price_window(when_utc: datetime) -> str:
    """'peak' or 'off_peak' for a call made at when_utc."""
    weekday = when_utc.weekday() < 5          # Mon=0 .. Fri=4
    return "peak" if weekday and when_utc.hour in PEAK_HOURS_UTC else "off_peak"


def _estimate_cost(model: str, hit: int, miss: int, out: int,
                   window: str) -> float | None:
    """Estimated USD for one call; None if the model has no price entry."""
    table = PRICES.get(model)
    if not table:
        return None
    p = table[window]
    return (hit * p["hit"] + miss * p["miss"] + out * p["out"]) / 1_000_000


def call_llm(prompt: str, run_label: str) -> tuple[str, dict]:
    """
    Call the DeepSeek API. Returns (response_text, metadata).
    Metadata records elapsed time, token usage, cost, finish reason and
    retries — all of which the evaluation needs.
    """
    global SESSION_COST_USD
    import openai

    client = _get_client()
    print(f"  [API] Calling {MODEL} for: {run_label}")

    retries    = 0      # retries after 429/5xx/timeouts (excluding 503)
    overloads  = 0      # 503 overload responses
    wait_total = 0.0
    start      = time.time()
    called_at  = datetime.now(timezone.utc)

    while True:
        try:
            called_at = datetime.now(timezone.utc)
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            break
        except openai.APIStatusError as e:
            code = e.status_code
            if code == 402:
                sys.exit("[ERROR] DeepSeek reports insufficient balance (402).\n"
                         "        Top up at https://platform.deepseek.com, then "
                         "resume with --start-rep.")
            if code in (401, 403):
                sys.exit(f"[ERROR] DeepSeek rejected the API key ({code}). "
                         "Check DEEPSEEK_API_KEY.")
            if code == 503:
                overloads += 1
                if overloads > len(OVERLOAD_WAITS):
                    sys.exit(f"[ERROR] {MODEL} still overloaded (503) after "
                             f"{overloads} attempts.\n"
                             f"        Try again later. Completed runs are "
                             f"unaffected; use --start-rep to resume.")
                backoff = OVERLOAD_WAITS[overloads - 1]
                print(f"  [API] server overloaded (503), waiting {backoff}s "
                      f"({overloads}/{len(OVERLOAD_WAITS)})")
            else:
                transient = code in (429, 500, 502, 504)
                if not transient or retries >= MAX_RETRIES:
                    raise
                backoff = min(MAX_BACKOFF,
                              _retry_delay(e) or 2 ** (retries + 1))
                print(f"  [API] transient error {code}, retrying in "
                      f"{backoff:.0f}s ({retries + 1}/{MAX_RETRIES})")
                retries += 1
            time.sleep(backoff)
            wait_total += backoff
        except (openai.APITimeoutError, openai.APIConnectionError) as e:
            if retries >= MAX_RETRIES:
                raise
            backoff = 2 ** (retries + 1)
            print(f"  [API] {type(e).__name__}, retrying in {backoff}s "
                  f"({retries + 1}/{MAX_RETRIES})")
            time.sleep(backoff)
            wait_total += backoff
            retries    += 1

    elapsed = time.time() - start
    # Billable time excludes backoff waits, or the cost-of-automation
    # timings would be inflated by our own retry policy.
    billable = elapsed - wait_total

    choice  = response.choices[0] if response.choices else None
    finish  = (choice.finish_reason if choice is not None
               and choice.finish_reason else "no_choice")
    message = choice.message if choice is not None else None
    text    = (message.content or "") if message is not None else ""
    reasoning_text = (getattr(message, "reasoning_content", None) or ""
                      if message is not None else "")

    usage = response.usage
    input_tokens  = (usage.prompt_tokens or 0) if usage else 0
    output_tokens = (usage.completion_tokens or 0) if usage else 0
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    reasoning_tokens = (getattr(details, "reasoning_tokens", 0) or 0) if details else 0
    # DeepSeek-specific usage fields; fall back to "all miss" if absent.
    cache_hit  = (getattr(usage, "prompt_cache_hit_tokens", None) or 0) if usage else 0
    cache_miss = getattr(usage, "prompt_cache_miss_tokens", None) if usage else None
    if cache_miss is None:
        cache_miss = max(0, input_tokens - cache_hit)

    window = _price_window(called_at)
    cost = _estimate_cost(MODEL, cache_hit, cache_miss, output_tokens, window)
    if cost is not None:
        SESSION_COST_USD += cost

    meta = {
        "label":              run_label,
        "provider":           "deepseek",
        "model":              MODEL,
        "model_version":      getattr(response, "model", None),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "temperature":        TEMPERATURE,
        "max_tokens":         MAX_TOKENS,
        "elapsed_seconds":    round(elapsed, 3),
        "backoff_seconds":    round(wait_total, 3),
        "billable_seconds":   round(billable, 3),
        "retries":            retries,
        "overload_retries":   overloads,
        "attempts":           1 + retries + overloads,   # requests sent
        "input_tokens":       input_tokens,
        "cache_hit_tokens":   cache_hit,
        "cache_miss_tokens":  cache_miss,
        "output_tokens":      output_tokens,     # includes reasoning tokens
        "thinking_tokens":    reasoning_tokens,
        "reasoning_chars":    len(reasoning_text),
        "price_window":       window,
        "cost_usd":           round(cost, 6) if cost is not None else None,
        "stop_reason":        finish,
        "truncated":          finish == "length",
        "timestamp":          datetime.now().isoformat(timespec="seconds"),
    }

    cost_txt = f"${cost:.4f}" if cost is not None else "cost n/a"
    print(f"  [API] Done in {billable:.1f}s  "
          f"(in={input_tokens}, out={output_tokens}, think={reasoning_tokens}, "
          f"stop={finish})  {cost_txt} [{window}]  "
          f"session total ${SESSION_COST_USD:.4f}")
    if meta["truncated"]:
        print("  [WARN] Response hit max_tokens and is TRUNCATED. "
              "Recorded in meta.json; do not count this as a model failure.")
    if finish not in ("stop", "length"):
        print(f"  [WARN] Unusual finish (finish={finish}). Recorded in meta.json.")
    if not text.strip():
        print("  [WARN] Empty response content. Recorded; check response file.")

    return text, meta


def list_models():
    client = _get_client()
    print("DeepSeek models available to this key:")
    for m in client.models.list():
        print(f"  {m.id}")


def mock_call(prompt: str, run_label: str) -> tuple[str, dict]:
    MOCK_DIR.mkdir(exist_ok=True)
    safe = run_label.replace(" ", "_").replace("/", "-")
    path = MOCK_DIR / f"{safe}.txt"
    path.write_text(prompt)
    print(f"  [MOCK] Saved prompt ({len(prompt)} chars) -> {path}")
    return (
        json.dumps({"scenarios": []}, indent=2),
        {"label": run_label, "mock": True, "elapsed_seconds": 0.0,
         "billable_seconds": 0.0, "input_tokens": 0, "output_tokens": 0,
         "thinking_tokens": 0, "cost_usd": 0.0,
         "stop_reason": "mock", "truncated": False, "retries": 0},
    )


# ---------------------------------------------------------------------------
# Run recording
# ---------------------------------------------------------------------------

class Run:
    """One generation run and its artefact directory."""

    def __init__(self, run_id: str, mode: str, mock: bool):
        self.run_id = run_id
        self.mode   = mode
        self.mock   = mock
        self.dir    = RUNS_DIR / run_id
        self.tests  = self.dir / "tests"
        self.tests.mkdir(parents=True, exist_ok=True)
        self.calls  = []
        self.started = time.time()

    def call(self, prompt: str, stage: str, label: str) -> str:
        fn = mock_call if self.mock else call_llm
        (self.dir / f"prompt_{stage}.txt").write_text(prompt)
        text, meta = fn(prompt, label)
        (self.dir / f"response_{stage}.txt").write_text(text)
        meta["stage"] = stage
        self.calls.append(meta)
        return text

    def write_tests(self, filename: str, code: str) -> Path:
        path = self.tests / filename
        path.write_text(strip_fences(code) + "\n")
        print(f"  [OUT] {path}")
        return path

    def finish(self, extra: dict | None = None):
        total_billable = sum(c.get("billable_seconds", 0.0) for c in self.calls)
        timing = {
            "run_id":                self.run_id,
            "mode":                  self.mode,
            "stage":                 "test_generation",
            "wall_clock_seconds":    round(time.time() - self.started, 3),
            "billable_seconds":      round(total_billable, 3),
            "calls":                 self.calls,
        }
        (self.dir / "timing.json").write_text(json.dumps(timing, indent=2))

        meta = {
            "run_id":      self.run_id,
            "mode":        self.mode,
            "model":       MODEL,
            "temperature": TEMPERATURE,
            "max_tokens":  MAX_TOKENS,
            "mock":        self.mock,
            "timestamp":   datetime.now().isoformat(timespec="seconds"),
            "truncated_any": any(c.get("truncated") for c in self.calls),
            "input_tokens":  sum(c.get("input_tokens", 0) for c in self.calls),
            "output_tokens": sum(c.get("output_tokens", 0) for c in self.calls),
            "thinking_tokens": sum(c.get("thinking_tokens", 0) for c in self.calls),
            "cost_usd":    round(sum(c.get("cost_usd") or 0.0 for c in self.calls), 6),
            "model_version": next((c.get("model_version") for c in self.calls
                                   if c.get("model_version")), None),
            "provider":    "deepseek",
        }
        if extra:
            meta.update(extra)
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))

        RUNS_DIR.mkdir(exist_ok=True)
        manifest = RUNS_DIR / "manifest.csv"
        new = not manifest.exists()
        cols = ["run_id", "mode", "model", "timestamp", "scenarios",
                "parse_error", "truncated_any", "billable_seconds",
                "input_tokens", "output_tokens", "thinking_tokens",
                "cost_usd", "model_version"]
        with open(manifest, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            if new:
                w.writeheader()
            row = dict(meta)
            row["billable_seconds"] = timing["billable_seconds"]
            w.writerow(row)
        return timing


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------

def _cross_service_run(ir: dict, mode: str, repeat: int, mock: bool):
    """Shared body for full and ablated, which differ only in Prompt 1/2 input."""
    run = Run(f"{mode}-r{repeat}", mode, mock)
    print(f"\n[PIPELINE] Mode: {mode.upper()}  (repeat {repeat})  -> {run.dir}")

    p1 = (build_prompt_1_full(ir) if mode == "full"
          else build_prompt_1_ablated(ir))
    scenarios_raw = run.call(p1, "1", f"{mode}_r{repeat}_p1_scenarios")

    parsed = parse_scenarios(scenarios_raw)
    (run.dir / "scenarios.json").write_text(json.dumps(parsed, indent=2))
    if parsed.get("parse_error"):
        print(f"  [WARN] Scenario JSON did not parse: {parsed['parse_error']}")
        print("         Recorded and continuing; this is a result, not a bug.")
    else:
        print(f"  [OK] {len(parsed['scenarios'])} scenario(s) parsed")

    p2 = build_prompt_2_cross_service(scenarios_raw, ir, mode=mode)
    test_code = run.call(p2, "2", f"{mode}_r{repeat}_p2_tests")
    run.write_tests(f"test_{mode}_r{repeat}.py", test_code)

    run.finish({
        "scenarios":   len(parsed.get("scenarios", [])),
        "parse_error": parsed.get("parse_error", ""),
        "target_edges": sorted({
            s.get("target_edge", "") for s in parsed.get("scenarios", [])
            if s.get("target_edge")
        }),
    })


def run_full_pipeline(ir, repeat, mock):
    _cross_service_run(ir, "full", repeat, mock)


def run_ablated_pipeline(ir, repeat, mock):
    _cross_service_run(ir, "ablated", repeat, mock)


def run_baseline_pipeline(ir: dict, repeat: int, mock: bool):
    """One run per service per repetition. No cross-service context anywhere."""
    for service, spec in ir["services"].items():
        run = Run(f"baseline-{service}-r{repeat}", "baseline", mock)
        print(f"\n[PIPELINE] Mode: BASELINE  service={service}  "
              f"(repeat {repeat})  -> {run.dir}")

        p1 = build_prompt_1_baseline(service, spec)
        scenarios_raw = run.call(p1, "1", f"baseline_{service}_r{repeat}_p1")

        parsed = parse_scenarios(scenarios_raw)
        (run.dir / "scenarios.json").write_text(json.dumps(parsed, indent=2))
        if parsed.get("parse_error"):
            print(f"  [WARN] Scenario JSON did not parse: {parsed['parse_error']}")
        else:
            print(f"  [OK] {len(parsed['scenarios'])} scenario(s) parsed")

        p2 = build_prompt_2_baseline(scenarios_raw, service)
        test_code = run.call(p2, "2", f"baseline_{service}_r{repeat}_p2")
        run.write_tests(f"test_baseline_{service}_r{repeat}.py", test_code)

        run.finish({
            "service":     service,
            "scenarios":   len(parsed.get("scenarios", [])),
            "parse_error": parsed.get("parse_error", ""),
        })


def run_repair(test_file: Path, error_output: str, mock: bool):
    """
    Prompt 3 — repair a failing test file IN PLACE.

    The original is moved to <run>/pre_repair/ first. The previous version
    wrote <name>.repaired.py alongside the original, so pytest collected both
    and counted every test twice.
    """
    test_file = Path(test_file)
    print(f"\n[REPAIR] Repairing: {test_file}")
    test_code = test_file.read_text()

    run_dir = test_file.parent.parent
    backup_dir = run_dir / "pre_repair"
    backup_dir.mkdir(exist_ok=True)
    (backup_dir / test_file.name).write_text(test_code)
    print(f"  [BAK] Original -> {backup_dir / test_file.name}")

    fn = mock_call if mock else call_llm
    p3 = build_prompt_3(test_code, error_output)
    (run_dir / "prompt_3.txt").write_text(p3)
    repaired, meta = fn(p3, f"repair_{test_file.stem}")
    (run_dir / "response_3.txt").write_text(repaired)

    if not mock:
        test_file.write_text(strip_fences(repaired) + "\n")
        print(f"  [OUT] Repaired in place -> {test_file}")

    meta["stage"] = "3"
    repair_log = run_dir / "repair.json"
    existing = json.loads(repair_log.read_text()) if repair_log.exists() else []
    existing.append({"file": test_file.name, **meta})
    repair_log.write_text(json.dumps(existing, indent=2))
    if not mock:
        print(f"[REPAIR] Estimated cost: ${meta.get('cost_usd') or 0:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate integration tests from the IR and OpenAPI specs."
    )
    parser.add_argument("--ir",        required=False,
                        help="Path to ir.json produced by build_ir.py")
    parser.add_argument("--specs-dir", required=False,
                        help="Unused; retained for CLI compatibility")
    parser.add_argument("--mode",      required=False,
                        choices=["full", "ablated", "baseline"])
    parser.add_argument("--repeats",   type=int, default=3)
    parser.add_argument("--start-rep", type=int, default=1,
                        help="First repetition number (default 1). Use this to "
                             "add runs later without overwriting earlier ones.")
    parser.add_argument("--mock",      action="store_true",
                        help="Save prompts to disk instead of calling the API")
    parser.add_argument("--repair",    default=None,
                        help="Path to a failing test file to repair (Prompt 3)")
    parser.add_argument("--error",     default=None,
                        help="Path to a file containing the pytest output")
    parser.add_argument("--model",     default=DEFAULT_MODEL,
                        help=f"DeepSeek model ID (default {DEFAULT_MODEL}). "
                             "Keep it identical across all modes and repeats.")
    parser.add_argument("--list-models", action="store_true",
                        help="List DeepSeek models available to your key and exit")
    args = parser.parse_args()

    if args.list_models:
        list_models()
        return
    if not args.ir or not args.mode:
        parser.error("--ir and --mode are required (unless --list-models)")

    global MODEL
    MODEL = args.model

    with open(args.ir) as fh:
        ir = json.load(fh)

    active = [e for e in ir["edges"] if e["confidence"] != "REJECTED"]
    print(f"\n[GENERATE] Mode={args.mode}  Mock={args.mock}  "
          f"Repeats={args.repeats}  Model={MODEL}")
    print(f"           IR: {args.ir}  ({len(ir['services'])} services, "
          f"{len(active)} active edges)")

    if args.repair:
        if not args.error:
            sys.exit("[ERROR] --repair requires --error (path to pytest output)")
        err = Path(args.error)
        run_repair(Path(args.repair),
                   err.read_text() if err.exists() else args.error,
                   args.mock)
        return

    # Refuse to run if full and ablated have drifted apart.
    diff = assert_prompt_symmetry(ir)
    RUNS_DIR.mkdir(exist_ok=True)
    (RUNS_DIR / "prompt_symmetry.diff").write_text(diff)
    print(f"           Prompt symmetry: OK "
          f"(diff -> {RUNS_DIR / 'prompt_symmetry.diff'})")

    last = args.start_rep + args.repeats - 1
    for repeat in range(args.start_rep, last + 1):
        if args.mode == "full":
            run_full_pipeline(ir, repeat, args.mock)
        elif args.mode == "ablated":
            run_ablated_pipeline(ir, repeat, args.mock)
        else:
            run_baseline_pipeline(ir, repeat, args.mock)

    print(f"\n[GENERATE] Complete. Artefacts in: {RUNS_DIR}/")
    if not args.mock:
        print(f"[GENERATE] Estimated API cost this invocation: "
              f"${SESSION_COST_USD:.4f}")
    print(f"[GENERATE] Manifest: {RUNS_DIR / 'manifest.csv'}")
    if args.mock:
        print(f"[GENERATE] Prompts also saved in: {MOCK_DIR}/")


if __name__ == "__main__":
    main()