"""
build_ir.py  —  Intermediate Representation (IR) constructor
LLM-Based Integration Testing of Inter-Service Communication
in Microservice Architectures

Reads a directory of Swagger 2.0 / OpenAPI 3.x JSON specs and produces:
  1. ir.json        — full IR graph for inspection and LLM input
  2. ir_report.txt  — human-readable structural validation report

Usage:
    python build_ir.py --specs-dir ./specs --output ir.json

Confidence tiers assigned to each edge:
  CONFIRMED  — schema overlap AND known-call annotation agree on direction
  CANDIDATE  — schema overlap only, direction not independently verified
  KNOWN_CALL — known-call annotation only, no schema overlap found
  REJECTED   — schema overlap inferred wrong direction (corrected by known-call)

IR edge structure:
{
  "from_service": str,
  "to_service":   str,
  "confidence":   "CONFIRMED" | "CANDIDATE" | "KNOWN_CALL" | "REJECTED",
  "evidence":     str,
  "shared_fields": [str],
  "known_call_evidence": str   # present on CONFIRMED and KNOWN_CALL edges
}
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Sock Shop known inter-service call relationships
# Source: Weaveworks Sock Shop architecture documentation
# ---------------------------------------------------------------------------
KNOWN_CALLS = [
    {
        "from_service": "orders",
        "to_service":   "user",
        "evidence":     "Orders fetches customer details (name, address, card) from User on checkout",
        "verification": "OBSERVED: user service logged GetUsers, GetCards and "
                        "GetAddresses during a checkout (2026-09-19)",
    },
    {
        "from_service": "orders",
        "to_service":   "carts",
        "evidence":     "Orders reads the customer cart items from Carts on checkout",
        "verification": "UNVERIFIED: carts does not log individual requests; "
                        "relationship asserted from architecture documentation only",
    },
    {
        "from_service": "orders",
        "to_service":   "payment",
        "evidence":     "Orders calls Payment to authorise the card transaction",
        "verification": "UNVERIFIED: payment does not log individual requests; "
                        "relationship asserted from architecture documentation only",
    },
    # REMOVED 2026-09-19 after runtime observation:
    #   orders -> catalogue and carts -> catalogue were recorded on the basis
    #   that an itemId appears in payloads crossing those boundaries. Direct
    #   observation showed catalogue receives no request during checkout, and
    #   none when an item is added to a cart, while logging every request it
    #   does receive. Both were data references, not calls.
]

# Fields that are strong signals of cross-service coupling when shared.
# Generic names (id, name, status) are deliberately excluded to reduce
# false positives from coincidental name matches.
COUPLING_FIELDS = {
    "customerId", "itemId", "unitPrice", "quantity",
    "firstName", "lastName", "username",
    "addresses", "cards", "longNum", "expires", "ccv",
    "authorised",
}

# Ground-truth edge set derived from KNOWN_CALLS, used for validation.
GROUND_TRUTH_EDGES = {
    (kc["from_service"], kc["to_service"]) for kc in KNOWN_CALLS
}


# ---------------------------------------------------------------------------
# Spec loading and parsing
# ---------------------------------------------------------------------------

def load_specs(specs_dir: str) -> dict[str, dict]:
    specs = {}
    p = Path(specs_dir)
    if not p.is_dir():
        sys.exit(f"[ERROR] {specs_dir} is not a directory")
    for f in sorted(p.glob("*.json")):
        service = f.stem
        with open(f) as fh:
            specs[service] = json.load(fh)
        print(f"  [+] Loaded: {service} ({f.name})")
    if not specs:
        sys.exit(f"[ERROR] No JSON files found in {specs_dir}")
    return specs


def resolve_ref(ref: str, definitions: dict) -> dict:
    if not ref.startswith("#/definitions/"):
        return {}
    return definitions.get(ref.split("/")[-1], {})


def flatten_fields(schema: dict, definitions: dict, depth: int = 0) -> set[str]:
    if depth > 3 or not isinstance(schema, dict):
        return set()
    if "$ref" in schema:
        schema = resolve_ref(schema["$ref"], definitions)
    fields = set()
    for name, sub in schema.get("properties", {}).items():
        fields.add(name)
        fields |= flatten_fields(sub, definitions, depth + 1)
    items = schema.get("items", {})
    if isinstance(items, dict):
        fields |= flatten_fields(items, definitions, depth + 1)
    return fields


def extract_endpoints(spec: dict) -> list[dict]:
    definitions = spec.get("definitions", {})
    endpoints = []
    for path, path_item in spec.get("paths", {}).items():
        path_params = re.findall(r"\{(\w+)\}", path)
        for method, op in path_item.items():
            if method not in {"get", "post", "put", "patch", "delete", "options", "head"}:
                continue
            body_schema = None
            for param in op.get("parameters", []):
                if param.get("in") == "body" and "schema" in param:
                    raw = param["schema"]
                    body_schema = resolve_ref(raw["$ref"], definitions) if "$ref" in raw else raw
            response_schema = None
            for code, resp in op.get("responses", {}).items():
                if str(code).startswith("2") and "schema" in resp:
                    raw = resp["schema"]
                    response_schema = resolve_ref(raw["$ref"], definitions) if "$ref" in raw else raw
                    break
            endpoints.append({
                "operation_id":  op.get("operationId", f"{method.upper()} {path}"),
                "method":        method.upper(),
                "path":          path,
                "path_params":   path_params,
                "body_schema":   body_schema,
                "response_schema": response_schema,
                "summary":       op.get("description", "").strip(),
            })
    return endpoints


# ---------------------------------------------------------------------------
# Edge inference with confidence tiers
# ---------------------------------------------------------------------------

def infer_schema_overlap(services_ir: dict) -> dict[tuple, dict]:
    """
    Returns a dict keyed by (from_service, to_service) for every
    candidate edge found by schema-overlap analysis.
    Direction heuristic: if service A's response schema contains fields
    that appear in service B's request schema, B is the caller (B → A).
    """
    candidates = {}
    names = list(services_ir.keys())

    for i, svc_a in enumerate(names):
        for svc_b in names[i + 1:]:
            defs_a = services_ir[svc_a]["definitions"]
            defs_b = services_ir[svc_b]["definitions"]

            produced_a, consumed_a = set(), set()
            produced_b, consumed_b = set(), set()

            for ep in services_ir[svc_a]["endpoints"]:
                if ep["response_schema"]:
                    produced_a |= flatten_fields(ep["response_schema"], defs_a)
                if ep["body_schema"]:
                    consumed_a |= flatten_fields(ep["body_schema"], defs_a)

            for ep in services_ir[svc_b]["endpoints"]:
                if ep["response_schema"]:
                    produced_b |= flatten_fields(ep["response_schema"], defs_b)
                if ep["body_schema"]:
                    consumed_b |= flatten_fields(ep["body_schema"], defs_b)

            # B consumes fields produced by A  →  B calls A  →  edge B → A
            overlap_b_calls_a = (produced_a & consumed_b) & COUPLING_FIELDS
            if overlap_b_calls_a:
                candidates[(svc_b, svc_a)] = {
                    "from_service":  svc_b,
                    "to_service":    svc_a,
                    "shared_fields": sorted(overlap_b_calls_a),
                    "evidence":      f"Schema overlap: {svc_b} consumes fields produced by {svc_a}",
                }

            # A consumes fields produced by B  →  A calls B  →  edge A → B
            overlap_a_calls_b = (produced_b & consumed_a) & COUPLING_FIELDS
            if overlap_a_calls_b:
                candidates[(svc_a, svc_b)] = {
                    "from_service":  svc_a,
                    "to_service":    svc_b,
                    "shared_fields": sorted(overlap_a_calls_b),
                    "evidence":      f"Schema overlap: {svc_a} consumes fields produced by {svc_b}",
                }

    return candidates


def assign_confidence_tiers(
    schema_candidates: dict[tuple, dict],
    known_calls: list[dict],
) -> list[dict]:
    """
    Merge schema overlap candidates with known-call annotations and
    assign a confidence tier to every edge.

    Tier rules:
      CONFIRMED  — (from, to) appears in both schema candidates AND known calls
      CANDIDATE  — (from, to) appears in schema candidates only
      KNOWN_CALL — (from, to) appears in known calls only (no schema evidence)
      REJECTED   — the reverse (to, from) appears in schema candidates
                   but (from, to) is in known calls: direction was wrong,
                   correct direction is emitted as KNOWN_CALL
    """
    known_map = {
        (kc["from_service"], kc["to_service"]): kc["evidence"]
        for kc in known_calls
    }
    verif_map = {
        (kc["from_service"], kc["to_service"]): kc.get("verification", "")
        for kc in known_calls
    }

    edges = []
    emitted = set()

    # Process schema candidates
    for key, cand in schema_candidates.items():
        frm, to = key
        reverse = (to, frm)

        if key in known_map:
            # Schema and known-call agree on direction
            edges.append({
                "from_service":        frm,
                "to_service":          to,
                "confidence":          "CONFIRMED",
                "evidence":            cand["evidence"],
                "shared_fields":       cand["shared_fields"],
                "known_call_evidence": known_map[key],
                "verification":        verif_map.get(key, ""),
            })
            emitted.add(key)

        elif reverse in known_map:
            # Schema inferred wrong direction — emit a REJECTED note
            # and let the known-call pass add the correct direction
            edges.append({
                "from_service":  frm,
                "to_service":    to,
                "confidence":    "REJECTED",
                "evidence":      (
                    f"Schema overlap recovered node pair ({frm}, {to}) but "
                    f"direction is incorrect. Known-call annotation confirms "
                    f"correct direction is {to} → {frm}."
                ),
                "shared_fields": cand["shared_fields"],
            })
            emitted.add(key)

        else:
            # Schema candidate with no known-call confirmation
            edges.append({
                "from_service":  frm,
                "to_service":    to,
                "confidence":    "CANDIDATE",
                "evidence":      cand["evidence"],
                "shared_fields": cand["shared_fields"],
            })
            emitted.add(key)

    # Add known-call edges not already emitted (CONFIRMED already handled above)
    for key, kc_evidence in known_map.items():
        if key not in emitted:
            frm, to = key
            edges.append({
                "from_service":        frm,
                "to_service":          to,
                "confidence":          "KNOWN_CALL",
                "evidence":            kc_evidence,
                "shared_fields":       [],
                "known_call_evidence": kc_evidence,
                "verification":        verif_map.get(key, ""),
            })

    return edges


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------

def validate_structure(edges: list[dict]) -> dict:
    """
    Compare the IR graph against the Sock Shop ground-truth edge set.
    Returns a validation summary dict and prints a report.
    """
    confirmed = {
        (e["from_service"], e["to_service"])
        for e in edges
        if e["confidence"] == "CONFIRMED"
    }
    known_call = {
        (e["from_service"], e["to_service"])
        for e in edges
        if e["confidence"] == "KNOWN_CALL"
    }
    in_graph = confirmed | known_call
    candidates = {
        (e["from_service"], e["to_service"])
        for e in edges
        if e["confidence"] == "CANDIDATE"
    }
    rejected = [e for e in edges if e["confidence"] == "REJECTED"]

    recovered     = in_graph & GROUND_TRUTH_EDGES
    missed        = GROUND_TRUTH_EDGES - in_graph
    spurious      = candidates - GROUND_TRUTH_EDGES

    return {
        "total_edges":           len(edges),
        "confirmed_edges":       len(confirmed),
        "known_call_edges":      len(known_call),
        "candidate_edges":       len(candidates),
        "rejected_edges":        len(rejected),
        "ground_truth_total":    len(GROUND_TRUTH_EDGES),
        "correctly_recovered":   sorted(recovered),
        "missed_edges":          sorted(missed),
        "spurious_candidates":   sorted(spurious),
        "direction_errors":      [
            {"inferred": (e["from_service"], e["to_service"]),
             "correct":  (e["to_service"],   e["from_service"])}
            for e in rejected
        ],
    }


def print_report(ir: dict, validation: dict, output_path: str) -> str:
    """Build and return the structural validation report as a string."""
    lines = []
    sep  = "=" * 62
    sep2 = "-" * 62

    lines += [
        sep,
        "  IR STRUCTURAL VALIDATION REPORT",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"  Output:    {output_path}",
        sep, "",
        "SERVICES EXTRACTED",
        sep2,
    ]
    for svc, data in ir["services"].items():
        lines.append(
            f"  {svc:<12}  {len(data['endpoints']):>2} endpoint(s)"
            f"  host={data['host']}"
        )

    lines += [
        "",
        f"  Total endpoints across all services: "
        f"{sum(len(d['endpoints']) for d in ir['services'].values())}",
        "",
        "EDGE SUMMARY",
        sep2,
        f"  Total edges inferred:   {validation['total_edges']}",
        f"  CONFIRMED  (overlap + known-call agree):  "
        f"{validation['confirmed_edges']}",
        f"  CANDIDATE  (overlap only, unverified):    "
        f"{validation['candidate_edges']}",
        f"  KNOWN_CALL (annotation only, no overlap): "
        f"{validation['known_call_edges']}",
        f"  REJECTED   (wrong direction from overlap):{validation['rejected_edges']}",
        "",
        "EDGE DETAIL",
        sep2,
    ]
    for e in ir["edges"]:
        badge = f"[{e['confidence']}]"
        lines.append(f"  {badge:<12}  {e['from_service']} -> {e['to_service']}")
        lines.append(f"               {e['evidence']}")
        if e.get("shared_fields"):
            lines.append(f"               shared fields: {e['shared_fields']}")
        if e.get("verification"):
            lines.append(f"               {e['verification']}")
        lines.append("")

    lines += [
        "STRUCTURAL VALIDATION vs GROUND TRUTH",
        sep2,
        f"  Ground-truth edges (Sock Shop architecture): "
        f"{validation['ground_truth_total']}",
        f"  Correctly recovered: {len(validation['correctly_recovered'])}",
        f"  Missed:              {len(validation['missed_edges'])}",
        f"  Spurious candidates: {len(validation['spurious_candidates'])}",
        "",
        "  NOTE: the ground-truth set is derived from the known-call",
        "  annotations, which are emitted into the graph unconditionally.",
        "  Recovery of these edges is therefore guaranteed by construction",
        "  and is not evidence about inference quality. The direction",
        "  result below is.",
        "",
    ]

    if validation["correctly_recovered"]:
        lines.append("  Recovered edges:")
        for frm, to in validation["correctly_recovered"]:
            lines.append(f"    [OK]  {frm} -> {to}")
        lines.append("")

    if validation["missed_edges"]:
        lines.append("  Missed edges (in ground truth but not in IR):")
        for frm, to in validation["missed_edges"]:
            lines.append(f"    [MISS]  {frm} -> {to}")
        lines.append("")

    if validation["spurious_candidates"]:
        lines.append("  Spurious candidates (not in ground truth):")
        for frm, to in validation["spurious_candidates"]:
            lines.append(f"    [?]  {frm} -> {to}")
        lines.append("")

    if validation["direction_errors"]:
        lines += [
            "DIRECTION ERRORS (key finding for RQ3 / paper)",
            sep2,
            "  Schema-overlap inference recovered the correct node pair",
            "  but inferred the wrong call direction. The known-call",
            "  annotation layer was required to correct these edges.",
            "",
        ]
        for err in validation["direction_errors"]:
            frm, to = err["inferred"]
            cfrm, cto = err["correct"]
            lines.append(f"  Inferred:  {frm} -> {to}  [WRONG DIRECTION]")
            lines.append(f"  Correct:   {cfrm} -> {cto}  [from known-call annotation]")
            lines.append("")

    lines += [
        sep,
        "  SUMMARY",
        sep2,
        f"  The graph contains {validation['ground_truth_total']} call "
        f"relationship(s), all supplied by the known-call annotation layer.",
        f"  Edges reaching CONFIRMED (schema and annotation agree): "
        f"{validation['confirmed_edges']}.",
    ]

    if validation["direction_errors"]:
        n = len(validation["direction_errors"])
        lines.append(
            f"  Schema-overlap alone produced {n} direction error(s)."
        )
        lines.append(
            "  Known-call annotations were necessary to correct edge direction,"
        )
        lines.append(
            "  confirming that structural schema matching alone is insufficient"
        )
        lines.append(
            "  to infer cross-service call semantics. (See paper Section 5 / 7.3)"
        )

    lines += ["", sep]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# IR builder
# ---------------------------------------------------------------------------

def build_ir(specs: dict[str, dict]) -> dict:
    print("\n[IR] Extracting endpoints...")
    services_ir = {}
    for service, spec in specs.items():
        endpoints = extract_endpoints(spec)
        services_ir[service] = {
            "host":       spec.get("host", service),
            "base_path":  spec.get("basePath", "/"),
            "endpoints":  endpoints,
            "definitions": spec.get("definitions", {}),
        }
        print(f"  [+] {service}: {len(endpoints)} endpoint(s)")

    print("\n[IR] Inferring cross-service edges...")
    schema_candidates = infer_schema_overlap(services_ir)
    edges = assign_confidence_tiers(schema_candidates, KNOWN_CALLS)

    for e in edges:
        badge = f"[{e['confidence']}]"
        fields = f"  fields={e['shared_fields']}" if e["shared_fields"] else ""
        print(f"  {badge:<12}  {e['from_service']} -> {e['to_service']}{fields}")

    # Strip definitions — needed for inference only
    for svc in services_ir:
        del services_ir[svc]["definitions"]

    return {"services": services_ir, "edges": edges}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build the IR from OpenAPI specs."
    )
    parser.add_argument("--specs-dir", required=True,
                        help="Directory containing JSON spec files")
    parser.add_argument("--output", default="ir.json",
                        help="Output JSON file (default: ir.json)")
    parser.add_argument("--report", default="ir_report.txt",
                        help="Output validation report (default: ir_report.txt)")
    args = parser.parse_args()

    print(f"\n[IR] Loading specs from: {args.specs_dir}")
    _t0 = time.time()
    specs = load_specs(args.specs_dir)

    ir = build_ir(specs)
    _elapsed = time.time() - _t0

    # Structural validation
    print("\n[IR] Running structural validation...")
    validation = validate_structure(ir["edges"])
    report_text = print_report(ir, validation, args.output)

    # Write outputs
    with open(args.output, "w") as fh:
        json.dump(ir, fh, indent=2)

    report_path = Path(args.output).parent / args.report
    with open(report_path, "w") as fh:
        fh.write(report_text)

    print("\n" + report_text)
    timing_path = Path(args.output).parent / "ir.timing.json"
    with open(timing_path, "w") as fh:
        json.dump({
            "stage": "ir_construction",
            "wall_clock_seconds": round(_elapsed, 3),
            "services": len(ir["services"]),
            "endpoints": sum(len(d["endpoints"]) for d in ir["services"].values()),
            "edges": len(ir["edges"]),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }, fh, indent=2)

    print(f"\n[IR] ir.json    -> {args.output}")
    print(f"[IR] ir_report  -> {report_path}")
    print(f"[IR] ir.timing  -> {timing_path}  ({_elapsed:.2f}s)")

    return ir


if __name__ == "__main__":
    main()