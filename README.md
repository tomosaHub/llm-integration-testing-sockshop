# LLM-Based Integration Testing in Microservice Architectures

**BSc (Hons) Computer Science research project, University of Pretoria, 2026**
Mosa Phalane · Supervisor: Prof. L. Marshall · Co-supervisor: Mr C. Redelinghuys

Can a large language model (LLM), given only the OpenAPI specifications of several microservices, write integration tests for the calls *between* those services? And does giving it a dependency graph of who calls whom help?

This repository contains everything needed to inspect or reproduce the study: the pipeline scripts, every prompt sent to the model, every response, every generated test, and the measured results.

## Key results

Tested on [Weaveworks Sock Shop](https://github.com/microservices-demo/microservices-demo) (5 services), using DeepSeek-V4-Pro, three conditions, five repetitions each.

| | Full (specs + graph) | Ablated (specs only) | Baseline (one service at a time) |
|---|---|---|---|
| Tests generated | 34 | 35 | 121 |
| Pass rate, first attempt | 6.2% | 2.9% | 16.5% |
| Pass rate, after one repair | 23.3% | 17.6% | 57.9% |
| Call relationships covered (after repair) | 53.3% | 26.7% | n/a |
| Tests aimed at a call that does not exist | 0 | 4 | n/a |
| Mean reasoning tokens to plan scenarios | 2,453 | 10,304 | n/a |

- The model wrote **190 runnable tests** for a total model cost of **US$3.89**.
- With the dependency graph, **every** cross-service test targeted a real service call.
- No cross-service test completed a successful checkout. The `orders` service requires a request body that its OpenAPI specification does not document, so the limiting factor was information missing from the specifications, not the model's ability to write test code.

The full analysis is in the report: [`report/LLM_Integration_Testing_Report.pdf`](report/LLM_Integration_Testing_Report.pdf).

## How the pipeline works

```
Sock Shop (Docker) → OpenAPI specs → dependency graph (ir.json) → LLM: scenarios → test code → repair → pytest on the live system
```

1. **`build_ir.py`** reads the five OpenAPI specifications and known-call annotations and builds the intermediate representation (`ir.json`): a directed, confidence-tiered graph of which service calls which.
2. **`generate_tests.py`** runs the three-prompt chain (scenario extraction, test generation, repair) against the DeepSeek API, in `full`, `ablated` or `baseline` mode, and records every prompt, response, token count and cost in `runs/`.
3. **`router.py`** starts an nginx routing layer on port 80 that sends each request to the service whose specification declares its path (Sock Shop's public entry point does not expose the backend APIs).
4. **`compute_metrics.py`** runs every generated test with pytest and computes success rate and edge coverage per run and per mode.
5. **`repair_all.py`** applies one repair attempt to every test file; **`restore_invalid_repairs.py`** detects and undoes repairs made against environment errors.
6. **`audit_repairs.py`** compares each test before and after repair and reports what the repair changed.

## Repository layout

```
.
├── build_ir.py, generate_tests.py, compute_metrics.py, router.py,
│   repair_all.py, restore_invalid_repairs.py, audit_repairs.py, verify_specs.py
├── specs/          OpenAPI (Swagger 2.0) specifications of the five services
├── ir.json         the dependency graph used in the study (+ ir_report.txt, ir.timing.json)
├── runs/           one folder per generation run: prompts, responses, scenarios,
│                   generated tests, pre-repair originals, metadata (tokens, cost, timing)
├── results/        per-test, per-run and summary CSVs, plus the repair audit
├── report/         the final report (PDF)
└── poster/         the Poster Day poster (PDF)
```

## Reproducing the study

Requirements: Python 3.10+, Docker, and a DeepSeek API key.

```bash
# 1. Start Sock Shop
git clone https://github.com/microservices-demo/microservices-demo
cd microservices-demo/deploy/docker-compose && docker compose up -d && cd -

# 2. Install dependencies
python3 -m pip install -r requirements.txt

# 3. Set your API key (never commit it)
export DEEPSEEK_API_KEY="your-key-here"

# 4. Generate tests (5 repetitions per mode)
python3 generate_tests.py --ir ir.json --mode full     --repeats 5
python3 generate_tests.py --ir ir.json --mode ablated  --repeats 5
python3 generate_tests.py --ir ir.json --mode baseline --repeats 5

# 5. Route requests to the backend services and measure
python3 router.py up
python3 compute_metrics.py --ir ir.json --mode full

# 6. Repair once, then measure again
python3 repair_all.py --ir ir.json --mode full
python3 compute_metrics.py --ir ir.json --mode full --rerun

# 7. Restore Sock Shop's normal entry point
python3 router.py down
```

`generate_tests.py --mock` writes the prompts to disk without calling the API, which is a free way to inspect exactly what the model receives. Model outputs are non-deterministic, so a re-run will produce different tests; the `runs/` folder holds the exact outputs reported in the study.

## Citation

> M. Phalane, "LLM-Based Integration Testing in Microservice Architectures," BSc (Hons) research report, Department of Computer Science, University of Pretoria, 2026.

The Sock Shop OpenAPI specifications in `specs/` are from Weaveworks' [microservices-demo](https://github.com/microservices-demo) project and remain under their original Apache 2.0 licence.
