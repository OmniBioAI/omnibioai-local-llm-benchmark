#!/usr/bin/env python3
"""
Sequential local Ollama scientific-model evaluation for OmniBioAI.

Runs one model at a time against the same fixed scientific prompt set.
Writes raw responses and timing/token metrics to JSONL and CSV files.

Requires Python 3.9+ (standard library only).
"""

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MODELS = [
    "gpt-oss:120b",
    "hf.co/DuoNeural/medgemma-1.5-4b-it-LiteRT:Q4_K_M",
    "phi4:14b",
    "qwen3:14b",
    "deepseek-r1:14b",
    "gpt-oss:20b",
    "gemma3:12b",
    "qwen3:8b",
    "gemma3:4b",
    "qwen3:4b",
    "llama3:8b",
]

# Synthetic examples only. No patient data or clinical decisions.
CASES = [
    {
        "id": "rnaseq_design",
        "domain": "RNA-seq",
        "prompt": """You are assisting with a research RNA-seq analysis.
There are 3 biological replicates in each of two groups (Control and Treatment).
Counts are raw gene-level counts from the same sequencing protocol.
Describe a defensible differential-expression workflow from count matrix to
interpretable results. Address count filtering, normalization, the statistical
model, multiple-testing correction, diagnostics, and limitations. Do not invent
results. Keep the answer structured and concise."""
    },
    {
        "id": "variant_interpretation",
        "domain": "Variant interpretation",
        "prompt": """Consider this synthetic variant: a rare heterozygous missense variant
in a disease-associated gene. It is absent from a small control cohort, has
conflicting in-silico predictions, no functional assay, and no convincing
segregation evidence. Explain what can and cannot be concluded. Distinguish
variant rarity, computational evidence, functional evidence, segregation,
phenotype fit, and clinical classification. Do not assign a definitive
pathogenic classification from the information given."""
    },
    {
        "id": "statistical_genetics",
        "domain": "Statistical genetics",
        "prompt": """A case-control association study has 40 cases and 60 controls.
The phenotype is imbalanced, several covariates may confound association, and
the study tests 500,000 variants. Explain an appropriate analysis strategy,
including quality control, population structure, covariates, rare-variant
considerations, multiple testing, power, and replication. Identify what cannot
be inferred without the genotype and phenotype data."""
    },
    {
        "id": "single_cell",
        "domain": "Single-cell / transcriptomics",
        "prompt": """Outline a reproducible single-cell RNA-seq workflow for comparing
two biological conditions with multiple donors per condition. Explain quality
control, normalization, integration/batch effects, cell annotation, differential
expression, and why cells should not automatically be treated as independent
biological replicates. Include a suitable donor-aware approach."""
    },
    {
        "id": "biomedical_evidence",
        "domain": "Biomedical literature",
        "prompt": """A researcher asks whether Gene X is a validated therapeutic target
for Disease Y. No papers, database records, or experimental data have been
provided in this conversation. Explain how you would assess the claim using
primary literature and authoritative databases. Clearly separate evidence
needed for genetic association, mechanism, target engagement, preclinical
validation, and clinical validation. Do not fabricate papers, citations, or
database findings."""
    },
    {
        "id": "scientific_reasoning",
        "domain": "Scientific reasoning",
        "prompt": """A differential-expression analysis reports 2,000 significant genes
at adjusted p < 0.05, but sample clustering is dominated by sequencing batch
rather than treatment. Explain why the result may be misleading and propose
diagnostic checks and corrective actions. State when the study design may make
the treatment effect impossible to separate from batch."""
    },
]


def post_json(url, payload, timeout):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url, timeout):
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def check_server(base_url, timeout):
    try:
        data = get_json(base_url.rstrip("/") + "/api/tags", timeout)
        available = {item.get("name") for item in data.get("models", [])}
        return available
    except Exception as exc:
        raise SystemExit(
            f"Cannot reach Ollama at {base_url}. "
            f"Check that the Ollama API is listening. Details: {exc}"
        )


def run_case(base_url, model, case, args):
    payload = {
        "model": model,
        "prompt": case["prompt"],
        "stream": False,
        "keep_alive": 0,  # unload after each request to limit memory pressure
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": args.context,
            "num_predict": args.max_tokens,
        },
    }

    started = time.perf_counter()
    try:
        result = post_json(
            base_url.rstrip("/") + "/api/generate",
            payload,
            args.timeout,
        )
        wall_seconds = time.perf_counter() - started
        eval_count = result.get("eval_count", 0) or 0
        eval_duration = result.get("eval_duration", 0) or 0
        tokens_per_second = (
            eval_count / (eval_duration / 1e9)
            if eval_duration > 0 else None
        )
        return {
            "status": "ok",
            "wall_seconds": round(wall_seconds, 3),
            "load_seconds": round((result.get("load_duration", 0) or 0) / 1e9, 3),
            "prompt_tokens": result.get("prompt_eval_count"),
            "generated_tokens": eval_count,
            "generation_seconds": round(eval_duration / 1e9, 3),
            "tokens_per_second": round(tokens_per_second, 2)
                if tokens_per_second is not None else None,
            "response": result.get("response", ""),
            "error": "",
        }
    except Exception as exc:
        return {
            "status": "error",
            "wall_seconds": round(time.perf_counter() - started, 3),
            "load_seconds": None,
            "prompt_tokens": None,
            "generated_tokens": None,
            "generation_seconds": None,
            "tokens_per_second": None,
            "response": "",
            "error": str(exc),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run a sequential scientific benchmark against local Ollama models."
    )
    parser.add_argument("--url", default="http://localhost:11434",
                        help="Ollama API base URL (default: http://localhost:11434)")
    parser.add_argument("--output-dir", default="omnibioai-model-eval",
                        help="Directory for CSV and JSONL results")
    parser.add_argument("--context", type=int, default=8192,
                        help="Context window for every model (default: 8192)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Maximum generated tokens per test (default: 512)")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Per-request timeout in seconds (default: 3600)")
    parser.add_argument("--pause", type=float, default=2.0,
                        help="Pause between model tests in seconds (default: 2)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Repeat each case per model (default: 1)")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Optional explicit model list; defaults to dashboard inventory")
    parser.add_argument("--skip-missing", action="store_true",
                        help="Skip models not currently present in Ollama")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = output_dir / f"results_{stamp}.jsonl"
    csv_path = output_dir / f"results_{stamp}.csv"

    available = check_server(args.url, args.timeout)
    print(f"Ollama reachable: {args.url}")
    print(f"Models reported by Ollama: {len(available)}")
    print(f"Scientific cases: {len(CASES)} | repeats: {args.repeats}")
    print(f"Sequential execution; context={args.context}, max_tokens={args.max_tokens}")
    print(f"Results: {output_dir.resolve()}")
    print()

    rows = []
    total = len(args.models) * len(CASES) * args.repeats
    current = 0

    for model in args.models:
        if model not in available:
            message = f"Model not found in Ollama: {model}"
            if args.skip_missing:
                print(f"SKIP: {message}")
                continue
            print(f"WARNING: {message}; requests may fail.")
        print(f"\n{'=' * 76}\nMODEL: {model}\n{'=' * 76}")

        for repeat in range(1, args.repeats + 1):
            for case in CASES:
                current += 1
                print(
                    f"[{current}/{total}] {model} | {case['id']} "
                    f"| repeat {repeat}/{args.repeats}",
                    flush=True,
                )
                metrics = run_case(args.url, model, case, args)
                record = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "model": model,
                    "case_id": case["id"],
                    "domain": case["domain"],
                    "repeat": repeat,
                    **metrics,
                }
                rows.append(record)

                with jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

                speed = (
                    f"{metrics['tokens_per_second']} tok/s"
                    if metrics["tokens_per_second"] is not None else "n/a"
                )
                print(
                    f"  {metrics['status']} | {metrics['wall_seconds']}s "
                    f"| {speed}"
                    + (f" | {metrics['error']}" if metrics["error"] else ""),
                    flush=True,
                )
                time.sleep(args.pause)

    fieldnames = [
        "timestamp_utc", "model", "case_id", "domain", "repeat", "status",
        "wall_seconds", "load_seconds", "prompt_tokens", "generated_tokens",
        "generation_seconds", "tokens_per_second", "response", "error",
        "human_score_0_to_4", "critical_error", "review_notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print("\nCompleted.")
    print(f"JSONL: {jsonl_path.resolve()}")
    print(f"CSV:   {csv_path.resolve()}")
    print("CSV includes blank columns for manual scientific scoring.")


if __name__ == "__main__":
    main()
