#!/usr/bin/env python3
"""
Sequential local Ollama scientific-model evaluation for OmniBioAI.
Includes Qwen3.5 122B in the default model inventory.

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
    "qwen3.5:122b",
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
