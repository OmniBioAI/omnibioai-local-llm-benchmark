# Local LLM Benchmark for Computational Biology

A preliminary evaluation of locally running large language models (LLMs) for computational biology tasks using NVIDIA DGX Spark and Ollama.

This benchmark is part of the ongoing development and hardening of the OmniBioAI platform. Its goal is to explore how different local models perform across scientific tasks, with particular attention to inference speed, response quality, and scientific reliability.

## Objectives

The benchmark aims to:

- Compare local LLM inference performance.
- Evaluate model responses to representative computational biology questions.
- Identify potential scientific inaccuracies and hallucinations.
- Understand the trade-offs between model size, speed, and response quality.
- Support evidence-based model selection for scientific AI applications.

## Hardware and Software

| Component | Configuration |
|---|---|
| Compute platform | NVIDIA DGX Spark |
| GPU architecture | NVIDIA GB10 Grace Blackwell |
| Memory | 128 GB unified memory |
| Model runtime | Ollama |
| Evaluation language | Python |
| Output formats | CSV and JSONL |

The benchmark runs models locally through the Ollama API.

## Models Evaluated

Five locally available models were included in this evaluation:

| Model | Parameters |
|---|---:|
| GPT-OSS | 120B |
| Qwen3 | 14B |
| Phi-4 | 14B |
| DeepSeek-R1 | 14B |
| MedGemma 1.5 | 4B |

The MedGemma model used in this experiment is a community-provided GGUF conversion:

`hf.co/DuoNeural/medgemma-1.5-4b-it-LiteRT:Q4_K_M`

Model tags and quantization details should be verified against the local Ollama installation when reproducing the benchmark.

## Scientific Evaluation Domains

Six computational biology and biomedical reasoning scenarios were evaluated:

1. RNA-seq experimental design and differential expression
2. Variant interpretation
3. Statistical genetics
4. Single-cell transcriptomics
5. Biomedical literature and therapeutic target evaluation
6. Scientific reasoning and batch-effect interpretation

Each model was evaluated against the same set of scientific scenarios.

## Benchmark Results

The initial evaluation produced the following average inference performance:

| Model | Average generation speed (tokens/sec) | Average wall time (sec) |
|---|---:|---:|
| MedGemma 1.5 4B | 75.83 | 9.96 |
| GPT-OSS 120B | 42.97 | 26.50 |
| DeepSeek-R1 14B | 24.17 | 25.00 |
| Qwen3 14B | 23.59 | 25.90 |
| Phi-4 14B | 23.54 | 25.87 |

These values are preliminary measurements from one evaluation run per model and scientific scenario.

**Important:** Inference speed does not indicate scientific accuracy or reliability. The results should not be interpreted as a model intelligence ranking.

## Methodology

The evaluation script communicates with the local Ollama API.

The benchmark records:

- Model name
- Scientific scenario
- Execution status
- Total wall-clock time
- Model loading duration
- Prompt token count
- Generated token count
- Generation duration
- Generation speed
- Model response
- Error information

The current configuration uses:

- Temperature: 0
- Random seed: 42
- Context window: 8192 tokens
- Maximum generated tokens: 512
- Sequential model evaluation
- One repeat per scenario in the published initial run

The script uses `keep_alive=0`, which unloads models after requests. Consequently, wall-clock measurements include model-loading behavior and should not be interpreted as generation-only latency.

## Repository Structure

```text
omnibioai-local-llm-benchmark/
├── data/
│   └── raw/
│       ├── results_20261116_235755.csv
│       └── results_20261116_235755.jsonl
├── results/
├── scripts/
│   └── omnibioai_model_eval.py
├── README.md
├── LICENSE
└── CITATION.cff
```

### Data files

- `CSV`: Tabular benchmark records for analysis.
- `JSONL`: Detailed evaluation records, including model responses and execution metadata.

The files in `data/raw/` represent the original benchmark output.

## Requirements

- Python 3.9 or later
- Ollama
- Locally installed benchmark models
- NVIDIA DGX Spark or another system capable of running the selected models

The evaluation script uses Python standard-library modules and does not require additional Python packages.

## Running the Benchmark

Start Ollama and verify that the required models are available:

```bash
ollama list
```

Run the evaluation script:

```bash
python scripts/omnibioai_model_eval.py \
  --models \
  gpt-oss:120b \
  qwen3:14b \
  phi4:14b \
  deepseek-r1:14b \
  hf.co/DuoNeural/medgemma-1.5-4b-it-LiteRT:Q4_K_M
```

Check the script's command-line options before running if your local version uses different argument names:

```bash
python scripts/omnibioai_model_eval.py --help
```

The script generates timestamped CSV and JSONL output files. Preserve these original outputs when conducting further analysis.

## Limitations

This is an initial exploratory benchmark, not a comprehensive scientific validation study.

Current limitations include:

- Only one evaluation repeat per model and scenario.
- Maximum response length limited to 512 generated tokens.
- Differences in model architecture and reasoning behavior.
- Model-loading overhead affects total wall-clock time.
- Human scientific-quality scoring is not yet complete.
- Some model outputs contain questionable recommendations or potentially incorrect technical details.
- Complete scientific reliability cannot be established from a small set of prompts.

Further evaluation should include repeated runs, independent scientific review, critical-error classification, structured-output validation, and tool-use testing.

## Scientific Reliability

A fast response is not necessarily a scientifically correct response.

For computational biology, model evaluation must consider:

- Statistical correctness
- Biological plausibility
- Appropriate use of scientific tools
- Evidence-based interpretation
- Reproducibility
- Hallucination and critical-error detection

Model selection should therefore consider the intended scientific responsibility rather than inference speed alone.

## Future Work

Planned extensions include:

- Repeated evaluation runs
- Human scoring of scientific responses
- Critical-error tracking
- Automated response validation
- Structured JSON generation tests
- Tool-calling evaluation
- Additional biomedical and computational biology scenarios
- Comparison across inference engines and configurations

## Intended Use

This repository is intended for research, experimentation, and benchmarking of locally deployed language models.

It is not a clinical validation study and must not be used as evidence of clinical safety, diagnostic accuracy, or suitability for patient-care decisions.

Users are responsible for reviewing the licenses and usage conditions of individual models before deployment, particularly in commercial environments.

## About OmniBioAI

This benchmark is associated with OmniBioAI, an AI-native computational biology platform focused on integrating scientific workflows, biomedical knowledge, local AI models, and reproducible computational infrastructure.

Website: https://omnibioai.org

GitHub: https://github.com/OmniBioAI

## License

The benchmark code and documentation are distributed under the repository license.

Individual language models and their weights remain subject to their respective licenses and usage terms. See the relevant model repositories for details.