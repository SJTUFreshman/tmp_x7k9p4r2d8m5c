<h1 align="center">Reallocating Reasoning across Models<br>via Learned Collaboration</h1>

<p align="center">
  <strong>Collaborative Learning (CL)</strong><br>
  Heterogeneous models · Learned task-state handoffs · Five reasoning benchmarks
</p>

<p align="center">
  <a href="#reported-results">Results</a> ·
  <a href="#quickstart-score-the-stored-math-results">Quickstart</a>
</p>

## Overview

This repository accompanies **Reallocating Reasoning across Models via Learned Collaboration**. It contains the curated experiment code, evaluation records, analysis artifacts, training configurations, and logs associated with the paper. Large retained artifacts are distributed through **Git LFS**.

Collaborative Learning trains smaller models to reassess, extend, and correct one another's reasoning through explicit task-state handoffs. It combines:

- **Model slicing:** Qwen3-1.7B, Qwen3-4B, and Qwen3-8B form a heterogeneous pool with 13.7B total parameters, comparable to Qwen3-14B. These are independently parameterized models; slicing does not partition a single checkpoint's weights.
- **Pyramid supervision:** protocol distillation initializes collaboration, followed by step-level feedback and task rewards that refine reasoning and interaction decisions.

The paper evaluates multi-hop question answering, mathematical reasoning, code generation, and constrained generation using MuSiQue, GSM-Hard, MATH, an eight-language MultiPL-E subset, and Conifer.

## Reported results

The following values reproduce Table 1 of the manuscript. All scores are percentages; higher is better. The first three benchmarks report **accuracy / F1**, MultiPL-E reports **weighted / macro-language pass@1**, and Conifer reports **Coverage / Explicit**.

| Method | MuSiQue | GSM-Hard | MATH | MultiPL-E | Conifer |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-14B | 39.55 / 49.70 | 68.94 / 80.69 | 74.60 / 82.86 | 55.18 / 57.34 | 88.00 / 95.72 |
| SE-RL | 41.76 / 52.73 | 68.18 / 79.67 | 76.00 / 80.70 | 60.43 / 62.55 | 88.00 / 95.81 |
| MAD | 39.51 / 50.59 | 66.67 / 74.18 | 72.40 / 81.32 | 55.03 / 55.35 | 88.55 / 95.84 |
| AgentVerse | 41.99 / 52.85 | 71.97 / 78.44 | 68.20 / 79.42 | 59.25 / 59.83 | 87.74 / 95.86 |
| AFlow | 41.83 / 50.36 | 68.18 / 76.04 | 65.00 / 76.53 | 42.83 / 42.46 | 87.92 / 96.51 |
| GPTSwarm | 36.20 / 47.37 | 70.45 / 78.39 | 69.20 / 79.90 | 51.63 / 52.43 | 88.17 / 95.96 |
| MAGRPO | 40.13 / 51.74 | 65.91 / 76.44 | 65.00 / 76.01 | 49.93 / 49.93 | 88.58 / 95.82 |
| MAPoRL | 40.59 / 52.74 | 63.64 / 75.34 | 70.80 / 80.87 | 53.25 / 53.27 | 86.37 / 89.19 |
| AT-GRPO | 41.54 / 53.51 | 64.39 / 76.86 | 74.20 / 83.70 | 57.91 / 58.74 | 86.60 / 95.85 |
| Homo. MAS | 26.40 / 32.82 | 71.21 / 79.93 | 68.20 / 76.41 | 56.36 / 57.20 | 88.02 / 94.27 |
| Homo. CL | 36.00 / 47.19 | 60.61 / 76.51 | 68.60 / 77.95 | 53.33 / 52.65 | 88.15 / 95.22 |
| **CL (ours)** | **42.37 / 53.43** | **71.97 / 81.15** | **78.40 / 85.25** | **59.39 / 59.63** | **88.90 / 95.90** |

CL achieves the best or joint-best score on six of the ten metrics and improves over Qwen3-14B on all ten, with a mean absolute gain of 2.38 percentage points. These are the paper's reported results, not results from rerunning every experiment during repository preparation.

## Repository layout

```text
reproduction/
  code/                   Experiment implementations and runtime dependencies
    jca/                  CL, task evaluators, analysis code, and MAS baselines
    jca_homo_gsm/          Homogeneous GSM-Hard implementation
    jca_homo_math/         Homogeneous MATH implementation
    jca_homo_launchers/    Homogeneous experiment launchers
    MAGRPO/               MAGRPO experiment implementation
    MAPoRL/               MAPoRL experiment implementation
    AT-GRPO/              AT-GRPO experiment implementation
    math_se_rl/           MATH self-evaluated RL implementation
  results/                Selected evaluation records, configurations, and logs
  training/               Training configurations, logs, and summaries
  analysis/               Saved analysis results
```

`results/` and `training/` are organized by benchmark: `MuSiQue`, `GSM-Hard`, `MATH`, `MultiPL-E`, and `Conifer`. Analysis scripts live primarily in [`reproduction/code/jca/analysis/`](reproduction/code/jca/analysis/); [`reproduction/analysis/`](reproduction/analysis/) holds saved analysis artifacts.

The release excludes model weights, adapter checkpoints, benchmark datasets, teacher pools, and SFT/RL training inputs. Obtain these external resources separately before training or generating new predictions. Stored evaluation outputs retain the information present in their source records. Some experiments survive as aggregate results or logs rather than complete per-example trajectories.

## Download the artifacts

Install [Git LFS](https://git-lfs.com/) before cloning:

```bash
git lfs install
git clone https://github.com/SJTUFreshman/tmp_x7k9p4r2d8m5c.git
cd tmp_x7k9p4r2d8m5c
git lfs pull
```

Run the examples below from the repository root. A checkout containing LFS pointer text instead of JSONL or other artifact content is not ready for scoring.

## Quickstart: score the stored MATH results

This path recomputes the MATH CL main-table metrics without a GPU, model server, or benchmark download. Use Python 3.10 or newer and install SymPy for symbolic answer comparison:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install sympy

python reproduction/code/jca/experiments/math_specific_sft_rl_v1/evaluate_main_table.py \
  summarize \
  --input reproduction/results/MATH/CL/math_specific_sft_rl_20260830/eval/results.jsonl \
  --output runs/math-main-table-summary.log
```

Expected result: **500 problems, 392 correct, accuracy 78.40%, F1 85.25%**. The summary includes the input hash, scorer version, aggregate metrics, and per-problem scores. Its output path must not already exist.

### MATH scoring convention

The main-table rule retains the final answer within three recorded protocol turns. A trajectory with more than three turns, or exactly three turns without a successful stop, is scored using its first tentative answer. The scorer preserves the original result file and writes a separate scoring summary.

[`evaluate_main_table.py`](reproduction/code/jca/experiments/math_specific_sft_rl_v1/evaluate_main_table.py) and its [`06_eval_suite.sh`](reproduction/code/jca/experiments/math_specific_sft_rl_v1/06_eval_suite.sh) wrapper provide the consolidated entry point. The retained [`evaluation.log`](reproduction/results/MATH/CL/math_specific_sft_rl_20260830/logs/evaluation.log) is derived from the stored result records and explicitly labeled as a scoring summary, rather than an execution transcript.

## Run a new MATH CL evaluation

The endpoint client requires the external MATH parquet files, the original `shard_04.jsonl`, and three already-running OpenAI-compatible endpoints serving the intended final adapters. Install parquet support in the client environment:

```bash
python -m pip install pandas pyarrow

python reproduction/code/jca/experiments/math_specific_sft_rl_v1/evaluate_main_table.py run \
  --source-root /path/to/MATH \
  --shard-file /path/to/test_10x500_seed42/shard_04.jsonl \
  --output-root runs/math-cl-new \
  --api-base-a1 http://127.0.0.1:8201/v1 --api-model-a1 A1 \
  --api-base-a2 http://127.0.0.1:8212/v1 --api-model-a2 A2 \
  --api-base-a3 http://127.0.0.1:8223/v1 --api-model-a3 A3
```

`--source-root` contains one subject directory per MATH subject, each with `test-00000-of-00001.parquet`. The shard must match the recorded SHA-256 `3d4b0649a1a4f6198ed22b138fb82b31d7339be65997af4175bf9bdcc6343183`. The endpoint model names must refer to the trained adapters: A1 is Qwen3-1.7B, A2 is Qwen3-4B, and A3 is Qwen3-8B for this MATH run. The endpoints must support the runner's structured JSON output requests. Set `OPENAI_API_KEY` if authentication is required.

The command builds the fixed evaluation shard, creates fresh state, runs the protocol, finalizes raw records, and appends the paper-rule scores to `evaluation.log`. It fixes A3 bootstrap and initial A3 handoff, enables incumbent-preservation guidance, disables thinking, uses one rollout, and limits execution to three recorded protocol turns. It refuses an existing output directory. The shell wrapper accepts the same arguments after the `run` subcommand has been omitted.

Start model servers and prepare the trained adapters before invoking this entry point. New generations depend on the supplied checkpoints and serving environment. The stored-result scoring command has been verified; the model evaluation has not been rerun during repository preparation.

## Other benchmarks and training

The code retains the experiment-specific organization and selected configurations. Useful starting points include:

| Area | Location |
| --- | --- |
| CL and shared evaluation utilities | [`reproduction/code/jca/scripts/`](reproduction/code/jca/scripts/) |
| MATH CL | [`reproduction/code/jca/experiments/math_specific_sft_rl_v1/`](reproduction/code/jca/experiments/math_specific_sft_rl_v1/) |
| Conifer | [`reproduction/code/jca/conifer_training_hub/`](reproduction/code/jca/conifer_training_hub/) |
| MultiPL-E execution evaluators | [`reproduction/code/jca/Code/MultiPL-E/`](reproduction/code/jca/Code/MultiPL-E/) |
| MAS baseline implementations | [`reproduction/code/jca/baseline/`](reproduction/code/jca/baseline/) |
| Per-benchmark training records | [`reproduction/training/`](reproduction/training/) |

Before running an experiment, use its retained configuration and logs to set dataset, checkpoint, interpreter, and output paths for your environment. Original server paths and experiment-specific defaults remain in many scripts; a generic launcher default is not necessarily the setting behind a paper result. Keep bundled task-specific evaluators with their corresponding runs, particularly the MATH evaluator variants.

Embedded API-key and authenticated-proxy defaults have been cleared in the public code. Supply credentials through the environment variables supported by the selected script; do not commit credentials.

There is no single environment lockfile for all experiments. Training and local model serving additionally require a compatible CUDA/PyTorch stack and the packages used by the selected implementation, such as Transformers, PEFT, Accelerate, and vLLM. MultiPL-E execution also needs the relevant language runtimes or its evaluator container. The paper reports training on eight NVIDIA A800-SXM4-80GB GPUs; this is the experimental setup, not a requirement for the CPU-only summary command.

## Manuscript

The manuscript TeX sources are intentionally kept outside this reproduction repository. The benchmark results and method names above follow the paper supplied with the artifact; the repository does not infer author identities or publication status.
