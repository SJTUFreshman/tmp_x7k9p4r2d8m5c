<h1 align="center">Reallocating Reasoning across Models<br>via Learned Collaboration</h1>

<p align="center">
  <strong>Collaborative Learning (CL)</strong><br>
  Model slicing · Protocol distillation · Pyramid supervision
</p>

<p align="center">
  <a href="https://github.com/SJTUFreshman/tmp_x7k9p4r2d8m5c"><img src="https://img.shields.io/badge/Repository-code%20%26%20artifacts-111827?style=flat-square" alt="Repository"></a>
  <a href="#quickstart"><img src="https://img.shields.io/badge/Quickstart-CPU%20scoring-2563eb?style=flat-square" alt="Quickstart"></a>
  <a href="#evaluation"><img src="https://img.shields.io/badge/Benchmarks-5-f59e0b?style=flat-square" alt="Five benchmarks"></a>
  <a href="https://git-lfs.com/"><img src="https://img.shields.io/badge/Git%20LFS-required-7c3aed?style=flat-square" alt="Git LFS required"></a>
</p>

<p align="center">
  <a href="assets/figures/Figure1.pdf"><img src="assets/figures/previews/Figure1-1.png" alt="Collaborative Learning overview: model slicing and learned collaboration" width="960"></a>
</p>

<p align="center">
  <b>A heterogeneous team of small language models that learns when to hand off, verify, and preserve reasoning.</b>
</p>

<p align="center">
  <a href="https://github.com/SJTUFreshman/tmp_x7k9p4r2d8m5c">Code and results</a>
  &nbsp;·&nbsp;
  <a href="https://anonymous.4open.science/r/modelcrew">Paper</a>
  &nbsp;·&nbsp;
  <a href="#quickstart">Quickstart</a>
  &nbsp;·&nbsp;
  <a href="#evaluation">Evaluation</a>
  &nbsp;·&nbsp;
  <a href="#documentation">Documentation</a>
</p>

This repository is the public code and artifact release for **Collaborative Learning (CL)**. CL reallocates a comparable parameter budget across three independently parameterized Qwen3 agents and trains them to exchange intermediate task state. The release contains selected training and evaluation code, benchmark scorers, result records, configurations, analysis outputs, and rendered paper figures.

> **Release status.** The checked-in records reproduce the reported tables. A new end-to-end training run still requires the external datasets, model weights, adapters, and serving infrastructure described in [System requirements and release scope](#system-requirements-and-release-scope).

## ✨ Overview

CL uses three collaborators with different capacities:

| Agent | Backbone | Collaboration role |
| --- | --- | --- |
| **A1** | Qwen3-1.7B | Produces an initial solution or focused correction |
| **A2** | Qwen3-4B | Rechecks the current state and develops the next step |
| **A3** | Qwen3-8B | Higher-capacity collaborator that may verify or finalize under the learned protocol |

These are three independently parameterized models, not slices of one checkpoint. Together they contain **13.7B parameters**, comparable to Qwen3-14B, while each forward pass uses one collaborator. Every agent receives the current task state and chooses one protocol action:

- `continue`: extend or refine the current reasoning;
- `handoff(j)`: pass the state to another collaborator;
- `terminate`: emit the final answer and stop.

The training protocol combines:

1. **Explicit handoffs:** an agent receives the current task state and may extend, revise, verify, or preserve it.
2. **Protocol distillation:** useful collaboration traces supervise how agents interact.
3. **Step-level supervision:** intermediate reasoning quality is scored during the interaction.
4. **Task-level rewards:** the final answer remains tied to the benchmark evaluator.

The resulting supervision forms a pyramid: protocol behavior at the bottom, deliberative process signals in the middle, and task success at the top.

<p align="center">
  <a href="assets/figures/Figure2.pdf">
    <img src="assets/figures/previews/Figure2-1.png" alt="Collaborative learning and pyramid supervision" width="920">
  </a>
</p>

## 🧪 Evaluation

The release covers four task categories and five benchmarks:

| Category | Benchmark | Capability tested | Code and artifact paths |
| --- | --- | --- | --- |
| Multi-hop QA | [MuSiQue](https://github.com/StonyBrookNLP/musique) | Compositional retrieval and reasoning | <code>reproduction/code/jca/</code>, <code>reproduction/results/MuSiQue/</code> |
| Mathematical reasoning | **GSM-Hard** | Robust numerical reasoning | <code>reproduction/code/jca_homo_gsm/</code>, <code>reproduction/results/GSM-Hard/</code> |
| Mathematical reasoning | **MATH** | Competition-level problem solving | <code>reproduction/code/jca/experiments/math_specific_sft_rl_v1/</code>, <code>reproduction/results/MATH/</code> |
| Code generation | [MultiPL-E](https://github.com/nuprl/MultiPL-E) | Pass@1 across eight languages | <code>reproduction/code/jca/Code/MultiPL-E/</code>, <code>reproduction/results/MultiPL-E/</code> |
| Constrained generation | **Conifer** | Coverage and explicit constraint satisfaction | <code>reproduction/code/jca/conifer_training_hub/</code>, <code>reproduction/results/Conifer/</code> |

### 📊 Reported results

The table reports the paper's main comparison. Values are percentages. MuSiQue, GSM-Hard, and MATH use **accuracy / F1**; MultiPL-E uses **weighted / macro-language pass@1**; Conifer uses **Coverage / Explicit**.

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

<p align="center">
  <a href="assets/figures/Figure3.pdf">
    <img src="assets/figures/previews/Figure3-1.png" alt="Ablation study across the five benchmarks" width="760">
  </a>
</p>

The retained CL records contain 2,417 MuSiQue examples, 132 GSM-Hard examples, 500 MATH problems, 1,352 MultiPL-E test problems, and 1,402 Conifer trajectories. CL is best or joint-best on **6 of 10 metrics**, improves over Qwen3-14B on all ten, and gains an average of **2.38 percentage points** over that single-model reference. The stored MATH evaluation scores **392/500 correct**, or **78.40% accuracy and 85.25% F1**, under the paper scorer.

> **Result provenance.** These are the paper's reported results and the retained evaluation records. Preparing this repository does not claim to rerun every historical training or baseline experiment.

### 📉 Efficiency and task performance

The paper also reports the trade-off between task performance, parameter-weighted output cost, and average model invocations. Each panel compares CL with established multi-agent baselines on one benchmark.

<p align="center">
  <a href="assets/figures/Figure4.pdf"><img src="assets/figures/previews/Figure4-1.png" alt="Computation cost versus task performance across five benchmarks" width="960"></a>
</p>

### 🧭 Emergent interaction topology

The learned policy does not force one universal agent order. Its interaction topology changes with the task: agents learn when to draft, review, refine, enrich, or terminate.

<p align="center">
  <a href="assets/figures/Figure5.pdf"><img src="assets/figures/previews/Figure5-1.png" alt="Task-dependent learned interaction topology" width="960"></a>
</p>

## 🧭 Workflow

```text
task query
    |
    v
 A1: Qwen3-1.7B ---- continue ----+
    |                              |
    +--------- handoff(j) ---------+--> A2: Qwen3-4B --> A3: Qwen3-8B
                                      |                    |
                                      +---- handoff -------+
                                                           |
                                      terminate <-----------+
                                                           |
                                                           v
                                                  task-specific evaluator
```

Protocol distillation supplies coordination traces, the large-model inspector scores each reasoning and interaction step, and the benchmark grader supplies the final task reward. The learned protocol can produce different interaction patterns for QA, math, code, and constrained generation.

## 📦 Repository layout

~~~
assets/
  figures/                         Paper figures (PDF) and GitHub previews (PNG)
reproduction/
  code/
    jca/                           CL implementation, utilities, evaluators, analysis
    jca_homo_gsm/                  Homogeneous GSM-Hard implementation
    jca_homo_math/                 Homogeneous MATH implementation
    jca_homo_launchers/            Cluster launchers and helpers
    AT-GRPO/ MAGRPO/ MAPoRL/       Baseline implementations
    math_se_rl/                    MATH self-evaluated RL baseline
  results/                         Outputs, configs, manifests, and logs
  training/                        Selected SFT/RL records and metrics
  analysis/                        Saved analysis artifacts
~~~

Each benchmark result tree keeps the method name, run configuration, raw or scored records, and summaries together.

## 🚀 Quickstart

### 1️⃣ Download the artifact release

Eight large Conifer scored files are tracked with Git LFS.

~~~
git lfs install
git clone https://github.com/SJTUFreshman/tmp_x7k9p4r2d8m5c.git
cd tmp_x7k9p4r2d8m5c
git lfs pull
~~~

### 2️⃣ Re-score the stored MATH result

This CPU-only path needs no benchmark download, model server, or GPU.

~~~
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install sympy

python reproduction/code/jca/experiments/math_specific_sft_rl_v1/evaluate_main_table.py \
  summarize \
  --input reproduction/results/MATH/CL/math_specific_sft_rl_20260830/eval/results.jsonl \
  --output runs/math-main-table-summary.log
~~~

The output path must be new. The scorer retains the final answer within three recorded protocol turns; if a trajectory exceeds the limit or does not stop by turn three, it falls back to the first tentative answer. The fixed shard has SHA-256 <code>3d4b0649a1a4f6198ed22b138fb82b31d7339be65997af4175bf9bdcc6343183</code>.

### 3️⃣ Run a fresh MATH evaluation

Supply the external MATH parquet root, the original <code>shard_04.jsonl</code>, and three OpenAI-compatible endpoints serving the final A1/A2/A3 adapters.

~~~
python -m pip install pandas pyarrow

python reproduction/code/jca/experiments/math_specific_sft_rl_v1/evaluate_main_table.py \
  run \
  --source-root /path/to/MATH \
  --shard-file /path/to/test_10x500_seed42/shard_04.jsonl \
  --output-root runs/math-cl-new \
  --api-base-a1 http://127.0.0.1:8201/v1 --api-model-a1 A1 \
  --api-base-a2 http://127.0.0.1:8212/v1 --api-model-a2 A2 \
  --api-base-a3 http://127.0.0.1:8223/v1 --api-model-a3 A3
~~~

The source root must contain one subject directory per MATH subject, each with <code>test-00000-of-00001.parquet</code>. The runner creates a new output root, executes shard construction, initialization, evaluation, and finalization, then writes the score summary. Authentication is supplied through the client environment or <code>--api-key</code>; credentials are not stored here.

## 🧰 Training and other evaluations

| Goal | Entry point |
| --- | --- |
| Shared CL utilities and analysis | <code>reproduction/code/jca/scripts/</code> |
| MATH CL training and evaluation | <code>reproduction/code/jca/experiments/math_specific_sft_rl_v1/</code> |
| Conifer pipeline | <code>reproduction/code/jca/conifer_training_hub/</code> |
| MultiPL-E scoring | <code>reproduction/code/jca/Code/MultiPL-E/</code> |
| Homogeneous controls | <code>reproduction/code/jca_homo_gsm/</code>, <code>reproduction/code/jca_homo_math/</code> |
| Baseline methods | <code>reproduction/code/AT-GRPO/</code>, <code>reproduction/code/MAGRPO/</code>, <code>reproduction/code/MAPoRL/</code> |
| Saved training records | <code>reproduction/training/</code> |

There is no single environment lockfile. A full run may require Python 3.10+, PyTorch/CUDA, Transformers, PEFT, Accelerate, vLLM, language runtimes, and benchmark-specific evaluators. Use the checked-in configs and logs for paths, seeds, model servers, and output conventions.

The MATH training chain is an external-resource workflow: `00_build_sft_data.sh` → `01_train_sft.sh` → `02_reuse_sampled_rl.sh` → `03_score_rl.sh` → `04_prepare_rl.sh` → `05_train_rl.sh` → `06_eval_suite.sh`. These stages expect model checkpoints, benchmark data, GPU workers, and endpoint configuration from `config.env`; the CPU-only scorer below is the reproducible starting point for a fresh checkout.

## 🖼️ Figures

All five paper figures are included as standalone PDFs with PNG previews for GitHub.

| Figure | Contents | Files |
| --- | --- | --- |
| 1 | Model slicing and learned collaboration | [PDF](assets/figures/Figure1.pdf), [preview](assets/figures/previews/Figure1-1.png) |
| 2 | Collaboration protocol and pyramid supervision | [PDF](assets/figures/Figure2.pdf), [preview](assets/figures/previews/Figure2-1.png) |
| 3 | Benchmark ablations | [PDF](assets/figures/Figure3.pdf), [preview](assets/figures/previews/Figure3-1.png) |
| 4 | Computation cost versus task performance | [PDF](assets/figures/Figure4.pdf), [preview](assets/figures/previews/Figure4-1.png) |
| 5 | Learned interaction topology | [PDF](assets/figures/Figure5.pdf), [preview](assets/figures/previews/Figure5-1.png) |

## 📚 Documentation

- [`reproduction/code/jca/experiments/math_specific_sft_rl_v1/evaluate_main_table.py`](reproduction/code/jca/experiments/math_specific_sft_rl_v1/evaluate_main_table.py): the consolidated MATH `run` and `summarize` entry point.
- [`reproduction/code/jca/experiments/math_specific_sft_rl_v1/06_eval_suite.sh`](reproduction/code/jca/experiments/math_specific_sft_rl_v1/06_eval_suite.sh): shell wrapper for a fresh MATH evaluation.
- [`reproduction/code/jca/scripts/`](reproduction/code/jca/scripts/): shared rollout, serving, evaluation, and analysis utilities.
- [Anonymous paper page](https://anonymous.4open.science/r/modelcrew): method, experimental protocol, and full discussion.

## 🛠️ System requirements and release scope

Included:

- selected implementation code, benchmark evaluators, launchers, and vendor utilities;
- paper-result records, manifests, configurations, logs, and selected training metrics;
- five rendered paper figures and GitHub-friendly previews;
- public code with embedded API keys and authenticated proxy defaults removed.

Obtained separately for full training or fresh evaluation:

- base-model weights, LoRA/adapters, and checkpoints;
- benchmark datasets, teacher-model pools, and external SFT/RL inputs;
- manuscript TeX/BibTeX source and unrelated historical material;
- GPU resources and model-serving endpoints.

Do not commit credentials or private dataset paths. Use environment variables or a local untracked configuration when a selected script requires authentication.
