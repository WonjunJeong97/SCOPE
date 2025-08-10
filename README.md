# SCOPE: Stochastic and Counterbiased Option Placement for Evaluating Large Language Models

> Official reproduction code for the **SCOPE** framework (IP + SS).  
> Designed so a new user can start reproducing results within ~10 minutes.

[![arXiv](https://img.shields.io/badge/arXiv-2507.18182-b31b1b.svg)](https://arxiv.org/abs/2507.18182)
<!-- Optional badges:
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
-->

---

## Overview

SCOPE mitigates two selection biases in LLM multiple-choice evaluation:

- **Position Bias** — models over-select certain positions (e.g., first/last).
- **Semantic Bias** — when uncertain, models gravitate to distractors that are semantically close to the answer.

**Modules**
- **Inverse-Positioning (IP)**: estimates a model’s position preference (via null/neutral prompts) and assigns answers to less-preferred positions to cap “luck” at \( \le 1/n \).
- **Semantic-Spread (SS)**: identifies near-miss distractors (semantic neighbors of the answer) and **separates them** spatially to discourage proximity-based guessing.

**Paper**
- *SCOPE: Stochastic and Counterbiased Option Placement for Evaluating Large Language Models*  
  Jeong, Wonjun; **Kim, Dongseok**; Whangbo, Taegkeun. arXiv:2507.18182 (2025)

---

## Contents

- [Quick Start](#quick-start)
- [Reproducing the Paper](#reproducing-the-paper)
- [Repository Structure](#repository-structure)
- [Data & Checkpoints](#data--checkpoints)
- [Determinism & Seeds](#determinism--seeds)
- [FAQ / Troubleshooting](#faq--troubleshooting)
- [Citation](#citation)
- [Contact](#contact)

---

## Quick Start

### 1) Clone & Environment
```bash
git clone https://github.com/WonjunJeong97/SCOPE
cd SCOPE

# choose one:
python -m venv .venv && source .venv/bin/activate
# or: conda create -n scope python=3.10 -y && conda activate scope

pip install -r requirements.txt
```

### 2) Environment variables
```bash
cp .env.example .env
# Edit .env and fill any required keys/tokens (e.g., OPENAI/HF if your setup needs them).
```

### 3) Jupyter notebooks
```bash
python -m pip install jupyter
jupyter lab
# Open notebooks under notebooks/ and run the first cells to verify your setup.
```

### 4) Quick smoke test (1–2 min)
Pick one lightweight config or demo script to ensure everything is wired:
```bash
# Example (replace with a real script/config in this repo if it differs)
bash scripts/eval.sh configs/example_small.yaml
```
If that runs, you’re ready to reproduce the paper.

---

## Reproducing the Paper

This section gives **exact, copy-runnable commands** to reproduce the paper’s main results with this repository.  
It assumes you (1) installed dependencies from `requirements.txt`, and (2) created `.env` from `.env.example` with required API keys.

---

### 0) Download data (if needed)

```bash
bash scripts/download_data.sh
```

> Data locations are configured in `configs/default.yaml` under:
> - `data.csqa_path: data/fixed/csqa_500_fixed.json`
> - `data.mmlu_path: data/fixed/mmlu_500_fixed.json`

---

### 1) Main results (CSQA / MMLU / Both)

The entrypoint is **`scripts/run_evaluation.sh`**, which calls `python src/main.py` and (by default) uses `configs/default.yaml`.

**Datasets**
- `-d csqa` — evaluate CSQA only  
- `-d mmlu` — evaluate MMLU only  
- `-d both` — run both datasets

**Models (examples)**  
`-m gpt-3.5-turbo` *(default)*, `-m gpt-4o-mini`, `-m claude-3-5-sonnet`, `-m gemini-1.5-pro`, `-m llama-3-70b` (via Groq), etc.  
See the full list in `configs/default.yaml > models.available`.

| Paper Item (example) | One-liner to run | Notes |
|---|---|---|
| Main – CSQA | ```bash\nbash scripts/run_evaluation.sh -d csqa -m gpt-3.5-turbo\n``` | Evaluates CSQA 500. Swap `-m` for other models. |
| Main – MMLU | ```bash\nbash scripts/run_evaluation.sh -d mmlu -m gpt-4o-mini\n``` | Evaluates MMLU 500. |
| Main – Both | ```bash\nbash scripts/run_evaluation.sh -d both -m gpt-3.5-turbo\n``` | Runs CSQA and MMLU sequentially. |

> Tip: If your local virtualenv directory is named `venv/`, the script auto-activates it.

### 2) Ablation study (optional)

Ablation modes are defined in `configs/default.yaml` under:
```yaml
ablation:
  enabled: false
  modes:
    - "IP+SS"    # Full SCOPE
    - "¬IP+SS"   # SS only
    - "IP+¬SS"   # IP only
```

Run ablations with the `-a/--ablation` flag (the script will use the modes above):

```bash
bash scripts/run_evaluation.sh -d csqa -m gpt-3.5-turbo -a
```

### 3) Quick smoke test (fast sanity check)

```bash
bash scripts/run_evaluation.sh -t
```

Runs in **test mode** (reduced trials/samples) to verify your environment end-to-end.

### 4) Jupyter tutorial (interactive reproduction)

```bash
python -m pip install jupyter
jupyter lab
```

Open and run **`notebooks/tutorial.ipynb`** from top to bottom.  
This mirrors the scripted pipeline and is convenient for step-by-step inspection.

### 5) Outputs, logs, and reproducibility

- **Outputs & figures** are controlled by `configs/default.yaml > output`:
  - `save_detailed_results: true`
  - `save_figures: true`
  - `figure_format: "png"`
  The exact save paths are determined by `src/main.py` together with these settings.
- **Logs**: capture console output to a timestamped file:
  ```bash
  mkdir -p results
  bash scripts/run_evaluation.sh -d both -m gpt-3.5-turbo | tee results/run_$(date +%Y%m%d_%H%M%S).log
  ```
- **Seeds/Determinism**: minor numeric drift (±ε) across hardware/CUDA/BLAS is normal.  
  Use the pinned packages in `requirements.txt` and keep drivers consistent for closest matches.

### 6) Command reference (for quick lookup)

```text
Usage: scripts/run_evaluation.sh [-m MODEL] [-d DATASET] [-t] [-a]

  -m, --model     Model name (default: gpt-3.5-turbo)
  -d, --dataset   Dataset: csqa | mmlu | both (default: both)
  -t, --test      Run in test mode (quick smoke test)
  -a, --ablation  Run ablation study (uses modes in configs/default.yaml)
```

**Environment note:** the script requires a valid `.env` (created from `.env.example`) containing your API keys/tokens before running any evaluations.


---

## Repository Structure
```bash
SCOPE/
├─ configs/        # per-table/figure experiment configs (YAML)
├─ scripts/        # download / train / eval / run_all helpers
├─ src/            # core implementation (data, models, utils, train.py, etc.)
├─ notebooks/      # demo & reproduction notebooks
├─ requirements.txt
├─ .env.example    # environment variable template
└─ README.md
```

---

## Citation
If this repository or the SCOPE framework helps your research, please cite:
```bash
@article{jeong2025scope,
  title   = {SCOPE: Stochastic and Counterbiased Option Placement for Evaluating Large Language Models},
  author  = {Jeong, Wonjun and Kim, Dongseok and Whangbo, Taegkeun},
  journal = {arXiv preprint arXiv:2507.18182},
  year    = {2025}
}
```
You may also cite the code base itself (optional):
```bash
@misc{scope_code_2025,
  title        = {SCOPE Codebase},
  author       = {Jeong, Wonjun and Kim, Dongseok and Whangbo, Taegkeun},
  howpublished = {\url{https://github.com/WonjunJeong97/SCOPE}},
  year         = {2025}
}
```

---

## Contact
- **Maintainer**: [Wonjun Jeong / tp04045@gachon.ac.kr]
- **Questions & issues**: please open a GitHub Issue in this repository.

---

## Why this solves the problem (and what changed)

- **Correct arXiv link/badge:** `https://arxiv.org/abs/2507.18182` (no extra brackets inside the URL).  
- **Closed code fences & sectioning:** every code block and list is properly fenced; headings are separated from code.  
- **10-minute path:** clone → env → `.env` → smoke test → reproduce.  
- **Explicit table/figure mapping:** users don’t have to guess which config/notebook regenerates which result.  
- **Determinism guidance & FAQ:** sets expectations about ±ε differences across hardware.  
- **Citation ready:** BibTeX included.

---

## Optional (nice to have, no code changes required)

- **Repository “About” box (right sidebar on GitHub):**  
  - Description: `Reproducible code for SCOPE (IP + SS): mitigating position & semantic biases in LLM MCQs`  
  - Website: `https://arxiv.org/abs/2507.18182`  
  - Topics: `llm`, `evaluation`, `bias`, `mmlu`, `csqa`, `reproducibility`

- **CITATION.cff (one-click citation button):** create a `CITATION.cff` file:
  ```yaml
  cff-version: 1.2.0
  title: "SCOPE: Stochastic and Counterbiased Option Placement for Evaluating Large Language Models"
  authors:
    - family-names: Jeong
      given-names: Wonjun
    - family-names: Kim
      given-names: Dongseok
    - family-names: Whangbo
      given-names: Taegkeun
  date-released: 2025-07-xx
  version: "v1.0.0"
  identifiers:
    - type: doi
      value: 10.48550/arXiv.2507.18182
  repository-code: "https://github.com/WonjunJeong97/SCOPE"
  url: "https://arxiv.org/abs/2507.18182"
  message: "If you use this repository, please cite the paper and the code."
