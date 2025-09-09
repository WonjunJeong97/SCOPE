<p align="center">
  <a href="https://arxiv.org/abs/2507.18182">
    <img alt="arXiv: 2507.18182"
         src="https://img.shields.io/badge/arXiv%3A%202507.18182-b31b1b?style=flat-square&logo=arXiv&logoColor=white&labelColor=b31b1b">
  </a>
  <img alt="Python 3.10+"
       src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white&labelColor=3776AB">
  <a href="LICENSE">
    <img alt="License"
         src="https://img.shields.io/badge/License-MIT-16a34a?style=flat-square&logo=opensourceinitiative&logoColor=white&labelColor=16a34a">
  </a>
</p>

<h1 align="center">SCOPE: Stochastic and Counterbiased Option Placement for Evaluating Large Language Models</h1>

<p align="center">
  <a href="#-quick-start">Quick start</a> •
  <a href="#-repository-structure">Repository</a> •
  <a href="#-citation">Cite</a> •
  <a href="#-contact">Contact</a>
</p>

> A framework for multiple-choice evaluation that **mitigates selection bias** by counterbalancing position and semantic preferences in language models.

- **Paper**: _SCOPE: Stochastic and Counterbiased Option Placement for Evaluating Large Language Models_ (**https://arxiv.org/abs/2507.18182**)  
- **Core idea**: use **Inverse-Positioning (IP)** to offset models’ positional biases and **Semantic-Spread (SS)** to spatially separate similar distractors, reducing guesswork.

---

## ✨ TL;DR

- **Position bias**: models disproportionately select certain answer slots (e.g., first/last); IP offsets this by placing the true answer in a less-preferred position.  
- **Semantic bias**: models tend to choose semantically similar distractors when uncertain; SS identifies near-miss distractors and **spreads them apart** to prevent clustering.  
- **General**: jointly applying IP + SS yields a fairer multiple-choice benchmark for large language models.

---

<p align="center">
  <img src="figures/pipeline.png" alt="SCOPE Pipeline (IP + SS)" width="96%">
</p>

---

## 🛠️ Quick start

All scripts are designed for ease of reproducibility; you should be able to run the benchmarks within a few minutes.

### Clone & setup

```bash
# 1) clone
git clone https://github.com/WonjunJeong97/SCOPE.git
cd SCOPE

# 2) Python deps (3.10+)
python -m venv .venv && source .venv/bin/activate   # or: conda create -n scope python=3.10 -y && conda activate scope
pip install -r requirements.txt
````

### Environment variables

```bash
cp .env.example .env
# Edit .env with any required API keys/tokens (e.g., OpenAI, HuggingFace) if your model requires them.
```

### Jupyter notebooks

```bash
python -m pip install jupyter
jupyter lab
# Open notebooks under notebooks/ and run the first cells to verify your setup.
```

### Quick smoke test (1–2 min)

Run the built-in **test mode** to verify your installation end to end:

```bash
bash scripts/run_evaluation.sh -t
# Optionally pin dataset/model (same test mode, just more explicit):
bash scripts/run_evaluation.sh -t -d csqa -m gpt-3.5-turbo
```

If it completes without errors, you’re ready to reproduce the paper.

> Note: This assumes `.env` is set up and the fixed datasets exist at the paths in `configs/default.yaml`.

---

## 📁 Repository structure

```
SCOPE/
├─ configs/        # per-table/figure experiment configs (YAML)
├─ figures/        # static images for README/docs (pipeline, schematics)
├─ scripts/        # download / train / eval / run_all helpers
├─ src/            # core implementation (data, models, utils, train.py, etc.)
├─ notebooks/      # demo & reproduction notebooks
├─ requirements.txt
├─ .env.example    # environment variable template
└─ README.md
```

---

## 📚 Citation

If this repository or the SCOPE framework helps your research, please cite:

```
@article{jeong2025scope,
  title   = {SCOPE: Stochastic and Counterbiased Option Placement for Evaluating Large Language Models},
  author  = {Jeong, Wonjun and Kim, Dongseok and Whangbo, Taegkeun},
  journal = {arXiv preprint arXiv:2507.18182},
  year    = {2025}
}
```

You may also cite the code base itself (optional):

```
@misc{scope_code_2025,
  title        = {SCOPE Codebase},
  author       = {Jeong, Wonjun and Kim, Dongseok and Whangbo, Taegkeun},
  howpublished = {\url{https://github.com/WonjunJeong97/SCOPE}},
  year         = {2025}
}
```

---

## 🤝 Contact

* **Maintainer**: Wonjun Jeong ([tp04045@gachon.ac.kr](mailto:tp04045@gachon.ac.kr))
* **Questions & issues**: please open a GitHub Issue in this repository.

---

## 📝 License

This project is released under the terms of the license in `LICENSE`.

```
```
