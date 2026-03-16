# Sequential Chain-of-Verification for Mathematical Reasoning

An adaptation of the Chain-of-Verification (CoVe) method for multi-step mathematical problem solving, evaluated on the [Hendrycks MATH dataset](https://github.com/hendrycks/math).

## Overview

This project extends CoVe — originally designed for factual recall — to mathematical reasoning by generating and verifying individual solution steps sequentially. Rather than verifying a complete response all at once, the adapted algorithm verifies each step before using it as context to generate the next, building a sound reasoning chain from the ground up.

## Repository Structure

```
.
├── final_project.ipynb       # Main notebook: runs baseline and CoVe experiments
├── math_parser.py            # Parses and extracts boxed final answers from model output
├── few_shot_examples.json    # Few-shot prompts for step generation, verification planning,
│                             #   verification execution, and revision
├── base_result.pkl           # Results dataframe for the baseline (direct prompting) model
├── cove_result.pkl           # Results dataframe for the adapted CoVe model
└── environment.yml           # Has packages information for the project conda environment
```

## File Descriptions

### `final_project.ipynb`
The main experiment notebook. Runs both the baseline and CoVe models on a stratified sample of 50 problems from the MATH dataset and evaluates accuracy based on the extracted final answer.

### `math_parser.py`
A utility module that extracts and parses the boxed final answer (e.g., `\boxed{42}`) from raw model output strings. Used to determine correctness for both models.

### `few_shot_examples.json`
Contains few-shot examples used to prompt the model at each stage of the CoVe pipeline:
- **Step generation** — generating the next reasoning step
- **Verification planning** — generating verification questions for a step
- **Verification execution** — answering verification questions independently
- **Revision** — revising a step based on verification results

### `base_result.pkl` / `cove_result.pkl`
Pandas DataFrames storing results for the baseline and CoVe models respectively. Each row corresponds to one problem and contains:
- `is_correct` (`bool`) — whether the model's final answer was correct
- `response` (`str`) — the full model output; steps are formatted as markdown headers (`## Step X:`) separated by double newlines (`\n\n`)

## Usage

1. Clone the repository and install dependencies (see below).
2. Open `final_project.ipynb` and run all cells.
3. Results are saved to `base_result.pkl` and `cove_result.pkl`.
4. To inspect existing results without re-running:

```python
import pandas as pd

base = pd.read_pickle("base_result.pkl")
cove = pd.read_pickle("cove_result.pkl")

print(f"Baseline accuracy: {base['is_correct'].mean():.2%}")
print(f"CoVe accuracy:     {cove['is_correct'].mean():.2%}")
```

## Requirements

Clone the repository and recreate the conda environment:
```bash
conda env create -f environment.yml
conda activate stats_425_final
```

The key dependencies include:

| Package | Version |
|---------|---------|
| Python | 3.14.3 |
| together | 2.2.0 |
| pandas | 3.0.1 |
| numpy | 2.4.2 |
| jupyterlab | 4.5.5 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.10.8 |

> **Note:** A [Together AI](https://www.together.ai/) API key is required. Set it as an environment variable before running the notebook

## Results

Evaluated on a stratified sample of 50 problems from the MATH dataset across topics (algebra, geometry, number theory, precalculus, probability) and difficulty levels.

| Model    | Accuracy |
|----------|----------|
| Baseline | 88% (44/50) |
| CoVe     | 90% (45/50) |

## Reference

This project is based on:

> Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu, Roberta Raileanu, Xian Li, Asli Celikyilmaz, and Jason Weston. 2024. [Chain-of-Verification Reduces Hallucination in Large Language Models](https://aclanthology.org/2024.findings-acl.212/). In *Findings of the Association for Computational Linguistics: ACL 2024*, pages 3563–3578, Bangkok, Thailand. Association for Computational Linguistics.
