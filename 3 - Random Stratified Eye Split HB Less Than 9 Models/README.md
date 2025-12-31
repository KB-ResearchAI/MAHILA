# README.md

# Random-Stratified Experiments

This folder contains random-stratified experiments conducted as a robustness
evaluation of the chronological models.

All model architectures, training logic, and evaluation metrics remain identical
to the chronological setup. The only differences are the dataset splitting
strategy and selected hyperparameters (where applicable).

Chronological evaluation remains the primary and deployment-relevant analysis.

## Models Covered
- 6 × Single-eye models
- 2 × Tri-eye models
- 1 × Hexa-eye model (shared backbone)

## How to Run
The same training code used for chronological experiments is executed with
random-stratified dataset splitting and a fixed random seed allthough the changes were made in how the data were spitted and adjustment of hyperparameters.

All executions and results are available in:
`all_models_random_stratified_eye_split_python_notebook.ipynb`

## Reproducibility
- Fixed random seed
- Deterministic training
- No data leakage
