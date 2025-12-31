# Supplementary Methods: Chronological Multi-Threshold Dataset Construction and Model Variants

## Purpose of This Supplementary Material
This repository accompanies the manuscript submitted for publication and documents **all non-primary model variants** derived from the primary conjunctiva-based anemia prediction model. The objective of this supplement is to provide **full transparency, reproducibility, and methodological justification** for experiments conducted across multiple hemoglobin (Hb) thresholds, image resolutions, and multi-image configurations.

All model notebooks corresponding to the experiments described below have been **uploaded as supplementary reference files**.

## Primary Reference Model
The **primary model** reported in the main manuscript used:

- **Chronological (time-based) data split**
- **Hb < 9.0 g/dL** as the anemia definition in the primary model
- **Conjunctiva region-of-interest (ROI) images**
- Fixed training pipeline and hyperparameters

This model serves as the **anchor experiment**. All other models are derived from it using controlled and minimal changes.


## Why Chronological Splitting Was Used
Chronological splitting was chosen to:
- Prevent information leakage across time
- Reflect real-world deployment where future samples are unseen
- Ensure fair evaluation on temporally later beneficiaries

For every eye stream, samples were sorted by `sample_date` and split as:
- **Train:** earliest 75%
- **Validation:** next 12.5%
- **Test:** most recent 12.5%

These split memberships were **computed once and frozen**.


## Why Multiple Hb Thresholds Were Evaluated
Hb < 9.0 g/dL is clinically relevant and was selected for the primary analysis. However, anemia definitions vary across public-health programs and clinical contexts.

To assess robustness, additional datasets were constructed for:


Hb < 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0 g/dL



**Critical design principle:**  
> Changing the Hb threshold modifies **only the class label**, not the Train/Val/Test split.

This ensures that differences in performance are attributable to the anemia definition itself rather than differences in sample composition.



## Dataset Materialization Strategy
The script  
`metadata_chronological_multithreshold_split.py`  
implements the following logic:

1. Uses the **Hb < 9.0 conjunctiva ROI folders** as the base image inventory.
2. Creates a **fixed chronological split** for each eye stream.
3. Reuses the same split for all Hb thresholds.
4. Writes threshold-specific datasets in a standardized folder structure.

### Corrected Folder Naming
Output folders are written as:


left_eye_3_hb_less_than_8_5


This preserves a clean and interpretable dataset lineage.



## Standard Dataset Folder Structure
For each eye stream and Hb threshold:


<eye>*hb_less_than*<threshold>/
└── conjunctiva_extracted/
├── anemic_train_roi/
├── anemic_val_roi/
├── anemic_test_roi/
├── anemic_not_train_roi/
├── anemic_not_val_roi/
└── anemic_not_test_roi/



This structure is identical across all thresholds and model variants.



## Image Curation and Quality Control
- Conjunctiva ROIs were extracted algorithmically and **manually inspected**.
- Images with unsatisfactory ROI extraction were removed.
- Manual inspection was performed across approximately **3,021 beneficiaries**.

### Multi-Image Inclusion Rules
- **Tri-eye models:** a beneficiary was included only if **all three images** for that eye side passed ROI QC.
- **Hex-eye models:** a beneficiary was included only if **all six eye images** passed ROI QC.

This ensured consistent multi-view inputs without missing images.



## Model Variants and How to Run Them
All model notebooks have been **uploaded as supplementary reference files**.  
Each variant was run by making **minimal changes** to the primary chronological Hb < 9.0 notebook.

### 1. Single-Eye Models
Examples:
- Right Eye 1 — Chronological — 224 resolution — Hb < 9
- Right Eye 1 — Chronological — 448 resolution — Hb < 9
- Right Eye 1 — Chronological — 780 resolution — Hb < 7 / 7.5 / 8 / 8.5 / 9.5 / 10

**How to run:**
- Change only:
  - dataset root path (Hb threshold folder), and/or
  - image resize parameter (224 / 448 / 780)
- All other code and hyperparameters remain unchanged.



### 2. Tri-Eye Models (3 Images)
- **Right-eye tri:** right_eye_1 + right_eye_2 + right_eye_3
- **Left-eye tri:** left_eye_1 + left_eye_2 + left_eye_3

Tri-eye models were evaluated across Hb thresholds:


7.0 → 10.0 g/dL



**How to run:**
- Point the training notebook to the appropriate tri-eye threshold folders.
- No architectural or training changes were required.



### 3. Hex-Eye Models (All 6 Images)
Hex-eye models used all six conjunctiva images per beneficiary.

- Evaluated primarily at **224 resolution**
- Hb thresholds: **7.0 → 10.0 g/dL**

**How to run:**
- Use the corresponding threshold folders for all six eyes.
- Training pipeline identical to the primary model.



## Summary of Experimental Design
- One **fixed chronological split**
- Multiple **Hb thresholds**
- Multiple **image resolutions**
- Single-eye, tri-eye, and hex-eye configurations
- **No hyperparameter tuning between variants**

This design ensures that observed differences in performance are driven by:
- Hb threshold definition,
- number of images used,
- image resolution,

and **not** by dataset construction artifacts.



## Data Availability Statement
Original full-field clinical images cannot be shared due to privacy and ethical constraints. Only **algorithm-extracted and manually inspected ROIs** are included for reproducibility.



