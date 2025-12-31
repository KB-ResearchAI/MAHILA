
# ROI Extraction and Dataset Materialization (Tongue + Conjunctiva)

This repository contains code and reference notebooks used to:
1) extract **tongue** and **conjunctiva (eye)** regions-of-interest (ROIs), and  
2) organize conjunctiva ROI datasets into **chronological Train/Val/Test** splits across multiple **Hb thresholds**, enabling comparable model runs.



## Data access and privacy
The underlying clinical dataset (raw images + linked metadata) is restricted and cannot be shared publicly. All processing in this repository assumes access to the original secured dataset environment.

This repository provides:
- ROI extraction code (tongue + conjunctiva)
- dataset materialization code (chronological splits across Hb thresholds)
- model training/evaluation notebooks (uploaded) that consume these ROI datasets


# 1) Tongue ROI extraction (SAM)

## What the tongue pipeline does
- Loads study records (filtered to valid sample IDs).
- Assigns anemia labels using **Hb < 9.0 g/dL**. (Primary Model)
- Extracts tongue images from per-record ZIP files.
- Runs **Segment Anything Model (SAM)** to produce a tongue mask and ROI.
- Saves ROIs to split folders (`anemic_train_roi`, `anemic_test_roi`, etc.).
- In the improved version, rejected ROIs are saved into `*_fallback` folders for review.

## SAM settings
- Checkpoint: `sam_vit_b_01ec64.pth`
- Model type: `vit_b` (must match checkpoint)

## Tongue ROI extraction quality
Tongue ROI extraction achieved approximately **97%** acceptable ROI quality (internal validation + review).



# 2) Conjunctiva (eye) ROI extraction (consistent segmentation + QC)

## What the conjunctiva pipeline does
- Processes 6 eye streams:
  - `left_eye_1`, `left_eye_2`, `left_eye_3`
  - `right_eye_1`, `right_eye_2`, `right_eye_3`
- For each input image:
  - Applies a consistent HSV-based segmentation rule for conjunctiva-like regions
  - Uses a spatial prior (lower region) and morphology
  - Selects the largest contour
  - Saves extracted ROI images
  - Writes a per-image summary log CSV (area, mean color, saved path)

## Manual inspection (QC filtering)
After automated extraction, ROIs were manually reviewed and images where conjunctiva was not correctly extracted were removed. This QC step is necessary to prevent non-conjunctiva regions from entering training/evaluation sets.

## Conjunctiva ROI extraction quality
Conjunctiva ROI extraction achieved approximately **93%** acceptable ROI quality (internal validation + review).



# 3) Chronological dataset materialization across Hb thresholds (conjunctiva ROIs)

## Why chronological splits are used
A time-based split helps reduce leakage and reflects real deployment more closely, where models are evaluated on later-in-time samples. For sensitivity analysis, datasets are materialized across multiple anemia thresholds while keeping the split membership fixed.

## Key principle: fixed split membership, variable label definition
For each eye stream independently:
1) Sort samples by `sample_date` (ascending).
2) Create a **fixed chronological split** once:
   - Train: earliest 75%
   - Val: next 12.5%
   - Test: latest 12.5%
3) For each Hb threshold **T**, define labels:
   - **Anemic**: Hb < T  
   - **Non-anemic**: Hb ≥ T  
4) Reuse the same Train/Val/Test membership for every threshold; only labels change.

This ensures that performance differences across thresholds are attributable to the Hb cutoff and class balance, not to different samples being used in Train/Val/Test.

## Hb thresholds materialized
The dataset builder creates folders for:
- `7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0`

## Output structure
For each eye × threshold:


<eye>*hb_less_than*<thr>/conjunctiva_extracted/
anemic_train_roi/
anemic_val_roi/
anemic_test_roi/
anemic_not_train_roi/
anemic_not_val_roi/
anemic_not_test_roi/



Example:
- `left_eye_3_hb_less_than_8_5/conjunctiva_extracted/...`

### Important naming detail (avoiding double suffix)
If the source eye folders include a suffix like `_hb_less_than_9_0`, the dataset builder strips the existing suffix before appending the new threshold.  
This prevents outputs like:
- `left_eye_3_hb_less_than_9_0_hb_less_than_8_5`  
and instead produces:
- `left_eye_3_hb_less_than_8_5`

## Auditing output
A combined CSV summary is written containing per-eye/per-threshold counts:
- totals per split
- anemic vs non-anemic per split
- output paths

Example:


tri_split_summary_all_eyes_all_thresholds_timebased.csv





# 4) Running the model notebooks (minimal changes from the primary model)

Model training/evaluation notebooks for each configuration have been uploaded to this repository for reference.

All experiments listed below follow the same chronological split logic and use the same conjunctiva ROI folder schema. To run different configurations from the primary (baseline) notebook, change only:

1) **Dataset path** (point to the correct Hb threshold folder)
2) **Resolution** (resize parameter)
3) **Input type** (single-eye vs tri-eye vs hex-eye)

No other changes should be required unless explicitly noted in that notebook.

## Baseline (Hb < 9.0, chronological)
Single-eye models at:
- 224 resolution: Right Eye 1/2/3, Left Eye 1/2/3
- 448 resolution: Right Eye 1/2/3, Left Eye 1/2/3

## Multi-threshold models (chronological)
For Hb thresholds:
- 7.0, 7.5, 8.0, 8.5, 9.5, 10.0

Configurations include:
- Hex model (all 6 images) at 224 resolution
- Tri models (right 3 images / left 3 images) at 224 resolution
- Single-eye models at 780 resolution for each eye stream

### How to switch threshold in a notebook
Update the dataset root used by the dataloader to the desired threshold folder, e.g.:
- from `.../right_eye_1_hb_less_than_9_0/...`
- to   `.../right_eye_1_hb_less_than_8_5/...`

### How to switch resolution
Update the preprocessing resize parameter in the notebook, e.g.:
- `224` → `448` → `780`

### How to switch model input type
- Single-eye: use one eye stream folder
- Tri-eye: use three eye streams from one side (left or right)
- Hex-eye: use all six eye streams



## Repository contents (high level)
- Tongue ROI extraction (SAM + improved post-filtering)
- Conjunctiva ROI extraction (consistent segmentation + QC)
- Chronological dataset builder across Hb thresholds
- Uploaded training/evaluation notebooks for all model variants



