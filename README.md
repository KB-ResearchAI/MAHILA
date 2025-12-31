# MAHILA

**Moderate Maternal Anemia Heuristics using Imaging Learning Algorithms**
MAHILA focuses on non-invasive anemia screening using routinely captured eye and tongue images in maternal health settings.


This repository contains code, notebooks, and scripts for **region-of-interest (ROI) extraction**, **dataset construction**, and **model training/evaluation** for anemia classification using **eye conjunctiva** and **tongue** images.

The work focuses on **chronological (time-based) learning**, multiple **hemoglobin (Hb) thresholds**, and consistent dataset construction to support fair evaluation and longitudinal robustness.

> **Data access note**
> The original clinical images and linked patient metadata are not included in this repository due to data-governance constraints. All scripts assume access to the original dataset through approved channels.



## Repository Structure

```
.
├── 1 - Primary Model - Chronological Split Eye HB Less Than 9 Models
├── 2 - Model Evaluation Chronological Split HB Less Than 9 Via Tflite
├── 3 - Random Stratified Eye Split HB Less Than 9 Models
├── 4 - Metadata Imputation
├── 5 - Chronlogical Split Eye HB Less Than 9 Models With Meta Data
├── 6 - Chronological Split Tongue HB Less Than 9 Models
├── 7 - Other Chronological Split Model At Different HB Thresholds & Various Resolution
├── Additional Material - Extraction of Eye Conjunctiva and Tongue Region From Original Images
├── Data
├── Master Results.xlsx
├── requirements.txt
└── README.md
```



## Environment Setup

### Tested Environment

* Python **3.12.x**
* IPython **9.x**

### Install Dependencies

```bash
pip install -r requirements.txt
```

> GPU acceleration (PyTorch / CUDA) is optional. Scripts can be executed in CPU-only mode by disabling CUDA where applicable.



## Data Organization (Expected Structure)

The scripts expect datasets to follow a **standardized ROI-based folder layout**.

### Conjunctiva ROIs (Example: Hb < 9.0)

```
Data/
└── left_eye_1_hb_less_than_9_0/
    └── conjunctiva_extracted/
        ├── anemic_train_roi/
        ├── anemic_val_roi/
        ├── anemic_test_roi/
        ├── anemic_not_train_roi/
        ├── anemic_not_val_roi/
        └── anemic_not_test_roi/
```

### Tongue ROIs (Example: Tongue-1, Hb < 9.0)

```
Data/
└── tongue_1_hb_less_than_9_0/
    └── tongue_extracted/
        ├── anemic_train_roi/
        ├── anemic_val_roi/
        ├── anemic_test_roi/
        ├── anemic_not_train_roi/
        ├── anemic_not_val_roi/
        └── anemic_not_test_roi/
```



## ROI Extraction

### Eye Conjunctiva Extraction

**Location**

```
Additional Material - Extraction of Eye Conjunctiva and Tongue Region From Original Images/conjunctiva_extraction.py
```

**Methodology**

* Consistent color- and morphology-based conjunctiva segmentation
* Spatial priors to isolate lower-palpebral conjunctiva
* Automated extraction with logging of region statistics
* Manual inspection performed post-extraction to remove failures

**Observed Quality**

* Conjunctiva ROI acceptance rate: **~93%**



### Tongue Extraction (SAM-based)

**Location**

```
Additional Material - Extraction of Eye Conjunctiva and Tongue Region From Original Images/tongue_extraction.py
```

**Methodology**

* Image extraction from archived ZIP files
* Segmentation using **Segment Anything Model (SAM)**
* Morphological refinement and HSV-based color validation
* Optional fallback storage for rejected masks
* Manual inspection performed post-extraction

**Observed Quality**

* Tongue ROI acceptance rate: **~97%**



## Chronological Dataset Construction Across Hb Thresholds

### Why Chronological Splits?

Chronological (time-based) splitting was used instead of random stratification to:

* Prevent temporal leakage
* Simulate real-world deployment conditions
* Ensure consistent comparisons across Hb thresholds



### Core Principle: Fixed Split, Variable Label

For each eye stream independently:

1. Samples are sorted by **sample_date** (ascending).
2. A single chronological split is created:

   * Train: 75%
   * Validation: 12.5%
   * Test: 12.5%
3. This split **never changes**.
4. For each Hb threshold **T**, labels are reassigned:

   * Anemic if `Hb < T`
   * Non-anemic otherwise

This ensures that performance differences across thresholds reflect **label definition changes only**, not data leakage or split variation.



### Hb Thresholds Evaluated

```
7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0
```

### Output Naming Convention

```
left_eye_3_hb_less_than_8_5/
right_eye_1_hb_less_than_10_0/
```

Existing suffixes are stripped automatically to avoid nested threshold names.

A combined split summary is saved as:

```
tri_split_summary_all_eyes_all_thresholds_timebased.csv
```



## Model Training and Evaluation

### Primary Eye Models (Hb < 9.0)

**Location**

```
1 - Primary Model - Chronological Split Eye HB Less Than 9 Models/
```

Includes:

* Single-eye models (left/right eye 1, 2, 3)
* Tri-eye models (tri left / tri right)
* Hex-eye model (all six eyes)

**Run Example**

```bash
python "right eye 1/scripts/right_eye_1.py"
```

Artifacts (metrics, ONNX, TFLite) are saved under each model’s `outputs/` directory.



### TFLite Evaluation

**Location**

```
2 - Model Evaluation Chronological Split HB Less Than 9 Via Tflite/
```

Run via:

* `tflite_evaluation_python_file.py`
* or `tflite_evaluation_notebook_file.ipynb`



### Random Stratified Baselines

**Location**

```
3 - Random Stratified Eye Split HB Less Than 9 Models/
```

Provided for comparison against chronological splitting.



### Other Hb Thresholds and Resolutions

**Location**

```
7 - Other Chronological Split Model At Different HB Thresholds & Various Resolution/
```

Includes:

* Dataset generation script
* Executed notebooks for:

  * Multiple Hb thresholds
  * Multiple image resolutions (224, 448, 780)

To run a different Hb threshold:

1. Update dataset root path
2. Adjust input resolution if required
3. Keep the chronological split logic unchanged



## Results

* Per-model CSVs: predictions, metrics, cross-validation indices
* Exported models: `.onnx`, `.tflite`
* Aggregated metrics: **Master Results.xlsx**



## Reproducibility Notes

This repository provides:

* Complete preprocessing logic
* Deterministic dataset construction
* Fully reproducible training pipelines

The original dataset is not redistributed. Researchers with approved access can reproduce results by following the documented pipeline.

