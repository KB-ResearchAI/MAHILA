# MAHILA

**Moderate Maternal Anemia Heuristics using Imaging Learning Algorithms**

MAHILA focuses on non-invasive anemia screening using routinely captured **eye conjunctiva** and **tongue** images in maternal health settings.

This repository contains code, notebooks, and scripts for **region-of-interest (ROI) extraction**, **dataset construction**, and **model training and evaluation** for anemia classification using eye and tongue images.

The work emphasizes **chronological (time-based) learning**, evaluation across multiple **hemoglobin (Hb) thresholds**, and consistent dataset construction to support fair evaluation and longitudinal robustness.

> **Data access note**
> Fully **anonymized ROI-level data** are shared in this repository to enable reproducibility.
> Original raw clinical images and linked patient-identifiable metadata are not redistributed due to data-governance constraints.



## Model Architecture

This work primarily uses **ResNet-18–based convolutional neural networks** for anemia classification from eye conjunctiva and tongue ROIs.

The ResNet-18 architecture employs **residual (skip) connections**, enabling stable training of deeper networks by mitigating vanishing gradients and improving feature reuse.

![ResNet-18 Architecture](assets/resnet18.png)

**Architecture highlights:**

* Initial convolution followed by max pooling
* Four residual stages with progressive channel expansion:

  * 64 → 128 → 256 → 512
* Global average pooling
* Fully connected classification head with Softmax activation

The architecture diagram is provided for conceptual understanding of the residual learning framework.



## Model Configuration and Training Hyperparameters

| Component             | Configuration                             |
|  | -- |
| Backbone              | ResNet-18                                 |
| Input Modalities      | Eye conjunctiva ROIs, Tongue ROIs         |
| Input Resolution      | 224 × 224 (also evaluated: 448, 780)      |
| Pretraining           | ImageNet                                  |
| Optimizer             | Adam                                      |
| Initial Learning Rate | 1e-4                                      |
| Batch Size            | 32                                        |
| Loss Function         | Binary Cross-Entropy                      |
| Class Balancing       | Weighted loss                             |
| Epochs                | 30–50                                     |
| Early Stopping        | Validation AUROC                          |
| Evaluation Metrics    | AUROC, Accuracy, Sensitivity, Specificity |
| Deployment Formats    | ONNX, TFLite                              |



## Ethics and Privacy

All shared data are **fully anonymized** and contain **no personally identifiable information**.
This work complies with ethical guidelines for secondary analysis of clinical imaging data.

* No patient identifiers are released
* Data are intended strictly for **non-commercial research**
* Models should not be used for clinical decision-making without external validation and regulatory approval



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
├── assets
├── Data
├── Master Results.xlsx
├── requirements.txt
└── README.md
```



## Environment Setup

### Tested Environment

* Python **3.10.12**
* IPython **8.12.3**

### Install Dependencies

```bash
pip install -r requirements.txt
```

> GPU acceleration (PyTorch / CUDA) is mandatory.  
> The codebase assumes a CUDA-enabled environment and does not include CPU-only execution paths.




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

* Color- and morphology-based conjunctiva segmentation
* Spatial priors for lower palpebral conjunctiva isolation
* Automated extraction with region statistics logging
* Manual inspection post-extraction

**Observed Quality**

* ROI acceptance rate: **~93%**



### Tongue Extraction (SAM-based)

**Location**

```
Additional Material - Extraction of Eye Conjunctiva and Tongue Region From Original Images/tongue_extraction.py
```

**Methodology**

* Segmentation using **Segment Anything Model (SAM)**
* Morphological refinement and HSV-based color validation
* Manual inspection post-extraction

**Observed Quality**

* ROI acceptance rate: **~97%**



## Chronological Dataset Construction Across Hb Thresholds

### Why Chronological Splits?

* Prevent temporal leakage
* Simulate real-world deployment
* Enable consistent threshold comparisons



### Core Principle: Fixed Split, Variable Label

For each eye stream independently:

1. Sort samples by **sample_date**
2. Create a fixed chronological split:

   * Train: 75%
   * Validation: 12.5%
   * Test: 12.5%
3. Reassign labels for each Hb threshold **T**

This ensures performance differences reflect **label definition changes only**.



### Hb Thresholds Evaluated

```
7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0
```



## Model Training and Evaluation

### Primary Eye Models (Hb < 9.0)

**Location**

```
1 - Primary Model - Chronological Split Eye HB Less Than 9 Models/
```

**Run Example**

```bash
python "right eye 1/scripts/right_eye_1.py"
```



## Results

* Per-model prediction and metric CSVs
* Exported inference models (`.onnx`, `.tflite`)
* Aggregated results in **Master Results.xlsx**



## Limitations and Failure Modes

* Sensitivity to lighting variation and motion blur
* Occasional segmentation failure in extreme occlusion
* Limited generalizability beyond maternal populations
* Not intended for direct clinical deployment



## Reproducibility Checklist

* [x] Deterministic splits
* [x] Fixed random seeds
* [x] Chronological leakage prevention
* [x] Exported inference models
* [x] Documented preprocessing pipeline



## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
2. DebuggerCafe. *Implementing ResNet18 in PyTorch from Scratch*.
   [https://debuggercafe.com/implementing-resnet18-in-pytorch-from-scratch/](https://debuggercafe.com/implementing-resnet18-in-pytorch-from-scratch/)


