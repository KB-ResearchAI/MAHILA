# Supplementary Methods: Chronological Split Tongue Models (Hb < 9 g/dL)

Refer : Chronological Split Tongue HB less than 9 Models.ipynb for detailed information


## Purpose
This folder contains a **secondary, comparative analysis** using tongue
images under a **chronological (time-ordered) data split** (Hb < 9 g/dL).

These experiments were performed **only to compare against the eye
conjunctiva–based models**, with the specific aim of evaluating whether
tongue images provide meaningful predictive signal for anemia detection.

**Tongue models are not primary models of the study.**

The results demonstrate that **tongue-based models perform substantially
worse than eye conjunctiva models**, reinforcing the conclusion that **eye
images alone are sufficient and superior for anemia prediction**.



## Relationship to Primary Models
| Aspect | Eye Conjunctiva Models | Tongue Models |
||--||
| Visual modality | Palpebral conjunctiva | Tongue |
| Split strategy | Chronological | Chronological |
| Role in study | Primary | Comparative |
| Predictive performance | Strong | Weak / unstable |



## Dataset Split Strategy
All tongue images were split **strictly by acquisition time** into:
- Training
- Validation
- Test

No shuffling or stratification was applied.  
The split boundaries were identical to those used for eye-based models,
ensuring a fair comparison.



## Tongue Image Processing
- Tongue regions were **algorithmically extracted**
- All extracted tongue regions underwent **manual visual inspection**
- Images were excluded if:
  - Tongue region was not fully visible
  - Severe blur or occlusion was present
  - Extraction was incomplete or inaccurate

The **extracted tongue regions (ROI images)** are acceptable for sharing,
whereas original raw images remain restricted due to privacy concerns.



## Models Evaluated
The following tongue-based models were evaluated:

- **Tongue 1 model**
- **Tongue 2 model**
- **Tongue 3 model**
- **Tri-tongue model** (fusion of three tongue images)

All models were trained and evaluated independently using identical
architectures and hyperparameters as the eye-based experiments.



## Test Set Performance Summary (Unseen Chronological Test Data)

Tongue models showed **low sensitivity, low recall, and poor F1-scores**
despite high apparent accuracy driven by class imbalance.

| Model | Precision | Recall | F1 | Accuracy | AUC |
|--|--|--|-|-|--|
| Tri-tongue | 0.33 | 0.17 | 0.23 | 0.84 | 0.64 |
| Tongue 1 | 0.31 | 0.19 | 0.24 | 0.84 | 0.68 |
| Tongue 2 | 0.50 | 0.02 | 0.04 | 0.87 | 0.66 |
| Tongue 3 | 0.20 | 0.23 | 0.22 | 0.78 | 0.60 |

Key observations:
- **Recall (sensitivity) is extremely low**, particularly for Tongue 2
- Apparent accuracy is misleading due to class imbalance
- Discriminative ability (AUC) remains modest
- Fusion (tri-tongue) does not meaningfully improve performance



## Interpretation
Compared to eye conjunctiva models, tongue-based models exhibit:
- Poor sensitivity to anemic cases
- Unstable precision–recall trade-offs
- Limited clinical utility

These findings indicate that **tongue images do not encode sufficient
visual information for reliable anemia prediction**, even after region
extraction and manual quality control.



## Summary
- Tongue models were evaluated as a **comparative analysis only**
- Chronological split mirrors eye-based experiments
- Tongue regions were algorithmically extracted and manually inspected
- Performance on unseen test data was **consistently inferior** to eye
  conjunctiva models
- Results support the central conclusion that **eye conjunctiva images
  alone are sufficient and superior for anemia prediction**


