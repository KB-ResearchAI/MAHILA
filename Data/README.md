# Supplementary Methods: Dataset Curation and Image Counts  
**Chronological Split Eye-Image Dataset (Hb < 9 g/dL)**

## Overview
This document describes the dataset curation, quality control procedures,
image inclusion criteria, and image count verification for the anemia
classification models developed using eye images under a chronological
data split (Hb < 9 g/dL). 

## Data Availability Statement
The original eye images cannot be publicly shared due to ethical and
privacy constraints, as they are directly linked to identifiable clinical
records. However, conjunctiva-extracted images are acceptable for sharing
and has been made available. These extracted images
do not contain identifiable facial features and are sufficient to
reproduce the reported experiments.

## Source Population
Images were collected from 3021 beneficiaries enrolled in the study. Each
beneficiary contributed multiple images across different anatomical
views (left and right eyes), acquired at the time of hemoglobin
measurement.

## Manual Image Inspection and Quality Control
All images underwent manual visual inspection prior to inclusion in model
development. Images were excluded if the conjunctiva extraction algorithm
failed, or if the extracted conjunctiva region was incomplete, blurred,
occluded, or otherwise visually unsatisfactory for analysis. Only images
with satisfactory conjunctiva extraction were retained.

## Conjunctiva Extraction Criteria
All eye-based models rely exclusively on algorithmically extracted
conjunctiva regions, followed by manual verification. Images were removed
if conjunctiva extraction failed entirely, produced partial or
mislocalized regions, or did not adequately capture the palpebral
conjunctiva.

## Model-Specific Inclusion Criteria

### Single-Eye Models
For single-eye models, each image was evaluated independently. Images
with satisfactory conjunctiva extraction were retained, and beneficiaries
could contribute images even if other eye views failed.

### Tri-Eye Models
For tri-eye models (tri-left-eye or tri-right-eye), images were included
only if conjunctiva extraction was successful for all three images
belonging to the same anatomical group. For example, in the tri-right-eye
model, a beneficiary was included only if conjunctiva extraction
succeeded for right_eye_1, right_eye_2, and right_eye_3. If extraction
failed for any one of the three images, all three images from that
beneficiary were excluded.

### Hexa-Eye Models
For hexa-eye models combining six eye images, a beneficiary was included
only if conjunctiva extraction was satisfactory for all six eye images
(three left-eye and three right-eye images). If extraction failed for any
one image, the entire six-image set for that beneficiary was excluded.
This ensured complete six-image inputs for multi-input models.

## Dataset Organization
Conjunctiva-extracted images are organized within the Data/ directory
according to anatomical view, anemia label, and chronological split:

```

Data/
├── left_eye_**hb_less_than_9_0/
│    └── conjunctiva_extracted/
│         ├── anemic_train_roi/
│         ├── anemic_val_roi/
│         ├── anemic_test_roi/
│         ├── anemic_not_train_roi/
│         ├── anemic_not_val_roi/
│         └── anemic_not_test_roi/
│
└── right_eye**_hb_less_than_9_0/
└── conjunctiva_extracted/
└── (same structure as above)

Identically the same steps were performed for tongue

## Image Count Verification
Image counts were verified programmatically using recursive directory
traversal and counting valid image files (.png) in each leaf
folder. Counts were computed per anatomical view, anemia label, and
chronological split.

## Class Distribution
Non-anemic images substantially outnumber anemic images. This imbalance
reflects real-world hemoglobin prevalence. No oversampling or image
duplication was applied, and chronological splitting preserves temporal
integrity.



