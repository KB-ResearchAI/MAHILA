# Chronological Split Eye Models (Hb < 9 g/dL) with Metadata

## Purpose
This folder contains a **secondary, comparative analysis** in which
structured maternal metadata were added to **multi-image eye models**
trained using a **chronological (time-ordered) split** (Hb < 9 g/dL).

These experiments were conducted **solely to compare against the
corresponding image-only multi-image models**, with the explicit objective
of assessing whether metadata provide additional predictive value beyond
eye images alone.

**These are not primary models of the study.**

The primary intent of this analysis is to support the conclusion that
**eye images alone are sufficient to predict anemia**, even in
multi-image settings.



## Relationship to Primary Models
| Aspect | Primary Models | This Folder |
|||-|
| Inputs | Eye images only | Eye images + metadata |
| Model types | Tri-eye, Hexa-eye | Tri-eye, Hexa-eye |
| Split strategy | Chronological | Chronological |
| Role | Main analysis | Comparative |
| Interpretation | Core findings | Supportive |



## Dataset Split Strategy
All samples were split **strictly by acquisition time** into training,
validation, and test sets.

- No shuffling
- No stratification
- Identical split boundaries to the image-only models

This preserves temporal integrity and ensures a fair comparison.



## Models Evaluated
Only **multi-image models** were evaluated in this comparison:

- **Tri-left-eye model**
- **Tri-right-eye model**
- **Hexa-eye model (six-eye input)**

Single-eye models were **not included**, as the comparison was intended
specifically for **multi-image fusion architectures**.



## Metadata Included
The following maternal metadata were incorporated:

- Age at registration (years)
- Days since last menstrual period (days)
- Per-image lighting indicators

Metadata were concatenated at the **feature fusion stage only**.
No architectural changes were made.



## Metadata Handling
- Raw timestamps converted from Unix format
- Clinically implausible values filtered
- Missing values imputed using a **MICE-style Iterative Imputer**
- Observed values always preferred over imputed values
- Final variables:
  - `age_at_registration_final`
  - `days_since_lmp_final`

Metadata processing is identical across all three models.



## Image Quality Control
- All original images were manually inspected
- Images were excluded if conjunctiva extraction failed or was inadequate


## Multi-Image Inclusion Rules

### Tri-Eye Models
A beneficiary was included **only if conjunctiva extraction succeeded for
all three images** of the corresponding eye (left or right).

If extraction failed for any one image, **all three images were excluded**.

### Hexa-Eye Model
A beneficiary was included **only if conjunctiva extraction succeeded for
all six eye images** (three left-eye and three right-eye images).

Partial six-image sets were excluded.

These strict criteria ensured **complete and consistent multi-image inputs**.



## Training and Evaluation
- Identical training pipeline to image-only models
- Same architectures, hyperparameters, and optimization
- Five-fold cross-validation on training data
- Best fold selected using precision–recall balance
- Final evaluation on held-out chronological test set

## Interpretation
Across all three multi-image models, performance with metadata was
**comparable to the corresponding image-only models**.

This indicates that:
- Eye images already encode the dominant predictive signal
- Metadata provide limited incremental benefit
- Image-only models are sufficient for accurate anemia prediction

Please refer the notebook - Chronlogical_Split_Eye_HB_less_than_9_Models_With_Meta_Data.ipynb
