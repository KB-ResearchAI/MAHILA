
# Supplementary Methods: Metadata Imputation for Anemia Eye-Image Study

## Overview
This supplementary document describes the preprocessing and imputation
procedures applied to maternal metadata used in the anemia eye-image study.
The code accompanying this document is provided **solely to document the
methodological approach** and does not include any model training or
evaluation steps.


## Data Availability
The underlying dataset is **not publicly available** due to ethical and
privacy constraints. Specifically:
- The analysis uses PostgreSQL database.


Accordingly, only the **metadata handling and imputation methodology**
is shared to enable reproducibility of the analytical approach.



## Metadata Variables
The following metadata fields were used.

### Source Variables
- `date_of_birth`
- `lmp_date` (last menstrual period)
- `sample_date`
- `hb_value` (hemoglobin concentration)

### Derived Variables
- **Age at registration (years)**
- **Days since last menstrual period (days)**


## Timestamp Processing
All timestamps were stored as Unix epoch values (seconds) and converted
to datetime format using `pd.to_datetime` with coercion of invalid values
to missing. Sample collection timestamps were normalized to **Indian
Standard Time (UTC + 5:30)**.



## Derivation of Clinical Variables

### Age at Registration
Age at registration was computed as:
```

(sample_date − date_of_birth) / 365.25

and rounded down to whole years.

### Days Since Last Menstrual Period
Days since last menstrual period were computed as:


sample_date − lmp_date


expressed in days.


## Clinical Plausibility Constraints
To ensure biological plausibility, derived values were constrained prior
to imputation as follows:

| Variable | Plausible Range |
|--------|-----------------|
| Age at registration | 15–49 years |
| Days since LMP | 0–280 days |

Values falling outside these ranges were treated as missing.



## Missing Data Characteristics
A total of **3030 records** were included in the metadata preprocessing
pipeline.

| Variable | Missing (n) | Missing (%) |
|--------|-------------|-------------|
| Age at registration | 294 | 9.7 |
| Days since LMP | 291 | 9.6 |

Missingness was attributable to invalid or absent timestamps as well as
the application of clinical plausibility constraints. All other metadata
fields exhibited complete observations.



## Imputation Methodology
Missing values were imputed using a **multivariate imputation by chained
equations (MICE)** framework implemented via `IterativeImputer`
(scikit-learn).

Imputation parameters were:
- `random_state = 42`
- `max_iter = 20`
- `sample_posterior = True`

Hemoglobin concentration (`hb_value`) was included as an auxiliary
predictor to inform the imputation model.



## Post-Imputation Processing
Following imputation:
- Values were clipped to the original clinical plausibility ranges.
- Observed values were retained preferentially over imputed values.
- Final variables used for downstream analysis were:
  - `age_at_registration_final`
  - `days_since_lmp_final`



## Summary
Approximately **10%** of maternal metadata required imputation. A
clinically constrained MICE-based approach was applied to preserve
realistic variability while ensuring biological plausibility. The
workflow is fully auditable and reproducible, while the dataset itself
remains restricted to protect participant privacy.

