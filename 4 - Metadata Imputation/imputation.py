# metadata_imputation.py
# =====================================================================
# Metadata Imputation Pipeline (Documentation Script)
#
# NOTE:
# - This script is shared ONLY to document how metadata imputation
#   was performed.
# - The underlying dataset and images CANNOT be shared.
# - Analysis is performed on ORIGINAL EYE IMAGES (not conjunctiva ROI).
# - No model training or evaluation is performed here.
# =====================================================================

import psycopg2
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

# ---------------------------------------------------------------------
# Load metadata from PostgreSQL
# ---------------------------------------------------------------------
anemia_image = pd.read_sql_query(
    'SELECT * FROM "anemia_image_samples"', con=newc
)

# Retain only valid image-linked samples
anemia_image = anemia_image[
    anemia_image['sample_id'].astype(str).str.contains("LID")
]

# Convert sample_date to datetime (IST)
anemia_image['sample_date'] = pd.to_datetime(
    anemia_image['sample_date'], origin='unix', unit='s'
)
anemia_image['sample_date'] = (
    pd.DatetimeIndex(anemia_image['sample_date']) +
    timedelta(hours=5, minutes=30)
)

# Restrict to clinically valid Hb values
data = anemia_image[
    (anemia_image['hb_value'] > 4) &
    (anemia_image['hb_value'] <= 18)
].copy()

# ---------------------------------------------------------------------
# Convert maternal timestamps
# ---------------------------------------------------------------------
df = data.copy()
df['lmp_date'] = pd.to_datetime(df['lmp_date'], unit='s', errors='coerce')
df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], unit='s', errors='coerce')
df['sample_date'] = pd.to_datetime(df['sample_date'], errors='coerce')

# ---------------------------------------------------------------------
# Derive clinical variables
# ---------------------------------------------------------------------
df['age_at_registration'] = (
    (df['sample_date'] - df['date_of_birth']).dt.days / 365.25
).apply(np.floor)

df['days_since_lmp'] = (
    df['sample_date'] - df['lmp_date']
).dt.days

# ---------------------------------------------------------------------
# Clinical plausibility filtering
# ---------------------------------------------------------------------
df.loc[
    (df['age_at_registration'] < 15) |
    (df['age_at_registration'] > 49),
    'age_at_registration'
] = np.nan

df.loc[
    (df['days_since_lmp'] < 0) |
    (df['days_since_lmp'] > 280),
    'days_since_lmp'
] = np.nan

# ---------------------------------------------------------------------
# Missingness summary (for audit)
# ---------------------------------------------------------------------
missing_summary = pd.DataFrame({
    "missing_count": df[['age_at_registration', 'days_since_lmp']].isna().sum(),
    "percentage": df[['age_at_registration', 'days_since_lmp']]
        .isna().mean() * 100
})
print("\n--- Missingness Before Imputation ---")
print(missing_summary)

# ---------------------------------------------------------------------
# Imputation (MICE-style)
# ---------------------------------------------------------------------
impute_cols = ['age_at_registration', 'days_since_lmp', 'hb_value']
impute_df = df[impute_cols].copy()

imp = IterativeImputer(
    random_state=42,
    max_iter=20,
    sample_posterior=True
)

imputed_array = imp.fit_transform(impute_df)
imputed_df = pd.DataFrame(
    imputed_array, columns=impute_cols, index=df.index
)

# Clip imputed values to clinical ranges
imputed_df['age_at_registration'] = imputed_df['age_at_registration'].clip(15, 49)
imputed_df['days_since_lmp'] = imputed_df['days_since_lmp'].clip(0, 280)

# ---------------------------------------------------------------------
# Final variables (observed preferred over imputed)
# ---------------------------------------------------------------------
df['age_at_registration_final'] = np.where(
    df['age_at_registration'].isna(),
    imputed_df['age_at_registration'],
    df['age_at_registration']
)

df['days_since_lmp_final'] = np.where(
    df['days_since_lmp'].isna(),
    imputed_df['days_since_lmp'],
    df['days_since_lmp']
)

print("\n--- Missingness After Imputation ---")
print(df[['age_at_registration_final', 'days_since_lmp_final']].isna().sum())

# ---------------------------------------------------------------------
# Export (restricted internal use only)
# ---------------------------------------------------------------------
df.to_csv("dataset_imputed.csv", index=False)
