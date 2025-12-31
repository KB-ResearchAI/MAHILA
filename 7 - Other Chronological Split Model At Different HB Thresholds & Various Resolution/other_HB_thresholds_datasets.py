# metadata_chronological_multithreshold_split.py
# -*- coding: utf-8 -*-
"""
Chronological (time-based) dataset builder for conjunctiva ROI images across multiple Hb thresholds.

FIX APPLIED (your request)
--------------------------
Previously, output folder names became:
  left_eye_3_hb_less_than_9_0_hb_less_than_8_5

Now, output folder names will be:
  left_eye_3_hb_less_than_8_5

We do this by "stripping" the existing suffix `_hb_less_than_<x>` from the
SOURCE folder name before appending the new threshold.

Everything else in your logic is unchanged.
"""

import os
import re
import shutil
from datetime import timedelta

import numpy as np
import pandas as pd
import psycopg2

# =========================
# 1) LOAD METADATA (DB)
# =========================
import pandas as pd

# Add the 'r' prefix here
anemia_image = pd.read_csv(r"C:\Users\sarfr\Downloads\Anemia AI Code Repo - Supplement Materials\Data\metadata\dataset_imputed.csv")
anemia_image = anemia_image[anemia_image["sample_id"].astype(str).str.contains("LID")].copy()

# Remove origin and unit since the date is already a string, not a timestamp
anemia_image["sample_date"] = pd.to_datetime(anemia_image["sample_date"])
anemia_image["sample_date"] = anemia_image["sample_date"] + timedelta(hours=5, minutes=30)

metadata = anemia_image[(anemia_image["hb_value"] > 3) & (anemia_image["hb_value"] < 20)].copy()
metadata = metadata.sort_values("sample_date", ascending=True).reset_index(drop=True)

# =========================
# 2) CONFIG (EDIT PATHS)
# =========================
SRC_TREE_ROOT = r"C:\Users\sarfr\Downloads\Anemia AI Code Repo - Supplement Materials\Data"  # <-- EDIT
OUT_LEFT_ROOT = r"C:\Users\sarfr\Downloads\Anemia AI Code Repo - Supplement Materials\Data"  # <-- EDIT
OUT_RIGHT_ROOT = r"C:\Users\sarfr\Downloads\Anemia AI Code Repo - Supplement Materials\Data"  # <-- EDIT

# ✅ DO NOT CHANGE SOURCE FOLDER NAMES (as you requested)
EYE_FOLDERS = [
    "left_eye_1_hb_less_than_9_0",
    "right_eye_1_hb_less_than_9_0",
    "left_eye_2_hb_less_than_9_0",
    "right_eye_2_hb_less_than_9_0",
    "left_eye_3_hb_less_than_9_0",
    "right_eye_3_hb_less_than_9_0",
]

# ✅ Map: existing folder name -> DB metadata column name
FOLDER_TO_META_COL = {
    "left_eye_1_hb_less_than_9_0": "left_eye_1",
    "right_eye_1_hb_less_than_9_0": "right_eye_1",
    "left_eye_2_hb_less_than_9_0": "left_eye_2",
    "right_eye_2_hb_less_than_9_0": "right_eye_2",
    "left_eye_3_hb_less_than_9_0": "left_eye_3",
    "right_eye_3_hb_less_than_9_0": "right_eye_3",
}

THRESHOLDS = [7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]

TRAIN_FRAC = 0.75
VAL_FRAC = 0.125
TEST_FRAC = 0.125

VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")

# =========================
# 3) METADATA SHAPE + SANITY
# =========================
meta_cols_needed = ["sample_id", "sample_date", "hb_value"] + list(FOLDER_TO_META_COL.values())
metadata = metadata[meta_cols_needed].copy()

missing_cols = [c for c in meta_cols_needed if c not in metadata.columns]
if missing_cols:
    raise ValueError(f"Missing columns in metadata: {missing_cols}")

os.makedirs(OUT_LEFT_ROOT, exist_ok=True)
os.makedirs(OUT_RIGHT_ROOT, exist_ok=True)

all_counts = []

# =========================
# 4) HELPERS
# =========================
def copy_partition(part_df: pd.DataFrame, split_root: str, split_name: str) -> None:
    """Copy files for one split partition into anemic vs non-anemic folders."""
    for _, row in part_df.iterrows():
        src = row["src_path"]
        dst_dir = os.path.join(
            split_root,
            f"anemic_{split_name}_roi" if row["label"] == 1 else f"anemic_not_{split_name}_roi",
        )
        dst = os.path.join(dst_dir, os.path.basename(src))
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"❌ Copy failed ({split_name}): {src} -> {dst} | {e}")

def strip_existing_threshold(folder_name: str) -> str:
    """
    Remove trailing '_hb_less_than_<number>' from folder name, if present.
    Example:
      left_eye_3_hb_less_than_9_0  -> left_eye_3
      right_eye_1                 -> right_eye_1 (unchanged)
    """
    return re.sub(r"_hb_less_than_\d+(?:_\d+)?$", "", folder_name)

def thr_to_token(thr: float) -> str:
    """Convert threshold float to folder-safe token: 8.5 -> '8_5'."""
    s = str(thr)
    return s.replace(".", "_")

# =========================
# 5) BUILD DATASETS
# =========================
for folder_name in EYE_FOLDERS:
    meta_col = FOLDER_TO_META_COL[folder_name]
    print(f"\n========== SOURCE FOLDER: {folder_name} | METADATA COL: {meta_col} ==========")

    # (a) Scan the EXISTING folder subtree ONCE to build filename -> path lookup
    eye_root = os.path.join(SRC_TREE_ROOT, folder_name)
    if not os.path.isdir(eye_root):
        print(f"⚠️ Eye root not found: {eye_root} — skipping.")
        continue

    filename_to_path = {}
    for root, _, files in os.walk(eye_root):
        for f in files:
            if f.lower().endswith(VALID_EXTS):
                filename_to_path[f] = os.path.join(root, f)

    if not filename_to_path:
        print(f"⚠️ No files found under {eye_root} — skipping.")
        continue

    # (b) Subset metadata rows that contain filenames for this eye stream (DB column)
    df = metadata[metadata[meta_col].notna()].copy()
    if df.empty:
        print(f"⚠️ No filenames listed in metadata for {meta_col} — skipping.")
        continue

    df[meta_col] = df[meta_col].astype(str)

    # (c) Enforce suffix convention based on the DB column name (NOT the folder name)
    expected_suffix = f"_{meta_col}.png"
    suffix_mask = df[meta_col].str.endswith(expected_suffix, na=False)
    bad_suffix = (~suffix_mask).sum()
    if bad_suffix:
        print(f"⚠️ {bad_suffix} rows do not end with '{expected_suffix}' — skipped.")

    df = df[suffix_mask].copy()
    if df.empty:
        print(f"❌ No rows with filenames ending '{expected_suffix}' for {meta_col} — skipping.")
        continue

    # (d) Map metadata filename -> disk path (from the scanned folder)
    df["src_path"] = df[meta_col].map(filename_to_path)
    miss = df["src_path"].isna().sum()
    if miss:
        print(f"⚠️ {miss} filenames not found under {eye_root} — skipped.")
    df = df[df["src_path"].notna()].copy()
    if df.empty:
        print(f"❌ After mapping, nothing left to process for {folder_name}.")
        continue

    # (e) Chronological order within this stream
    df = df.sort_values("sample_date", ascending=True).reset_index(drop=True)

    # (f) Fixed chronological split indices ONCE (do NOT change per threshold)
    n = len(df)
    n_train = int(np.round(TRAIN_FRAC * n))
    n_val = int(np.round(VAL_FRAC * n))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n - n_train

    train_pos = list(range(0, n_train))
    val_pos = list(range(n_train, n_train + n_val))
    test_pos = list(range(n_train + n_val, n))

    # (g) For each threshold: ONLY label changes, split membership stays constant
    base_name = strip_existing_threshold(folder_name)  # ✅ FIX: prevents double "hb_less_than"
    for thr in THRESHOLDS:
        print(f"--- Threshold: {thr} ---")
        df["label"] = (df["hb_value"] < float(thr)).astype(int)

        out_root = OUT_LEFT_ROOT if folder_name.startswith("left") else OUT_RIGHT_ROOT

        # ✅ FIXED TAG:
        # left_eye_3 + hb_less_than_8_5  => left_eye_3_hb_less_than_8_5
        tag = f"{base_name}_hb_less_than_{thr_to_token(thr)}"
        split_root = os.path.join(out_root, tag, "conjunctiva_extracted")

        for d in [
            "anemic_train_roi",
            "anemic_val_roi",
            "anemic_test_roi",
            "anemic_not_train_roi",
            "anemic_not_val_roi",
            "anemic_not_test_roi",
        ]:
            os.makedirs(os.path.join(split_root, d), exist_ok=True)

        copy_partition(df.iloc[train_pos], split_root, "train")
        copy_partition(df.iloc[val_pos], split_root, "val")
        copy_partition(df.iloc[test_pos], split_root, "test")

        train_df = df.iloc[train_pos]
        val_df = df.iloc[val_pos]
        test_df = df.iloc[test_pos]

        t_total = len(train_df); t_pos = int((train_df["label"] == 1).sum()); t_neg = t_total - t_pos
        v_total = len(val_df);   v_pos = int((val_df["label"] == 1).sum());   v_neg = v_total - v_pos
        s_total = len(test_df);  s_pos = int((test_df["label"] == 1).sum());  s_neg = s_total - s_pos

        print(f"Train(seq) total={t_total}, anemic={t_pos}, non-anemic={t_neg}")
        print(f"Val  (seq) total={v_total}, anemic={v_pos}, non-anemic={v_neg}")
        print(f"Test (seq) total={s_total}, anemic={s_pos}, non-anemic={s_neg}")
        print(f"📁 Saved to: {split_root}")

        all_counts.append(
            dict(
                source_folder_name=folder_name,
                output_base_name=base_name,
                metadata_column=meta_col,
                threshold=thr,
                train_total=t_total,
                train_anemic=t_pos,
                train_non_anemic=t_neg,
                val_total=v_total,
                val_anemic=v_pos,
                val_non_anemic=v_neg,
                test_total=s_total,
                test_anemic=s_pos,
                test_non_anemic=s_neg,
                output_root=split_root,
            )
        )

# =========================
# 6) WRITE SUMMARY CSV
# =========================
if all_counts:
    summary_df = pd.DataFrame(all_counts)
    summary_csv = os.path.join(OUT_RIGHT_ROOT, "tri_split_summary_all_eyes_all_thresholds_timebased.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n✅ Combined sequential-split summary saved at: {summary_csv}")
else:
    print("\n⚠️ No splits were created (check mapping/suffixes/paths).")
