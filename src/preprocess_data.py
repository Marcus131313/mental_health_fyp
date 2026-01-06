import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = BASE_DIR / "data" / "students_mental_health.csv"
CLEAN_DATA_PATH = BASE_DIR / "data" / "mental_health_clean.csv"


def winsorize_series(s: pd.Series, lower=0.01, upper=0.99):
    """
    Simple outlier handling:
    cap values below 1st percentile and above 99th percentile.
    """
    low_val = s.quantile(lower)
    high_val = s.quantile(upper)
    return s.clip(lower=low_val, upper=high_val)


def safe_mode(s: pd.Series):
    """Return the most frequent value in a Series."""
    return s.mode().iloc[0]


def main():
    print("📂 Loading raw dataset from:", RAW_DATA_PATH)
    df = pd.read_csv(RAW_DATA_PATH)
    print("Original shape:", df.shape)

    # ---------------------------------
    # 1. Remove duplicate rows
    # ---------------------------------
    before = df.shape[0]
    df = df.drop_duplicates().reset_index(drop=True)
    after = df.shape[0]
    print(f"🧹 Removed {before - after} duplicate rows. New shape: {df.shape}")

    # ---------------------------------
    # 2. Handle missing values
    # ---------------------------------
    # Numeric columns -> fill with median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    print("\nNumeric columns:", num_cols)
    print("Categorical columns:", cat_cols)

    # Fill numeric
    for col in num_cols:
        missing = df[col].isna().sum()
        if missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"🔧 Filled {missing} missing values in numeric column '{col}' with median {median_val:.2f}.")

    # Fill categorical
    for col in cat_cols:
        missing = df[col].isna().sum()
        if missing > 0:
            mode_val = safe_mode(df[col])
            df[col] = df[col].fillna(mode_val)
            print(f"🔧 Filled {missing} missing values in categorical column '{col}' with mode '{mode_val}'.")

    # ---------------------------------
    # 3. Simple outlier handling
    #    (Age, CGPA, Semester_Credit_Load)
    # ---------------------------------
    for col in ["Age", "CGPA", "Semester_Credit_Load"]:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            before_min, before_max = df[col].min(), df[col].max()
            df[col] = winsorize_series(df[col])
            after_min, after_max = df[col].min(), df[col].max()
            print(
                f"📊 Winsorized {col}: "
                f"[{before_min:.2f}, {before_max:.2f}] -> [{after_min:.2f}, {after_max:.2f}]"
            )
        else:
            print(f"⚠️ Column '{col}' not found or not numeric, skipping winsorization.")

    # ---------------------------------
    # 4. Ordinal encoding for ordered categories
    #    (Sleep_Quality, Physical_Activity, etc.)
    #    Adjust mappings if your actual categories differ.
    # ---------------------------------
    ordinal_mappings = {
        "Sleep_Quality": {"Poor": 1, "Average": 2, "Good": 3},
        "Physical_Activity": {"Low": 1, "Moderate": 2, "High": 3},
        "Diet_Quality": {"Poor": 1, "Average": 2, "Good": 3},
        "Social_Support": {"Low": 1, "Moderate": 2, "High": 3},
        "Extracurricular_Involvement": {"Low": 1, "Moderate": 2, "High": 3},
        "Substance_Use": {"Never": 1, "Occasionally": 2, "Frequently": 3},
        "Counseling_Service_Use": {
            "Never": 1,
            "Rarely": 2,
            "Sometimes": 3,
            "Often": 4,
        },
    }

    for col, mapping in ordinal_mappings.items():
        if col in df.columns:
            new_col = f"{col}_enc"
            df[new_col] = df[col].map(mapping)
            print(f"✅ Encoded '{col}' into '{new_col}'.")
        else:
            print(f"⚠️ Ordinal column '{col}' not found in dataset, skipping.")

    # ---------------------------------
    # 5. Save cleaned data
    # ---------------------------------
    df.to_csv(CLEAN_DATA_PATH, index=False)
    print("\n✅ Saved cleaned dataset to:", CLEAN_DATA_PATH)
    print("Final shape:", df.shape)


if __name__ == "__main__":
    main()