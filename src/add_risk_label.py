import pandas as pd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
CLEAN_DATA_PATH = BASE_DIR / "data" / "mental_health_clean.csv"


def classify_risk(row):
    """
    Simple rule-based risk classification.
    Assumes Stress_Level, Depression_Score, Anxiety_Score are on a 0–4 scale.
    You can adjust the thresholds later if needed.
    """
    stress = row["Stress_Level"]
    depression = row["Depression_Score"]
    anxiety = row["Anxiety_Score"]

    scores = [stress, depression, anxiety]

    # High risk: at least one score >= 3
    if any(s >= 3 for s in scores):
        return "High"
    # Moderate risk: no score >= 3, but at least one score == 2
    elif any(s == 2 for s in scores):
        return "Moderate"
    # Low risk: all scores 0 or 1
    else:
        return "Low"


def main():
    print("📂 Loading cleaned dataset from:", CLEAN_DATA_PATH)
    df = pd.read_csv(CLEAN_DATA_PATH)
    print("Current shape:", df.shape)

    # Check required columns exist
    required_cols = ["Stress_Level", "Depression_Score", "Anxiety_Score"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ Required column '{col}' not found in dataset.")

    # Apply classification row by row
    print("🧠 Creating Risk_Level label...")
    df["Risk_Level"] = df.apply(classify_risk, axis=1)

    # Check distribution
    print("\n📊 Risk_Level value counts:")
    print(df["Risk_Level"].value_counts())

    # Save back to same file (overwrite)
    df.to_csv(CLEAN_DATA_PATH, index=False)
    print("\n✅ Saved updated dataset (with Risk_Level) to:", CLEAN_DATA_PATH)
    print("New shape:", df.shape)


if __name__ == "__main__":
    main()