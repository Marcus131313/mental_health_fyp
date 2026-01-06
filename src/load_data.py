import pandas as pd
from pathlib import Path

# Get base project directory (mental_health_fyp)
BASE_DIR = Path(__file__).resolve().parents[1]

# Build the path to your CSV inside the data/ folder
DATA_PATH = BASE_DIR / "data" / "mental_health_clean.csv"

def main():
    print("📂 Loading dataset from:", DATA_PATH)

    # Read CSV into a DataFrame
    df = pd.read_csv(DATA_PATH)

    # Show basic info
    print("✅ Loaded successfully!")
    print("Shape (rows, columns):", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

if __name__ == "__main__":
    main()