import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.impute import SimpleImputer
import joblib

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "mental_health_clean.csv"

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "mental_health_model.pkl"

def main():
    print("📂 Loading cleaned dataset from:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    print("Shape:", df.shape)

        # Make sure models/ folder exists
    MODELS_DIR.mkdir(exist_ok=True)

    # -------------------------
    # 1. Separate features (X) and target (y)
    # -------------------------
    TARGET_COL = "Risk_Level"

    if TARGET_COL not in df.columns:
        raise ValueError(f"❌ '{TARGET_COL}' column not found. Did you run add_risk_label.py?")

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    print("\n✅ Separated X (features) and y (Risk_Level)")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # -------------------------
    # 2. Identify numeric and categorical columns
    # -------------------------
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    print("\nNumeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    # -------------------------
    # 3. Preprocessor:
    #    - scale numeric
    #    - one-hot encode categorical
    # -------------------------
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # -------------------------
    # 4. Define model (Logistic Regression)
    # -------------------------
    clf = LogisticRegression(max_iter=1000, multi_class="auto")

    # Full pipeline = preprocessing + model
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    # -------------------------
    # 5. Train/test split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,        # 20% test, 80% train
        random_state=42,
        stratify=y,           # keep class balance
    )

    print("\n🔀 Split data:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test :", X_test.shape, "y_test :", y_test.shape)

    # -------------------------
    # 6. Train the model
    # -------------------------
    print("\n🚀 Training Logistic Regression model...")
    model.fit(X_train, y_train)
    print("✅ Training complete.")

    # -------------------------
    # 7. Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
    # -------------------------
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n📊 Evaluation on test set:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")

    # Per-class metrics
    print("\n📋 Classification report (per class):")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix
    print("🧩 Confusion Matrix (rows = true, columns = predicted):")
    print(confusion_matrix(y_test, y_pred))

    # -------------------------
    # 8. Save the trained model pipeline
    # -------------------------
    joblib.dump(model, MODEL_PATH)
    print(f"\n💾 Saved trained model pipeline to: {MODEL_PATH}")

if __name__ == "__main__":
    main()