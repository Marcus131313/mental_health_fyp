import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "mental_health_model.pkl"
DATA_PATH = BASE_DIR / "data" / "mental_health_clean.csv"


@st.cache_resource
def train_model(df: pd.DataFrame):
    """
    Train a Logistic Regression pipeline on the cleaned mental health dataset.
    Cached so it only trains once per session on Streamlit Cloud.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    # Separate features and target
    X = df.drop(columns=["Risk_Level"])
    y = df["Risk_Level"]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    clf = LogisticRegression(max_iter=1000)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    # Train/test split (for training only – we don’t use X_test here)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    return model


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def main():
    st.title("Mental Health Risk Prediction – University Students")

    st.markdown(
        """
This is a **prototype decision-support dashboard** for predicting student mental health risk  
using survey data.  

⚠️ **Disclaimer:** This is *not* a medical diagnosis tool.  
It is only for academic and decision-support purposes.
"""
    )

    # Load data then train model on the server
    df = load_data()
    model = train_model(df)

    # -----------------------------
    # Overview section
    # -----------------------------
    st.header("📊 Dataset Overview")

    st.write("Number of records:", df.shape[0])
    st.write("Number of columns:", df.shape[1])

    st.write("Preview of data:")
    st.dataframe(df.head())

    st.write("Risk_Level distribution:")
    st.bar_chart(df["Risk_Level"].value_counts())

    # -----------------------------
    # Prediction section
    # -----------------------------
    st.header("🧠 Try a Risk Prediction")

    # We'll build a "typical" row first (median/mode), then override with user inputs.
    X = df.drop(columns=["Risk_Level"])
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Start with default values
    default_row = {}

    for col in numeric_cols:
        default_row[col] = float(df[col].median())

    for col in categorical_cols:
        default_row[col] = df[col].mode().iloc[0]

    # -----------------------------
    # Sidebar preset profiles
    # -----------------------------
    st.sidebar.header("🎛️ Quick Profiles")

    profile = st.sidebar.selectbox(
        "Choose a preset",
        ["Custom", "Balanced student", "High stress & poor sleep", "Financially stressed"],
    )

    # start from data-based defaults
    stress_default = int(default_row.get("Stress_Level", 1))
    depression_default = int(default_row.get("Depression_Score", 1))
    anxiety_default = int(default_row.get("Anxiety_Score", 1))
    financial_default = int(default_row.get("Financial_Stress", 1))

    if profile == "Balanced student":
        stress_default = 1
        depression_default = 1
        anxiety_default = 1
        financial_default = 1
    elif profile == "High stress & poor sleep":
        stress_default = 4
        depression_default = 3
        anxiety_default = 4
        financial_default = 2
    elif profile == "Financially stressed":
        stress_default = 3
        depression_default = 2
        anxiety_default = 3
        financial_default = 4
    # "Custom" → keep data-based defaults

    # -----------------------------
    # User input widgets
    # -----------------------------
    st.subheader("Basic Information")

    age = st.number_input("Age", min_value=16, max_value=40, value=int(default_row.get("Age", 21)))
    cgpa = st.number_input(
        "CGPA",
        min_value=0.0,
        max_value=4.0,
        value=float(round(default_row.get("CGPA", 3.0), 2)),
        step=0.01,
    )

    stress = st.slider("Stress_Level (0–4)", 0, 4, stress_default)
    depression = st.slider("Depression_Score (0–4)", 0, 4, depression_default)
    anxiety = st.slider("Anxiety_Score (0–4)", 0, 4, anxiety_default)
    financial_stress = st.slider("Financial_Stress (0–4)", 0, 4, financial_default)

    st.subheader("Lifestyle & Support")

    sleep_quality = st.selectbox(
        "Sleep_Quality",
        ["Poor", "Average", "Good"],
        index=["Poor", "Average", "Good"].index(default_row.get("Sleep_Quality", "Average")),
    )
    physical_activity = st.selectbox(
        "Physical_Activity",
        ["Low", "Moderate", "High"],
        index=["Low", "Moderate", "High"].index(default_row.get("Physical_Activity", "Moderate")),
    )
    diet_quality = st.selectbox(
        "Diet_Quality",
        ["Poor", "Average", "Good"],
        index=["Poor", "Average", "Good"].index(default_row.get("Diet_Quality", "Average")),
    )
    social_support = st.selectbox(
        "Social_Support",
        ["Low", "Moderate", "High"],
        index=["Low", "Moderate", "High"].index(default_row.get("Social_Support", "Moderate")),
    )

    relationship_status = st.selectbox(
        "Relationship_Status",
        df["Relationship_Status"].dropna().unique().tolist(),
        index=0,
    )

    course = st.selectbox(
        "Course",
        df["Course"].dropna().unique().tolist(),
        index=0,
    )

    residence_type = st.selectbox(
        "Residence_Type",
        df["Residence_Type"].dropna().unique().tolist(),
        index=0,
    )

    family_history = st.selectbox(
        "Family_History",
        df["Family_History"].dropna().unique().tolist(),
        index=0,
    )

    chronic_illness = st.selectbox(
        "Chronic_Illness",
        df["Chronic_Illness"].dropna().unique().tolist(),
        index=0,
    )

    extracurricular = st.selectbox(
        "Extracurricular_Involvement",
        df["Extracurricular_Involvement"].dropna().unique().tolist(),
        index=0,
    )

    substance_use = st.selectbox(
        "Substance_Use",
        df["Substance_Use"].dropna().unique().tolist(),
        index=0,
    )

    counseling_use = st.selectbox(
        "Counseling_Service_Use",
        df["Counseling_Service_Use"].dropna().unique().tolist(),
        index=0,
    )

    # -----------------------------
    # Build the final input row
    # -----------------------------
    input_row = default_row.copy()

    input_row.update(
        {
            "Age": age,
            "CGPA": cgpa,
            "Stress_Level": stress,
            "Depression_Score": depression,
            "Anxiety_Score": anxiety,
            "Financial_Stress": financial_stress,
            "Sleep_Quality": sleep_quality,
            "Physical_Activity": physical_activity,
            "Diet_Quality": diet_quality,
            "Social_Support": social_support,
            "Relationship_Status": relationship_status,
            "Course": course,
            "Residence_Type": residence_type,
            "Family_History": family_history,
            "Chronic_Illness": chronic_illness,
            "Extracurricular_Involvement": extracurricular,
            "Substance_Use": substance_use,
            "Counseling_Service_Use": counseling_use,
        }
    )

    # Make sure all expected feature columns exist
    input_df = pd.DataFrame([input_row], columns=X.columns)

    st.subheader("🔍 Input Summary")
    st.write(input_df)

    # -----------------------------
    # Prediction & recommendations
    # -----------------------------
    if st.button("Predict Risk Level"):
        pred = model.predict(input_df)[0]

        # Emoji + short description + suggestions
        if pred == "Low":
            emoji = "🟢"
            desc = "Low mental health risk based on the current inputs."
            bullets = [
                "Maintain healthy routines (sleep, exercise, socialising).",
                "Continue promoting awareness of mental health resources.",
                "Encourage peer support and regular check-ins among students.",
            ]
        elif pred == "Moderate":
            emoji = "🟠"
            desc = "Moderate risk – some patterns suggest stress or emotional strain."
            bullets = [
                "Monitor stress and academic workload for this group of students.",
                "Share information about counselling and support services more actively.",
                "Encourage time management, breaks, and self-care activities.",
            ]
        else:  # High
            emoji = "🔴"
            desc = "High risk – patterns suggest significant stress or distress."
            bullets = [
                "Flag this risk group for closer attention at an aggregate level (not individual targeting).",
                "Consider proactive outreach from counsellors, mentors, or support units.",
                "Review academic load, financial assistance, and available support programmes.",
            ]

        st.subheader("🧠 Predicted Mental Health Risk")
        st.markdown(f"### {emoji} {pred} Risk")
        st.write(desc)
        st.caption("Reminder: this is a prototype decision-support tool, not a diagnosis.")

        st.subheader("📚 Support & Action Suggestions")
        for item in bullets:
            st.markdown(f"- {item}")

    st.markdown("---")
    st.caption(
        "Mental Health Risk Prediction Prototype | FYP 2025 | "
        "For academic and decision-support purposes only."
    )


if __name__ == "__main__":
    main()