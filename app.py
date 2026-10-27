#!/usr/bin/env python3
"""
FDS Case Study 3: Titanic Passenger Survival Prediction Using Logistic Regression
Streamlit Web Application (app.py)

Run locally with:
    streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]
TARGET = "Survived"
RANDOM_STATE = 42


def train_fallback_pipeline() -> Pipeline:
    """Trains the exact scikit-learn pipeline from titanic.csv if the joblib artifact is missing or from a different sklearn version."""
    candidates = [
        "data/titanic.csv",
        "titanic.csv",
        "../data/titanic.csv",
    ]
    data_path = next((p for p in candidates if os.path.isfile(p)), None)
    if data_path is None:
        raise FileNotFoundError(
            "Model artifact and 'titanic.csv' were not found. Please place 'titanic_logistic_model.joblib' in 'models/' or 'titanic.csv' in 'data/'."
        )
    df = pd.read_csv(data_path)
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/titanic_logistic_model.joblib")
    return pipeline


@st.cache_resource
def load_trained_model() -> Pipeline:
    """Loads the saved scikit-learn Pipeline or retrains cleanly if cross-environment version mismatch occurs."""
    model_paths = [
        "models/titanic_logistic_model.joblib",
        "titanic_logistic_model.joblib",
    ]
    for path in model_paths:
        if os.path.isfile(path):
            try:
                model = joblib.load(path)
                # Verify compatibility with a 1-row validation check
                sample_df = pd.DataFrame(
                    [
                        {
                            "Pclass": 3,
                            "Sex": "male",
                            "Age": 22.0,
                            "SibSp": 1,
                            "Parch": 0,
                            "Fare": 7.25,
                            "Embarked": "S",
                        }
                    ]
                )
                _ = model.predict_proba(sample_df)
                return model
            except Exception:
                break
    return train_fallback_pipeline()


def validate_inputs(
    pclass: int,
    sex: str,
    age: float,
    sibsp: int,
    parch: int,
    fare: float,
    embarked: str,
) -> list[str]:
    """Validates passenger input values before running model inference."""
    errors = []
    if pclass not in (1, 2, 3):
        errors.append("Passenger Class (Pclass) must be 1, 2, or 3.")
    if sex not in ("female", "male"):
        errors.append("Sex must be 'female' or 'male'.")
    if not (0.1 <= age <= 100.0):
        errors.append("Age must be between 0.1 and 100.0 years.")
    if not (0 <= sibsp <= 10):
        errors.append("Siblings/Spouses Aboard (SibSp) must be between 0 and 10.")
    if not (0 <= parch <= 10):
        errors.append("Parents/Children Aboard (Parch) must be between 0 and 10.")
    if not (0.0 <= fare <= 600.0):
        errors.append("Ticket Fare must be between £0.00 and £600.00.")
    if embarked not in ("C", "Q", "S"):
        errors.append("Port of Embarkation must be 'C', 'Q', or 'S'.")
    return errors


def predict_passenger_survival(
    model: Pipeline,
    pclass: int,
    sex: str,
    age: float,
    sibsp: int,
    parch: int,
    fare: float,
    embarked: str,
) -> dict:
    """Runs preprocessing and logistic regression inference for a single passenger record."""
    input_df = pd.DataFrame(
        [
            {
                "Pclass": int(pclass),
                "Sex": str(sex),
                "Age": float(age),
                "SibSp": int(sibsp),
                "Parch": int(parch),
                "Fare": float(fare),
                "Embarked": str(embarked),
            }
        ]
    )
    probs = model.predict_proba(input_df)[0]
    pred_class = int(model.predict(input_df)[0])
    return {
        "input_df": input_df,
        "predicted_class": pred_class,
        "survival_probability": float(probs[1]),
        "non_survival_probability": float(probs[0]),
    }


def main() -> None:
    st.set_page_config(
        page_title="Titanic Survival Predictor | FDS Case Study 3",
        page_icon="🚢",
        layout="wide",
    )

    st.title("🚢 Titanic Passenger Survival Prediction")
    st.markdown(
        "**Fundamentals of Data Science (FDS) — Case Study 3: Logistic Regression Classifier**  \n"
        "Estimate the probability that a Titanic passenger survived using a leakage-free `scikit-learn` "
        "`Pipeline` (`ColumnTransformer` + L2-regularized `LogisticRegression`)."
    )

    try:
        model = load_trained_model()
    except Exception as exc:
        st.error(f"Error loading trained model pipeline: {exc}")
        st.stop()

    col_form, col_result = st.columns([1.15, 1.0], gap="large")

    with col_form:
        st.subheader("Passenger Profile Inputs")
        with st.form("passenger_prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                pclass = st.selectbox(
                    "Ticket Class (Pclass)",
                    options=[1, 2, 3],
                    index=2,
                    format_func=lambda x: {
                        1: "1st Class (Upper Deck)",
                        2: "2nd Class (Middle Deck)",
                        3: "3rd Class (Lower Deck)",
                    }[x],
                )
                sex = st.selectbox(
                    "Sex",
                    options=["female", "male"],
                    index=0,
                    format_func=lambda x: "Female" if x == "female" else "Male",
                )
                age = st.number_input(
                    "Age (Years)",
                    min_value=0.1,
                    max_value=100.0,
                    value=28.0,
                    step=1.0,
                    help="Dataset range: 0.42 to 80.0 years (Training median: 28.5 years).",
                )
                embarked = st.selectbox(
                    "Port of Embarkation",
                    options=["S", "C", "Q"],
                    index=0,
                    format_func=lambda x: {
                        "S": "Southampton (S)",
                        "C": "Cherbourg (C)",
                        "Q": "Queenstown (Q)",
                    }[x],
                )
            with c2:
                sibsp = st.number_input(
                    "Siblings / Spouses Aboard (SibSp)",
                    min_value=0,
                    max_value=10,
                    value=0,
                    step=1,
                )
                parch = st.number_input(
                    "Parents / Children Aboard (Parch)",
                    min_value=0,
                    max_value=10,
                    value=0,
                    step=1,
                )
                fare = st.number_input(
                    "Ticket Fare (£)",
                    min_value=0.0,
                    max_value=600.0,
                    value=14.45,
                    step=1.0,
                    help="Dataset range: £0.00 to £512.33 (Training median: £14.45).",
                )

            submitted = st.form_submit_button("Predict Survival Probability", use_container_width=True)

    with col_result:
        st.subheader("Prediction Result & Model Interpretation")
        errors = validate_inputs(pclass, sex, age, sibsp, parch, fare, embarked)
        if errors:
            for err in errors:
                st.error(err)
        else:
            try:
                res = predict_passenger_survival(
                    model, pclass, sex, age, sibsp, parch, fare, embarked
                )
                surv_prob = res["survival_probability"]
                non_surv_prob = res["non_survival_probability"]
                pred_class = res["predicted_class"]

                if pred_class == 1:
                    st.success(
                        f"**Predicted Class: SURVIVED (1)**  \n"
                        f"Estimated Survival Probability: **{surv_prob * 100:.2f}%**"
                    )
                else:
                    st.warning(
                        f"**Predicted Class: DID NOT SURVIVE (0)**  \n"
                        f"Estimated Survival Probability: **{surv_prob * 100:.2f}%**"
                    )

                m1, m2 = st.columns(2)
                m1.metric("Survival Probability P(Y=1|X)", f"{surv_prob * 100:.2f}%")
                m2.metric("Non-Survival Probability P(Y=0|X)", f"{non_surv_prob * 100:.2f}%")

                st.progress(min(max(surv_prob, 0.0), 1.0))
                st.caption(
                    "Classification decision boundary: P(Survived = 1 | X) >= 0.50 -> Survived (1); otherwise Did Not Survive (0)."
                )

                st.markdown("#### Input Feature Summary Passed to Pipeline")
                st.dataframe(res["input_df"], use_container_width=True, hide_index=True)

            except Exception as pred_err:
                st.error(f"Prediction failed: {pred_err}")

    st.divider()
    with st.expander("📊 Held-Out Test Set Performance & Fitted Model Coefficients"):
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown(
                """
                **5-Fold Stratified Cross-Validation (Training Set, n = 712):**
                - **CV Accuracy:** 79.49% ± 2.73%
                - **CV ROC-AUC:** 0.8549 ± 0.0212
                - **CV F1-Score:** 0.7246 ± 0.0335

                **Held-Out Test Set Evaluation (n = 179, Stratified 80/20 Split):**
                - **Accuracy:** 80.45% (144 / 179) | *Train: 80.76%*
                - **Precision (Survived = 1):** 79.31% | *Train: 76.98%*
                - **Recall / Sensitivity (Survived = 1):** 66.67% | *Train: 71.06%*
                - **Specificity (Survived = 0):** 89.09% | *Train: 86.79%*
                - **F1-Score (Survived = 1):** 0.7244 | *Train: 0.7390*
                - **ROC-AUC Score:** 0.8433 | *Train: 0.8594*
                - **Test Confusion Matrix:** TN = 98, FP = 12, FN = 23, TP = 46
                """
            )
        with c_right:
            feature_names = list(model.named_steps["preprocessor"].get_feature_names_out())
            coefs = model.named_steps["classifier"].coef_.ravel()
            coef_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Log-Odds Coefficient (β)": np.round(coefs, 4),
                    "Odds Ratio (e^β)": np.round(np.exp(coefs), 4),
                }
            ).sort_values(by="Log-Odds Coefficient (β)", key=abs, ascending=False)
            st.dataframe(coef_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
