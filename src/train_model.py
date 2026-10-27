#!/usr/bin/env python3
"""
FDS Case Study 3: Titanic Passenger Survival Prediction Using Logistic Regression
Reproducible Training, Cross-Validation, Evaluation, and Serialization Script (src/train_model.py)
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

RANDOM_STATE = 42
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]
TARGET = "Survived"
CV_SCORING = ("accuracy", "roc_auc", "f1")


def build_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """Constructs a leakage-free scikit-learn preprocessing and Logistic Regression pipeline."""
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

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    return model_pipeline


def locate_dataset() -> str:
    """Locates titanic.csv across standard project paths."""
    candidates = [
        "data/titanic.csv",
        "titanic.csv",
        "../data/titanic.csv",
        "/working_dir/c_78a76925d800ecf0/user_uploaded_files/ttl=63d/user_uploaded/titanic.csv",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Could not find 'titanic.csv'. Place 'titanic.csv' in the 'data/' directory or project root."
    )


def main() -> None:
    print("=== SOFTWARE ENVIRONMENT ===")
    print(f"Python:       {sys.version.split()[0]}")
    print(f"numpy:        {np.__version__}")
    print(f"pandas:       {pd.__version__}")
    print(f"scipy:        {scipy.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"joblib:       {joblib.__version__}\n")

    data_path = locate_dataset()
    print(f"[1/5] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"      Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Separate predictors (X) and target (y)
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Stratified 80/20 train-test split to preserve survival class ratio
    print("[2/5] Performing stratified 80/20 train-test split (random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"      Training set: {X_train.shape[0]} rows (Survived rate: {y_train.mean()*100:.2f}%)")
    print(f"      Testing set:  {X_test.shape[0]} rows (Survived rate: {y_test.mean()*100:.2f}%)")

    # Build and fit pipeline on training data ONLY
    print("[3/5] Running 5-fold StratifiedKFold cross_validate & fitting pipeline on training set...")
    pipeline = build_pipeline(random_state=RANDOM_STATE)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=skf,
        scoring=CV_SCORING,
        return_train_score=False,
    )
    cv_acc = cv_results["test_accuracy"]
    cv_auc = cv_results["test_roc_auc"]
    cv_f1 = cv_results["test_f1"]

    print(f"      CV Accuracy: {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}  (Folds: {np.round(cv_acc, 4)})")
    print(f"      CV ROC-AUC:  {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}  (Folds: {np.round(cv_auc, 4)})")
    print(f"      CV F1-Score: {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}  (Folds: {np.round(cv_f1, 4)})")

    pipeline.fit(X_train, y_train)

    # Evaluate on both training and held-out test data
    print("\n[4/5] Evaluating model on Training Set (n=712) and Held-Out Test Set (n=179)...")
    y_train_pred = pipeline.predict(X_train)
    y_train_prob = pipeline.predict_proba(X_train)[:, 1]
    cm_train = confusion_matrix(y_train, y_train_pred)
    tn_tr, fp_tr, fn_tr, tp_tr = cm_train.ravel()

    y_test_pred = pipeline.predict(X_test)
    y_test_prob = pipeline.predict_proba(X_test)[:, 1]
    cm_test = confusion_matrix(y_test, y_test_pred)
    tn_te, fp_te, fn_te, tp_te = cm_test.ravel()

    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision (Survived = 1)",
                "Recall / Sensitivity (Survived = 1)",
                "Specificity / TNR (Survived = 0)",
                "F1-Score (Survived = 1)",
                "ROC-AUC Score",
            ],
            "Training Set (n=712)": [
                round(accuracy_score(y_train, y_train_pred), 4),
                round(precision_score(y_train, y_train_pred), 4),
                round(recall_score(y_train, y_train_pred), 4),
                round(tn_tr / (tn_tr + fp_tr), 4),
                round(f1_score(y_train, y_train_pred), 4),
                round(roc_auc_score(y_train, y_train_prob), 4),
            ],
            "Held-Out Test Set (n=179)": [
                round(accuracy_score(y_test, y_test_pred), 4),
                round(precision_score(y_test, y_test_pred), 4),
                round(recall_score(y_test, y_test_pred), 4),
                round(tn_te / (tn_te + fp_te), 4),
                round(f1_score(y_test, y_test_pred), 4),
                round(roc_auc_score(y_test, y_test_prob), 4),
            ],
        }
    )
    print(metrics_df.to_string(index=False))

    print(f"\n      Training Confusion Matrix (n=712): TN={tn_tr}, FP={fp_tr}, FN={fn_tr}, TP={tp_tr}")
    print(f"      Testing Confusion Matrix  (n=179): TN={tn_te}, FP={fp_te}, FN={fn_te}, TP={tp_te}")
    print("\n      Held-Out Test Set Classification Report:")
    print(
        classification_report(
            y_test,
            y_test_pred,
            target_names=["Did Not Survive (0)", "Survived (1)"],
            digits=4,
        )
    )

    # Extract coefficients, odds ratios, and Wald statistics
    feature_names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    coefs = pipeline.named_steps["classifier"].coef_.ravel()
    intercept = float(pipeline.named_steps["classifier"].intercept_[0])
    odds_ratios = np.exp(coefs)

    X_tr_trans = pipeline.named_steps["preprocessor"].transform(X_train)
    X_design = np.column_stack([np.ones(len(X_tr_trans)), X_tr_trans])
    W = y_train_prob * (1.0 - y_train_prob)
    reg_mat = np.diag([0.0] + [1.0] * len(coefs))
    cov_mat = np.linalg.inv((X_design.T * W) @ X_design + reg_mat)
    se_all = np.sqrt(np.diag(cov_mat))
    se_coefs = se_all[1:]
    wald_z = coefs / se_coefs
    p_values = 2.0 * (1.0 - stats.norm.cdf(np.abs(wald_z)))

    coef_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": np.round(coefs, 4),
            "Std_Error": np.round(se_coefs, 4),
            "Wald_z": np.round(wald_z, 3),
            "p_value": [f"{p:.4e}" if p < 0.001 else f"{p:.4f}" for p in p_values],
            "Odds_Ratio": np.round(odds_ratios, 4),
        }
    ).sort_values(by="Coefficient", key=abs, ascending=False)
    print(f"\n      Fitted Logistic Regression Coefficients (Intercept beta_0 = {intercept:.4f}, SE = {se_all[0]:.4f}):")
    print(coef_df.to_string(index=False))

    # Save trained pipeline artifact
    print("\n[5/5] Saving trained model artifact...")
    os.makedirs("models", exist_ok=True)
    model_out_paths = [
        "models/titanic_logistic_model.joblib",
        "/working_dir/c_78a76925d800ecf0/artifacts/file_generation/ttl=63d/output/titanic_logistic_model.joblib",
    ]
    for out_path in model_out_paths:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            joblib.dump(pipeline, out_path)
            print(f"      Saved: {out_path}")
        except OSError:
            pass


if __name__ == "__main__":
    main()
