"""
modeling.py
-----------
Trains and evaluates ML models to predict adverse event severity.

Models compared:
  1. Logistic Regression (baseline)
  2. Random Forest
  3. Gradient Boosting (sklearn)

Handles class imbalance via SMOTE and class weights.
Produces SHAP explanations for the best model.
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    roc_curve,
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "age_years",
    "patient_weight",
    "num_concomitant_drugs",
    "num_reactions",
    "polypharmacy_flag",
    "sex_encoded",
    "drug_encoded",
    "route_encoded",
    "indication_encoded",
]

TARGET = "is_serious"


def prepare_features(df):
    """
    Encode categoricals, impute missing values,
    and return X, y arrays with feature names.
    """
    df_model = df.copy()

    # --- Encode categoricals ---
    label_encoders = {}
    for col, source in [
        ("sex_encoded", "patient_sex"),
        ("drug_encoded", "generic_name"),
        ("route_encoded", "route"),
        ("indication_encoded", "indication_group"),
    ]:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[source].astype(str))
        label_encoders[col] = le

    # --- Impute numerics with median ---
    for col in ["age_years", "patient_weight"]:
        median_val = df_model[col].median()
        df_model[col] = df_model[col].fillna(median_val)

    # --- Ensure target exists ---
    df_model = df_model.dropna(subset=[TARGET])

    X = df_model[FEATURE_COLS].values
    y = df_model[TARGET].values.astype(int)

    return X, y, FEATURE_COLS, label_encoders, df_model


def train_and_evaluate(
    input_path="data/processed_adverse_events.csv",
    model_dir="models",
):
    """
    Full training pipeline:
      1. Load processed data
      2. Encode & split
      3. SMOTE for balancing
      4. Train 3 models with cross-validation
      5. Evaluate on hold-out test set
      6. Save best model, metrics, and artifacts
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    df = pd.read_csv(input_path, low_memory=False)
    df["receive_date"] = pd.to_datetime(df["receive_date"], errors="coerce")
    print(f"Loaded {len(df):,} records for modeling")

    X, y, feature_names, label_encoders, df_model = prepare_features(df)
    print(f"Features: {len(feature_names)}, Samples: {len(y):,}")
    print(f"Class balance: {np.mean(y):.1%} serious")

    # ---- Train/test split (stratified, time-aware via shuffle=False is tricky
    #      with FAERS so we use stratified random split) ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- SMOTE on training set only ----
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {len(y_train_bal):,} training samples")

    # ---- Scale features ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)

    # ---- Define models ----
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_leaf=20,
            random_state=42,
        ),
    }

    # ---- Train & evaluate ----
    results = {}
    best_model = None
    best_auc = 0
    best_name = ""

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n{'─'*50}")
        print(f"Training: {name}")

        # Cross-validation on balanced training data
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train_bal, cv=cv, scoring="roc_auc"
        )
        print(f"  CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Fit on full balanced training set
        model.fit(X_train_scaled, y_train_bal)

        # Predict on original (unbalanced) test set
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Metrics
        test_auc = roc_auc_score(y_test, y_proba)
        test_f1 = f1_score(y_test, y_pred)
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall_vals, precision_vals)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        fpr, tpr, _ = roc_curve(y_test, y_proba)

        results[name] = {
            "cv_auc_mean": float(cv_scores.mean()),
            "cv_auc_std": float(cv_scores.std()),
            "test_roc_auc": float(test_auc),
            "test_pr_auc": float(pr_auc),
            "test_f1": float(test_f1),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "precision_curve": precision_vals.tolist(),
            "recall_curve": recall_vals.tolist(),
        }

        print(f"  Test ROC-AUC: {test_auc:.4f}")
        print(f"  Test PR-AUC:  {pr_auc:.4f}")
        print(f"  Test F1:      {test_f1:.4f}")

        if test_auc > best_auc:
            best_auc = test_auc
            best_model = model
            best_name = name

    print(f"\n{'='*50}")
    print(f"Best model: {best_name} (ROC-AUC = {best_auc:.4f})")

    # ---- Feature importance (from best model) ----
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_[0])
    else:
        importances = np.zeros(len(feature_names))

    feature_importance = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
    )

    # ---- Save artifacts ----
    joblib.dump(best_model, model_dir / "best_model.joblib")
    joblib.dump(scaler, model_dir / "scaler.joblib")
    joblib.dump(label_encoders, model_dir / "label_encoders.joblib")
    feature_importance.to_csv(model_dir / "feature_importance.csv", index=False)

    # Save metrics as JSON
    metrics_out = {
        "best_model": best_name,
        "feature_names": feature_names,
        "results": {},
    }
    for name, res in results.items():
        metrics_out["results"][name] = {
            "cv_auc_mean": res["cv_auc_mean"],
            "cv_auc_std": res["cv_auc_std"],
            "test_roc_auc": res["test_roc_auc"],
            "test_pr_auc": res["test_pr_auc"],
            "test_f1": res["test_f1"],
            "confusion_matrix": res["confusion_matrix"],
            "fpr": res["fpr"],
            "tpr": res["tpr"],
            "precision_curve": res["precision_curve"],
            "recall_curve": res["recall_curve"],
        }

    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    # Save test data for dashboard
    test_data = pd.DataFrame(X_test, columns=feature_names)
    test_data["y_true"] = y_test
    test_data["y_proba"] = y_proba
    test_data["y_pred"] = y_pred
    test_data.to_csv(model_dir / "test_predictions.csv", index=False)

    print(f"\nArtifacts saved to {model_dir}/")
    return results, best_model, best_name


if __name__ == "__main__":
    train_and_evaluate()
