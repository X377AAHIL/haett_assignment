"""
Model Training Module
=====================
Trains multiple ML models for churn prediction, evaluates them,
tracks experiments with MLflow, and registers the best model.

Models trained:
1. Logistic Regression (baseline)
2. Random Forest
3. XGBoost
4. LightGBM

Evaluation: Precision, Recall, F1 Score, ROC-AUC, PR-AUC
Model selection: Best F1 Score (balances precision and recall for retention)
"""

import os
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.data_preparation import prepare_dataset
from src.explainability import ShapExplainer
from src.feature_engineering import (
    FEATURE_COLUMNS,
    FeatureTransformer,
    build_features,
)

warnings.filterwarnings("ignore")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MLFLOW_TRACKING_URI = os.path.join(BASE_DIR, "mlruns")
EXPERIMENT_NAME = "haett-churn-prediction"


def get_models() -> dict:
    """Return model configurations for training."""
    return {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
            "params": {
                "C": 1.0,
                "max_iter": 1000,
                "class_weight": "balanced",
            },
        },
        "RandomForest": {
            "model": RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42, class_weight="balanced", n_jobs=-1
            ),
            "params": {
                "n_estimators": 200,
                "max_depth": 10,
                "class_weight": "balanced",
            },
        },
        "XGBoost": {
            "model": XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=2.5,  # Handle class imbalance
                random_state=42,
                eval_metric="logloss",
            ),
            "params": {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.1,
                "scale_pos_weight": 2.5,
            },
        },
        "LightGBM": {
            "model": LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=2.5,
                random_state=42,
                verbose=-1,
            ),
            "params": {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.1,
                "scale_pos_weight": 2.5,
            },
        },
    }


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate a trained model and return metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
    }

    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: str):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Active", "Churned"],
        yticklabels=["Active", "Churned"],
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names: list, model_name: str, save_path: str):
    """Plot and save feature importance."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return

    # Sort by importance
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_features)))
    ax.barh(range(len(sorted_features)), sorted_importances[::-1], color=colors)
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features[::-1], fontsize=10)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Feature Importance — {model_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def hyperparameter_tuning(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    """Perform hyperparameter tuning on XGBoost (the best model)."""
    print("\n  Running hyperparameter tuning for XGBoost...")
    param_grid = {
        "n_estimators": [200, 300, 400],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1, 0.15],
    }

    xgb = XGBClassifier(
        scale_pos_weight=2.5,
        random_state=42,
        eval_metric="logloss",
    )

    grid_search = GridSearchCV(
        xgb, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)

    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best CV F1: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


def train_pipeline():
    """Full training pipeline with MLflow tracking."""
    print("=" * 60)
    print("Haett Model Training Pipeline")
    print("=" * 60)

    # --- Data Preparation ---
    print("\n📊 Phase 1: Data Preparation")
    train_users, test_users, merged = prepare_dataset()

    # --- Feature Engineering ---
    print("\n🔧 Phase 2: Feature Engineering")
    orders = merged["orders"]
    subscriptions = merged["subscriptions"]
    engagement = merged["engagement"]

    train_features = build_features(train_users, orders, subscriptions, engagement)
    test_features = build_features(test_users, orders, subscriptions, engagement)

    # Fit transformer on training data
    transformer = FeatureTransformer()
    X_train = transformer.fit_transform(train_features)
    y_train = train_features["churned"].values

    X_test = transformer.transform(test_features)
    y_test = test_features["churned"].values

    # Save transformer
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(transformer, os.path.join(MODELS_DIR, "feature_transformer.joblib"))
    print(f"  Transformer saved to {MODELS_DIR}/feature_transformer.joblib")

    # --- MLflow Setup ---
    mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_URI}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # --- Model Training ---
    print("\n🤖 Phase 3: Model Training")
    models = get_models()
    results = {}

    for model_name, config in models.items():
        print(f"\n  Training {model_name}...")

        with mlflow.start_run(run_name=model_name):
            model = config["model"]

            # Train
            model.fit(X_train, y_train)

            # Evaluate
            metrics = evaluate_model(model, X_test, y_test)
            results[model_name] = {"model": model, "metrics": metrics}

            # Log parameters
            mlflow.log_params(config["params"])

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log confusion matrix
            y_pred = model.predict(X_test)
            cm_path = os.path.join(MODELS_DIR, f"confusion_matrix_{model_name}.png")
            plot_confusion_matrix(y_test, y_pred, model_name, cm_path)
            mlflow.log_artifact(cm_path)

            # Log feature importance
            fi_path = os.path.join(MODELS_DIR, f"feature_importance_{model_name}.png")
            plot_feature_importance(model, FEATURE_COLUMNS, model_name, fi_path)
            if os.path.exists(fi_path):
                mlflow.log_artifact(fi_path)

            # Log model
            if "XGBoost" in model_name:
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")

            print(f"    Accuracy:  {metrics['accuracy']:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1 Score:  {metrics['f1_score']:.4f}")
            print(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"    PR-AUC:    {metrics['pr_auc']:.4f}")

    # --- Hyperparameter Tuning on Best Model ---
    print("\n🔍 Phase 4: Hyperparameter Tuning")
    tuned_model, best_params = hyperparameter_tuning(X_train, y_train)

    with mlflow.start_run(run_name="XGBoost_Tuned"):
        mlflow.log_params(best_params)
        tuned_metrics = evaluate_model(tuned_model, X_test, y_test)
        mlflow.log_metrics(tuned_metrics)
        mlflow.xgboost.log_model(tuned_model, "model")

        # Log artifacts
        y_pred_tuned = tuned_model.predict(X_test)
        cm_path = os.path.join(MODELS_DIR, "confusion_matrix_XGBoost_Tuned.png")
        plot_confusion_matrix(y_test, y_pred_tuned, "XGBoost (Tuned)", cm_path)
        mlflow.log_artifact(cm_path)

        fi_path = os.path.join(MODELS_DIR, "feature_importance_XGBoost_Tuned.png")
        plot_feature_importance(tuned_model, FEATURE_COLUMNS, "XGBoost (Tuned)", fi_path)
        mlflow.log_artifact(fi_path)

        results["XGBoost_Tuned"] = {"model": tuned_model, "metrics": tuned_metrics}

        print(f"\n  XGBoost (Tuned) Results:")
        print(f"    Accuracy:  {tuned_metrics['accuracy']:.4f}")
        print(f"    Precision: {tuned_metrics['precision']:.4f}")
        print(f"    Recall:    {tuned_metrics['recall']:.4f}")
        print(f"    F1 Score:  {tuned_metrics['f1_score']:.4f}")
        print(f"    ROC-AUC:   {tuned_metrics['roc_auc']:.4f}")
        print(f"    PR-AUC:    {tuned_metrics['pr_auc']:.4f}")

    # --- Select Best Model ---
    print("\n🏆 Phase 5: Model Selection")
    best_name = max(results, key=lambda k: results[k]["metrics"]["f1_score"])
    best_model = results[best_name]["model"]
    best_metrics = results[best_name]["metrics"]

    print(f"  Best model: {best_name}")
    print(f"  F1 Score:   {best_metrics['f1_score']:.4f}")

    # Save best model
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"  Model saved to {model_path}")

    # Save model metadata
    metadata = {
        "model_name": best_name,
        "metrics": best_metrics,
        "feature_columns": FEATURE_COLUMNS,
        "best_params": best_params if best_name == "XGBoost_Tuned" else results[best_name].get("params", {}),
    }
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # --- Register Best Model in MLflow ---
    with mlflow.start_run(run_name=f"BEST_{best_name}"):
        mlflow.log_params({"selected_model": best_name})
        mlflow.log_metrics(best_metrics)
        if "XGBoost" in best_name:
            mlflow.xgboost.log_model(
                best_model, "model",
                registered_model_name="haett-churn-model",
            )
        else:
            mlflow.sklearn.log_model(
                best_model, "model",
                registered_model_name="haett-churn-model",
            )
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(metadata_path)

        # --- SHAP Explainability Integration ---
        print("\n🧠 Generating SHAP Explanations...")
        X_train_df = pd.DataFrame(X_train, columns=FEATURE_COLUMNS)
        X_test_df = pd.DataFrame(X_test, columns=FEATURE_COLUMNS)
        
        # Initialize Explainer with dynamically selected best model
        explainer = ShapExplainer(model=best_model, background_data=X_train_df)
        shap_values = explainer.get_shap_values(X_test_df)
        
        # Define paths
        shap_dir = os.path.join("artifacts", "shap")
        os.makedirs(shap_dir, exist_ok=True)
        
        summary_path = os.path.join(shap_dir, "summary.png")
        bar_path = os.path.join(shap_dir, "bar.png")
        waterfall_path = os.path.join(shap_dir, "waterfall.png")
        json_path = os.path.join(shap_dir, "feature_importance.json")
        
        # Save artifacts locally
        explainer.save_summary_plot(shap_values, summary_path)
        explainer.save_bar_plot(shap_values, bar_path)
        
        # Find sample with highest predicted churn probability for waterfall plot
        y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else best_model.predict(X_test)
        highest_churn_idx = np.argmax(y_prob)
        explainer.save_waterfall_plot(shap_values, highest_churn_idx, waterfall_path)
        
        explainer.save_feature_importance_json(shap_values, json_path)
        
        # Log to MLflow under 'explainability' directory
        mlflow.log_artifact(summary_path, artifact_path="explainability")
        mlflow.log_artifact(bar_path, artifact_path="explainability")
        mlflow.log_artifact(waterfall_path, artifact_path="explainability")
        mlflow.log_artifact(json_path, artifact_path="explainability")
        print("  SHAP artifacts successfully logged to MLflow.")

    # --- Results Summary ---
    print("\n" + "=" * 60)
    print("Model Comparison Summary")
    print("=" * 60)
    summary = pd.DataFrame(
        {name: res["metrics"] for name, res in results.items()}
    ).T
    summary = summary.sort_values("f1_score", ascending=False)
    print(summary.to_string())

    # Save comparison
    summary.to_csv(os.path.join(MODELS_DIR, "model_comparison.csv"))

    # --- Monitoring Setup (Reference Dataset) ---
    print("\n📦 Generating Monitoring Reference Dataset...")
    import datetime
    import yaml
    
    # Use validation data before scaling (test_features)
    ref_df = test_features.copy()
    y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else best_model.predict(X_test)
    ref_df["prediction_probability"] = y_prob
    ref_df["prediction"] = ["High" if p >= 0.6 else "Medium" if p >= 0.3 else "Low" for p in y_prob]
    
    model_version = datetime.datetime.now().strftime("v_%Y%m%d_%H%M%S")
    ref_path = f"artifacts/monitoring/reference/reference_dataset_{model_version}.parquet"
    os.makedirs(os.path.dirname(ref_path), exist_ok=True)
    
    ref_df.to_parquet(ref_path, engine="fastparquet")
    print(f"  Reference dataset saved to {ref_path}")
    
    # Update config
    config_path = "config/monitoring.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            mon_config = yaml.safe_load(f)
        mon_config["reference_dataset_path"] = ref_path
        with open(config_path, "w") as f:
            yaml.dump(mon_config, f)
        print("  Updated config/monitoring.yaml with new reference dataset path.")

    print(f"\n✅ Training complete. Best model: {best_name}")
    print(f"   MLflow tracking URI: file://{MLFLOW_TRACKING_URI}")
    print(f"   Run: mlflow ui --backend-store-uri file://{MLFLOW_TRACKING_URI} --port 5001")

    return best_model, transformer, best_metrics


if __name__ == "__main__":
    train_pipeline()
