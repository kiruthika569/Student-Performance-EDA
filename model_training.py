"""
Student Performance Analysis - Model Training & Evaluation Study
================================================================
A comprehensive, beginner-friendly machine learning and deep learning
investigation to predict student 'Pass_Fail' status while strictly preventing target leakage.

Models Evaluated:
1. Baseline: Dummy Classifier (Most Frequent class prediction)
2. Linear Model: Logistic Regression (Balanced class weights)
3. Tree Ensemble: Random Forest Classifier (Tuned on validation set, balanced class weights)
4. Deep Learning: Multilayer Perceptron (MLP) with TensorFlow/Keras & EarlyStopping

Key Enhancements:
- Outlier handling via training-fitted IQR capping (no data leakage).
- Random Forest hyperparameter tuning evaluated exclusively on validation set.
- Feature importance extraction and ranking from Random Forest.
- ROC Curve & ROC-AUC evaluation for both MLP and Random Forest.
- Artifact persistence in 'models/' and visualizations in 'visualizations/'.
"""

import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# Paths and configuration
DATASET_PATH = "Student_Performance_Dataset.csv"
MODELS_DIR = "models"
VIS_DIR = "visualizations"
RANDOM_SEED = 42

# Ensure reproducibility
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def print_header(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def cap_outliers_iqr(X_train, X_val, X_test, numerical_cols):
    """
    Fits IQR bounds exclusively on training data to prevent leakage,
    and applies capping (winsorization) to train, validation, and test sets.
    """
    X_train_capped = X_train.copy()
    X_val_capped = X_val.copy()
    X_test_capped = X_test.copy()

    capping_summary = []

    for col in numerical_cols:
        q1 = X_train[col].quantile(0.25)
        q3 = X_train[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        train_outlier_count = ((X_train[col] < lower_bound) | (X_train[col] > upper_bound)).sum()

        # Apply capping
        X_train_capped[col] = X_train_capped[col].clip(lower_bound, upper_bound)
        X_val_capped[col] = X_val_capped[col].clip(lower_bound, upper_bound)
        X_test_capped[col] = X_test_capped[col].clip(lower_bound, upper_bound)

        capping_summary.append({
            "Feature": col,
            "Q1": round(q1, 2),
            "Q3": round(q3, 2),
            "IQR": round(iqr, 2),
            "Lower Bound": round(lower_bound, 2),
            "Upper Bound": round(upper_bound, 2),
            "Train Outliers": train_outlier_count,
        })

    return X_train_capped, X_val_capped, X_test_capped, pd.DataFrame(capping_summary)


def tune_random_forest(X_train_proc, y_train, X_val_proc, y_val):
    """
    Evaluates small grid of Random Forest hyperparameters exclusively
    on the validation set to select optimal parameters without test leakage.
    """
    param_grid = [
        {"n_estimators": 50, "max_depth": 4, "min_samples_split": 5},
        {"n_estimators": 100, "max_depth": 4, "min_samples_split": 5},
        {"n_estimators": 50, "max_depth": 6, "min_samples_split": 2},
        {"n_estimators": 100, "max_depth": 6, "min_samples_split": 2},
        {"n_estimators": 100, "max_depth": 8, "min_samples_split": 2},
        {"n_estimators": 150, "max_depth": 8, "min_samples_split": 5},
    ]

    tuning_results = []
    best_val_f1 = -1.0
    best_params = None
    best_model = None

    for params in param_grid:
        rf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        rf.fit(X_train_proc, y_train)
        val_preds = rf.predict(X_val_proc)
        val_acc = accuracy_score(y_val, val_preds)
        val_macro_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
        val_fail_f1 = f1_score(y_val, val_preds, pos_label=0, zero_division=0)

        tuning_results.append({
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "min_samples_split": params["min_samples_split"],
            "Val Accuracy": round(val_acc, 4),
            "Val Macro F1": round(val_macro_f1, 4),
            "Val Fail F1": round(val_fail_f1, 4),
        })

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_params = params
            best_model = rf

    return best_model, best_params, pd.DataFrame(tuning_results)


def plot_mlp_history(history, save_path):
    """Plot Training vs Validation Loss and Accuracy across epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history["loss"], label="Training Loss", color="#2980b9", lw=2)
    ax1.plot(
        history.history["val_loss"],
        label="Validation Loss",
        color="#e74c3c",
        lw=2,
        linestyle="--",
    )
    ax1.set_title("MLP Training & Validation Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Binary Crossentropy Loss", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.plot(history.history["accuracy"], label="Training Accuracy", color="#27ae60", lw=2)
    ax2.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy",
        color="#f39c12",
        lw=2,
        linestyle="--",
    )
    ax2.set_title("MLP Training & Validation Accuracy", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved MLP training history plot: {save_path}")


def plot_confusion_matrices(model_results, save_path):
    """Plot a 2x2 grid of confusion matrices for all evaluated models."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (name, res) in enumerate(model_results.items()):
        ax = axes[idx]
        cm = res["confusion_matrix"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Pred Fail (0)", "Pred Pass (1)"],
            yticklabels=["Act Fail (0)", "Act Pass (1)"],
            annot_kws={"size": 13, "weight": "bold"},
        )
        acc = res["accuracy"] * 100
        f1_fail = res["fail_f1"] * 100
        auc_val = res.get("roc_auc", None)
        auc_str = f" | AUC: {auc_val:.3f}" if auc_val is not None else ""
        ax.set_title(f"{name}\nAcc: {acc:.1f}% | Fail F1: {f1_fail:.1f}%{auc_str}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Actual Label", fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrices plot: {save_path}")


def plot_metric_comparison(comparison_df, save_path):
    """Plot bar chart comparing models on Accuracy, Fail Recall, Fail Precision, Macro F1, and ROC-AUC."""
    metrics_to_plot = ["Accuracy", "Fail Recall", "Fail Precision", "Macro F1", "ROC-AUC"]
    plot_df = comparison_df.set_index("Model")[metrics_to_plot]

    fig, ax = plt.subplots(figsize=(13, 6))
    plot_df.plot(kind="bar", ax=ax, width=0.75, colormap="viridis")
    ax.set_title("Model Metric Comparison on Stratified Test Set", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", linestyle=":", alpha=0.7)
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.legend(title="Metrics", fontsize=10, title_fontsize=11)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved model comparison plot: {save_path}")


def plot_feature_importance(importance_df, save_path):
    """Plot horizontal bar chart of Random Forest feature importances."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=importance_df,
        palette="crest_r",
        hue="Feature",
        legend=False,
        ax=ax,
    )
    ax.set_title("Random Forest Feature Importance (Transformed Features)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Gini Importance Score", fontsize=12)
    ax.set_ylabel("Feature Name", fontsize=12)
    ax.grid(axis="x", linestyle=":", alpha=0.7)

    for p in ax.patches:
        width = p.get_width()
        ax.annotate(
            f"{width:.4f}",
            (width, p.get_y() + p.get_height() / 2.0),
            ha="left",
            va="center",
            fontsize=9,
            xytext=(4, 0),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved feature importance plot: {save_path}")


def plot_roc_curves(roc_data, save_path):
    """Plot ROC curves comparing MLP, Random Forest, and other models."""
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = {
        "Random Forest": "#2ecc71",
        "Deep Learning (MLP)": "#e74c3c",
        "Logistic Regression": "#3498db",
        "Baseline (Dummy)": "#7f8c8d",
    }

    for name, data in roc_data.items():
        fpr = data["fpr"]
        tpr = data["tpr"]
        auc_val = data["auc"]
        color = colors.get(name, "#34495e")
        linestyle = "--" if name == "Baseline (Dummy)" else "-"
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=color, lw=2.2, linestyle=linestyle)

    # Diagonal random guess line
    ax.plot([0, 1], [0, 1], color="black", linestyle=":", lw=1.5, label="Random Guess (AUC = 0.500)")

    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10, frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved ROC curves plot: {save_path}")


def main():
    print_header("Student Performance Analysis - Model Training & Evaluation Study")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    # -------------------------------------------------------------
    # Step 1: Load Dataset & Target Leakage Prevention
    # -------------------------------------------------------------
    print_header("Step 1: Loading Dataset & Target Leakage Check")
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.\n")

    print("[CRITICAL] Target Leakage Analysis:")
    print("  * 'Final_Percentage' is the exact mathematical average of Math, Science, and English scores.")
    print("  * 'Pass_Fail' is directly derived from 'Final_Percentage' (>= 50% = Pass).")
    print("  * If current exam scores or Final_Percentage were included, models would achieve trivial ~100%")
    print("    accuracy simply by memorizing the arithmetic threshold.")
    print("  * Therefore, we strictly exclude: Student_ID, Math_Score, Science_Score, English_Score,")
    print("    Final_Percentage, and Performance_Level.")
    print("  * Our models predict outcomes using ONLY prior behavioral, demographic, and historical factors.")

    # -------------------------------------------------------------
    # Step 2: Feature Selection & Target Definition
    # -------------------------------------------------------------
    print_header("Step 2: Defining Features and Target")

    numerical_features = [
        "Age",
        "Class",
        "Study_Hours_Per_Day",
        "Attendance_Percentage",
        "Previous_Year_Score",
    ]

    categorical_features = [
        "Gender",
        "Parental_Education",
        "Internet_Access",
        "Extracurricular_Activities",
    ]

    feature_columns = numerical_features + categorical_features
    target_column = "Pass_Fail"

    print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    print(f"Total input features: {len(feature_columns)}")

    X = df[feature_columns].copy()
    y = (df[target_column] == "Pass").astype(int).values

    pass_count = np.sum(y == 1)
    fail_count = np.sum(y == 0)
    print(f"\nTarget Class Distribution:")
    print(f"  - Pass (1): {pass_count} ({pass_count / len(y) * 100:.2f}%)")
    print(f"  - Fail (0): {fail_count} ({fail_count / len(y) * 100:.2f}%)")
    print("  Notice: Severe class imbalance (~94.7% Pass vs ~5.3% Fail).")

    # -------------------------------------------------------------
    # Step 3: Stratified Train / Validation / Test Splitting
    # -------------------------------------------------------------
    print_header("Step 3: Stratified Train / Validation / Test Splitting")

    # 70% Train, 30% Temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # 15% Val, 15% Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    print(f"Training set   : {X_train.shape[0]} samples (70%) | Fail: {np.sum(y_train == 0)}, Pass: {np.sum(y_train == 1)}")
    print(f"Validation set : {X_val.shape[0]} samples (15%)  | Fail: {np.sum(y_val == 0)}, Pass: {np.sum(y_val == 1)}")
    print(f"Testing set    : {X_test.shape[0]} samples (15%)  | Fail: {np.sum(y_test == 0)}, Pass: {np.sum(y_test == 1)}")

    # -------------------------------------------------------------
    # Step 4: Outlier Handling via Training-Fitted IQR Capping
    # -------------------------------------------------------------
    print_header("Step 4: Outlier Handling via IQR Capping (Fitted on Train Only)")

    X_train_capped, X_val_capped, X_test_capped, capping_df = cap_outliers_iqr(
        X_train, X_val, X_test, numerical_features
    )
    print(capping_df.to_string(index=False))
    print("\nObservation: Capping limits fitted on training set and applied across splits to prevent data leakage.")

    # -------------------------------------------------------------
    # Step 5: Preprocessing Pipeline (Scale & One-Hot Encode)
    # -------------------------------------------------------------
    print_header("Step 5: Preprocessing Pipeline")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    # Fit ONLY on the training set to prevent data leakage
    X_train_proc = preprocessor.fit_transform(X_train_capped)
    X_val_proc = preprocessor.transform(X_val_capped)
    X_test_proc = preprocessor.transform(X_test_capped)

    raw_feature_names = preprocessor.get_feature_names_out()
    clean_feature_names = [name.replace("num__", "").replace("cat__", "") for name in raw_feature_names]
    input_dim = X_train_proc.shape[1]
    print(f"Transformed input dimensions: {input_dim} features")
    print(f"Transformed feature names ({len(clean_feature_names)}): {clean_feature_names}")

    # Save preprocessor pipeline
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Saved preprocessing pipeline to: {preprocessor_path}")

    # Compute balanced class weights for training
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {int(cls): float(w) for cls, w in zip(classes, weights)}
    print(f"Computed Class Weights: Class 0 (Fail): {class_weight_dict[0]:.4f}, Class 1 (Pass): {class_weight_dict[1]:.4f}")

    # Dictionary to collect test results for all models
    model_results = {}
    roc_data = {}

    # -------------------------------------------------------------
    # Step 6: Train & Evaluate Baseline (Dummy Classifier)
    # -------------------------------------------------------------
    print_header("Step 6: Model 1 - Baseline (Dummy Classifier: Most Frequent)")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train_proc, y_train)
    dummy_preds = dummy.predict(X_test_proc)
    dummy_probs = dummy.predict_proba(X_test_proc)[:, 1]
    dummy_auc = roc_auc_score(y_test, dummy_probs)
    fpr_dummy, tpr_dummy, _ = roc_curve(y_test, dummy_probs)

    model_results["Baseline (Dummy)"] = {
        "model": dummy,
        "preds": dummy_preds,
        "probs": dummy_probs,
        "accuracy": accuracy_score(y_test, dummy_preds),
        "fail_precision": precision_score(y_test, dummy_preds, pos_label=0, zero_division=0),
        "fail_recall": recall_score(y_test, dummy_preds, pos_label=0, zero_division=0),
        "fail_f1": f1_score(y_test, dummy_preds, pos_label=0, zero_division=0),
        "pass_recall": recall_score(y_test, dummy_preds, pos_label=1, zero_division=0),
        "macro_f1": f1_score(y_test, dummy_preds, average="macro", zero_division=0),
        "roc_auc": dummy_auc,
        "confusion_matrix": confusion_matrix(y_test, dummy_preds),
    }
    roc_data["Baseline (Dummy)"] = {"fpr": fpr_dummy, "tpr": tpr_dummy, "auc": dummy_auc}

    # -------------------------------------------------------------
    # Step 7: Train & Evaluate Linear Model (Logistic Regression)
    # -------------------------------------------------------------
    print_header("Step 7: Model 2 - Logistic Regression (Balanced)")
    log_reg = LogisticRegression(class_weight="balanced", random_state=RANDOM_SEED, max_iter=1000)
    log_reg.fit(X_train_proc, y_train)
    log_reg_preds = log_reg.predict(X_test_proc)
    log_reg_probs = log_reg.predict_proba(X_test_proc)[:, 1]
    log_auc = roc_auc_score(y_test, log_reg_probs)
    fpr_log, tpr_log, _ = roc_curve(y_test, log_reg_probs)

    model_results["Logistic Regression"] = {
        "model": log_reg,
        "preds": log_reg_preds,
        "probs": log_reg_probs,
        "accuracy": accuracy_score(y_test, log_reg_preds),
        "fail_precision": precision_score(y_test, log_reg_preds, pos_label=0, zero_division=0),
        "fail_recall": recall_score(y_test, log_reg_preds, pos_label=0, zero_division=0),
        "fail_f1": f1_score(y_test, log_reg_preds, pos_label=0, zero_division=0),
        "pass_recall": recall_score(y_test, log_reg_preds, pos_label=1, zero_division=0),
        "macro_f1": f1_score(y_test, log_reg_preds, average="macro", zero_division=0),
        "roc_auc": log_auc,
        "confusion_matrix": confusion_matrix(y_test, log_reg_preds),
    }
    roc_data["Logistic Regression"] = {"fpr": fpr_log, "tpr": tpr_log, "auc": log_auc}

    # -------------------------------------------------------------
    # Step 8: Random Forest Hyperparameter Tuning & Evaluation
    # -------------------------------------------------------------
    print_header("Step 8: Model 3 - Random Forest Hyperparameter Tuning on Validation Set")
    print("Evaluating parameter combinations strictly on Validation set (no test data used):")

    best_rf, best_rf_params, tuning_df = tune_random_forest(
        X_train_proc, y_train, X_val_proc, y_val
    )
    print(tuning_df.to_string(index=False))
    print(f"\nOptimal Hyperparameters Selected via Validation Macro F1: {best_rf_params}")

    # Evaluate tuned Random Forest on test set
    rf_preds = best_rf.predict(X_test_proc)
    rf_probs = best_rf.predict_proba(X_test_proc)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_probs)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)

    model_results["Random Forest"] = {
        "model": best_rf,
        "preds": rf_preds,
        "probs": rf_probs,
        "accuracy": accuracy_score(y_test, rf_preds),
        "fail_precision": precision_score(y_test, rf_preds, pos_label=0, zero_division=0),
        "fail_recall": recall_score(y_test, rf_preds, pos_label=0, zero_division=0),
        "fail_f1": f1_score(y_test, rf_preds, pos_label=0, zero_division=0),
        "pass_recall": recall_score(y_test, rf_preds, pos_label=1, zero_division=0),
        "macro_f1": f1_score(y_test, rf_preds, average="macro", zero_division=0),
        "roc_auc": rf_auc,
        "confusion_matrix": confusion_matrix(y_test, rf_preds),
    }
    roc_data["Random Forest"] = {"fpr": fpr_rf, "tpr": tpr_rf, "auc": rf_auc}

    # -------------------------------------------------------------
    # Step 9: Feature Importance Analysis (Random Forest)
    # -------------------------------------------------------------
    print_header("Step 9: Feature Importance Analysis (Random Forest)")
    importances = best_rf.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": clean_feature_names,
        "Importance": importances,
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    print("Ranked Feature Importances:")
    print(importance_df.to_string(index=False))

    feat_plot_path = os.path.join(VIS_DIR, "9_feature_importance.png")
    plot_feature_importance(importance_df, feat_plot_path)

    # -------------------------------------------------------------
    # Step 10: Train & Evaluate Deep Learning MLP
    # -------------------------------------------------------------
    print_header("Step 10: Model 4 - Deep Learning MLP (TensorFlow/Keras)")

    mlp = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu", name="hidden_layer_1"),
            layers.Dropout(0.3, name="dropout_1"),
            layers.Dense(32, activation="relu", name="hidden_layer_2"),
            layers.Dropout(0.2, name="dropout_2"),
            layers.Dense(1, activation="sigmoid", name="output_layer"),
        ],
        name="student_pass_fail_mlp",
    )

    mlp.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    history = mlp.fit(
        X_train_proc,
        y_train,
        validation_data=(X_val_proc, y_val),
        epochs=50,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[early_stopping],
        verbose=0,
    )

    mlp_probs = mlp.predict(X_test_proc, verbose=0).flatten()
    mlp_preds = (mlp_probs >= 0.5).astype(int)
    mlp_auc = roc_auc_score(y_test, mlp_probs)
    fpr_mlp, tpr_mlp, _ = roc_curve(y_test, mlp_probs)

    model_results["Deep Learning (MLP)"] = {
        "model": mlp,
        "preds": mlp_preds,
        "probs": mlp_probs,
        "accuracy": accuracy_score(y_test, mlp_preds),
        "fail_precision": precision_score(y_test, mlp_preds, pos_label=0, zero_division=0),
        "fail_recall": recall_score(y_test, mlp_preds, pos_label=0, zero_division=0),
        "fail_f1": f1_score(y_test, mlp_preds, pos_label=0, zero_division=0),
        "pass_recall": recall_score(y_test, mlp_preds, pos_label=1, zero_division=0),
        "macro_f1": f1_score(y_test, mlp_preds, average="macro", zero_division=0),
        "roc_auc": mlp_auc,
        "confusion_matrix": confusion_matrix(y_test, mlp_preds),
    }
    roc_data["Deep Learning (MLP)"] = {"fpr": fpr_mlp, "tpr": tpr_mlp, "auc": mlp_auc}

    mlp_save_path = os.path.join(MODELS_DIR, "student_pass_fail_mlp.keras")
    mlp.save(mlp_save_path)
    print(f"Saved MLP model to: {mlp_save_path}")

    # -------------------------------------------------------------
    # Step 11: Comprehensive Comparative Reporting (With ROC-AUC)
    # -------------------------------------------------------------
    print_header("Step 11: Comparative Evaluation on Unseen Stratified Test Set (750 Samples)")

    comparison_rows = []
    for name, res in model_results.items():
        cm = res["confusion_matrix"]
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        comparison_rows.append({
            "Model": name,
            "Accuracy": round(res["accuracy"], 4),
            "Fail Recall": round(res["fail_recall"], 4),
            "Fail Precision": round(res["fail_precision"], 4),
            "Fail F1": round(res["fail_f1"], 4),
            "Pass Recall": round(res["pass_recall"], 4),
            "Macro F1": round(res["macro_f1"], 4),
            "ROC-AUC": round(res["roc_auc"], 4),
            "True Fail (0)": tn,
            "False Pass (0 as 1)": fp,
            "False Alarm (1 as 0)": fn,
            "True Pass (1)": tp,
        })

    comparison_df = pd.DataFrame(comparison_rows)
    print(comparison_df.to_string(index=False))

    metrics_csv_path = os.path.join(MODELS_DIR, "model_comparison_metrics.csv")
    comparison_df.to_csv(metrics_csv_path, index=False)
    print(f"\nSaved updated evaluation metrics to: {metrics_csv_path}")

    # Print individual classification reports
    for name, res in model_results.items():
        print("-" * 65)
        print(f"Detailed Classification Report - {name}:")
        print("-" * 65)
        print(
            classification_report(
                y_test,
                res["preds"],
                target_names=["Fail (Class 0)", "Pass (Class 1)"],
                digits=4,
                zero_division=0,
            )
        )

    # -------------------------------------------------------------
    # Step 12: Best Model Selection and Persistence
    # -------------------------------------------------------------
    print_header("Step 12: Best Model Selection & Persistence")

    best_model_name = comparison_df.sort_values(by=["Macro F1", "ROC-AUC"], ascending=False).iloc[0]["Model"]
    print(f"Selected Best Performing Model based on Macro F1 & ROC-AUC: '{best_model_name}'")

    if best_model_name == "Random Forest":
        best_model_path = os.path.join(MODELS_DIR, "best_model_random_forest.joblib")
        joblib.dump(best_rf, best_model_path)
        print(f"Saved best model (Random Forest) to: {best_model_path}")
    elif best_model_name == "Deep Learning (MLP)":
        print(f"MLP is saved as best model to: {mlp_save_path}")
    else:
        best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
        joblib.dump(model_results[best_model_name]["model"], best_model_path)
        print(f"Saved best model to: {best_model_path}")

    # -------------------------------------------------------------
    # Step 13: Generating All Evaluation Visualizations
    # -------------------------------------------------------------
    print_header("Step 13: Generating Evaluation Visualizations")

    # 1. MLP Training History (Loss and Accuracy)
    history_plot_path = os.path.join(VIS_DIR, "6_mlp_training_history.png")
    plot_mlp_history(history, history_plot_path)

    # 2. Metric Comparison Bar Chart
    comparison_plot_path = os.path.join(VIS_DIR, "7_model_metric_comparison.png")
    plot_metric_comparison(comparison_df, comparison_plot_path)

    # 3. Confusion Matrix Grid
    cm_plot_path = os.path.join(VIS_DIR, "8_model_confusion_matrices.png")
    plot_confusion_matrices(model_results, cm_plot_path)

    # 4. ROC Curves
    roc_plot_path = os.path.join(VIS_DIR, "10_roc_curve.png")
    plot_roc_curves(roc_data, roc_plot_path)

    print("\n" + "=" * 75)
    print("All requirements completed successfully! Visualizations saved in 'visualizations/'.")
    print("=" * 75)


if __name__ == "__main__":
    main()
