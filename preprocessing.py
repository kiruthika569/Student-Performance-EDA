"""
Data Preprocessing & Feature Preparation - Student Performance Dataset
=====================================================================
A beginner-friendly script to perform outlier detection using the IQR method,
inspect unique categorical values, and remove non-predictive identifiers.
"""

import numpy as np
import pandas as pd

# Path to original dataset
DATASET_PATH = "Student_Performance_Dataset.csv"


def detect_outliers_iqr(df, numerical_columns):
    """
    Detect outliers for numerical columns using the Interquartile Range (IQR) method.
    Formula:
      IQR = Q3 (75th percentile) - Q1 (25th percentile)
      Lower Bound = Q1 - 1.5 * IQR
      Upper Bound = Q3 + 1.5 * IQR
    Any value below Lower Bound or above Upper Bound is an outlier.
    """
    outlier_summary = []

    for col in numerical_columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_percentage = (outlier_count / len(df)) * 100

        outlier_summary.append({
            "Column": col,
            "Q1 (25%)": round(q1, 2),
            "Q3 (75%)": round(q3, 2),
            "IQR": round(iqr, 2),
            "Lower Bound": round(lower_bound, 2),
            "Upper Bound": round(upper_bound, 2),
            "Outlier Count": outlier_count,
            "Outlier %": round(outlier_percentage, 2)
        })

    return pd.DataFrame(outlier_summary)


def main():
    print("=" * 70)
    print("Step 1: Loading Dataset")
    print("=" * 70)

    # 1. Load the CSV file into a pandas DataFrame
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset successfully loaded from: {DATASET_PATH}")
    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}\n")

    # 2. Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    print("=" * 70)
    print("Step 2: Checking Numerical Columns for Outliers (IQR Method)")
    print("=" * 70)
    print(f"Numerical columns detected ({len(numerical_cols)}): {numerical_cols}\n")

    outlier_df = detect_outliers_iqr(df, numerical_cols)
    print(outlier_df.to_string(index=False))
    print("\nObservation:")
    total_outliers = outlier_df["Outlier Count"].sum()
    print(f"Total outlier points detected across all numerical columns: {total_outliers}\n")

    print("=" * 70)
    print("Step 3: Unique Values in Categorical Columns")
    print("=" * 70)

    # Note: We display unique values for all categorical columns
    for col in categorical_cols:
        unique_vals = df[col].unique()
        print(f"Column: '{col}' ({len(unique_vals)} unique values)")
        # If there are many values (like Student_ID), show sample instead of flooding screen
        if len(unique_vals) > 10:
            print(f"  Sample values: {list(unique_vals[:5])} ... (total: {len(unique_vals)})")
        else:
            print(f"  Values: {list(unique_vals)}")
            # Show value counts for better context
            counts = df[col].value_counts().to_dict()
            print(f"  Frequencies: {counts}")
        print("-" * 50)
    print()

    print("=" * 70)
    print("Step 4: Removing 'Student_ID' From Features")
    print("=" * 70)

    if "Student_ID" in df.columns:
        features_df = df.drop(columns=["Student_ID"])
        print("Removed 'Student_ID' successfully.")
        print(f"Original shape : {df.shape}")
        print(f"New feature shape: {features_df.shape}")
        print(f"Remaining feature columns ({len(features_df.columns)}):")
        for i, col_name in enumerate(features_df.columns, start=1):
            print(f"  {i}. {col_name}")
    else:
        print("'Student_ID' column was not found in dataset.")

    print("\n" + "=" * 70)
    print("Preprocessing checks completed successfully! Original CSV unchanged.")
    print("=" * 70)


if __name__ == "__main__":
    main()
