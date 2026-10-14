"""
Exploratory Data Analysis (EDA) - Student Performance Dataset
============================================================
A beginner-friendly script to inspect and summarize student performance data
using Pandas and NumPy.
"""

import numpy as np
import pandas as pd

# Define the dataset path
DATASET_PATH = "Student_Performance_Dataset.csv"


def main():
    print("=" * 60)
    print("Loading Dataset...")
    print("=" * 60)

    # 1. Load the CSV file into a pandas DataFrame
    df = pd.read_csv(DATASET_PATH)
    print("Dataset loaded successfully!\n")

    # 2. Dataset Shape (Number of rows and columns)
    print("-" * 60)
    print("1. DATASET SHAPE")
    print("-" * 60)
    rows, cols = df.shape
    print(f"Total Rows (Records)   : {rows}")
    print(f"Total Columns (Features): {cols}\n")

    # 3. First Five Rows
    print("-" * 60)
    print("2. FIRST 5 ROWS (HEAD)")
    print("-" * 60)
    print(df.head())
    print("\n")

    # 4. Column Names
    print("-" * 60)
    print("3. COLUMN NAMES")
    print("-" * 60)
    for idx, col in enumerate(df.columns, start=1):
        print(f"  {idx}. {col}")
    print("\n")

    # 5. Data Types of Columns
    print("-" * 60)
    print("4. DATA TYPES")
    print("-" * 60)
    print(df.dtypes)
    print("\n")

    # 6. Missing Values Analysis
    print("-" * 60)
    print("5. MISSING VALUES PER COLUMN")
    print("-" * 60)
    missing_values = df.isnull().sum()
    print(missing_values)
    total_missing = missing_values.sum()
    print(f"\nTotal Missing Values in Dataset: {total_missing}\n")

    # 7. Duplicate Rows Count
    print("-" * 60)
    print("6. DUPLICATE ROWS COUNT")
    print("-" * 60)
    duplicate_count = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_count}\n")

    # 8. Statistical Summary
    print("-" * 60)
    print("7. STATISTICAL SUMMARY (NUMERICAL FEATURES)")
    print("-" * 60)
    print(df.describe())
    print("\n")

    print("-" * 60)
    print("8. STATISTICAL SUMMARY (CATEGORICAL FEATURES)")
    print("-" * 60)
    print(df.describe(include=["object"]))
    print("\n")

    print("=" * 60)
    print("Inspection complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
