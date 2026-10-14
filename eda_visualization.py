"""
Exploratory Data Analysis (EDA) - Visualizations
================================================
A beginner-friendly script to generate and save 5 essential exploratory
data analysis charts for the Student Performance Dataset using Matplotlib and Seaborn.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# File paths
DATASET_PATH = "Student_Performance_Dataset.csv"
OUTPUT_DIR = "visualizations"


def setup_environment():
    """Ensure the output directory exists and configure visual styling."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def main():
    print("=" * 70)
    print("Student Performance Dataset - EDA Visualizations")
    print("=" * 70)

    # 1. Load dataset
    print(f"\nLoading dataset from '{DATASET_PATH}'...")
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded successfully! {df.shape[0]} rows, {df.shape[1]} columns.\n")

    setup_environment()

    # -------------------------------------------------------------
    # Chart 1: Histogram of Study_Hours_Per_Day
    # -------------------------------------------------------------
    print("-" * 70)
    print("Chart 1: Histogram of Daily Study Hours")
    print("-" * 70)
    explanation_1 = (
        "Explanation:\n"
        "- Shows the distribution and frequency of daily study hours among students.\n"
        "- Helps identify whether study habits are evenly spread, normally distributed, or skewed."
    )
    print(explanation_1)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        df["Study_Hours_Per_Day"],
        bins=20,
        kde=True,
        color="#2b5c8f",
        edgecolor="black",
        ax=ax,
    )
    ax.set_title("Distribution of Daily Study Hours", fontsize=14, fontweight="bold")
    ax.set_xlabel("Study Hours Per Day", fontsize=12)
    ax.set_ylabel("Number of Students", fontsize=12)

    chart1_path = os.path.join(OUTPUT_DIR, "1_study_hours_histogram.png")
    plt.savefig(chart1_path, dpi=300)
    print(f"Saved: {chart1_path}")
    plt.show()
    plt.close()

    # -------------------------------------------------------------
    # Chart 2: Bar Chart of Performance_Level
    # -------------------------------------------------------------
    print("-" * 70)
    print("Chart 2: Bar Chart of Student Performance Levels")
    print("-" * 70)
    explanation_2 = (
        "Explanation:\n"
        "- Displays the total count of students categorized into each performance tier\n"
        "  (e.g., Poor, Average, Good, Excellent).\n"
        "- Highlights the class balance and overall academic standing of the cohort."
    )
    print(explanation_2)

    fig, ax = plt.subplots(figsize=(8, 5))
    # Count values in logical order if present
    order = ["Poor", "Average", "Good", "Excellent"]
    filtered_order = [lvl for lvl in order if lvl in df["Performance_Level"].unique()]
    palette = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]

    sns.countplot(
        x="Performance_Level",
        data=df,
        order=filtered_order,
        palette=palette[:len(filtered_order)],
        hue="Performance_Level",
        legend=False,
        ax=ax,
    )
    ax.set_title("Number of Students per Performance Level", fontsize=14, fontweight="bold")
    ax.set_xlabel("Performance Level", fontsize=12)
    ax.set_ylabel("Student Count", fontsize=12)

    # Annotate counts on top of bars
    for p in ax.patches:
        height = int(p.get_height())
        ax.annotate(
            f"{height}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
            xytext=(0, 3),
            textcoords="offset points",
        )

    chart2_path = os.path.join(OUTPUT_DIR, "2_performance_level_barchart.png")
    plt.savefig(chart2_path, dpi=300)
    print(f"Saved: {chart2_path}")
    plt.show()
    plt.close()

    # -------------------------------------------------------------
    # Chart 3: Boxplot of Subject Scores
    # -------------------------------------------------------------
    print("-" * 70)
    print("Chart 3: Boxplot of Math, Science, and English Scores")
    print("-" * 70)
    explanation_3 = (
        "Explanation:\n"
        "- Compares the score distributions across Math, Science, and English.\n"
        "- Displays median lines, interquartile ranges (middle 50%), and potential outliers\n"
        "  for each subject side-by-side."
    )
    print(explanation_3)

    fig, ax = plt.subplots(figsize=(8, 5))
    subject_data = df[["Math_Score", "Science_Score", "English_Score"]].rename(
        columns={
            "Math_Score": "Math",
            "Science_Score": "Science",
            "English_Score": "English",
        }
    )
    sns.boxplot(
        data=subject_data,
        palette=["#5dade2", "#48c9b0", "#f5b041"],
        width=0.5,
        ax=ax,
    )
    ax.set_title("Score Distribution Across Core Subjects", fontsize=14, fontweight="bold")
    ax.set_xlabel("Subject", fontsize=12)
    ax.set_ylabel("Score (0 - 100)", fontsize=12)

    chart3_path = os.path.join(OUTPUT_DIR, "3_subject_scores_boxplot.png")
    plt.savefig(chart3_path, dpi=300)
    print(f"Saved: {chart3_path}")
    plt.show()
    plt.close()

    # -------------------------------------------------------------
    # Chart 4: Scatterplot of Study Hours vs Final Percentage
    # -------------------------------------------------------------
    print("-" * 70)
    print("Chart 4: Scatterplot of Study Hours vs. Final Percentage")
    print("-" * 70)
    explanation_4 = (
        "Explanation:\n"
        "- Examines the relationship between daily study duration and final academic performance.\n"
        "- Helps determine if higher study hours correlate with higher final percentages."
    )
    print(explanation_4)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        x="Study_Hours_Per_Day",
        y="Final_Percentage",
        data=df,
        alpha=0.5,
        color="#8e44ad",
        edgecolor="w",
        s=40,
        ax=ax,
    )
    ax.set_title("Study Hours per Day vs. Final Percentage", fontsize=14, fontweight="bold")
    ax.set_xlabel("Study Hours Per Day", fontsize=12)
    ax.set_ylabel("Final Percentage (%)", fontsize=12)

    chart4_path = os.path.join(OUTPUT_DIR, "4_study_hours_vs_final_percentage.png")
    plt.savefig(chart4_path, dpi=300)
    print(f"Saved: {chart4_path}")
    plt.show()
    plt.close()

    # -------------------------------------------------------------
    # Chart 5: Correlation Heatmap for Numerical Columns
    # -------------------------------------------------------------
    print("-" * 70)
    print("Chart 5: Correlation Heatmap (Numerical Columns)")
    print("-" * 70)
    explanation_5 = (
        "Explanation:\n"
        "- Displays Pearson correlation coefficients between all numerical features.\n"
        "- Values near +1 show strong positive relationships, near -1 show negative relationships,\n"
        "  and values near 0 mean little to no linear association."
    )
    print(explanation_5)

    fig, ax = plt.subplots(figsize=(10, 8))
    numerical_df = df.select_dtypes(include=[np.number])
    corr_matrix = numerical_df.corr()

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Correlation Heatmap of Numerical Features", fontsize=14, fontweight="bold")

    chart5_path = os.path.join(OUTPUT_DIR, "5_correlation_heatmap.png")
    plt.savefig(chart5_path, dpi=300)
    print(f"Saved: {chart5_path}")
    plt.show()
    plt.close()

    print("\n" + "=" * 70)
    print(f"All 5 visualizations generated and saved in: '{OUTPUT_DIR}'")
    print("=" * 70)


if __name__ == "__main__":
    main()
