"""Explore the Dry Bean dataset before model development."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "dry_bean.csv"


def explore_dataset():
    """Display dataset structure, quality checks, and summary statistics."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            "Run download_dataset.py first."
        )

    dataframe = pd.read_csv(DATASET_PATH)

    target_column = "Class"
    features = dataframe.drop(columns=[target_column])

    print("=" * 60)
    print("DRY BEAN DATASET EXPLORATION")
    print("=" * 60)

    print(f"\nDataset shape: {dataframe.shape}")
    print(f"Number of input features: {features.shape[1]}")
    print(f"Number of target classes: {dataframe[target_column].nunique()}")

    print("\nColumn data types:")
    print(dataframe.dtypes)

    print("\nMissing values per column:")
    print(dataframe.isnull().sum())

    print("\nDuplicate rows:", dataframe.duplicated().sum())

    print("\nClass distribution:")
    class_counts = dataframe[target_column].value_counts().sort_index()
    print(class_counts)

    print("\nClass distribution (%):")
    print((class_counts / len(dataframe) * 100).round(2))

    print("\nFeature summary statistics:")
    print(features.describe().round(3).T)

    print("\nCorrelation with target-independent features:")
    print(features.corr().round(2))

    print("\nDataset exploration completed.")


if __name__ == "__main__":
    explore_dataset()