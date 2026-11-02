"""Download and validate the UCI Dry Bean classification dataset."""

from pathlib import Path

from ucimlrepo import fetch_ucirepo


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIRECTORY / "dry_bean.csv"


def download_dataset():
    """Fetch the dataset, save it locally, and validate its contents."""

    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("Downloading the UCI Dry Bean dataset...")

    dry_bean = fetch_ucirepo(id=602)

    features = dry_bean.data.features.copy()
    targets = dry_bean.data.targets.copy()

    if len(targets.columns) != 1:
        raise ValueError("Expected exactly one target column.")

    target_column = targets.columns[0]

    dataframe = features.copy()
    dataframe[target_column] = targets[target_column].to_numpy()

    # Validate the dataset before saving it.
    instance_count = len(dataframe)
    feature_count = features.shape[1]
    missing_count = int(dataframe.isnull().sum().sum())
    class_count = dataframe[target_column].nunique()

    print("\nDataset validation results")
    print("-" * 40)
    print(f"Instances: {instance_count}")
    print(f"Input features: {feature_count}")
    print(f"Target column: {target_column}")
    print(f"Number of classes: {class_count}")
    print(f"Missing values: {missing_count}")

    assert instance_count >= 500, (
        "The dataset must contain at least 500 instances."
    )
    assert feature_count >= 12, (
        "The dataset must contain at least 12 input features."
    )
    assert missing_count == 0, (
        "Missing values were found. Investigate before proceeding."
    )
    assert class_count >= 2, "At least two target classes are required."

    dataframe.to_csv(DATASET_PATH, index=False)

    print("\nClass distribution:")
    print(dataframe[target_column].value_counts().sort_index())

    print(f"\nDataset saved to: {DATASET_PATH}")
    print("\nFirst five rows:")
    print(dataframe.head())

    print("\nDataset validation passed.")


if __name__ == "__main__":
    download_dataset()