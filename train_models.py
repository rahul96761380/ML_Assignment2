"""Train and evaluate five classifiers on the Dry Bean dataset."""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "dry_bean.csv"
MODEL_DIRECTORY = PROJECT_ROOT / "model"
RESULTS_DIRECTORY = PROJECT_ROOT / "results"

TARGET_COLUMN = "Class"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def build_models():
    """Create the five required classification pipelines."""

    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                (
                    "classifier",
                    DecisionTreeClassifier(
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    KNeighborsClassifier(n_neighbors=5),
                ),
            ]
        ),
        "Naive Bayes": Pipeline(
            [
                ("classifier", GaussianNB()),
            ]
        ),
        "Support Vector Machine": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        probability=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def calculate_auc(model, x_test, y_test, class_labels):
    """Calculate multiclass one-vs-rest macro AUC."""

    probabilities = model.predict_proba(x_test)

    return roc_auc_score(
        y_test,
        probabilities,
        labels=class_labels,
        multi_class="ovr",
        average="macro",
    )


def train_and_evaluate():
    """Train models, calculate metrics, and save artifacts."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}. "
            "Run download_dataset.py first."
        )

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(DATASET_PATH)

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    original_count = len(dataframe)

    # Remove exact duplicate rows before splitting.
    dataframe = dataframe.drop_duplicates().reset_index(drop=True)

    print(f"Original rows: {original_count}")
    print(f"Rows after removing duplicates: {len(dataframe)}")
    print(f"Duplicate rows removed: {original_count - len(dataframe)}")

    x = dataframe.drop(columns=[TARGET_COLUMN])
    y = dataframe[TARGET_COLUMN].astype(str)

    # Encode class names into numeric labels for consistent evaluation.
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    class_labels = list(range(len(label_encoder.classes_)))

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    print(f"\nTraining rows: {len(x_train)}")
    print(f"Test rows: {len(x_test)}")
    print(f"Input features: {x.shape[1]}")
    print(f"Classes: {list(label_encoder.classes_)}")

    # Export test features with their original readable class labels.
    test_data = x_test.copy()
    test_data[TARGET_COLUMN] = label_encoder.inverse_transform(y_test)
    test_data.to_csv(PROJECT_ROOT / "test_data.csv", index=False)

    joblib.dump(
        label_encoder,
        MODEL_DIRECTORY / "label_encoder.joblib",
    )

    results = []
    models = build_models()

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        model.fit(x_train, y_train)

        predictions = model.predict(x_test)

        metrics = {
            "ML Model Name": model_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "AUC": calculate_auc(
                model,
                x_test,
                y_test,
                class_labels,
            ),
            "Precision": precision_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "Recall": recall_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "F1": f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "MCC": matthews_corrcoef(y_test, predictions),
        }

        results.append(metrics)

        model_filename = (
            model_name.lower()
            .replace(" ", "_")
            + ".joblib"
        )

        joblib.dump(
            model,
            MODEL_DIRECTORY / model_filename,
        )

        print(
            f"Accuracy: {metrics['Accuracy']:.4f} | "
            f"AUC: {metrics['AUC']:.4f} | "
            f"F1: {metrics['F1']:.4f} | "
            f"MCC: {metrics['MCC']:.4f}"
        )

    results_dataframe = pd.DataFrame(results)

    results_path = RESULTS_DIRECTORY / "model_comparison.csv"
    results_dataframe.to_csv(results_path, index=False)

    print("\n" + "=" * 90)
    print("MODEL COMPARISON")
    print("=" * 90)
    print(results_dataframe.round(4).to_string(index=False))

    print(f"\nComparison table saved to: {results_path}")
    print(f"Test data saved to: {PROJECT_ROOT / 'test_data.csv'}")
    print(f"Saved models in: {MODEL_DIRECTORY}")

    print("\nAll five models trained and evaluated successfully.")


if __name__ == "__main__":
    train_and_evaluate()