"""Interactive Streamlit application for Dry Bean classification."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# --------------------------------------------------
# Application configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIRECTORY = PROJECT_ROOT / "model"
SAMPLE_DATA_PATH = PROJECT_ROOT / "test_data.csv"
TARGET_COLUMN = "Class"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(
    page_title="Dry Bean Classification Lab",
    page_icon="🌱",
    layout="wide",
)


# --------------------------------------------------
# Load saved artifacts
# --------------------------------------------------

@st.cache_resource
def load_artifacts():
    """Load the saved models and label encoder."""

    encoder_path = MODEL_DIRECTORY / "label_encoder.joblib"

    if not encoder_path.exists():
        raise FileNotFoundError(
            "The saved label encoder was not found. "
            "Run train_models.py first."
        )

    models = {}

    for model_name, filename in MODEL_FILES.items():
        model_path = MODEL_DIRECTORY / filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Saved model not found: {model_path}"
            )

        models[model_name] = joblib.load(model_path)

    label_encoder = joblib.load(encoder_path)

    return models, label_encoder


# --------------------------------------------------
# Dataset validation
# --------------------------------------------------

def validate_test_data(dataframe, label_encoder, expected_features):
    """Validate the CSV and return features and encoded targets."""

    if dataframe.empty:
        raise ValueError("The uploaded CSV contains no rows.")

    if dataframe.columns.duplicated().any():
        raise ValueError("The CSV contains duplicate column names.")

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"The CSV must contain the target column '{TARGET_COLUMN}'."
        )

    uploaded_features = [
        column for column in dataframe.columns
        if column != TARGET_COLUMN
    ]

    missing_features = sorted(
        set(expected_features) - set(uploaded_features)
    )

    unexpected_features = sorted(
        set(uploaded_features) - set(expected_features)
    )

    if missing_features or unexpected_features:
        messages = []

        if missing_features:
            messages.append(
                "Missing features: " + ", ".join(missing_features)
            )

        if unexpected_features:
            messages.append(
                "Unexpected features: " + ", ".join(unexpected_features)
            )

        raise ValueError(" ".join(messages))

    features = dataframe[expected_features].copy()

    # Reject non-numeric values instead of silently modifying the data.
    for column in expected_features:
        features[column] = pd.to_numeric(
            features[column],
            errors="raise",
        )

    if features.isnull().any().any():
        raise ValueError(
            "The test data contains missing feature values. "
            "Please correct the CSV and upload it again."
        )

    if dataframe[TARGET_COLUMN].isnull().any():
        raise ValueError(
            f"The '{TARGET_COLUMN}' column contains missing values."
        )

    target_names = dataframe[TARGET_COLUMN].astype(str)
    known_classes = set(label_encoder.classes_)
    uploaded_classes = set(target_names.unique())

    unknown_classes = uploaded_classes - known_classes

    if unknown_classes:
        raise ValueError(
            "Unknown target labels: "
            + ", ".join(sorted(unknown_classes))
        )

    targets = label_encoder.transform(target_names)

    return features, targets


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate_model(model, features, targets, class_labels):
    """Calculate all six assignment metrics."""

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    metrics = {
        "Accuracy": accuracy_score(targets, predictions),
        "AUC": roc_auc_score(
            targets,
            probabilities,
            labels=class_labels,
            multi_class="ovr",
            average="macro",
        ),
        "Precision": precision_score(
            targets,
            predictions,
            labels=class_labels,
            average="macro",
            zero_division=0,
        ),
        "Recall": recall_score(
            targets,
            predictions,
            labels=class_labels,
            average="macro",
            zero_division=0,
        ),
        "F1": f1_score(
            targets,
            predictions,
            labels=class_labels,
            average="macro",
            zero_division=0,
        ),
        "MCC": matthews_corrcoef(targets, predictions),
    }

    return metrics, predictions


# --------------------------------------------------
# User interface
# --------------------------------------------------

st.title("🌱 Dry Bean Classification Lab")

st.markdown(
    """
    **Machine Learning Assignment 2 | M.Tech (AIML)**

    Explore seven dry bean varieties using five trained classification
    algorithms. Upload labelled test data to evaluate the saved models
    without retraining.
    """
)

st.info(
    "The uploaded CSV must contain all 16 input features and a "
    "'Class' target column with the original bean variety names."
)

try:
    models, label_encoder = load_artifacts()
except Exception as error:
    st.error(f"Unable to load the saved models: {error}")
    st.stop()


# Data source selection
data_source = st.radio(
    "Choose test data",
    ["Use bundled sample test data", "Upload a test CSV"],
    horizontal=True,
)

uploaded_file = None

if data_source == "Upload a test CSV":
    uploaded_file = st.file_uploader(
        "Upload your labelled test dataset",
        type=["csv"],
    )

    if uploaded_file is None:
        st.info("Upload a CSV file to begin evaluation.")
        st.stop()

    try:
        test_dataframe = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"Unable to read the uploaded CSV: {error}")
        st.stop()

else:
    if not SAMPLE_DATA_PATH.exists():
        st.error(
            "The bundled test_data.csv was not found. "
            "Choose the upload option and provide a test CSV."
        )
        st.stop()

    test_dataframe = pd.read_csv(SAMPLE_DATA_PATH)

    st.success("Using the bundled test dataset.")


# Validate test data
try:
    expected_features = list(
        next(iter(models.values())).feature_names_in_
    )
    x_test, y_test = validate_test_data(
        test_dataframe,
        label_encoder,
        expected_features,
    )
except Exception as error:
    st.error(f"Test data validation failed: {error}")
    st.stop()


st.subheader("Test dataset overview")

col1, col2, col3 = st.columns(3)

col1.metric("Test instances", f"{len(test_dataframe):,}")
col2.metric("Input features", f"{x_test.shape[1]}")
col3.metric("Bean varieties", f"{len(label_encoder.classes_)}")

with st.expander("Preview test data"):
    st.dataframe(test_dataframe.head(10), use_container_width=True)


# Evaluate every saved model on the same test data
class_labels = list(range(len(label_encoder.classes_)))
all_results = {}
all_predictions = {}

try:
    for model_name, model in models.items():
        metrics, predictions = evaluate_model(
            model,
            x_test,
            y_test,
            class_labels,
        )

        all_results[model_name] = metrics
        all_predictions[model_name] = predictions

except Exception as error:
    st.error(f"Model evaluation failed: {error}")
    st.stop()


st.subheader("Comparison of all five models")

comparison_dataframe = pd.DataFrame.from_dict(
    all_results,
    orient="index",
)

comparison_dataframe.index.name = "ML Model Name"

st.dataframe(
    comparison_dataframe.style.format("{:.4f}").highlight_max(
        axis=0,
        subset=[
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1",
            "MCC",
        ],
    ),
    use_container_width=True,
)

csv_data = comparison_dataframe.reset_index().to_csv(index=False)

st.download_button(
    label="Download model comparison (CSV)",
    data=csv_data,
    file_name="dry_bean_model_comparison.csv",
    mime="text/csv",
)


# Selected model
st.subheader("Explore an individual classifier")

selected_model_name = st.selectbox(
    "Select a classification model",
    list(MODEL_FILES.keys()),
)

selected_model = models[selected_model_name]
selected_predictions = all_predictions[selected_model_name]
selected_metrics = all_results[selected_model_name]

metric_columns = st.columns(6)

for column, (metric_name, metric_value) in zip(
    metric_columns,
    selected_metrics.items(),
):
    column.metric(metric_name, f"{metric_value:.4f}")


# Confusion matrix
st.subheader("Confusion matrix")

confusion = confusion_matrix(
    y_test,
    selected_predictions,
    labels=class_labels,
)

figure, axis = plt.subplots(figsize=(10, 7))

display = ConfusionMatrixDisplay(
    confusion_matrix=confusion,
    display_labels=label_encoder.classes_,
)

display.plot(
    ax=axis,
    xticks_rotation=45,
    colorbar=False,
    values_format="d",
)

axis.set_title(f"{selected_model_name} — Confusion Matrix")
figure.tight_layout()

st.pyplot(figure)
plt.close(figure)


# Classification report
st.subheader("Classification report")

report = classification_report(
    y_test,
    selected_predictions,
    labels=class_labels,
    target_names=label_encoder.classes_,
    output_dict=True,
    zero_division=0,
)

report_dataframe = pd.DataFrame(report).transpose()

st.dataframe(
    report_dataframe.style.format("{:.4f}"),
    use_container_width=True,
)
