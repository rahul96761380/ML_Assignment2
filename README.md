# Dry Bean Classification

## Introduction

The aim of this project is to classify dry beans into seven varieties using machine learning. I trained five classification models and compared their performance using different evaluation metrics.

I also developed a Streamlit application where users can compare the models, view their results and upload a labelled CSV file for evaluation.

## Dataset

I used the Dry Bean dataset from the UCI Machine Learning Repository.

- **Total records:** 13,611
- **Records after removing duplicates:** 13,543
- **Input features:** 16
- **Classes:** 7
- **Missing values:** None in the original dataset

The seven bean varieties are BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER and SIRA.

## Models Used

I trained and compared the following models:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

I removed duplicate rows before splitting the data. I used an 80:20 train-test split with stratification.

## Evaluation Metrics

I evaluated the models using the following metrics:

- Accuracy
- AUC
- Precision
- Recall
- F1-score
- Matthews Correlation Coefficient (MCC)

Macro averaging was used for Precision, Recall and F1-score. AUC was calculated using the one-vs-rest method.

## Results

The results on the test dataset are shown below.

| Model | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|:--|--:|--:|--:|--:|--:|--:|
| Logistic Regression | 0.9195 | 0.9947 | 0.9311 | 0.9303 | 0.9306 | 0.9028 |
| Decision Tree | 0.8955 | 0.9460 | 0.9094 | 0.9104 | 0.9098 | 0.8737 |
| KNN | 0.9155 | 0.9832 | 0.9290 | 0.9256 | 0.9270 | 0.8978 |
| Gaussian Naive Bayes | 0.7630 | 0.9670 | 0.7736 | 0.7685 | 0.7669 | 0.7143 |
| SVM | 0.9210 | 0.9945 | 0.9342 | 0.9317 | 0.9328 | 0.9045 |

SVM gave the highest accuracy at 92.10%. It also achieved the highest F1-score and MCC. Logistic Regression achieved the highest AUC.

## Streamlit Application

The Streamlit application includes:

- Comparison of all five models
- Selection of an individual model
- Display of the six evaluation metrics
- Confusion matrix
- Classification report
- CSV upload for evaluating labelled test data
- Option to download the model comparison results

The application uses the saved models to evaluate test data without training them again.

The uploaded CSV must contain all 16 input features and a `Class` column with the original bean variety names.

## Project Files

| File or Folder | Description |
|:--|:--|
| `download_dataset.py` | Downloads the dataset |
| `explore_dataset.py` | Explores the dataset |
| `train_models.py` | Trains and evaluates the models |
| `app.py` | Runs the Streamlit application |
| `data/` | Contains the dataset |
| `model/` | Contains the saved models and label encoder |
| `results/` | Contains the model comparison results |
| `test_data.csv` | Contains the test dataset |
| `requirements.txt` | Lists the required Python packages |

## How to Run

### 1. Install the required packages

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

```bash
python download_dataset.py
```

### 3. Explore the dataset

```bash
python explore_dataset.py
```

### 4. Train the models

```bash
python train_models.py
```

### 5. Start the Streamlit application

```bash
python -m streamlit run app.py
```

After running the last command, open the local URL displayed in the terminal.

## GitHub Repository

[ML Assignment 2 – Dry Bean Classification](https://github.com/rahul96761380/ML_Assignment2)

## Streamlit Deployment

The live application link will be added here after deployment.