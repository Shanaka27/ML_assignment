# Heart Disease Prediction Using Machine Learning

A machine learning project that predicts whether a patient may have
heart disease using clinical data. This project compares two supervised
classification algorithms: **Random Forest** and **Support Vector
Machine (SVM)**.

## Project Overview

-   **Course:** Fundamentals of Machine Learning -- Programming
    Assignment
-   **Task:** Binary classification
-   **Dataset:** UCI Heart Disease Data, downloaded from Kaggle
-   **Dataset file:** `heart_disease_uci.csv`
-   **Notebook:** `ML_Assignment.ipynb`

The original target column, `num`, has values from 0 to 4. In this
project, it is converted into a binary target:

-   `0` --- No heart disease
-   `1` --- Heart disease present

## Dataset

The notebook uses the `heart_disease_uci.csv` file from the [Kaggle
Heart Disease Data
dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data).

The dataset contains **920 records** from four sources:

-   Cleveland
-   Hungary
-   Switzerland
-   VA Long Beach

Download the CSV file from Kaggle and place it in the same folder as the
notebook before running the code.

### Data preparation

The notebook performs the following preparation steps:

-   Converts the original `num` target into a binary target.
-   Treats zero values in `trestbps` and `chol` as missing values.
-   Excludes `id`, `dataset`, and the original `num` column from the
    model features.
-   Uses the median to fill missing numeric values.
-   Scales numeric features with `StandardScaler`.
-   Treats missing categorical values as a separate `"missing"`
    category.
-   Converts categorical features into numeric columns using one-hot
    encoding.

The `id` and `dataset` columns are excluded because they do not
represent a patient's clinical measurements. Excluding the original
`num` column also prevents target leakage.

## Machine Learning Models

### 1. Random Forest

Random Forest combines predictions from multiple decision trees. The
model uses balanced class weights, and `GridSearchCV` searches for the
best hyperparameters using five-fold stratified cross-validation.

Parameters tested include:

-   Number of trees: `100`, `200`
-   Maximum depth: `None`, `4`, `8`
-   Minimum samples per leaf: `1`, `3`, `5`
-   Maximum features per split: `sqrt`, `log2`

### 2. Support Vector Machine (SVM)

SVM finds a decision boundary to separate the two classes. The notebook
uses `SVC` with probability estimates enabled.

Parameters tested include:

-   `C`: `0.1`, `1`, `10`, `100`
-   `gamma`: `scale`, `0.01`, `0.1`
-   Kernel: `rbf`, `linear`

Both models use preprocessing pipelines, and hyperparameter tuning is
performed on the training set only.

## Methodology

1.  Load the CSV dataset.
2.  Clean the target and prepare the features.
3.  Explore the data using summary statistics and visualizations.
4.  Split the data into **80% training** and **20% testing**, using
    stratified sampling.
5.  Build preprocessing and classification pipelines.
6.  Tune both models using five-fold stratified cross-validation and
    F1-score.
7.  Evaluate the selected models on the held-out test set.
8.  Estimate 95% bootstrap confidence intervals for test accuracy.
9.  Review Random Forest feature importance and permutation importance
    for both models.

The notebook uses a fixed random seed (`42`) to help make the results
reproducible.

## Results

The following results are from the notebook's held-out test set.

  Metric        Random Forest      SVM
  ----------- --------------- --------
  Accuracy             0.8533   0.8424
  Precision            0.8505   0.8174
  Recall               0.8922   0.9216
  F1-score             0.8708   0.8664
  ROC-AUC              0.9261   0.9152

### Best hyperparameters

**Random Forest**

``` text
max_depth = 8
max_features = sqrt
min_samples_leaf = 3
n_estimators = 200
```

**SVM**

``` text
C = 1
gamma = 0.1
kernel = rbf
```

### Results summary

-   Random Forest achieved higher test accuracy, precision, F1-score,
    and ROC-AUC.
-   SVM achieved higher recall, so it identified a larger proportion of
    patients with heart disease in the test set.
-   The best cross-validation F1-score was `0.8474` for Random Forest
    and `0.8591` for SVM.
-   The 95% bootstrap confidence intervals for test accuracy were
    `0.799–0.902` for Random Forest and `0.788–0.891` for SVM.

These results are for this dataset and test split only. The models are
for learning and research; they are **not medical diagnostic tools** and
should not replace advice from qualified healthcare professionals.

## Visualizations

The notebook creates visualizations for:

-   Disease rates by dataset source.
-   Numeric feature distributions by target class.
-   Categorical feature counts by target class, including missing
    values.
-   Correlations between numeric features and the target.
-   Confusion matrices for both models.
-   ROC curves.
-   Top Random Forest feature importances.
-   Permutation importance for both models.

## Requirements

Install the required Python libraries:

``` bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

## How to Run

1.  Clone or download this repository.

2.  Download `heart_disease_uci.csv` from the [Kaggle dataset
    page](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data).

3.  Place `heart_disease_uci.csv` in the same folder as
    `ML_Assignment.ipynb`.

4.  Install the required libraries.

5.  Start Jupyter Notebook:

    ``` bash
    jupyter notebook
    ```

6.  Open `ML_Assignment.ipynb`.

7.  Run the notebook cells from top to bottom.

## Project Structure

``` text
.
├── ML_Assignment.ipynb
├── heart_disease_uci.csv   # Download separately from Kaggle
└── README.md
```

## Team

-   Kusal Darshana Kaluarachchi
-   Shanaka Niluksha Ranathunga

## References

-   [UCI Machine Learning Repository --- Heart
    Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
-   [Kaggle --- Heart Disease
    Data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
-   [Scikit-learn documentation](https://scikit-learn.org/stable/)
