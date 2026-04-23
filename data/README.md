# Dataset Documentation

This directory contains the datasets used to test and evaluate the Agentic Data Scientist.


## 1. `example_dataset.csv` (Provided demo)

| Property | Value |
|---|---|
| Rows | 20 |
| Columns | 6 |
| Target | `label` (binary: 0/1) |
| Task | Binary classification |
| Imbalance | 1.0 (perfectly balanced) |
| Missing values | None |
| Source | Provided with the CE888 template |

**Features:** age, bmi, smoker (categorical), steps_per_day, cholesterol.  
**Purpose:** Smoke test for the pipeline. Very small dataset useful for verifying end-to-end execution.

---

## 2. `iris.csv` (Multi-class, clean)

| Property | Value |
|---|---|
| Rows | 150 |
| Columns | 5 |
| Target | `species` (3 classes: setosa, versicolor, virginica) |
| Task | Multi-class classification |
| Imbalance | 1.0 (perfectly balanced) |
| Missing values | None |
| Source | scikit-learn `load_iris()` — Fisher's classic Iris dataset |

**Features:** sepal length, sepal width, petal length, petal width (all numeric, in cm).  
**Purpose:** Tests multi-class handling, clean-data pipeline. Known to be easily separable, expects high accuracy.

---

## 3. `imbalanced_synthetic.csv` (Imbalanced binary)

| Property | Value |
|---|---|
| Rows | 2,000 |
| Columns | 13 |
| Target | `target` (binary: 0/1) |
| Task | Binary classification |
| Imbalance | ~5.2 (85% majority / 15% minority) |
| Missing values | ~8% in columns feat_0, feat_3, feat_7 |
| Source | scikit-learn `make_classification()` with injected missing values |

**Features:** 12 numeric features (6 informative, 2 redundant, 4 noise).  
**Purpose:** Tests the agent's imbalance-handling strategy (class weights, threshold tuning), missing-value imputation, and the reflector's ability to detect bias towards the majority class.

---

## 4. `mixed_type.csv` (Mixed types, missing data, weak signal)

| Property | Value |
|---|---|
| Rows | 800 |
| Columns | 7 |
| Target | `approved` (binary: 0/1) |
| Task | Binary classification |
| Imbalance | ~1.4 |
| Missing values | ~12% in income and education |
| Source | Synthetically generated with numpy |

**Features:** age (int), income (float), education (categorical, 4 levels), city (categorical, 5 levels), score_a (float), score_b (float).  
**Purpose:** Tests mixed-type preprocessing (numeric + categorical), missing-value handling, and the agent's behaviour when predictive signal is very weak. This dataset intentionally has near-random target labels, so it triggers the reflector's replanning logic and tests whether the agent gracefully handles low-performance situations.

---

## Dataset Diversity Summary

| Dataset | Size | Types | Classes | Imbalance | Missing | Signal |
|---|---|---|---|---|---|---|
| example_dataset | Tiny (20) | Mixed | 2 | None | None | Strong |
| iris | Small (150) | Numeric | 3 | None | None | Strong |
| imbalanced_synthetic | Medium (2k) | Numeric | 2 | High (5.2) | Moderate | Moderate |
| mixed_type | Small (800) | Mixed | 2 | Slight | Moderate | Weak |

This selection covers the key scenarios: multi-class, imbalance, missing values, categorical features, and weak signal, exercising all planning branches and reflection logic.
