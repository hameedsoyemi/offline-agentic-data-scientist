# Offline Agentic Data Scientist

An autonomous classification pipeline that profiles unseen datasets, generates adaptive execution plans, trains and evaluates models, reflects on results, and iteratively refines its strategy, all without LLM or cloud dependencies.


---

## Overview

Traditional ML pipelines apply the same steps regardless of the data. This project takes a different approach: it encodes the **judgement** of a data scientist into a rule-based autonomous agent that adapts its behaviour to whatever dataset it encounters.

The system follows a **plan → execute → reflect → replan** loop:

```
                    ┌─────────────┐
                    │  Load Data  │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │   Profile   │◄── data_profiler.py
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │ Check Memory│◄── memory.py
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │    Plan     │◄── planner.py
                    └──────┬──────┘
                           ▼
               ┌───────────────────────┐
               │  Execute (Preprocess  │◄── modelling.py
               │  + Train + Ensemble)  │
               └───────────┬───────────┘
                           ▼
                    ┌─────────────┐
                    │  Evaluate   │◄── evaluation.py
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
              ┌─────│   Reflect   │◄── reflector.py
              │     └──────┬──────┘
              │            │
         needs replan?     │ no
              │            ▼
              │     ┌─────────────┐
              │     │ Save Memory │
              │     │ + Report    │
              │     └─────────────┘
              │
              └──► adjust strategy → loop back to Plan
```

---

## Key Features

- **Adaptive Planning** : the planner inspects dataset characteristics (size, imbalance, missing values, cardinality, dimensionality) and conditionally assembles a tailored pipeline rather than running a fixed sequence
- **Multi-Criteria Reflection** : the reflector analyses results through multiple lenses: baseline comparison, model diversity, precision-recall gap, adaptive thresholds, and data quality flags
- **Persistent Memory** : a JSON-backed store that records every run's outcome keyed by dataset fingerprint, enabling meta-learning and strategy reuse across datasets
- **Graceful Failure Handling** : the agent recognises genuinely unpredictable data (weak signal) and stops after a bounded number of replan attempts rather than looping infinitely
- **Fully Offline** : no LLMs, no cloud APIs, no AutoML services, all decision logic is encoded as deterministic rules and heuristics

---

## Results

The agent was evaluated across four datasets designed to exercise different planning branches and failure modes:

| Dataset | Rows | Classes | Challenge | Best Model | Bal. Accuracy | Replans |
|---|---|---|---|---|---|---|
| example_dataset | 20 | 2 | Tiny, clean | LogisticRegression | 1.000 | 0 |
| iris | 150 | 3 | Small, multi-class | GradientBoosting | 0.967 | 0 |
| imbalanced_synthetic | 2,000 | 2 | 5.2× class imbalance | SVC (RBF) | 0.870 | 0 |
| mixed_type | 800 | 2 | Weak signal, noise | RandomForest | 0.544 | 2 (max) |

The imbalanced dataset demonstrates the planner correctly adding class weighting and threshold tuning. The weak-signal dataset demonstrates the agent recognising failure and stopping gracefully after exhausting replan strategies.

---

## Project Structure

```
offline-agentic-data-scientist/
│
├── run_agent.py                    # Entry point : CLI interface
├── agentic_data_scientist.py       # Orchestrator : coordinates the full pipeline
│
├── agents/
│   ├── __init__.py
│   ├── planner.py                  # Adaptive plan generation from dataset profile
│   ├── reflector.py                # Multi-criteria result analysis + replan strategy selection
│   └── memory.py                   # Persistent JSON store with similarity-based retrieval
│
├── tools/
│   ├── __init__.py
│   ├── data_profiler.py            # Rich EDA: missing values, outliers, correlations, skewness
│   ├── modelling.py                # Preprocessing, model selection, training, ensemble building
│   └── evaluation.py               # Metrics, confusion matrix, comparison charts, markdown report
│
├── data/
│   ├── example_dataset.csv
│   ├── iris.csv
│   ├── imbalanced_synthetic.csv
│   ├── mixed_type.csv
│   └── README.md                   # Dataset descriptions and diversity summary
│
├── tests/
│   ├── __init__.py
│   ├── sanity_check.py             # Quick structural validation
│   ├── test_planner.py             # 9 tests : conditional branch coverage
│   ├── test_reflector.py           # 11 tests : issue detection and strategy selection
│   ├── test_memory.py              # Persistence and similarity retrieval tests
│   ├── test_data_profiler.py       # Profiling edge case tests
│   ├── test_modelling.py           # Preprocessing, training, and ensemble tests
│   └── test_evaluation.py          # Metrics and report generation tests
│
├── report/
│   └── REPORT.md                   # Full technical report
│
├── outputs/                        # Auto-generated per run (reports, charts, matrices)
│   └── .gitkeep
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How It Works

### Planning

The planner selects from conditional plan templates based on the dataset profile:

| Condition | Adaptation |
|---|---|
| Rows < 1,000 | Add cross-validation, prefer simpler models |
| Rows > 50,000 | Add sub-sampling for profiling |
| Imbalance ratio ≥ 3.0 | Add class weighting, threshold tuning |
| Columns > 60 | Add feature selection, regularisation |
| Max missing > 15% | Add advanced imputation, missing indicators |
| High-cardinality categoricals | Add target encoding |
| Memory hint available | Prioritise previously successful model family |

Plans are deduplicated and dependency-ordered so prerequisite steps always execute first.

### Reflection

After each execution cycle, the reflector runs multi-criteria analysis:

1. **Baseline comparison** : does the best model meaningfully beat a dummy classifier?
2. **Model diversity** : are all models performing identically (weak signal indicator)?
3. **Adaptive thresholds** : F1 and balanced accuracy thresholds adjust for problem difficulty
4. **Precision-recall gap** : detects conservative or over-predicting models
5. **Imbalance-specific checks** : accuracy vs. balanced accuracy divergence
6. **Data quality flags** : small dataset and high missing-value warnings

Based on findings, the reflector selects from five named replan strategies: `address_imbalance`, `try_ensemble`, `tune_threshold`, `aggressive_replan`, or `conservative_replan`.

### Memory

Every run is persisted to a JSON file keyed by a dataset fingerprint (derived from shape and column names). The memory system supports:

- **Run history** per dataset : timestamps, best model, metrics
- **Global strategy log** : which replan strategies have been attempted and their outcomes
- **Similarity-based retrieval** : the planner can borrow strategies from datasets with comparable characteristics

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/hameedsoyemi/offline-agentic-data-scientist.git
cd offline-agentic-data-scientist
pip install -r requirements.txt
```

### Running the Agent

```bash
# Run on the Iris dataset
python run_agent.py --data data/iris.csv --target species

# Run on the imbalanced dataset
python run_agent.py --data data/imbalanced_synthetic.csv --target target

# Run on your own dataset
python run_agent.py --data path/to/your_data.csv --target your_target_column
```

Each run generates a timestamped folder in `outputs/` containing a markdown report, confusion matrix, model comparison chart, and feature importance plot.

### Running the Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=agents --cov=tools --cov-report=term-missing

# Quick sanity check
python tests/sanity_check.py
```

**Test suite:** 62 tests passing · 89% code coverage

---

## Model Candidates

The agent selects from the following classifiers based on dataset characteristics:

| Model | When Preferred |
|---|---|
| Logistic Regression | Small datasets, baseline comparisons |
| Random Forest | General-purpose, mixed feature types |
| Gradient Boosting | Strong signal datasets, high accuracy needed |
| K-Nearest Neighbours | Small to medium datasets with clear clusters |
| Extra Trees | Alternative ensemble, high variance data |
| AdaBoost | Boosting when gradient boosting overfits |
| SVC (RBF kernel) | Non-linear boundaries, moderate-size datasets |

A soft-voting ensemble is optionally built from the top-performing models when the reflector recommends it.

---

## Technologies

Python · scikit-learn · NumPy · Pandas · Matplotlib · pytest

---

## Limitations and Future Work

- **No hyperparameter tuning** : models use sensible defaults; adding grid/random search would improve accuracy
- **No automated feature engineering** : the agent uses raw and profiled features but doesn't generate interaction terms or polynomial features
- **Reflection is rule-based** : a more sophisticated reflector could use statistical hypothesis testing
- **Single-task** : currently supports classification only; extending to regression would require additional planning templates

---

## Author

**Hameed Soyemi**

---

## Licence

This project is released under the [MIT Licence](https://opensource.org/licenses/MIT).
