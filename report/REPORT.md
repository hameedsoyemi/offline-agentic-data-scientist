# Technical Report: Offline Agentic Data Scientist

---

## 1. Introduction and Problem Definition

The goal of this project is to design, implement, and evaluate an **Offline Agentic Data Scientist**, an autonomous system capable of performing end-to-end classification tasks on unseen datasets without relying on Large Language Models or cloud-based services. The system must demonstrate autonomy, planning, reflection, and system-level reasoning rather than simply maximising predictive accuracy.

The key challenge is building a system that can adapt its behaviour to the characteristics of any given dataset. Unlike a static machine learning pipeline that applies the same steps regardless of the data, an agentic system must inspect the data, reason about what it finds, decide on a strategy, execute that strategy, evaluate the outcome, and potentially revise its approach all without human intervention.

This report documents the architecture of the system, the design decisions behind each component, the results obtained across four diverse datasets, and a critical reflection on the system's strengths, limitations, and ethical implications.

---

## 2. Agent Architecture

The system follows a **plan–execute–reflect–replan** loop, which is a well-established pattern in autonomous agent design. The architecture is organised into three layers:

### 2.1 Agent Layer

The agent layer contains three components that handle the cognitive aspects of the system:

**Planner** (`agents/planner.py`): The planner analyses the dataset profile and generates an ordered list of execution steps. Rather than using a single fixed pipeline, it selects from conditional plan templates based on the dataset's characteristics. For instance, a small dataset (fewer than 1,000 rows) triggers cross-validation and a preference for simpler models, while a highly imbalanced dataset (ratio ≥ 3.0) triggers imbalance-handling steps such as balanced class weights and decision threshold tuning. The planner also consults the memory system: if a previous run on the same or similar dataset succeeded with a particular model family, that model is prioritised in the plan.

**Reflector** (`agents/reflector.py`): After each execution cycle, the reflector analyses the results through multiple lenses. It compares the best model's performance against a dummy baseline, checks whether all models perform identically (suggesting weak signal), examines the precision-recall gap, applies adaptive performance thresholds that account for problem difficulty, and flags data quality concerns such as very small sample sizes or high missing-value rates. Based on its analysis, the reflector outputs a structured reflection containing identified issues, prioritised suggestions, and a named replan strategy if the performance falls below acceptable thresholds.

**Memory** (`agents/memory.py`): The memory system is a JSON-backed persistent store that records the outcome of every run. Each dataset is identified by a fingerprint derived from its shape, target column, and column names. The memory stores the full run history for each fingerprint, including timestamps, the best model, and its metrics. It also maintains a global strategy log that records which replan strategies were attempted and their outcomes. A similarity-based retrieval function allows the planner to look up strategies that worked on datasets with similar characteristics (size, dimensionality, imbalance), enabling a basic form of meta-learning across runs.

### 2.2 Tool Layer

The tool layer provides the concrete data science capabilities:

**Data Profiler** (`tools/data_profiler.py`): The profiler goes beyond basic descriptive statistics. It computes missing-value percentages per column, identifies feature types (numeric vs. categorical), detects constant and near-constant columns, computes pairwise Pearson correlations to flag highly correlated feature pairs (threshold > 0.85), performs IQR-based outlier detection across all numeric columns, and summarises skewness. For classification targets, it computes class counts and the imbalance ratio. All findings are compiled into a structured profile dictionary that drives the planner's decisions.

**Modelling** (`tools/modelling.py`): The modelling tool builds a sklearn `ColumnTransformer` for preprocessing (median imputation + scaling for numerics; mode imputation + one-hot encoding for categoricals), then selects candidate models based on dataset characteristics. Smaller datasets get more candidates (including KNN and SVC), while larger datasets exclude expensive models. All candidates are trained and evaluated with 3-fold cross-validation scores. The tool also provides a voting ensemble builder that combines the top three non-dummy models, and a feature importance extractor that works with both tree-based models (via `feature_importances_`) and linear models (via `coef_`).

**Evaluation** (`tools/evaluation.py`): The evaluation tool computes a full classification report, generates a confusion matrix visualisation, produces a model comparison bar chart, and writes a comprehensive markdown report that includes plan rationale, per-model results, reflection output, and ethical considerations.

### 2.3 Orchestrator

The orchestrator (`agentic_data_scientist.py`) ties everything together. It manages the execution loop, handles errors at each stage, tracks timing information, and coordinates between agents and tools. If the reflector recommends replanning and the maximum replan count has not been reached, the orchestrator applies the replan strategy (which modifies the plan and/or the profile) and re-executes the pipeline.

---

## 3. Key Design Decisions

### 3.1 Template-Based Conditional Planning

Rather than a monolithic pipeline, the planner uses a base template augmented with conditional branches. This was chosen over a fully dynamic approach for several reasons: it is transparent (the plan can be logged and inspected), predictable (the same profile always produces the same plan), and extensible (new templates can be added without modifying existing logic). The trade-off is that the planner cannot discover entirely novel strategies (it can only combine pre-defined building blocks).

### 3.2 Adaptive Thresholds in Reflection

The reflector uses adaptive performance thresholds rather than fixed ones. For imbalanced datasets (ratio ≥ 3.0), the F1 and balanced accuracy thresholds are lowered from 0.60 to 0.55, reflecting the genuine difficulty of such problems. This prevents the agent from entering unnecessary replan cycles on inherently hard tasks.

### 3.3 Named Replan Strategies

Instead of generic replanning (which risks repeating the same actions), the reflector selects a specific named strategy based on its diagnosis. For example, if the analysis identifies imbalance as the primary issue, the `address_imbalance` strategy forces balanced class weights. If the model barely beats the baseline, the `try_ensemble` strategy adds a voting ensemble. This targeted approach ensures each replan attempt is meaningfully different from the previous one.

### 3.4 Ensemble as a Replan Action

The voting ensemble is not always included in the initial plan, it is added as a replan action when the reflector diagnoses that individual models are underperforming. This is deliberate: ensembles add computational cost and complexity, so they should only be deployed when there is evidence that the current approach is insufficient.

### 3.5 Memory-Guided Meta-Learning

The memory system enables the agent to learn across runs. If the agent has previously processed a dataset with the same fingerprint, the planner can skip exploratory model selection and prioritise the known best model. The similarity-based retrieval extends this to unseen datasets: if a new dataset resembles one in memory (similar size, dimensionality, and imbalance), the agent can borrow the successful strategy as a starting point.

---

## 4. Results

The agent was tested on four diverse datasets:

### 4.1 Example Dataset (20 rows, binary, balanced)

The agent correctly identified `label` as the target, noted the extremely small dataset size, and added cross-validation to the plan. LogisticRegression achieved a perfect balanced accuracy of 1.000, which is expected given the small size and clear separability. The reflector flagged that all models performed identically and that the small dataset makes results unstable, both appropriate observations.

### 4.2 Iris (150 rows, 3-class, balanced)

The planner added cross-validation for the small dataset. GradientBoosting achieved 0.967 balanced accuracy. The reflector noted the small dataset size as a concern but did not recommend replanning since performance was strong. Feature importance correctly identified petal width as the most discriminative feature, consistent with domain knowledge.

### 4.3 Imbalanced Synthetic (2,000 rows, binary, 5.2× imbalance)

The planner correctly detected the imbalance and added `consider_imbalance_strategy` and `adjust_threshold` to the plan. All applicable models used `class_weight='balanced'`. SVC with RBF kernel achieved 0.870 balanced accuracy. The reflector found no major issues, confirming that the imbalance-handling strategy was effective. Feature importance highlighted the informative features over the noise features, demonstrating sensible model behaviour.

### 4.4 Mixed-Type (800 rows, binary, weak signal)

This dataset has near-random target labels and was designed to stress-test the agent's failure-handling behaviour. The planner added cross-validation and simpler model preferences. RandomForest achieved only 0.544 balanced accuracy, barely above chance. The reflector correctly identified the marginal improvement over baseline, low F1, and low balanced accuracy, and recommended replanning with the `try_ensemble` strategy. After two replan attempts (adding ensemble, then retrying), performance did not improve, and the agent correctly stopped at the maximum replan count. This demonstrates that the agent handles genuinely unpredictable data gracefully rather than entering an infinite loop.

### 4.5 Summary of Results

| Dataset | Best Model | Bal. Acc | F1 Macro | Replans |
|---|---|---|---|---|
| example_dataset | LogisticRegression | 1.000 | 1.000 | 0 |
| iris | GradientBoosting | 0.967 | 0.967 | 0 |
| imbalanced_synthetic | SVC_RBF | 0.870 | 0.868 | 0 |
| mixed_type | RandomForest | 0.544 | 0.525 | 2 (max) |

---

## 5. Testing and Quality Assurance

The project includes a comprehensive test suite with **62 tests** across four test modules, achieving **89% code coverage** across the agents and tools packages.

### Test Categories

- **Unit tests for the planner** (9 tests): verify that each conditional branch fires correctly, that plans are deduplicated, and that dependency ordering is maintained.
- **Unit tests for the reflector** (11 tests): verify issue detection for low F1, precision-recall gaps, imbalance, small datasets, and model diversity; verify replan strategy selection.
- **Unit tests for memory and profiler** (17 tests): verify persistence, run history tracking, similarity retrieval, target inference, feature type detection, outlier detection, and correlation analysis.
- **Modelling and integration tests** (13 tests): verify preprocessing, model selection, training, ensemble building, feature importance extraction, and full end-to-end pipeline execution on multiple datasets.
- **Integration tests** (4 tests): end-to-end runs on the example dataset and Iris, a noise dataset to trigger replanning, and a memory persistence test across runs.

---

## 6. Advanced Features Implemented

Beyond the mandatory core tasks, the following advanced features were implemented:

1. **Voting Ensemble** : soft-voting ensemble of top-3 models, added dynamically during replanning when individual models underperform.

2. **Feature Importance Extraction and Visualisation** : works with tree-based models (feature_importances_) and linear models (coefficient magnitudes), producing a ranked horizontal bar chart.

3. **Cross-Validation Scoring** : 3-fold stratified cross-validation is performed for every candidate model during training, providing a more robust estimate than a single train-test split.

4. **Similarity-Based Memory Retrieval** : the memory system computes Euclidean distance over a normalised feature vector (log-rows, log-cols, imbalance ratio, numeric fraction) to find previously processed datasets with similar characteristics.

5. **Adaptive Reflection Thresholds** : F1 and balanced accuracy thresholds adjust based on problem difficulty (imbalance ratio), preventing unnecessary replanning on inherently hard tasks.

6. **Rich Data Profiling** : the profiler detects outliers (IQR method), highly correlated feature pairs, constant columns, and skewness, all of which inform the planner's decisions.

7. **Model Comparison Visualisation** : a grouped bar chart comparing balanced accuracy and macro F1 across all candidate models, saved as an artefact.

8. **Strategy Logging for Meta-Learning** : the memory system records which replan strategies were attempted and whether they led to improvement, enabling future agents to avoid strategies that previously failed.

---

## 7. Limitations and Future Work

### 7.1 Current Limitations

**No hyperparameter tuning:** The current system uses default or lightly configured hyperparameters. Adding grid search or Bayesian optimisation would likely improve performance, particularly on medium-sized datasets where the computational cost is acceptable.

**Limited feature engineering:** The agent does not generate new features (e.g., polynomial interactions, binning, or domain-specific transformations). The `feature_engineering` plan step exists but is not yet backed by concrete logic.

**Single evaluation metric for replan decisions:** The reflector primarily uses balanced accuracy and macro F1 to decide whether to replan. A more sophisticated approach might consider per-class F1 scores, confidence intervals, or learning curve analysis.

**No regression support:** The system is designed exclusively for classification. Extending it to regression would require different evaluation metrics, model candidates, and planning logic.

**Simple memory similarity:** The current distance heuristic uses only four features. A richer representation (e.g., including feature-type distribution, missing-value patterns, or dataset topic embedding) would improve cross-dataset generalisation.

### 7.2 Future Directions

- **Bayesian hyperparameter optimisation** using Optuna or scikit-optimize.
- **Automated feature engineering** with polynomial features, interaction terms, or target-encoding for high-cardinality categoricals.
- **Learning curve analysis** to distinguish overfitting from underfitting and guide model complexity decisions.
- **Per-class performance analysis** in the reflector to identify specific classes that are problematic.
- **Statistical significance testing** (e.g., paired t-tests on cross-validation folds) to determine whether the best model is genuinely superior or just lucky.

---

## 8. Ethical Considerations

### 8.1 Transparency and Auditability

Every decision the agent makes is logged and persisted. The plan, profile, metrics, reflection, and feature importances are all saved as human-readable JSON and markdown files. This ensures that a human reviewer can trace exactly why the agent chose a particular model and strategy.

### 8.2 Fairness

The automated handling of class imbalance (via balanced class weights) helps prevent the model from ignoring minority classes. However, the agent does not check for protected attributes or proxy discrimination. In a deployment context, the feature importance output should be reviewed to ensure that the model does not rely on sensitive or proxy variables.

### 8.3 Data Privacy

The agent runs entirely offline, no data is sent to external services. All processing happens locally, which is appropriate for sensitive datasets.

### 8.4 Limitations of Automation

The agent should not be treated as a replacement for human judgement. Its reflections are heuristic-based and may miss domain-specific issues. The markdown report is designed to support human review, not to bypass it.


---


## 9. Conclusion

This project demonstrates that an offline, rule-based agentic system can autonomously handle diverse classification tasks by combining conditional planning, structured reflection, persistent memory, and targeted replanning. The system successfully adapts its behaviour to small datasets, imbalanced classes, mixed feature types, and weak-signal scenarios. With 62 passing tests and 89% code coverage, the implementation is robust and well-documented. The primary areas for future improvement are hyperparameter tuning, automated feature engineering, and richer statistical analysis in the reflection stage.

---
