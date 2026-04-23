from typing import Any, Dict, List, Tuple
import numpy as np



def reflect(
    dataset_profile: Dict[str, Any],
    evaluation: Dict[str, Any],
    all_metrics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    best_model = evaluation.get("model", "unknown")
    bal_acc = float(evaluation.get("balanced_accuracy", 0.0))
    f1_macro = float(evaluation.get("f1_macro", 0.0))
    accuracy = float(evaluation.get("accuracy", 0.0))
    precision = float(evaluation.get("precision_macro", 0.0))
    recall = float(evaluation.get("recall_macro", 0.0))
    imb = float(dataset_profile.get("imbalance_ratio") or 1.0)
    rows = dataset_profile.get("shape", {}).get("rows", 0)

    issues: List[str] = []
    suggestions: List[str] = []
    analysis: Dict[str, Any] = {}

    # Baseline comparison
    dummy = next((m for m in all_metrics if "Dummy" in m.get("model", "")), None)
    if dummy is not None:
        dummy_ba = float(dummy.get("balanced_accuracy", 0.0))
        improvement = bal_acc - dummy_ba
        analysis["improvement_over_baseline"] = round(improvement, 4)

        if improvement < 0.02:
            issues.append(
                f"Best model improves only {improvement:.3f} over dummy baseline; "
                "signal may be very weak or the pipeline has issues."
            )
            suggestions.append(
                "Verify target column quality, check for data leakage, "
                "and consider more expressive features."
            )
        elif improvement < 0.05:
            issues.append(
                f"Marginal improvement ({improvement:.3f}) over baseline."
            )
            suggestions.append(
                "Try feature engineering or ensemble methods to lift performance."
            )

    # Model diversity analysis
    non_dummy = [m for m in all_metrics if "Dummy" not in m.get("model", "")]
    if len(non_dummy) >= 2:
        ba_scores = [float(m.get("balanced_accuracy", 0)) for m in non_dummy]
        spread = max(ba_scores) - min(ba_scores)
        analysis["model_spread_bal_acc"] = round(spread, 4)

        if spread < 0.01:
            issues.append(
                "All models perform almost identically; data may lack discriminative signal."
            )
            suggestions.append(
                "Investigate feature importance and consider collecting richer features."
            )

    # F1 / balanced-accuracy checks (adaptive thresholds)
    # Adaptive threshold: harder problems (high imbalance / few rows) get lower thresholds before we flag concern
    f1_threshold = 0.55 if imb >= 3.0 else 0.60
    ba_threshold = 0.55 if imb >= 3.0 else 0.60

    if f1_macro < f1_threshold:
        issues.append(f"Macro F1 ({f1_macro:.3f}) below adaptive threshold ({f1_threshold}).")
        suggestions.append(
            "Try hyperparameter tuning, ensemble methods, or additional preprocessing."
        )

    if bal_acc < ba_threshold:
        issues.append(f"Balanced accuracy ({bal_acc:.3f}) below threshold ({ba_threshold}).")

    # Precision-recall gap analysis
    pr_gap = abs(precision - recall)
    analysis["precision_recall_gap"] = round(pr_gap, 4)
    if pr_gap > 0.15:
        if precision > recall:
            issues.append(
                f"Large precision-recall gap ({pr_gap:.3f}). Model is conservative, "
                "missing many positive cases."
            )
            suggestions.append(
                "Lower the decision threshold or apply class weights to improve recall."
            )
        else:
            issues.append(
                f"Large precision-recall gap ({pr_gap:.3f}). Model over-predicts "
                "the positive class."
            )
            suggestions.append(
                "Raise the decision threshold or apply stricter regularisation."
            )

    # Imbalance-specific checks
    if imb >= 3.0:
        if accuracy > bal_acc + 0.10:
            issues.append(
                "Accuracy much higher than balanced accuracy. Model biased towards majority."
            )
        suggestions.append(
            "Class imbalance present (ratio {:.1f}): consider class_weight='balanced', "
            "threshold tuning, or SMOTE oversampling.".format(imb)
        )

    # Small dataset warning
    if rows < 500:
        issues.append(
            "Very small dataset (<500 rows), results may be unstable."
        )
        suggestions.append(
            "Use cross-validation and simpler models to reduce variance."
        )

    # Missing data quality check
    missing_pct = dataset_profile.get("missing_pct", {})
    high_missing_cols = [c for c, v in missing_pct.items() if v > 20]
    if high_missing_cols:
        issues.append(
            f"{len(high_missing_cols)} feature(s) have >20% missing values."
        )
        suggestions.append(
            "Consider advanced imputation (KNN, iterative) or dropping "
            "heavily missing features."
        )

    # Determine overall status and replan recommendation
    status = "needs_attention" if issues else "ok"

    replan_recommended = bool(issues and (f1_macro < f1_threshold or bal_acc < ba_threshold))

    replan_strategy = _select_replan_strategy(
        issues, f1_macro, bal_acc, imb, rows
    )

    return {
        "status": status,
        "best_model": best_model,
        "best_balanced_accuracy": round(bal_acc, 4),
        "best_f1_macro": round(f1_macro, 4),
        "issues": issues,
        "suggestions": suggestions,
        "replan_recommended": replan_recommended,
        "replan_strategy": replan_strategy,
        "analysis": analysis,
    }


# Replan strategy selection

def _select_replan_strategy(
    issues: List[str], f1: float, bal_acc: float, imb: float, rows: int,
) -> str:
    """Pick a concrete replan strategy keyword based on identified issues."""
    if not issues:
        return "none"

    # Priority order (pick the first match)
    issue_text = " ".join(issues).lower()

    if "imbalance" in issue_text or "majority" in issue_text:
        return "address_imbalance"
    if "baseline" in issue_text or "weak" in issue_text:
        return "try_ensemble"
    if "precision-recall" in issue_text:
        return "tune_threshold"
    if f1 < 0.50:
        return "aggressive_replan"
    return "conservative_replan"



def should_replan(reflection: Dict[str, Any]) -> bool:
    return bool(reflection.get("replan_recommended", False))


def apply_replan_strategy(
    plan: List[str],
    dataset_profile: Dict[str, Any],
    reflection: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    new_plan = list(plan)
    new_profile = dict(dataset_profile)

    strategy = reflection.get("replan_strategy", "conservative_replan")
    notes = list(new_profile.get("notes", []))
    notes.append(f"Replan: applying strategy '{strategy}'.")

    if strategy == "address_imbalance":
        if "consider_imbalance_strategy" not in new_plan:
            idx = new_plan.index("train_models") if "train_models" in new_plan else len(new_plan)
            new_plan.insert(idx, "consider_imbalance_strategy")
        new_profile["force_balanced_weights"] = True

    elif strategy == "try_ensemble":
        if "train_ensemble" not in new_plan:
            idx = (new_plan.index("evaluate") if "evaluate" in new_plan else len(new_plan))
            new_plan.insert(idx, "train_ensemble")

    elif strategy == "tune_threshold":
        if "adjust_threshold" not in new_plan:
            idx = (new_plan.index("evaluate") if "evaluate" in new_plan else len(new_plan))
            new_plan.insert(idx, "adjust_threshold")

    elif strategy == "aggressive_replan":
        for step in ["train_ensemble", "hyperparameter_tuning"]:
            if step not in new_plan:
                idx = (new_plan.index("evaluate") if "evaluate" in new_plan else len(new_plan))
                new_plan.insert(idx, step)
        new_profile["force_balanced_weights"] = True

    else:
        if "feature_engineering" not in new_plan:
            idx = (new_plan.index("build_preprocessor") if "build_preprocessor" in new_plan else 0)
            new_plan.insert(idx, "feature_engineering")

    new_plan.append("replan_attempt")
    new_profile["notes"] = notes
    return new_plan, new_profile