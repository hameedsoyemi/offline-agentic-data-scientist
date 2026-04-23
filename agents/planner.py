from typing import Any, Dict, List, Optional



_BASE_PLAN: List[str] = [
    "profile_dataset",
    "detect_data_quality_issues",
    "handle_missing_values",
    "build_preprocessor",
    "select_models",
    "train_models",
    "evaluate",
    "feature_importance_analysis",
    "reflect",
    "write_report",
]

_SMALL_DATASET_EXTRAS: List[str] = [
    "apply_cross_validation",
    "prefer_simpler_models",
]

_LARGE_DATASET_EXTRAS: List[str] = [
    "subsample_for_profiling",
]

_IMBALANCED_EXTRAS: List[str] = [
    "consider_imbalance_strategy",
    "adjust_threshold",
]

_HIGH_DIMENSIONAL_EXTRAS: List[str] = [
    "apply_feature_selection",
    "apply_regularisation",
]

_HEAVY_MISSING_EXTRAS: List[str] = [
    "handle_severe_missing_data",
    "flag_missing_indicators",
]

_HIGH_CARDINALITY_EXTRAS: List[str] = [
    "apply_target_encoding",
]



def _insert_before(plan: List[str], anchor: str, new_tasks: List[str]) -> List[str]:
    out: List[str] = []
    for step in plan:
        if step == anchor:
            for t in new_tasks:
                if t not in out:
                    out.append(t)
        if step not in out:
            out.append(step)
    return out


def _insert_after(plan: List[str], anchor: str, new_tasks: List[str]) -> List[str]:
    out: List[str] = []
    for step in plan:
        if step not in out:
            out.append(step)
        if step == anchor:
            for t in new_tasks:
                if t not in out:
                    out.append(t)
    return out


def _deduplicate(plan: List[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for s in plan:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out



def create_plan(
    dataset_profile: Dict[str, Any],
    memory_hint: Optional[Dict[str, Any]] = None,
) -> List[str]:
    plan = list(_BASE_PLAN)

    rows = dataset_profile.get("shape", {}).get("rows", 0)
    cols = dataset_profile.get("shape", {}).get("cols", 0)
    imb = float(dataset_profile.get("imbalance_ratio") or 1.0)
    missing_pct = dataset_profile.get("missing_pct", {})
    n_unique = dataset_profile.get("n_unique_by_col", {})
    cat_cols = dataset_profile.get("feature_types", {}).get("categorical", [])

    # Size-based adaptation
    if rows < 1000:
        plan = _insert_before(plan, "train_models", _SMALL_DATASET_EXTRAS)
    elif rows > 50_000:
        plan = _insert_before(plan, "profile_dataset", _LARGE_DATASET_EXTRAS)

    # Imbalance adaptation
    if imb >= 3.0:
        plan = _insert_before(plan, "train_models", _IMBALANCED_EXTRAS)

    # Dimensionality adaptation
    if cols > 60:
        plan = _insert_before(plan, "build_preprocessor", _HIGH_DIMENSIONAL_EXTRAS)

    # Missing data adaptation
    max_missing = max(missing_pct.values()) if missing_pct else 0.0
    if max_missing > 15.0:
        plan = _insert_before(plan, "build_preprocessor", _HEAVY_MISSING_EXTRAS)

    # High-cardinality categoricals
    high_card = [c for c in cat_cols if n_unique.get(c, 0) > 20]
    if high_card:
        plan = _insert_before(plan, "build_preprocessor", _HIGH_CARDINALITY_EXTRAS)

    # Memory-guided prioritisation
    if memory_hint and memory_hint.get("best_model"):
        best_prev = memory_hint["best_model"]
        plan = _insert_after(
            plan, "select_models",
            [f"prioritise_model:{best_prev}"],
        )

    plan = _deduplicate(plan)
    return plan


# Helpers for targeted re-planning

def add_ensemble_step(plan: List[str]) -> List[str]:
    return _insert_after(plan, "train_models", ["train_ensemble"])


def add_hyperparameter_tuning(plan: List[str]) -> List[str]:
    return _insert_before(plan, "evaluate", ["hyperparameter_tuning"])


def add_feature_engineering(plan: List[str]) -> List[str]:
    return _insert_before(plan, "build_preprocessor", ["feature_engineering"])


def estimate_plan_cost(plan: List[str], rows: int) -> str:
    expensive = {"train_ensemble", "hyperparameter_tuning", "apply_cross_validation"}
    n_expensive = sum(1 for s in plan if s in expensive)
    if rows > 50_000 or n_expensive >= 2:
        return "high"
    if rows > 10_000 or n_expensive >= 1:
        return "medium"
    return "low"