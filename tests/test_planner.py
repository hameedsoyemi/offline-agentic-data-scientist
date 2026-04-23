import pytest
from agents.planner import (
    create_plan,
    add_ensemble_step,
    add_hyperparameter_tuning,
    add_feature_engineering,
    estimate_plan_cost,
)


def _make_profile(rows=5000, cols=10, imbalance=1.0, missing_max=0.0, cat_cols=None):
    """Helper to build a minimal profile dict."""
    cat_cols = cat_cols or []
    num_cols = [f"num_{i}" for i in range(cols - 1 - len(cat_cols))]
    return {
        "shape": {"rows": rows, "cols": cols},
        "feature_types": {"numeric": num_cols, "categorical": cat_cols},
        "imbalance_ratio": imbalance,
        "missing_pct": {f"col_{i}": missing_max for i in range(cols)},
        "n_unique_by_col": {c: 5 for c in cat_cols},
        "notes": [],
    }


class TestCreatePlan:
    def test_base_plan_returned(self):
        profile = _make_profile()
        plan = create_plan(profile)
        assert isinstance(plan, list)
        assert len(plan) > 0
        assert "train_models" in plan
        assert "reflect" in plan

    def test_small_dataset_adds_cv(self):
        profile = _make_profile(rows=500)
        plan = create_plan(profile)
        assert "apply_cross_validation" in plan
        assert "prefer_simpler_models" in plan

    def test_large_dataset_adds_subsample(self):
        profile = _make_profile(rows=60000)
        plan = create_plan(profile)
        assert "subsample_for_profiling" in plan

    def test_imbalanced_adds_strategy(self):
        profile = _make_profile(imbalance=5.0)
        plan = create_plan(profile)
        assert "consider_imbalance_strategy" in plan
        assert "adjust_threshold" in plan

    def test_high_dimensional_adds_selection(self):
        profile = _make_profile(cols=80)
        plan = create_plan(profile)
        assert "apply_feature_selection" in plan

    def test_heavy_missing_adds_handling(self):
        profile = _make_profile(missing_max=25.0)
        plan = create_plan(profile)
        assert "handle_severe_missing_data" in plan

    def test_memory_hint_adds_prioritise(self):
        profile = _make_profile()
        hint = {"best_model": "RandomForest"}
        plan = create_plan(profile, memory_hint=hint)
        assert "prioritise_model:RandomForest" in plan

    def test_no_duplicates(self):
        profile = _make_profile(rows=500, imbalance=5.0, missing_max=30.0)
        plan = create_plan(profile)
        assert len(plan) == len(set(plan))

    def test_order_train_before_evaluate(self):
        profile = _make_profile()
        plan = create_plan(profile)
        assert plan.index("train_models") < plan.index("evaluate")
        assert plan.index("evaluate") < plan.index("reflect")


class TestPlanHelpers:
    def test_add_ensemble(self):
        plan = ["train_models", "evaluate"]
        updated = add_ensemble_step(plan)
        assert "train_ensemble" in updated
        assert updated.index("train_ensemble") == updated.index("train_models") + 1

    def test_add_tuning(self):
        plan = ["train_models", "evaluate"]
        updated = add_hyperparameter_tuning(plan)
        assert "hyperparameter_tuning" in updated

    def test_add_feature_eng(self):
        plan = ["build_preprocessor", "train_models"]
        updated = add_feature_engineering(plan)
        assert updated.index("feature_engineering") < updated.index("build_preprocessor")

    def test_cost_estimate(self):
        assert estimate_plan_cost(["train_models"], rows=100) == "low"
        assert estimate_plan_cost(["train_models", "apply_cross_validation"], rows=100) == "medium"
        assert estimate_plan_cost(["train_ensemble", "hyperparameter_tuning"], rows=100) == "high"
