import pytest
from agents.reflector import reflect, should_replan, apply_replan_strategy


def _make_profile(rows=1000, imbalance=1.0):
    return {
        "shape": {"rows": rows, "cols": 10},
        "imbalance_ratio": imbalance,
        "missing_pct": {},
    }


def _make_eval(model="RF", bal_acc=0.85, f1=0.84, acc=0.85, prec=0.84, rec=0.84):
    return {
        "model": model,
        "balanced_accuracy": bal_acc,
        "f1_macro": f1,
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
    }


class TestReflect:
    def test_ok_status_when_good_performance(self):
        profile = _make_profile()
        evaluation = _make_eval(bal_acc=0.90, f1=0.89)
        dummy = _make_eval(model="DummyMostFrequent", bal_acc=0.50, f1=0.33)
        result = reflect(profile, evaluation, [evaluation, dummy])
        assert result["status"] == "ok"
        assert result["replan_recommended"] is False

    def test_needs_attention_low_f1(self):
        profile = _make_profile()
        evaluation = _make_eval(bal_acc=0.55, f1=0.50)
        dummy = _make_eval(model="DummyMostFrequent", bal_acc=0.50, f1=0.33)
        result = reflect(profile, evaluation, [evaluation, dummy])
        assert result["status"] == "needs_attention"
        assert len(result["issues"]) > 0

    def test_replan_triggered_for_low_performance(self):
        profile = _make_profile()
        evaluation = _make_eval(bal_acc=0.52, f1=0.45)
        dummy = _make_eval(model="DummyMostFrequent", bal_acc=0.50, f1=0.33)
        result = reflect(profile, evaluation, [evaluation, dummy])
        assert result["replan_recommended"] is True

    def test_precision_recall_gap_detected(self):
        profile = _make_profile()
        evaluation = _make_eval(prec=0.90, rec=0.60)
        result = reflect(profile, evaluation, [evaluation])
        issue_text = " ".join(result["issues"])
        assert "precision-recall" in issue_text.lower()

    def test_imbalance_suggestion(self):
        profile = _make_profile(imbalance=5.0)
        evaluation = _make_eval()
        result = reflect(profile, evaluation, [evaluation])
        sug_text = " ".join(result["suggestions"])
        assert "imbalance" in sug_text.lower()

    def test_small_dataset_warning(self):
        profile = _make_profile(rows=200)
        evaluation = _make_eval()
        result = reflect(profile, evaluation, [evaluation])
        issue_text = " ".join(result["issues"])
        assert "small dataset" in issue_text.lower()

    def test_model_diversity_check(self):
        profile = _make_profile()
        m1 = _make_eval(model="RF", bal_acc=0.80)
        m2 = _make_eval(model="LR", bal_acc=0.805)
        result = reflect(profile, m1, [m1, m2])
        issue_text = " ".join(result["issues"])
        assert "identically" in issue_text.lower()

    def test_analysis_keys(self):
        profile = _make_profile()
        evaluation = _make_eval()
        dummy = _make_eval(model="DummyMostFrequent", bal_acc=0.50)
        result = reflect(profile, evaluation, [evaluation, dummy])
        assert "analysis" in result
        assert "improvement_over_baseline" in result["analysis"]


class TestShouldReplan:
    def test_true_when_recommended(self):
        assert should_replan({"replan_recommended": True}) is True

    def test_false_when_not_recommended(self):
        assert should_replan({"replan_recommended": False}) is False

    def test_false_when_missing(self):
        assert should_replan({}) is False


class TestApplyReplanStrategy:
    def test_returns_modified_plan_and_profile(self):
        plan = ["train_models", "evaluate"]
        profile = {"notes": []}
        reflection = {"replan_strategy": "try_ensemble"}
        new_plan, new_profile = apply_replan_strategy(plan, profile, reflection)
        assert "train_ensemble" in new_plan
        assert "replan_attempt" in new_plan

    def test_address_imbalance_strategy(self):
        plan = ["train_models", "evaluate"]
        profile = {"notes": []}
        reflection = {"replan_strategy": "address_imbalance"}
        new_plan, new_profile = apply_replan_strategy(plan, profile, reflection)
        assert new_profile.get("force_balanced_weights") is True

    def test_aggressive_replan(self):
        plan = ["train_models", "evaluate"]
        profile = {"notes": []}
        reflection = {"replan_strategy": "aggressive_replan"}
        new_plan, new_profile = apply_replan_strategy(plan, profile, reflection)
        assert "train_ensemble" in new_plan
        assert "hyperparameter_tuning" in new_plan
