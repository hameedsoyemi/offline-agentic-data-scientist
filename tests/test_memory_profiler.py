import os
import json
import tempfile
import pytest
import pandas as pd
import numpy as np

from agents.memory import JSONMemory
from tools.data_profiler import (
    infer_target_column,
    is_classification_target,
    dataset_fingerprint,
    profile_dataset,
)


# Memory tests

class TestJSONMemory:
    def _make_memory(self, tmp_path):
        return JSONMemory(str(tmp_path / "test_mem.json"))

    def test_empty_get(self, tmp_path):
        mem = self._make_memory(tmp_path)
        assert mem.get_dataset_record("nonexistent") is None

    def test_upsert_and_get(self, tmp_path):
        mem = self._make_memory(tmp_path)
        mem.upsert_dataset_record("fp_123", {"best_model": "RF", "best_metrics": {"balanced_accuracy": 0.9}})
        rec = mem.get_dataset_record("fp_123")
        assert rec is not None
        assert rec["best_model"] == "RF"

    def test_run_history_appended(self, tmp_path):
        mem = self._make_memory(tmp_path)
        mem.upsert_dataset_record("fp_1", {"best_model": "A", "best_metrics": {"balanced_accuracy": 0.8, "f1_macro": 0.7}})
        mem.upsert_dataset_record("fp_1", {"best_model": "B", "best_metrics": {"balanced_accuracy": 0.85, "f1_macro": 0.75}})
        rec = mem.get_dataset_record("fp_1")
        assert len(rec["run_history"]) == 2

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "persist.json")
        mem1 = JSONMemory(path)
        mem1.upsert_dataset_record("fp_x", {"best_model": "LR", "best_metrics": {}})
        mem2 = JSONMemory(path)
        assert mem2.get_dataset_record("fp_x") is not None

    def test_add_note(self, tmp_path):
        mem = self._make_memory(tmp_path)
        mem.add_note("test note")
        assert len(mem.data["notes"]) == 1

    def test_log_strategy(self, tmp_path):
        mem = self._make_memory(tmp_path)
        mem.log_strategy("fp_1", "try_ensemble", "improved")
        assert len(mem.data["global_strategy_log"]) == 1
        strategies = mem.get_successful_strategies()
        assert "try_ensemble" in strategies

    def test_find_similar(self, tmp_path):
        mem = self._make_memory(tmp_path)
        mem.upsert_dataset_record("fp_small", {
            "best_model": "LR", "shape": {"rows": 100, "cols": 5},
            "best_metrics": {"balanced_accuracy": 0.7, "f1_macro": 0.6},
        })
        mem.upsert_dataset_record("fp_big", {
            "best_model": "RF", "shape": {"rows": 50000, "cols": 100},
            "best_metrics": {"balanced_accuracy": 0.9, "f1_macro": 0.9},
        })
        similar = mem.find_similar({"rows": 120, "cols": 6}, imbalance_ratio=1.0, numeric_frac=0.8)
        assert len(similar) >= 1


# Data profiler tests

class TestInferTarget:
    def test_common_names(self):
        df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
        assert infer_target_column(df) == "target"

    def test_label_column(self):
        df = pd.DataFrame({"x": [1, 2], "Label": [0, 1]})
        assert infer_target_column(df) == "Label"

    def test_last_column_fallback(self):
        df = pd.DataFrame({"feat_a": range(100), "feat_b": range(100), "cls": [0, 1] * 50})
        assert infer_target_column(df) == "cls"


class TestIsClassification:
    def test_string_is_classification(self):
        assert is_classification_target(pd.Series(["a", "b", "c"])) is True

    def test_few_unique_ints(self):
        assert is_classification_target(pd.Series([0, 1, 2, 0, 1])) is True

    def test_many_unique_not_classification(self):
        assert is_classification_target(pd.Series(range(100))) is False


class TestFingerprint:
    def test_deterministic(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        fp1 = dataset_fingerprint(df, "b")
        fp2 = dataset_fingerprint(df, "b")
        assert fp1 == fp2

    def test_different_targets_differ(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        assert dataset_fingerprint(df, "a") != dataset_fingerprint(df, "b")


class TestProfileDataset:
    def _make_df(self):
        return pd.DataFrame({
            "age": [25, 30, 35, 40, 45],
            "city": ["A", "B", "A", "C", "B"],
            "label": [0, 1, 0, 1, 0],
        })

    def test_basic_keys(self):
        df = self._make_df()
        p = profile_dataset(df, "label")
        assert "shape" in p
        assert "missing_pct" in p
        assert "feature_types" in p
        assert "is_classification" in p
        assert "notes" in p

    def test_feature_types(self):
        df = self._make_df()
        p = profile_dataset(df, "label")
        assert "age" in p["feature_types"]["numeric"]
        assert "city" in p["feature_types"]["categorical"]

    def test_class_counts(self):
        df = self._make_df()
        p = profile_dataset(df, "label")
        assert p["class_counts"]["0"] == 3
        assert p["class_counts"]["1"] == 2

    def test_missing_detected(self):
        df = self._make_df()
        df.loc[0, "age"] = np.nan
        p = profile_dataset(df, "label")
        assert p["missing_pct"]["age"] > 0

    def test_outlier_detection(self):
        df = pd.DataFrame({
            "x": list(range(100)) + [1000],
            "label": [0, 1] * 50 + [0],
        })
        p = profile_dataset(df, "label")
        assert "x" in p["outlier_counts"]

    def test_correlation_pairs(self):
        np.random.seed(0)
        x = np.random.randn(100)
        df = pd.DataFrame({"a": x, "b": x * 1.01 + 0.001, "label": [0, 1] * 50})
        p = profile_dataset(df, "label")
        assert len(p["high_correlation_pairs"]) >= 1

    def test_invalid_target_raises(self):
        df = self._make_df()
        with pytest.raises(ValueError):
            profile_dataset(df, "nonexistent")
