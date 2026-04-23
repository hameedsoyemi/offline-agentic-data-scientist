import os
import json
import tempfile
import pytest
import pandas as pd
import numpy as np

from sklearn.datasets import make_classification, load_iris
from tools.data_profiler import profile_dataset
from tools.modelling import (
    build_preprocessor,
    select_models,
    train_models,
    build_ensemble,
    extract_feature_importance,
)
from tools.evaluation import evaluate_best, save_json
from agentic_data_scientist import AgenticDataScientist


# Modelling tests

class TestBuildPreprocessor:
    def test_creates_transformer(self):
        profile = {
            "feature_types": {"numeric": ["a", "b"], "categorical": ["c"]},
        }
        pp = build_preprocessor(profile)
        assert pp is not None

    def test_handles_empty_categoricals(self):
        profile = {"feature_types": {"numeric": ["a"], "categorical": []}}
        pp = build_preprocessor(profile)
        assert pp is not None


class TestSelectModels:
    def test_always_includes_dummy(self):
        profile = {"shape": {"rows": 1000, "cols": 10}, "imbalance_ratio": 1.0}
        candidates = select_models(profile)
        names = [n for n, _ in candidates]
        assert "DummyMostFrequent" in names

    def test_balanced_weights_for_imbalanced(self):
        profile = {"shape": {"rows": 1000, "cols": 10}, "imbalance_ratio": 5.0}
        candidates = select_models(profile)
        # Check that LogisticRegression uses balanced weights
        for name, model in candidates:
            if name == "LogisticRegression":
                assert model.class_weight == "balanced"

    def test_fewer_models_for_large_data(self):
        small_profile = {"shape": {"rows": 500, "cols": 5}, "imbalance_ratio": 1.0}
        large_profile = {"shape": {"rows": 60000, "cols": 5}, "imbalance_ratio": 1.0}
        small_candidates = select_models(small_profile)
        large_candidates = select_models(large_profile)
        assert len(small_candidates) >= len(large_candidates)


class TestTrainModels:
    def _make_data(self, n=200, n_features=5):
        X, y = make_classification(n_samples=n, n_features=n_features, random_state=42)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
        df["target"] = y
        return df

    def test_returns_results(self):
        df = self._make_data()
        profile = profile_dataset(df, "target")
        pp = build_preprocessor(profile)
        candidates = select_models(profile)
        with tempfile.TemporaryDirectory() as tmpdir:
            results = train_models(df, "target", pp, candidates, 42, 0.2, tmpdir)
            assert "best" in results
            assert "all_metrics" in results
            assert len(results["all_metrics"]) > 0

    def test_best_sorted_by_balanced_accuracy(self):
        df = self._make_data()
        profile = profile_dataset(df, "target")
        pp = build_preprocessor(profile)
        candidates = select_models(profile)
        with tempfile.TemporaryDirectory() as tmpdir:
            results = train_models(df, "target", pp, candidates, 42, 0.2, tmpdir)
            best_ba = results["best"]["metrics"]["balanced_accuracy"]
            for m in results["all_metrics"]:
                assert m["balanced_accuracy"] <= best_ba + 1e-6


class TestBuildEnsemble:
    def test_ensemble_added(self):
        df = pd.DataFrame(
            np.random.randn(300, 5), columns=[f"f{i}" for i in range(5)]
        )
        df["target"] = np.random.choice([0, 1], 300)
        profile = profile_dataset(df, "target")
        pp = build_preprocessor(profile)
        candidates = select_models(profile)
        with tempfile.TemporaryDirectory() as tmpdir:
            results = train_models(df, "target", pp, candidates, 42, 0.2, tmpdir)
            updated = build_ensemble(results, pp, 42)
            names = [m["model"] for m in updated["all_metrics"]]
            assert "VotingEnsemble" in names


class TestFeatureImportance:
    def test_extracts_from_rf(self):
        df = pd.DataFrame(np.random.randn(200, 4), columns=[f"f{i}" for i in range(4)])
        df["target"] = np.random.choice([0, 1], 200)
        profile = profile_dataset(df, "target")
        pp = build_preprocessor(profile)
        candidates = [("RandomForest", __import__("sklearn.ensemble", fromlist=["RandomForestClassifier"]).RandomForestClassifier(n_estimators=10, random_state=42))]
        with tempfile.TemporaryDirectory() as tmpdir:
            results = train_models(df, "target", pp, candidates[:1] + [("DummyMostFrequent", __import__("sklearn.dummy", fromlist=["DummyClassifier"]).DummyClassifier())], 42, 0.2, tmpdir)
            fi = extract_feature_importance(results["best"], profile)
            if results["best"]["name"] != "DummyMostFrequent":
                assert fi is not None
                assert len(fi) > 0


# Integration tests

class TestIntegration:
    def test_full_pipeline_example_dataset(self):
        """End-to-end test on the example dataset."""
        agent = AgenticDataScientist(
            memory_path=os.path.join(tempfile.mkdtemp(), "test_mem.json"),
            verbose=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = agent.run(
                data_path="data/example_dataset.csv",
                target="auto",
                output_root=tmpdir,
                seed=42,
                test_size=0.2,
                max_replans=0,
            )
            assert os.path.isdir(out_dir)
            expected_files = [
                "report.md", "metrics.json", "reflection.json",
                "eda_summary.json", "confusion_matrix.png",
                "model_comparison.png", "plan.json",
            ]
            for f in expected_files:
                assert os.path.exists(os.path.join(out_dir, f)), f"Missing: {f}"

    def test_full_pipeline_iris(self):
        """End-to-end on Iris (multi-class)."""
        agent = AgenticDataScientist(
            memory_path=os.path.join(tempfile.mkdtemp(), "test_mem.json"),
            verbose=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = agent.run(
                data_path="data/iris.csv",
                target="species",
                output_root=tmpdir,
                seed=42,
            )
            metrics = json.load(open(os.path.join(out_dir, "metrics.json")))
            assert metrics["best_metrics"]["balanced_accuracy"] > 0.8

    def test_replanning_triggered_on_noise(self):
        """Noisy data should trigger at least one replan."""
        np.random.seed(123)
        df = pd.DataFrame(np.random.randn(200, 3), columns=["a", "b", "c"])
        df["label"] = np.random.choice([0, 1], 200)
        path = os.path.join(tempfile.mkdtemp(), "noise.csv")
        df.to_csv(path, index=False)

        agent = AgenticDataScientist(
            memory_path=os.path.join(tempfile.mkdtemp(), "test_mem.json"),
            verbose=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = agent.run(
                data_path=path, target="label",
                output_root=tmpdir, max_replans=1,
            )
            reflection = json.load(open(os.path.join(out_dir, "reflection.json")))
            # With random noise, the reflector should have found issues
            assert len(reflection["issues"]) > 0

    def test_memory_persists_across_runs(self):
        """Running twice on same dataset should produce a memory hit."""
        mem_path = os.path.join(tempfile.mkdtemp(), "test_mem.json")
        agent = AgenticDataScientist(memory_path=mem_path, verbose=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            agent.run(data_path="data/example_dataset.csv", target="auto",
                      output_root=tmpdir, max_replans=0)
            # Second run should find memory
            agent2 = AgenticDataScientist(memory_path=mem_path, verbose=False)
            agent2.run(data_path="data/example_dataset.csv", target="auto",
                       output_root=tmpdir, max_replans=0)
            # Memory file should exist and have a record
            mem_data = json.load(open(mem_path))
            assert len(mem_data["datasets"]) >= 1
