from typing import Any, Dict, List, Tuple, Optional
import os
import traceback

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold,
)

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    VotingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def build_preprocessor(profile: Dict[str, Any]) -> ColumnTransformer:
    num_cols = profile["feature_types"]["numeric"]
    cat_cols = profile["feature_types"]["categorical"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True)),
    ])

    # scikit-learn renamed `sparse` -> `sparse_output` (v1.2+). Support both.
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=30)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )


def select_models(profile: Dict[str, Any], seed: int = 42,) -> List[Tuple[str, Any]]:
    rows = profile["shape"]["rows"]
    cols = profile["shape"]["cols"]
    imb = float(profile.get("imbalance_ratio") or 1.0)
    force_balanced = profile.get("force_balanced_weights", False)
    class_weight = "balanced" if (imb >= 3.0 or force_balanced) else None

    candidates: List[Tuple[str, Any]] = [
        ("DummyMostFrequent", DummyClassifier(strategy="most_frequent")),
        ("LogisticRegression", LogisticRegression(
            max_iter=2000, class_weight=class_weight, random_state=seed,
        )),
        ("RandomForest", RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=-1,
            class_weight=class_weight, max_depth=None,
        )),
        ("ExtraTrees", ExtraTreesClassifier(
            n_estimators=200, random_state=seed, n_jobs=-1,
            class_weight=class_weight,
        )),
    ]

    if rows <= 50000:
        candidates.append(("GradientBoosting", GradientBoostingClassifier(
            n_estimators=150, random_state=seed, max_depth=5,
        )))

    if rows <= 10000:
        candidates.append(("KNN", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)))

    if rows <= 20000 and cols <= 200:
        candidates.append(("AdaBoost", AdaBoostClassifier(
            n_estimators=100, random_state=seed,
        )))

    # SVC can be expensive after one-hot; keep for smaller problems
    if rows <= 10000 and cols <= 100:
        candidates.append(("SVC_RBF", SVC(
            kernel="rbf", probability=True, class_weight=class_weight,
            random_state=seed,
        )))

    return candidates


def train_models(
    df: pd.DataFrame,
    target: str,
    preprocessor: ColumnTransformer,
    candidates: List[Tuple[str, Any]],
    seed: int,
    test_size: float,
    output_dir: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found.")

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # Drop rows with missing target
    mask = ~y.isna()
    X, y = X.loc[mask], y.loc[mask]

    # Stratified split
    stratify = y if (y.nunique(dropna=True) > 1 and y.value_counts().min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify,
    )

    results: List[Dict[str, Any]] = []

    for name, model in candidates:
        if verbose:
            print(f"  [Modelling] Training: {name}")

        try:
            pipe = Pipeline(steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            metrics = _compute_metrics(name, y_test, y_pred)

            # Cross-validation score (3-fold for speed)
            try:
                cv_scores = cross_val_score(
                    pipe, X_train, y_train, cv=3,
                    scoring="balanced_accuracy", n_jobs=-1,
                )
                metrics["cv_mean_bal_acc"] = round(float(cv_scores.mean()), 4)
                metrics["cv_std_bal_acc"] = round(float(cv_scores.std()), 4)
            except Exception:
                metrics["cv_mean_bal_acc"] = None
                metrics["cv_std_bal_acc"] = None

            results.append({
                "name": name,
                "pipeline": pipe,
                "metrics": metrics,
                "X_test": X_test,
                "y_test": y_test,
                "y_pred": y_pred,
                "X_train": X_train,
                "y_train": y_train,
            })

        except Exception as exc:
            if verbose:
                print(f"  [Modelling] FAILED {name}: {exc}")
            continue

    if not results:
        raise RuntimeError("All model candidates failed during training.")

    # Sort by balanced accuracy then F1
    results.sort(
        key=lambda r: (r["metrics"]["balanced_accuracy"], r["metrics"]["f1_macro"]),
        reverse=True,
    )

    return {
        "results": results,
        "best": results[0],
        "all_metrics": [r["metrics"] for r in results],
    }


def build_ensemble(
    training_payload: Dict[str, Any],
    preprocessor: ColumnTransformer,
    seed: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Build a soft-voting ensemble from the top 3 non-dummy models.
    Returns the updated training_payload with the ensemble appended.
    """
    results = training_payload["results"]
    non_dummy = [r for r in results if "Dummy" not in r["name"]]
    top3 = non_dummy[:3]

    if len(top3) < 2:
        if verbose:
            print("  [Ensemble] Not enough models for ensemble.")
        return training_payload

    X_train = top3[0]["X_train"]
    y_train = top3[0]["y_train"]
    X_test = top3[0]["X_test"]
    y_test = top3[0]["y_test"]

    estimators = [(r["name"], r["pipeline"].named_steps["model"]) for r in top3]

    try:
        voting = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
        ens_pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", voting),
        ])
        ens_pipe.fit(X_train, y_train)
        y_pred = ens_pipe.predict(X_test)
        metrics = _compute_metrics("VotingEnsemble", y_test, y_pred)

        if verbose:
            print(f"  [Ensemble] VotingEnsemble bal_acc={metrics['balanced_accuracy']:.3f}")

        entry = {
            "name": "VotingEnsemble",
            "pipeline": ens_pipe,
            "metrics": metrics,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "X_train": X_train,
            "y_train": y_train,
        }
        results.append(entry)
        results.sort(
            key=lambda r: (r["metrics"]["balanced_accuracy"], r["metrics"]["f1_macro"]),
            reverse=True,
        )
        training_payload["best"] = results[0]
        training_payload["all_metrics"] = [r["metrics"] for r in results]
    except Exception as exc:
        if verbose:
            print(f"  [Ensemble] Failed: {exc}")

    return training_payload


def extract_feature_importance(
    best_result: Dict[str, Any],
    profile: Dict[str, Any],
) -> Optional[Dict[str, float]]:
    """
    Extract feature importances from tree-based models or coefficients
    from linear models.  Returns {feature_name: importance} or None.
    """
    try:
        pipe = best_result["pipeline"]
        model = pipe.named_steps["model"]
        preprocess = pipe.named_steps["preprocess"]

        # Get transformed feature names
        try:
            feature_names = preprocess.get_feature_names_out().tolist()
        except Exception:
            feature_names = None

        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)

        if importances is not None:
            if feature_names and len(feature_names) == len(importances):
                imp_dict = {str(n): round(float(v), 6) for n, v in zip(feature_names, importances)}
            else:
                imp_dict = {f"feature_{i}": round(float(v), 6) for i, v in enumerate(importances)}
            # Sort descending
            imp_dict = dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True))
            return imp_dict
    except Exception:
        pass
    return None


def _compute_metrics(name: str, y_true, y_pred) -> Dict[str, Any]:
    return {
        "model": name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }
