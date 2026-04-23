from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd



def infer_target_column(df: pd.DataFrame) -> Optional[str]:
    """
    Heuristic target inference:
      - prefer common target-like column names
      - else last column if it has relatively low cardinality
    """
    candidates = ["target", "label", "class", "y", "outcome", "diagnosis", "species"]
    lower_map = {c.lower().strip(): c for c in df.columns}
    for k in candidates:
        if k in lower_map:
            return lower_map[k]

    last = df.columns[-1]
    uniq = df[last].nunique(dropna=True)
    n = len(df)
    if n > 0 and (uniq <= 50 or (uniq / max(n, 1) < 0.05)):
        return last
    return None


def is_classification_target(series: pd.Series) -> bool:
    if series.dtype == "object" or str(series.dtype).startswith("category"):
        return True
    uniq = series.nunique(dropna=True)
    return uniq <= 50



def dataset_fingerprint(df: pd.DataFrame, target: str) -> str:
    cols = ",".join(df.columns.astype(str).tolist())
    shape = f"{df.shape[0]}x{df.shape[1]}"
    base = f"{shape}|{target}|{cols}"
    h = abs(hash(base)) % (10**12)
    return f"fp_{h}"



def profile_dataset(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset columns.")

    y = df[target]
    profile: Dict[str, Any] = {}

    profile["shape"] = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    profile["columns"] = df.columns.astype(str).tolist()

    missing = (df.isna().mean() * 100).round(2).to_dict()
    profile["missing_pct"] = {str(k): float(v) for k, v in missing.items()}

    profile["target"] = str(target)
    profile["target_dtype"] = str(y.dtype)
    profile["is_classification"] = bool(is_classification_target(y))

    # Feature types
    X = df.drop(columns=[target])
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.astype(str).tolist()
    cat_cols = [c for c in X.columns.astype(str).tolist() if c not in numeric_cols]

    profile["feature_types"] = {"numeric": numeric_cols, "categorical": cat_cols}
    profile["n_unique_by_col"] = {str(c): int(df[c].nunique(dropna=True)) for c in df.columns.astype(str)}

    # Numeric summary
    num_df = df[numeric_cols] if numeric_cols else pd.DataFrame()
    if not num_df.empty:
        desc = num_df.describe().T
        profile["numeric_summary"] = {
            col: {
                "mean": round(float(desc.loc[col, "mean"]), 4) if col in desc.index else None,
                "std": round(float(desc.loc[col, "std"]), 4) if col in desc.index else None,
                "min": round(float(desc.loc[col, "min"]), 4) if col in desc.index else None,
                "max": round(float(desc.loc[col, "max"]), 4) if col in desc.index else None,
            }
            for col in numeric_cols if col in desc.index
        }
        # Skewness
        skew = num_df.skew(numeric_only=True)
        profile["skewness"] = {str(k): round(float(v), 4) for k, v in skew.items()}
    else:
        profile["numeric_summary"] = {}
        profile["skewness"] = {}

    # Correlation analysis
    if len(numeric_cols) >= 2:
        corr = num_df.corr(numeric_only=True)

        high_corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                r = abs(corr.iloc[i, j])
                if r > 0.85:
                    high_corr_pairs.append({
                        "col_a": str(corr.columns[i]),
                        "col_b": str(corr.columns[j]),
                        "abs_corr": round(float(r), 4),
                    })
        profile["high_correlation_pairs"] = high_corr_pairs
    else:
        profile["high_correlation_pairs"] = []

    # Outlier detection (IQR)
    outlier_counts: Dict[str, int] = {}
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < lower) | (s > upper)).sum())
        if n_out > 0:
            outlier_counts[col] = n_out
    profile["outlier_counts"] = outlier_counts

    # Constant / near-constant columns
    constant_cols = [
        str(c) for c in df.columns
        if df[c].nunique(dropna=True) <= 1
    ]
    profile["constant_cols"] = constant_cols

    # Richer Notes Generation
    notes: List[str] = []
    if profile["shape"]["rows"] < 1000:
        notes.append("Small dataset (<1k rows): prefer simpler models / guard against overfitting.")
    if profile["shape"]["cols"] > 100:
        notes.append("High dimensionality (>100 cols): watch one-hot expansion and overfitting.")
    if constant_cols:
        notes.append(f"Constant columns detected ({len(constant_cols)}): consider dropping them.")
    if profile["high_correlation_pairs"]:
        notes.append(f"{len(profile['high_correlation_pairs'])} highly correlated feature pair(s) (>0.85).")
    if outlier_counts:
        total_out = sum(outlier_counts.values())
        notes.append(f"Outliers detected: {total_out} total IQR-flagged values across {len(outlier_counts)} column(s).")
    profile["notes"] = notes

    if profile["is_classification"]:
        vc = y.value_counts(dropna=False)
        profile["class_counts"] = {str(k): int(v) for k, v in vc.items()}
        if len(vc) >= 2:
            ratio = float(vc.max() / max(vc.min(), 1))
        else:
            ratio = 1.0
        profile["imbalance_ratio"] = round(ratio, 3)
        profile["n_classes"] = int(len(vc))
        if ratio >= 3.0:
            notes.append("Imbalance detected (ratio >= 3.0): prioritise macro metrics / balanced accuracy.")
    else:
        profile["class_counts"] = None
        profile["imbalance_ratio"] = None
        profile["n_classes"] = None
        notes.append("Non-classification target: this template focuses on classification.")

    return profile