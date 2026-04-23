import os
import json
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report



def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)



def plot_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(max(4, len(labels)), max(4, len(labels))))
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    plt.colorbar(ax.images[0], ax=ax)
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)

    thresh = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(int(cm[i, j]), "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(all_metrics: List[Dict[str, Any]], out_path: str) -> None:
    """Bar chart comparing balanced accuracy and F1 for all models."""
    names = [m["model"] for m in all_metrics]
    bal_acc = [m.get("balanced_accuracy", 0) for m in all_metrics]
    f1_macro = [m.get("f1_macro", 0) for m in all_metrics]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 4))
    bars1 = ax.bar(x - width / 2, bal_acc, width, label="Balanced Accuracy", color="#4c72b0")
    bars2 = ax.bar(x + width / 2, f1_macro, width, label="Macro F1", color="#dd8452")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="grey", linestyle="--", linewidth=0.7, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(
    importance: Dict[str, float], out_path: str, top_n: int = 15,
) -> None:
    """Horizontal bar chart of top-N feature importances."""
    if not importance:
        return
    items = list(importance.items())[:top_n]
    items.reverse()  # so most important is at top
    names = [n for n, _ in items]
    vals = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(6, max(3, len(names) * 0.35)))
    ax.barh(names, vals, color="#55a868")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {min(top_n, len(importance))} Feature Importances")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)



def evaluate_best(training_payload: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    best = training_payload["best"]
    all_metrics = training_payload["all_metrics"]

    y_test = best["y_test"]
    y_pred = best["y_pred"]

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted([str(x) for x in y_test.dropna().unique().tolist()])
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, labels, cm_path, f"Confusion Matrix: {best['name']}")

    # Classification report (dict for JSON)
    cls_report_str = classification_report(y_test, y_pred, zero_division=0)
    cls_report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    # Model comparison plot
    comp_path = os.path.join(output_dir, "model_comparison.png")
    plot_model_comparison(all_metrics, comp_path)

    return {
        "best_metrics": best["metrics"],
        "all_metrics": all_metrics,
        "confusion_matrix_path": cm_path,
        "model_comparison_path": comp_path,
        "classification_report": cls_report_str,
        "classification_report_dict": cls_report_dict,
    }



def write_markdown_report(
    out_path: str,
    ctx: Any,
    fingerprint: str,
    dataset_profile: Dict[str, Any],
    plan: List[str],
    eval_payload: Dict[str, Any],
    reflection: Dict[str, Any],
    feature_importance: Optional[Dict[str, float]] = None,
) -> None:
    best = eval_payload["best_metrics"]

    def short_list(xs: List[str], n: int = 12) -> str:
        return ", ".join(xs[:n]) + (" ..." if len(xs) > n else "")

    numeric = dataset_profile.get("feature_types", {}).get("numeric", [])
    categorical = dataset_profile.get("feature_types", {}).get("categorical", [])
    notes = dataset_profile.get("notes", [])
    class_counts = dataset_profile.get("class_counts", {})

    # Build plan justification
    plan_rationale = _build_plan_rationale(dataset_profile, plan)

    md = f"""# Agentic Data Scientist Report

**Run ID:** `{ctx.run_id}`  
**Started (UTC):** {ctx.started_at}  
**Dataset:** `{ctx.data_path}`  
**Target:** `{ctx.target}`  
**Fingerprint:** `{fingerprint}`  

---

## 1. Dataset Profile

| Property | Value |
|---|---|
| Rows | {dataset_profile["shape"]["rows"]} |
| Columns | {dataset_profile["shape"]["cols"]} |
| Classification | {dataset_profile.get("is_classification")} |
| Imbalance ratio | {dataset_profile.get("imbalance_ratio")} |
| Number of classes | {dataset_profile.get("n_classes", "N/A")} |

**Feature Types**
- Numeric ({len(numeric)}): {short_list(numeric)}
- Categorical ({len(categorical)}): {short_list(categorical)}

**Class Distribution**
"""
    if class_counts:
        for cls, cnt in class_counts.items():
            md += f"- `{cls}`: {cnt}\n"
    else:
        md += "- (not applicable)\n"

    md += f"""
**Data Quality Notes**
"""
    for n in notes:
        md += f"- {n}\n"

    md += f"""
---

## 2. Execution Plan

{chr(10).join([f"{i+1}. `{t}`" for i, t in enumerate(plan)])}

**Plan Rationale:** {plan_rationale}

---

## 3. Results

### Best Model: `{best.get("model")}`

| Metric | Score |
|---|---|
| Accuracy | {best.get("accuracy", 0):.4f} |
| Balanced Accuracy | {best.get("balanced_accuracy", 0):.4f} |
| Macro F1 | {best.get("f1_macro", 0):.4f} |
| Macro Precision | {best.get("precision_macro", 0):.4f} |
| Macro Recall | {best.get("recall_macro", 0):.4f} |

### All Candidates

| Model | Bal. Acc | F1 Macro | Accuracy |
|---|---|---|---|
"""
    for m in eval_payload.get("all_metrics", []):
        md += f"| {m['model']} | {m.get('balanced_accuracy',0):.4f} | {m.get('f1_macro',0):.4f} | {m.get('accuracy',0):.4f} |\n"

    md += f"""
### Classification Report

```
{eval_payload.get("classification_report", "")}
```

---

## 4. Reflection

**Status:** {reflection.get("status", "N/A")}

**Issues identified:**
"""
    for issue in reflection.get("issues", []):
        md += f"- {issue}\n"
    if not reflection.get("issues"):
        md += "- None\n"

    md += """
**Suggestions:**
"""
    for sug in reflection.get("suggestions", []):
        md += f"- {sug}\n"
    if not reflection.get("suggestions"):
        md += "- None\n"

    md += f"""
**Replan recommended:** {reflection.get("replan_recommended", False)}  
**Replan strategy:** {reflection.get("replan_strategy", "none")}

---

## 5. Ethical Considerations

- The agent operates fully offline. No data leaves the local machine.
- Model decisions should be audited before deployment in any real-world setting.
- Class imbalance handling is automated but should be reviewed for fairness implications.
- Feature importance should be inspected for potential proxy discrimination.

---

## 6. Artefacts

- Confusion matrix: `confusion_matrix.png`
- Model comparison: `model_comparison.png`
"""
    if feature_importance:
        md += "- Feature importance: `feature_importance.png`\n"

    md += f"""
---

*Generated by the Agentic Data Scientist pipeline.*
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)


def _build_plan_rationale(profile: Dict[str, Any], plan: List[str]) -> str:
    """Generate a human-readable explanation of why this plan was chosen."""
    parts: List[str] = []
    rows = profile.get("shape", {}).get("rows", 0)
    imb = float(profile.get("imbalance_ratio") or 1.0)

    if rows < 1000:
        parts.append("small dataset size triggered cross-validation and simpler model preference")
    elif rows > 50_000:
        parts.append("large dataset triggered sub-sampling for profiling")

    if imb >= 3.0:
        parts.append("class imbalance detected, so balanced weights and threshold tuning were added")

    if "apply_feature_selection" in plan:
        parts.append("high dimensionality triggered feature selection steps")

    if "handle_severe_missing_data" in plan:
        parts.append("severe missing data triggered advanced imputation")

    if not parts:
        parts.append("standard pipeline: no special adaptations were needed")

    return "; ".join(parts) + "."
