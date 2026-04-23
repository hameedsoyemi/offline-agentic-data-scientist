import json
import os
import shutil
from typing import Any, Dict, List, Optional
from datetime import datetime


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class JSONMemory:
    def __init__(self, path: str = "agent_memory.json"):
        self.path = path
        self.data: Dict[str, Any] = {
            "datasets": {},
            "notes": [],
            "global_strategy_log": [],
        }
        self._load()


    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                # Merge in any missing top-level keys
                for key in ("datasets", "notes", "global_strategy_log"):
                    self.data.setdefault(key, loaded.get(key, self.data[key]))
                self.data.update(loaded)
        except Exception:
            backup = self.path + ".bak"
            if os.path.exists(self.path):
                shutil.copy(self.path, backup)
            self.data = {
                "datasets": {},
                "notes": [{"ts": now_iso(), "msg": f"Memory reset; backup at {backup}"}],
                "global_strategy_log": [],
            }

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)


    def get_dataset_record(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        return self.data.get("datasets", {}).get(fingerprint)

    def upsert_dataset_record(self, fingerprint: str, record: Dict[str, Any]) -> None:
        existing = self.data.setdefault("datasets", {}).get(fingerprint, {})
        # Preserve run_history
        history: List[Dict[str, Any]] = existing.get("run_history", [])
        history.append({
            "timestamp": now_iso(),
            "model": record.get("best_model", "unknown"),
            "bal_acc": record.get("best_metrics", {}).get("balanced_accuracy", 0),
            "f1_macro": record.get("best_metrics", {}).get("f1_macro", 0),
        })
        record["run_history"] = history
        self.data["datasets"][fingerprint] = record
        self.save()


    def add_note(self, msg: str) -> None:
        self.data.setdefault("notes", []).append({"ts": now_iso(), "msg": msg})
        self.save()


    def log_strategy(self, fingerprint: str, strategy: str, outcome: str) -> None:
        self.data.setdefault("global_strategy_log", []).append({
            "ts": now_iso(),
            "fingerprint": fingerprint,
            "strategy": strategy,
            "outcome": outcome,
        })
        self.save()


    def find_similar(
        self,
        shape: Dict[str, int],
        imbalance_ratio: float,
        numeric_frac: float,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
 
        import math

        target_vec = (
            math.log1p(shape.get("rows", 0)),
            math.log1p(shape.get("cols", 0)),
            imbalance_ratio,
            numeric_frac,
        )

        scored: List[tuple] = []
        for fp, rec in self.data.get("datasets", {}).items():
            s = rec.get("shape", {})
            ir = rec.get("best_metrics", {}).get("balanced_accuracy", 0.5)
            # We don't store numeric_frac in old records so default to 0.5
            vec = (
                math.log1p(s.get("rows", 0)),
                math.log1p(s.get("cols", 0)),
                float(rec.get("imbalance_ratio", 1.0) if "imbalance_ratio" in rec else 1.0),
                0.5,
            )
            dist = sum((a - b) ** 2 for a, b in zip(target_vec, vec)) ** 0.5
            scored.append((dist, rec))

        scored.sort(key=lambda x: x[0])
        return [rec for _, rec in scored[:top_k]]


    def get_successful_strategies(self) -> List[str]:
        return [
            entry["strategy"]
            for entry in self.data.get("global_strategy_log", [])
            if entry.get("outcome") == "improved"
        ]
