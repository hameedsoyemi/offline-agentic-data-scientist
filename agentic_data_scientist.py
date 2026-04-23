# Orchestrator for an "agentic" offline data scientist pipeline.
# Handles dataset loading, profiling, planning, training, evaluation, reflection,
# and optional re-planning cycles. Designed primarily for classification tasks.


import os
import json
import uuid
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

# Agent components and tooling used by the orchestrator
from agents.planner import create_plan, estimate_plan_cost
from agents.reflector import reflect, should_replan, apply_replan_strategy
from agents.memory import JSONMemory
from tools.data_profiler import profile_dataset, infer_target_column, dataset_fingerprint
from tools.modelling import (
    build_preprocessor,
    select_models,
    train_models,
    build_ensemble,
    extract_feature_importance,
)
from tools.evaluation import (
    evaluate_best,
    write_markdown_report,
    save_json,
    plot_feature_importance,
)

# Lightweight container for run metadata and parameters
@dataclass
class RunContext:
    run_id: str
    started_at: str
    data_path: str
    target: str
    output_dir: str
    seed: int
    test_size: float
    max_replans: int


def now_iso() -> str:
    """Return current UTC time in ISO 8601 format (no microseconds) with Z suffix."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class AgenticDataScientist:
    def __init__(self, memory_path: str = "agent_memory.json", verbose: bool = True):
        # Verbose controls logging output
        self.verbose = verbose
        # Simple persistent memory used to remember prior runs for a dataset fingerprint
        self.memory = JSONMemory(memory_path)

        # Context and transient state populated when run() is executed
        self.ctx: Optional[RunContext] = None
        self.state: Dict[str, Any] = {}
        self._timings: Dict[str, float] = {}


    def log(self, msg: str) -> None:
        """Print a log message when verbose is enabled."""
        if self.verbose:
            print(f"[AgenticDataScientist] {msg}")

    def _start_timer(self, label: str) -> None:
        self._timings[label] = time.time()

    def _stop_timer(self, label: str) -> float:
        elapsed = time.time() - self._timings.pop(label, time.time())
        self.log(f" {label}: {elapsed:.1f}s")
        return elapsed


    def load_data(self, path: str) -> pd.DataFrame:
        """Print a log message when verbose is enabled."""
        self.log(f"Loading dataset: {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")
        df = pd.read_csv(path)
        self.log(f"Loaded {df.shape[0]} rows x {df.shape[1]} cols")
        return df


    def run(
        self,
        data_path: str,
        target: str,
        output_root: str = "outputs",
        seed: int = 42,
        test_size: float = 0.2,
        max_replans: int = 1,
    ) -> str:
        run_start = time.time()

        # Create unique run id and output directory
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        output_dir = os.path.join(output_root, run_id)
        os.makedirs(output_dir, exist_ok=True)

        # Populate run context with parameters and metadata
        self.ctx = RunContext(
            run_id=run_id, started_at=now_iso(), data_path=data_path,
            target=target, output_dir=output_dir, seed=seed,
            test_size=test_size, max_replans=max_replans,
        )
        # Internal state used to track replanning attempts
        self.state = {"replan_count": 0, "stage_timings": {}}

        self.log(f" Run {run_id} ")

        # Load dataset into memory
        self._start_timer("load_data")
        df = self.load_data(data_path)
        self.state["stage_timings"]["load_data"] = self._stop_timer("load_data")

        # If client requested auto target detection, infer it from data
        if target.strip().lower() == "auto":
            inferred = infer_target_column(df)
            if not inferred:
                raise ValueError("Could not infer target column. Please provide --target <name>.")
            # Update context with inferred target name
            self.ctx.target = inferred
            self.log(f"Inferred target column: '{inferred}'")

        # Produce a dataset profile (EDA summary) and a fingerprint used for memory
        self._start_timer("profile")
        profile = profile_dataset(df, self.ctx.target)
        fp = dataset_fingerprint(df, self.ctx.target)
        self.state["stage_timings"]["profile"] = self._stop_timer("profile")
        self.log(f"Dataset fingerprint: {fp}")
        self.log(f"Shape: {profile['shape']}, Classes: {profile.get('n_classes', '?')}, "
                 f"Imbalance: {profile.get('imbalance_ratio', 'N/A')}")

        # Look up previous runs for the same dataset fingerprint (memory hint)
        prev = self.memory.get_dataset_record(fp)
        if prev:
            self.log(f"Memory hit: previously best={prev.get('best_model')} for fp={fp}")
        else:
            self.log("No memory record for this dataset fingerprint.")

        # Create an initial plan informed by the profile and optional memory hint
        self._start_timer("planning")
        plan = create_plan(profile, memory_hint=prev)
        cost = estimate_plan_cost(plan, profile["shape"]["rows"])
        self.state["stage_timings"]["planning"] = self._stop_timer("planning")
        self.log(f"Plan ({len(plan)} steps, est. cost={cost}): {plan}")

        # Execution loop: trains and evaluates, then optionally replans and repeats
        feature_importance = None

        while True:
            # Build preprocessing pipeline tailored to the profile
            self._start_timer("preprocessing")
            preprocessor = build_preprocessor(profile)
            self._stop_timer("preprocessing")

            # Choose candidate models to try based on the profile
            self._start_timer("training")
            candidates = select_models(profile, seed=self.ctx.seed)
            self.log(f"Candidate models: {[n for n, _ in candidates]}")

            # Train candidate models and persist intermediate artefacts
            try:
                results = train_models(
                    df=df, target=self.ctx.target, preprocessor=preprocessor,
                    candidates=candidates, seed=self.ctx.seed,
                    test_size=self.ctx.test_size, output_dir=self.ctx.output_dir,
                    verbose=self.verbose,
                )
            except RuntimeError as e:
                self.log(f"Training failed: {e}")
                break
            self.state["stage_timings"]["training"] = self._stop_timer("training")

            # Evaluates the response from the reflector and ensembles if necessary
            if "train_ensemble" in plan:
                self._start_timer("ensemble")
                self.log("Building voting ensemble...")
                results = build_ensemble(results, preprocessor, self.ctx.seed, self.verbose)
                self._stop_timer("ensemble")

            # Evaluate the trained models and pick the best one
            self._start_timer("evaluation")
            eval_payload = evaluate_best(results, output_dir=self.ctx.output_dir)
            self.state["stage_timings"]["evaluation"] = self._stop_timer("evaluation")

            self.log(f"Best model: {eval_payload['best_metrics']['model']} "
                     f"(bal_acc={eval_payload['best_metrics']['balanced_accuracy']:.3f}, "
                     f"f1={eval_payload['best_metrics']['f1_macro']:.3f})")

            # Decides which features matter most to the best model selected
            feature_importance = extract_feature_importance(results["best"], profile)
            if feature_importance:
                fi_path = os.path.join(self.ctx.output_dir, "feature_importance.png")
                plot_feature_importance(feature_importance, fi_path)
                save_json(
                    os.path.join(self.ctx.output_dir, "feature_importance.json"),
                    feature_importance,
                )
                self.log(f"Top feature: {list(feature_importance.keys())[0]}")

            # Reflect on the evaluation in the context of the dataset profile
            self._start_timer("reflection")
            reflection = reflect(
                dataset_profile=profile,
                evaluation=eval_payload["best_metrics"],
                all_metrics=eval_payload["all_metrics"],
            )
            self.state["stage_timings"]["reflection"] = self._stop_timer("reflection")

            if reflection["issues"]:
                self.log(f"Reflection issues: {reflection['issues']}")
            else:
                self.log("Reflection: no issues found.")

            # Persist core run artefacts for later review
            save_json(os.path.join(self.ctx.output_dir, "eda_summary.json"), profile)
            save_json(os.path.join(self.ctx.output_dir, "plan.json"), {
                "plan": plan, "cost_estimate": cost,
            })
            save_json(os.path.join(self.ctx.output_dir, "metrics.json"), eval_payload)
            save_json(os.path.join(self.ctx.output_dir, "reflection.json"), reflection)

            # Generate a human-readable markdown report summarising the run
            write_markdown_report(
                out_path=os.path.join(self.ctx.output_dir, "report.md"),
                ctx=self.ctx, fingerprint=fp, dataset_profile=profile,
                plan=plan, eval_payload=eval_payload, reflection=reflection,
                feature_importance=feature_importance,
            )

            # Update the memory store with outcomes from this run
            self.memory.upsert_dataset_record(fp, {
                "last_seen": now_iso(),
                "target": self.ctx.target,
                "shape": profile["shape"],
                "best_model": eval_payload["best_metrics"]["model"],
                "best_metrics": eval_payload["best_metrics"],
                "imbalance_ratio": profile.get("imbalance_ratio"),
            })

            # Decide whether the agent should attempt to re-plan and re-run
            if not should_replan(reflection):
                self.log("No replan needed, finishing run.")
                break

            # If we've already replanned the allowed number of times, stop
            if self.state["replan_count"] >= self.ctx.max_replans:
                self.log("Replan suggested but max_replans reached. Stopping.")
                break

            # Otherwise, increment replan counter and apply the replan strategy
            self.state["replan_count"] += 1
            strategy = reflection.get("replan_strategy", "unknown")
            self.log(f"Replanning #{self.state['replan_count']} with strategy '{strategy}'...")

            # apply_replan_strategy returns an updated (plan, profile) pair
            prev_best = eval_payload["best_metrics"].get("balanced_accuracy", 0)
            plan, profile = apply_replan_strategy(plan, profile, reflection)
            self.log(f"Updated plan: {plan}")

            # Log strategy for meta-learning
            self.memory.log_strategy(fp, strategy, "attempted")

        # Final log and return the directory containing run outputs
        total_time = time.time() - run_start
        self.log(f" Done in {total_time:.1f}s. Output → {self.ctx.output_dir} ")
        return self.ctx.output_dir