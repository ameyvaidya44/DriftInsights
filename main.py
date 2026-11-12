"""
DriftInsights -- CLI driver for the explainable drift-aware adaptive pipeline.

Demonstrates:
  baseline ingest -> M_0 training -> W_before SHAP -> streaming ->
  ADWIN detection -> Delta_E diagnosis -> adaptation -> hot-swap of M_{t+1}

Usage:
    python main.py
    python main.py --model lightgbm --max-samples 2500
    python main.py --legacy                 # previous Pipeline orchestrator
    python main.py --dataset sea --legacy
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.adaptation_engine import AdaptationEngine
from src.config import Config
from src.data_streamer import StreamSimulator
from src.drift_diagnostic import DriftDiagnostic
from src.model_registry import ModelRegistry
from src.pipeline_config import PipelineManager, generate_synthetic_drift_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DriftInsights -- Explainable Drift-Aware Adaptive ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", "-c", default="config/default.yaml")
    parser.add_argument("--dataset", "-d", choices=["sea", "elec2", "synthetic"])
    parser.add_argument(
        "--model",
        "-m",
        choices=["xgboost", "lightgbm", "random_forest"],
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--label-delay", type=int, default=None)
    parser.add_argument("--skip-stream", action="store_true")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Run the original src.pipeline.Pipeline orchestrator",
    )
    return parser.parse_args()


def _load_named_dataset(config: Config, name: str):
    from src.data_handler import DataHandler

    handler = DataHandler(config)
    df = handler.load_dataset(name)
    n = len(df)
    split = int(n * 0.6)
    return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)


def run_framework(args: argparse.Namespace) -> dict:
    config = Config(args.config)
    if args.model:
        config._data["model"]["type"] = args.model

    dataset = args.dataset or "synthetic"
    if dataset in {"sea", "elec2"}:
        df_base, df_stream = _load_named_dataset(config, dataset)
        target_column = "target"
    else:
        df_base, df_stream = generate_synthetic_drift_dataset(
            n_baseline=1800,
            n_stream=3200,
            drift_in_stream=700,
            noise=0.08,
        )
        target_column = "target"

    if args.max_samples:
        df_stream = df_stream.iloc[: args.max_samples].reset_index(drop=True)

    batch_size = args.batch_size or int(config.data.streaming.batch_size)
    label_delay = (
        args.label_delay
        if args.label_delay is not None
        else int(config.data.streaming.label_delay)
    )
    buffer_size = int(config.data.streaming.buffer_size)
    cooldown = int(config.decision.cooldown_samples)

    print("=" * 64)
    print("  DriftInsights -- Explainable Drift-Aware Adaptive Pipeline")
    print("=" * 64)
    print(f"  Dataset:     {dataset}  (baseline={len(df_base)}, stream={len(df_stream)})")
    print(f"  Model:       {config.model.type}")
    print(f"  Batch size:  {batch_size}   label delay: {label_delay}")
    print()

    manager = PipelineManager(config=config, model_type=config.model.type)
    print("[1] Loading baseline data...")
    manager.load_baseline_data(df_base, target_column=target_column)
    print(f"    features: {manager.feature_names}")

    print("[2] Training base model M_0...")
    manager.set_base_model()
    holdout_pred = manager.predict(manager.X_holdout)
    m0_acc = float(np.mean(holdout_pred == manager.y_holdout))
    print(f"    M_0 holdout accuracy: {m0_acc:.4f}")

    print("[3] Computing baseline SHAP (W_before)...")
    rankings = manager.compute_baseline_shap()
    for name, score in list(rankings.items())[:5]:
        print(f"    {name:12s}  |SHAP|={score:.4f}")

    if args.skip_stream:
        return {"skipped_stream": True, "m0_accuracy": m0_acc}

    streamer = StreamSimulator(
        df_stream,
        target_column=target_column,
        batch_size=batch_size,
        label_delay=label_delay,
        buffer_size=buffer_size,
        transform_fn=manager.transform_features,
    )
    diagnostic = DriftDiagnostic(manager)
    adapter = AdaptationEngine(manager)
    registry = ModelRegistry(manager, diagnostic)

    print("[4] Streaming production data...")
    processed = 0
    n_correct = 0
    n_labeled = 0
    last_adapt_idx = -cooldown
    diagnoses: list[dict] = []
    swaps: list[dict] = []

    for batch in streamer:
        y_pred = manager.predict(batch.X)
        pairs = streamer.record_predictions(batch, y_pred)
        processed += len(batch.X)

        for pair in pairs:
            n_labeled += 1
            n_correct += int(pair.y_true == pair.y_pred)
            error = 0 if pair.y_true == pair.y_pred else 1
            event = diagnostic.update(error, pair.index)
            if event is None:
                continue
            if pair.index - last_adapt_idx < cooldown:
                continue

            X_after, y_after = streamer.get_recent_window(diagnostic.window_size)
            if len(X_after) < 20:
                continue

            diagnosis = diagnostic.diagnose(X_after, event)
            diagnoses.append(diagnosis.to_dict())
            print(
                f"    drift @ {diagnosis.sample_idx}  "
                f"Delta_E={diagnosis.delta_e_total:.4f}  "
                f"top={diagnosis.top_features[0].feature if diagnosis.top_features else '-'}"
            )

            candidate = adapter.execute(diagnosis.delta_e_total, X_after, y_after)
            X_h, y_h = streamer.get_holdout_batch(size=min(80, len(y_after)))
            if len(X_h) < 10:
                X_h, y_h = manager.X_holdout, manager.y_holdout

            record = registry.commit(candidate, diagnosis, X_h, y_h, X_after=X_after)
            swaps.append(record)
            last_adapt_idx = pair.index
            print(
                f"    {record['drift_type']} -> {record['strategy']}  "
                f"{record['action']}  acc {record['old_accuracy']:.4f} -> "
                f"{record['new_accuracy']:.4f}  v{record['model_version']}"
            )

        if processed % 500 == 0 and processed > 0:
            acc = n_correct / n_labeled if n_labeled else 0.0
            print(f"    progress {processed}/{len(df_stream)}  stream_acc={acc:.4f}")

    stream_acc = n_correct / n_labeled if n_labeled else 0.0
    summary = {
        "dataset": dataset,
        "model_type": manager.model_type,
        "baseline_rows": len(df_base),
        "stream_rows": processed,
        "labeled_pairs": n_labeled,
        "stream_accuracy": round(stream_acc, 4),
        "m0_holdout_accuracy": round(m0_acc, 4),
        "drift_events": len(diagnoses),
        "hot_swaps": sum(1 for r in swaps if r.get("action") == "hot_swap"),
        "rejected": sum(1 for r in swaps if r.get("action") == "rejected"),
        "model_version": manager.model_version,
        "audit_log": str(registry.audit_path),
    }

    print()
    print("=" * 64)
    print("  PIPELINE SUMMARY")
    print("=" * 64)
    for key, value in summary.items():
        print(f"  {key:24s} {value}")
    print("=" * 64)
    return summary


def run_legacy(args: argparse.Namespace) -> None:
    from src.pipeline import Pipeline

    config = Config(args.config)
    if args.dataset and args.dataset != "synthetic":
        config._data["data"]["dataset"] = args.dataset
    if args.model:
        config._data["model"]["type"] = args.model

    pipeline = Pipeline(config=config)
    try:
        print("[>] Setting up legacy pipeline...")
        pipeline.setup()
        print("[>] Training base model...")
        metrics = pipeline.train()
        print(f"  [OK] Accuracy: {metrics['accuracy']:.4f}")
        if not args.skip_stream:
            print("[>] Starting streaming loop...")
            summary = pipeline.run_stream(max_samples=args.max_samples)
            print(f"  Samples: {summary['total_samples']}  drifts: {summary['drift_events']}")
    finally:
        pipeline.shutdown()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        if args.legacy:
            run_legacy(args)
        else:
            run_framework(args)
    except KeyboardInterrupt:
        print("\n[!] Pipeline interrupted by user")
    except Exception as exc:
        logging.getLogger("driftinsights").error("Pipeline error: %s", exc, exc_info=True)
        print(f"\n[X] Pipeline error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
