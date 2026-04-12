"""
DriftInsights -- CLI Entrypoint

Run the full ML pipeline from the command line.

Usage:
    python main.py                      # Run full pipeline with defaults
    python main.py --config custom.yaml # Use custom config
    python main.py --dataset sea        # Override dataset
    python main.py --model lightgbm     # Override model type
    python main.py --max-samples 5000   # Limit streaming samples
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.pipeline import Pipeline
from src.config import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description="DriftInsights -- Explainable & Adaptive ML System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --config config/default.yaml --dataset elec2
  python main.py --model xgboost --max-samples 10000
        """,
    )
    parser.add_argument(
        "--config", "-c",
        default="config/default.yaml",
        help="Path to YAML config file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--dataset", "-d",
        choices=["sea", "elec2"],
        help="Override dataset (default: from config)",
    )
    parser.add_argument(
        "--model", "-m",
        choices=["xgboost", "lightgbm", "random_forest"],
        help="Override model type (default: from config)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of streaming samples (default: all)",
    )
    parser.add_argument(
        "--skip-stream",
        action="store_true",
        help="Only train the model, skip streaming",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  DriftInsights -- Explainable & Adaptive ML System")
    print("=" * 60)
    print()

    # Load config
    config = Config(args.config)

    # Apply CLI overrides
    if args.dataset:
        config._data["data"]["dataset"] = args.dataset
    if args.model:
        config._data["model"]["type"] = args.model

    # Create and run pipeline
    pipeline = Pipeline(config=config)

    try:
        # Phase 1: Setup
        print("[>] Setting up pipeline...")
        pipeline.setup()
        print(f"  [OK] Dataset: {config.data.dataset}")
        print(f"  [OK] Model: {config.model.type}")
        print(f"  [OK] Session: {pipeline.session_id}")
        print()

        # Phase 2: Train
        print("[>] Training base model...")
        metrics = pipeline.train()
        print(f"  [OK] Accuracy: {metrics['accuracy']:.4f}")
        print(f"  [OK] F1 Score: {metrics['f1']:.4f}")
        roc = metrics.get('roc_auc')
        print(f"  [OK] ROC-AUC:  {f'{roc:.4f}' if roc else 'N/A'}")
        print()

        if not args.skip_stream:
            # Phase 3-5: Stream with drift detection + adaptation
            print("[>] Starting streaming loop...")
            print("  (drift detection + XAI + adaptation active)")
            print()

            summary = pipeline.run_stream(max_samples=args.max_samples)

            # Print summary
            print()
            print("=" * 60)
            print("  PIPELINE SUMMARY")
            print("=" * 60)
            fm = summary["final_metrics"]
            print(f"  Samples Processed:      {summary['total_samples']}")
            print(f"  Elapsed Time:           {summary['elapsed_seconds']}s")
            print(f"  Running Accuracy:       {fm['running_accuracy']:.4f}")
            print(f"  Windowed Error Rate:    {fm['windowed_error']:.4f}")
            print(f"  Drift Events:           {summary['drift_events']}")
            print(f"  Adaptations:            {summary['adaptations']}")
            print(f"  Successful Adaptations: {summary['successful_adaptations']}")
            print(f"  Failed Adaptations:     {summary.get('failed_adaptations', 0)}")
            print(f"  Rollbacks:              {summary.get('rollbacks', 0)}")
            print(f"  Model Version:          {summary.get('model_version_string', 'v1.0.0')}")
            print(f"  Log File:               {pipeline.event_logger.get_log_path()}")
            print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n[!] Pipeline interrupted by user")
    except Exception as e:
        logging.getLogger("driftinsights").error(f"Pipeline error: {e}", exc_info=True)
        print(f"\n[X] Pipeline error: {e}")
        sys.exit(1)
    finally:
        pipeline.shutdown()


if __name__ == "__main__":
    main()
