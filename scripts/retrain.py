"""
Retraining Pipeline for SNAP Prediction Model

Run this script when new SNAP application data arrives to rebuild the model
end-to-end. It runs the full pipeline in order:
  1. Aggregate Google Trends + SNAP data
  2. Interpolate missing SNAP values
  3. Scale training data (normalize, add features)
  4. Train the XGBoost model
  5. Generate predictions

Usage:
    python scripts/retrain.py
"""

import subprocess
import sys
import os

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)

PIPELINE_STEPS = [
    ("1. Aggregate trends data", "create_aggregate_trends.py"),
    ("2. Interpolate missing SNAP data", "interpolate_missing_snap_data.py"),
    ("3. Scale training data", "create_scaled_training_data.py"),
    ("4. Train model", "train_model.py"),
    ("5. Generate predictions", "predict.py"),
]


def run_step(description, script_name):
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"  SKIP: {script_path} not found")
        return False

    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  Running: {script_name}")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        print(f"  FAILED with exit code {result.returncode}")
        return False

    print(f"  DONE")
    return True


def main():
    print("SNAP Prediction Model — Full Retraining Pipeline")
    print("=" * 60)

    failed = []
    for description, script in PIPELINE_STEPS:
        success = run_step(description, script)
        if not success:
            failed.append(script)
            print(f"\nPipeline stopped at: {script}")
            break

    print("\n" + "=" * 60)
    if failed:
        print(f"Pipeline FAILED at: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("Pipeline completed successfully!")
        print("Model saved to: county_models/global_model.pkl")
        print("Predictions saved to: src/data/finalPrediction.csv")


if __name__ == "__main__":
    main()
