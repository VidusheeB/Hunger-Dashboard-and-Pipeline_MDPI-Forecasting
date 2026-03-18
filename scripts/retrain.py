"""
Retraining Pipeline for SNAP Prediction Model

Run this script when new SNAP application data arrives to rebuild the model
end-to-end.

  Step 1: build_training_data.py  — joins trends + SNAP + population/income,
                                    produces training_data.csv and
                                    trend_scaling_params.json
  Step 2: train_model.py          — trains XGBoost on training_data.csv,
                                    saves global_model.pkl
  Step 3: predict.py              — generates finalPrediction.csv using the
                                    new model and current prediction CSVs

NOTE: Before running this pipeline, if new prediction trend CSVs were uploaded,
run preprocess_prediction_data.py first to create any missing DMA placeholder files.

Usage:
    python scripts/retrain.py
"""

import subprocess
import sys
import os

SCRIPTS_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)

PIPELINE_STEPS = [
    ("1. Build training data", "build_training_data.py"),
    ("2. Train model",         "train_model.py"),
    ("3. Generate predictions","predict.py"),
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
        print("Training data: src/data/training_data.csv")
        print("Model saved:   county_models/global_model.pkl")
        print("Predictions:   src/data/finalPrediction.csv")


if __name__ == "__main__":
    main()
