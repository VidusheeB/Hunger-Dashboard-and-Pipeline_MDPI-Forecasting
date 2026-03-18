"""
run_pipeline.py — Single entry point for the fully reproducible SNAP prediction pipeline.

Usage:
    python run_pipeline.py                    # Run all stages
    python run_pipeline.py --skip-build       # Skip data build if training_data.csv exists
    python run_pipeline.py --skip-train       # Skip training if model pkl exists
    python run_pipeline.py --skip-predict     # Skip forward prediction
    python run_pipeline.py --stages 4,6       # Run only specific stages

Pipeline stages:
    2. build_features  — join SNAP + Trends + population/income → training_data.csv
    3. train           — fit XGBoost, save model + feature importance
    4. evaluate        — walk-forward validation → metrics JSON + per-month CSV
    5. predict         — forward predictions for next target month
    6. report          — figures and paper_summary.json

Note: Stage 1 (load_raw) has no standalone output; it is called by stage 2.
"""

import argparse
import logging
import os
import sys
import time

# Ensure the project root is on the path when run from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import config


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args():
    p = argparse.ArgumentParser(description="SNAP Prediction Pipeline")
    p.add_argument("--skip-build",   action="store_true",
                   help="Skip stage 2 if training_data.csv already exists")
    p.add_argument("--skip-train",   action="store_true",
                   help="Skip stage 3 if model pkl already exists")
    p.add_argument("--skip-predict", action="store_true",
                   help="Skip stage 5 (forward prediction)")
    p.add_argument("--stages",       type=str, default=None,
                   help="Comma-separated list of stage numbers to run, e.g. '4,6'")
    return p.parse_args()


def should_run(stage_num: int, args, specific_stages: set) -> bool:
    if specific_stages:
        return stage_num in specific_stages
    if stage_num == 2 and args.skip_build and os.path.exists(config.TRAINING_DATA_CSV):
        return False
    if stage_num == 3 and args.skip_train and os.path.exists(config.MODEL_PKL):
        return False
    if stage_num == 5 and args.skip_predict:
        return False
    return True


def run_stage(name: str, fn, stage_num: int, args, specific_stages: set):
    if not should_run(stage_num, args, specific_stages):
        logging.getLogger().info(f"  [SKIP] Stage {stage_num}: {name}")
        return None

    t0 = time.time()
    logging.getLogger().info(f"\n{'─' * 60}")
    try:
        result = fn()
        elapsed = time.time() - t0
        logging.getLogger().info(f"  Stage {stage_num} completed in {elapsed:.1f}s")
        return result
    except Exception as e:
        logging.getLogger().error(f"  Stage {stage_num} FAILED: {e}")
        raise


def main():
    setup_logging()
    args = parse_args()
    logger = logging.getLogger()

    specific_stages = set()
    if args.stages:
        specific_stages = {int(s.strip()) for s in args.stages.split(",")}

    # Ensure all output directories exist
    config.ensure_output_dirs()

    # Print banner
    logger.info("=" * 60)
    logger.info("  SNAP PREDICTION PIPELINE")
    logger.info(f"  Project: {config.PROJECT_ROOT}")
    logger.info(f"  Outputs: {config.OUTPUTS_ROOT}")
    logger.info("=" * 60)

    pipeline_start = time.time()

    # ── Stage 2: Build training data ─────────────────────────────────────────
    from pipeline.stage2_build_features import build_training_data
    run_stage("Build Features", build_training_data, 2, args, specific_stages)

    # ── Stage 3: Train model ──────────────────────────────────────────────────
    from pipeline.stage3_train import train_and_save
    run_stage("Train Model", train_and_save, 3, args, specific_stages)

    # ── Stage 4: Walk-forward evaluation ─────────────────────────────────────
    from pipeline.stage4_evaluate import evaluate
    run_stage("Evaluate (Walk-Forward)", evaluate, 4, args, specific_stages)

    # ── Stage 5: Forward prediction ───────────────────────────────────────────
    from pipeline.stage5_predict import predict
    run_stage("Predict", predict, 5, args, specific_stages)

    # ── Stage 6: Generate report figures ─────────────────────────────────────
    from pipeline.stage6_report import generate_all
    run_stage("Generate Report", generate_all, 6, args, specific_stages)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = time.time() - pipeline_start
    logger.info("\n" + "=" * 60)
    logger.info(f"  PIPELINE COMPLETE  ({total:.1f}s total)")
    logger.info("=" * 60)
    logger.info(f"  Training data:     {config.TRAINING_DATA_CSV}")
    logger.info(f"  Model:             {config.MODEL_PKL}")
    logger.info(f"  Walk-forward:      {config.WF_OVERALL_JSON}")
    logger.info(f"  Predictions:       {config.PREDICTIONS_CSV}")
    logger.info(f"  Figures:           {config.FIGURES_DIR}/")
    logger.info(f"  Paper summary:     {config.PAPER_SUMMARY_JSON}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
