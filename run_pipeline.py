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
import importlib
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


def run_stage_lazy(
    name: str,
    module_name: str,
    fn_name: str,
    stage_num: int,
    args,
    specific_stages: set,
):
    if not should_run(stage_num, args, specific_stages):
        logging.getLogger().info(f"  [SKIP] Stage {stage_num}: {name}")
        return None
    module = importlib.import_module(module_name)
    return run_stage(name, getattr(module, fn_name), stage_num, args, specific_stages)


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

    # ── Pipeline stages ──────────────────────────────────────────────────────
    run_stage_lazy("Build Features", "pipeline.stage2_build_features", "build_training_data", 2, args, specific_stages)
    run_stage_lazy("Feature Engineering", "pipeline.feature_engineering", "engineer_features", 25, args, specific_stages)
    run_stage_lazy("Train Model", "pipeline.stage3_train", "train_and_save", 3, args, specific_stages)
    run_stage_lazy("Evaluate (Walk-Forward)", "pipeline.stage4_evaluate", "evaluate", 4, args, specific_stages)
    run_stage_lazy("Predict", "pipeline.stage5_predict", "predict", 5, args, specific_stages)
    run_stage_lazy("Generate Report", "pipeline.stage6_report", "generate_all", 6, args, specific_stages)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = time.time() - pipeline_start
    logger.info("\n" + "=" * 60)
    logger.info(f"  PIPELINE COMPLETE  ({total:.1f}s total)")
    logger.info("=" * 60)
    logger.info(f"  Training data:     {config.TRAINING_DATA_CSV}")
    logger.info(f"  Feature dataset:   {config.FEATURES_CSV}")
    logger.info(f"  Feature registry:  {config.FEATURE_REGISTRY_CSV}")
    logger.info(f"  Model:             {config.MODEL_PKL}")
    logger.info(f"  Walk-forward:      {config.WF_OVERALL_JSON}")
    logger.info(f"  Predictions:       {config.PREDICTIONS_CSV}")
    logger.info(f"  Figures:           {config.FIGURES_DIR}/")
    logger.info(f"  Paper summary:     {config.PAPER_SUMMARY_JSON}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
