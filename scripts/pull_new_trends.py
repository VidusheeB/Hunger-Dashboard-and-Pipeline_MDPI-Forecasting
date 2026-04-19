"""
Pull Google Trends data for 3 new keywords across 14 California DMAs.

Keywords: "food stamps", "how to apply for calfresh", "food bank near me"
DMAs: 14 California-area DMAs (same as existing CalFresh/FoodBank data)
Date chunks: 10 overlapping ~13-month windows (2017-01-01 to 2025-12-31)

Saves CSVs matching existing format:
    src/data/trends/{FolderName}/{DMAName}/{DMAName}_chunk{N}.csv
    Header: "Time","<keyword>"

Pipeline reads all CSVs in the DMA folder and chain-stitches them,
so overlapping chunks are handled automatically.

Usage:
    python scripts/pull_new_trends.py
    python scripts/pull_new_trends.py --dry-run   # show plan without pulling
    python scripts/pull_new_trends.py --kw "food stamps"  # single keyword
"""

import argparse
import os
import sys
import time
import random
import logging
from datetime import datetime

import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRENDS_DIR = os.path.join(BASE_DIR, "src", "data", "trends")
FAILED_LOG = os.path.join(BASE_DIR, "scripts", "pull_new_trends_failed.log")

# Keywords: display name → (search term, folder name, CSV column header)
KEYWORDS = {
    "food stamps": {
        "search_term": "food stamps",
        "folder": "Food Stamps 2017-2025",
        "col_header": "food stamps",
    },
    "how to apply for calfresh": {
        "search_term": "how to apply for calfresh",
        "folder": "How To Apply For CalFresh 2017-2025",
        "col_header": "how to apply for calfresh",
    },
    "how to apply for food stamps": {
        "search_term": "how to apply for food stamps",
        "folder": "How To Apply For Food Stamps 2017-2025",
        "col_header": "how to apply for food stamps",
    },
    "food bank near me": {
        "search_term": "food bank near me",
        "folder": "Food Bank Near Me 2017-2025",
        "col_header": "food bank near me",
    },
    "snap topic": {
        "search_term": "/m/030gj7",  # SNAP Topic mid (aggregates food stamps, EBT, SNAP benefits, etc.)
        "folder": "SNAP Topic 2017-2025",
        "col_header": "snap topic",
    },
}

# 14 California-area DMAs: folder name → Google Trends geo code
DMAS = {
    "Bakersfield":                        "US-CA-800",
    "ChicoRedding":                       "US-CA-868",
    "Eureka":                             "US-CA-802",
    "FresnoVisalia":                      "US-CA-866",
    "LosAngeles":                         "US-CA-803",
    "MedfordKlamathFalls":                "US-OR-813",
    "MontereySalinas":                    "US-CA-828",
    "PalmSprings":                        "US-CA-804",
    "Reno":                               "US-NV-811",
    "SacramentoStocktonModesto":          "US-CA-862",
    "SanDiego":                           "US-CA-825",
    "SanFranciscoOaklandSanJose":         "US-CA-807",
    "SantaBarbaraSantaMariaSanLuisObispo":"US-CA-855",
    "YumaElCentro":                       "US-AZ-771",
}

# 10 overlapping ~13-month chunks (user-specified)
CHUNKS = [
    ("2017-01-01", "2017-12-31",  "chunk01_2017"),
    ("2017-12-01", "2018-11-30",  "chunk02_2017-12"),
    ("2018-11-01", "2019-10-31",  "chunk03_2018-11"),
    ("2019-10-01", "2020-09-30",  "chunk04_2019-10"),
    ("2020-09-01", "2021-08-31",  "chunk05_2020-09"),
    ("2021-08-01", "2022-07-31",  "chunk06_2021-08"),
    ("2022-07-01", "2023-06-30",  "chunk07_2022-07"),
    ("2023-06-01", "2024-05-31",  "chunk08_2023-06"),
    ("2024-05-01", "2025-04-30",  "chunk09_2024-05"),
    ("2025-04-01", "2025-12-31",  "chunk10_2025-04"),
]

# Rate-limiting: seconds between API calls (add jitter)
SLEEP_BASE = 8
SLEEP_JITTER = 4
MAX_RETRIES = 3
RETRY_WAIT = 60  # seconds on rate-limit error

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_dirs(path):
    os.makedirs(path, exist_ok=True)


def csv_path(kw_key, dma_name, chunk_label):
    folder = KEYWORDS[kw_key]["folder"]
    dma_dir = os.path.join(TRENDS_DIR, folder, dma_name)
    return os.path.join(dma_dir, f"{dma_name}_{chunk_label}.csv")


def already_downloaded(path):
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path)
        return len(df) > 0
    except Exception:
        return False


def log_failure(kw_key, dma_name, start, end, reason):
    with open(FAILED_LOG, "a") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{ts}  keyword={kw_key!r}  dma={dma_name}  {start}→{end}  reason={reason}\n")


def pull_chunk(pt, kw_key, dma_name, geo_code, start, end):
    """
    Pull one (keyword, DMA, date-chunk) from Google Trends.
    Returns pd.DataFrame with columns [Time, <col_header>] or None on failure.
    """
    search_term = KEYWORDS[kw_key]["search_term"]
    col_header = KEYWORDS[kw_key]["col_header"]
    timeframe = f"{start} {end}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            pt.build_payload(
                kw_list=[search_term],
                cat=0,
                timeframe=timeframe,
                geo=geo_code,
                gprop="",
            )
            df = pt.interest_over_time()

            if df is None or df.empty:
                log.warning(f"  Empty response: {kw_key} | {dma_name} | {start}→{end}")
                return None

            # Drop isPartial column if present
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])

            # Rename the value column to our header.
            # For topic mids the column name is the mid itself; for text keywords it's the search term.
            df = df.reset_index()
            value_col = [c for c in df.columns if c not in ("date", "Time")][0]
            df = df.rename(columns={"date": "Time", value_col: col_header})
            df["Time"] = df["Time"].dt.strftime("%Y-%m-%d")
            df = df[["Time", col_header]]

            return df

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "Too Many Requests" in err_str or "rate limit" in err_str.lower():
                wait = RETRY_WAIT * attempt + random.randint(0, 30)
                log.warning(f"  Rate limited (attempt {attempt}/{MAX_RETRIES}). Sleeping {wait}s …")
                time.sleep(wait)
            else:
                log.error(f"  Error on attempt {attempt}: {e}")
                if attempt == MAX_RETRIES:
                    return None
                time.sleep(SLEEP_BASE)

    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Show plan without pulling")
    parser.add_argument("--kw", default=None, help="Pull only this keyword (substring match)")
    parser.add_argument("--dma", default=None, help="Pull only this DMA (substring match)")
    args = parser.parse_args()

    # Filter keywords / DMAs if requested
    kw_keys = [k for k in KEYWORDS if args.kw is None or args.kw.lower() in k]
    dma_keys = [d for d in DMAS if args.dma is None or args.dma.lower() in d.lower()]

    total = len(kw_keys) * len(dma_keys) * len(CHUNKS)
    log.info(f"Plan: {len(kw_keys)} keywords × {len(dma_keys)} DMAs × {len(CHUNKS)} chunks = {total} pulls")

    if args.dry_run:
        for kw in kw_keys:
            for dma in dma_keys:
                for start, end, label in CHUNKS:
                    path = csv_path(kw, dma, label)
                    status = "SKIP (exists)" if already_downloaded(path) else "PULL"
                    print(f"  [{status}] {kw} | {dma} | {start}→{end} → {path}")
        return

    # ── Init pytrends ──────────────────────────────────────────────────────────
    try:
        from pytrends.request import TrendReq
    except ImportError:
        log.error("pytrends not installed. Run: pip install pytrends")
        sys.exit(1)

    # tz=360 = UTC-6 (Central); retries built into our logic
    pt = TrendReq(hl="en-US", tz=360, timeout=(10, 25))

    # Clear failed log for this run
    if os.path.exists(FAILED_LOG):
        os.remove(FAILED_LOG)

    pulled = 0
    skipped = 0
    failed = 0

    for kw_key in kw_keys:
        folder = KEYWORDS[kw_key]["folder"]
        for dma_name, geo_code in DMAS.items():
            if dma_name not in dma_keys:
                continue

            dma_dir = os.path.join(TRENDS_DIR, folder, dma_name)
            make_dirs(dma_dir)

            for chunk_idx, (start, end, label) in enumerate(CHUNKS, 1):
                path = csv_path(kw_key, dma_name, label)

                if already_downloaded(path):
                    log.info(f"  SKIP  {kw_key} | {dma_name} | chunk {chunk_idx}/10")
                    skipped += 1
                    continue

                log.info(f"  PULL  {kw_key} | {dma_name} | {start}→{end}  ({chunk_idx}/10)")
                df = pull_chunk(pt, kw_key, dma_name, geo_code, start, end)

                if df is not None:
                    df.to_csv(path, index=False, quoting=1)  # quoting=1 = QUOTE_ALL
                    log.info(f"        → saved {len(df)} rows → {os.path.relpath(path, BASE_DIR)}")
                    pulled += 1
                else:
                    log.error(f"  FAIL  {kw_key} | {dma_name} | {start}→{end}")
                    log_failure(kw_key, dma_name, start, end, "max retries exceeded")
                    failed += 1

                # Sleep between pulls (skip after last chunk of last DMA)
                sleep_s = SLEEP_BASE + random.uniform(0, SLEEP_JITTER)
                time.sleep(sleep_s)

    log.info("=" * 60)
    log.info(f"Done.  Pulled={pulled}  Skipped={skipped}  Failed={failed}")
    if failed > 0:
        log.warning(f"Failed chunks logged to: {FAILED_LOG}")
        log.warning("Re-run the script to retry, or pull manually from Google Trends.")


if __name__ == "__main__":
    main()
