"""
pull_foodbank_trends.py

Pulls weekly Google Trends data for "Food bank" (topic or keyword fallback)
for 14 California-covering DMAs, in overlapping ~12-month chunks.

Outputs:
  src/data/trends/FoodBank/<DMA>/<DMA>_<chunk_start>_<chunk_end>.csv  (per chunk)
  src/data/trends/FoodBank/<DMA>/<DMA>_combined.csv                   (per DMA)
  src/data/trends/FoodBank/failed_pulls.log
"""

import time
import random
import logging
from pathlib import Path

import pandas as pd
from pytrends.request import TrendReq

# ---------------------------------------------------------------------------
# Configuration — edit geo codes here if any need updating
# ---------------------------------------------------------------------------

# Nielsen DMA codes formatted as Google Trends geo strings.
# Verify at: https://trends.google.com (inspect network requests for geo param)
DMAS: dict[str, tuple[str, str]] = {
    #  slug                              display name                          geo code
    "Bakersfield":                     ("Bakersfield",                        "US-DMA-800"),
    "ChicoRedding":                    ("Chico–Redding",                      "US-DMA-868"),
    "Eureka":                          ("Eureka",                             "US-DMA-802"),
    "FresnoVisalia":                   ("Fresno–Visalia",                     "US-DMA-866"),
    "LosAngeles":                      ("Los Angeles",                        "US-DMA-803"),
    "MedfordKlamathFalls":             ("Medford–Klamath Falls",              "US-DMA-813"),
    "MontereySalinas":                 ("Monterey–Salinas",                   "US-DMA-828"),
    "PalmSprings":                     ("Palm Springs",                       "US-DMA-804"),
    "Reno":                            ("Reno",                               "US-DMA-811"),
    "SacramentoStocktonModesto":       ("Sacramento–Stockton–Modesto",        "US-DMA-862"),
    "SanDiego":                        ("San Diego",                          "US-DMA-825"),
    "SanFranciscoOaklandSanJose":      ("San Francisco–Oakland–San Jose",     "US-DMA-807"),
    "SantaBarbaraSantaMariaSanLuisObispo": (
                                        "Santa Barbara–Santa Maria–San Luis Obispo",
                                                                              "US-DMA-855"),
    "YumaElCentro":                    ("Yuma–El Centro",                     "US-DMA-771"),
}

# Overlapping ~12-month chunks
CHUNKS: list[tuple[str, str]] = [
    ("2017-01-01", "2017-12-31"),
    ("2017-12-01", "2018-11-30"),
    ("2018-11-01", "2019-10-31"),
    ("2019-10-01", "2020-09-30"),
    ("2020-09-01", "2021-08-31"),
    ("2021-08-01", "2022-07-31"),
    ("2022-07-01", "2023-06-30"),
    ("2023-06-01", "2024-05-31"),
    ("2024-05-01", "2025-04-30"),
    ("2025-04-01", "2025-12-31"),
]

SEARCH_TERM = "food bank"
OUTPUT_DIR = Path("src/data/trends/FoodBank")
LOG_FILE = OUTPUT_DIR / "failed_pulls.log"

SLEEP_MIN = 8
SLEEP_MAX = 12
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Topic lookup
# ---------------------------------------------------------------------------

def resolve_keyword(pytrends: TrendReq, term: str) -> str:
    """
    Try to find the Google Trends topic mid for `term`.
    Returns the mid string (e.g. '/m/02_qt8') if a matching Topic is found,
    otherwise returns the plain search term as a fallback.
    """
    try:
        suggestions = pytrends.suggestions(keyword=term)
        for s in suggestions:
            if s.get("type", "").lower() == "topic" and term.lower() in s.get("title", "").lower():
                mid = s["mid"]
                log.info(f"Topic found: '{s['title']}' → mid={mid}  (using topic entity)")
                return mid
        log.info(f"No topic entity found for '{term}'; falling back to keyword search.")
    except Exception as exc:
        log.warning(f"suggestions() call failed ({exc}); falling back to keyword search.")
    return term


# ---------------------------------------------------------------------------
# Data pull
# ---------------------------------------------------------------------------

def throttle() -> None:
    delay = random.uniform(SLEEP_MIN, SLEEP_MAX)
    time.sleep(delay)


def fetch_chunk(
    pytrends: TrendReq,
    keyword: str,
    geo: str,
    chunk_start: str,
    chunk_end: str,
) -> pd.DataFrame | None:
    """
    Fetch weekly Trends data for one DMA + chunk.
    Returns a raw DataFrame (index=date, column=keyword) or None on total failure.
    """
    timeframe = f"{chunk_start} {chunk_end}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            pytrends.build_payload(
                kw_list=[keyword],
                cat=0,
                timeframe=timeframe,
                geo=geo,
                gprop="",
            )
            df = pytrends.interest_over_time()
            if df is None or df.empty:
                log.warning(f"  Empty response (attempt {attempt}/{MAX_RETRIES})")
            else:
                return df
        except Exception as exc:
            wait = (2 ** attempt) + random.uniform(0, 2)
            log.warning(f"  Request failed (attempt {attempt}/{MAX_RETRIES}): {exc} — retrying in {wait:.1f}s")
            time.sleep(wait)

    return None


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def chunk_path(dma_dir: Path, slug: str, chunk_start: str, chunk_end: str) -> Path:
    return dma_dir / f"{slug}_{chunk_start}_{chunk_end}.csv"


def combined_path(dma_dir: Path, slug: str) -> Path:
    return dma_dir / f"{slug}_combined.csv"


def save_chunk(df_raw: pd.DataFrame, keyword: str, dma_name: str, geo: str,
               chunk_start: str, chunk_end: str, path: Path) -> pd.DataFrame:
    """Reshape raw pytrends DataFrame to tidy format and save."""
    col = keyword if keyword in df_raw.columns else df_raw.columns[0]
    tidy = (
        df_raw[[col]]
        .reset_index()
        .rename(columns={"date": "week_date", col: "trends_value"})
        .assign(
            dma_name=dma_name,
            dma_geo=geo,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
        )
        [["dma_name", "dma_geo", "chunk_start", "chunk_end", "week_date", "trends_value"]]
    )
    tidy["week_date"] = pd.to_datetime(tidy["week_date"]).dt.date
    path.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(path, index=False)
    return tidy


def build_combined(dma_dir: Path, slug: str, dma_name: str, geo: str) -> None:
    """Concatenate all saved chunk CSVs into one combined CSV for a DMA."""
    chunk_files = sorted(dma_dir.glob(f"{slug}_????-??-??_????-??-??.csv"))
    if not chunk_files:
        log.warning(f"  No chunk files found for {slug}; skipping combined save.")
        return

    frames = [pd.read_csv(f) for f in chunk_files]
    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["week_date"])
        .sort_values("week_date")
        .reset_index(drop=True)
    )
    out = combined_path(dma_dir, slug)
    combined.to_csv(out, index=False)
    log.info(f"  Combined CSV saved: {out}  ({len(combined)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25), retries=2, backoff_factor=0.5)

    log.info("=== Resolving keyword / topic ===")
    keyword = resolve_keyword(pytrends, SEARCH_TERM)
    throttle()

    failed: list[str] = []

    for slug, (dma_name, geo) in DMAS.items():
        dma_dir = OUTPUT_DIR / slug
        dma_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"\n{'='*60}")
        log.info(f"DMA: {dma_name}  ({geo})")

        for chunk_start, chunk_end in CHUNKS:
            out_path = chunk_path(dma_dir, slug, chunk_start, chunk_end)

            # --- Resumability: skip if already done ---
            if out_path.exists():
                log.info(f"  [SKIP] {chunk_start} → {chunk_end}  (file exists)")
                continue

            log.info(f"  Pulling {chunk_start} → {chunk_end} …")
            df_raw = fetch_chunk(pytrends, keyword, geo, chunk_start, chunk_end)

            if df_raw is None:
                msg = f"FAILED: {dma_name} | {chunk_start} → {chunk_end}"
                log.error(f"  {msg}")
                failed.append(msg)
            else:
                save_chunk(df_raw, keyword, dma_name, geo, chunk_start, chunk_end, out_path)
                log.info(f"  Saved {len(df_raw)} rows → {out_path.name}")

            throttle()

        # Build combined CSV after all chunks for this DMA
        build_combined(dma_dir, slug, dma_name, geo)

    # Final failure summary
    log.info(f"\n{'='*60}")
    if failed:
        log.warning(f"COMPLETED WITH {len(failed)} FAILED PULL(S):")
        for f in failed:
            log.warning(f"  • {f}")
        log.warning(f"Re-run the script to retry (failed chunks will be skipped if already saved).")
    else:
        log.info("All chunks pulled successfully.")


if __name__ == "__main__":
    main()
