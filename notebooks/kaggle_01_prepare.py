"""Phase A1: Optimized Multimodal Data Pipeline for Kaggle (150,000 Rows).
Features:
  - Network Backoff: urllib3.util.Retry with status_forcelist=[429, 500, 502, 503, 504].
  - Silent Warnings: Silences urllib3 dead-link warnings so output stays clean.
  - Sized Pool: pool_maxsize=32 matching ThreadPoolExecutor workers.
  - Deep Image Verification: PIL verify() and load() with auto-purge on corruption.
  - Sub-Linear Clustering: Multi-index hash dictionary lookup with Union-Find (zero false negatives).
  - Progress Tracking: tqdm on download, hashing, and clustering loops.
  - Memory-Optimized Contract: Export to table.parquet with categorical & downcasted dtypes.
"""

import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
import urllib3
from urllib3.util import Retry

# Silence urllib3 pool and dead-domain connection warnings
urllib3.disable_warnings()
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)

# ==============================================================================
# Configuration & Constants
# ==============================================================================
INPUT_TSV_PATH: str = "/kaggle/input/datasets/jarroud/fake-news/multimodal_train.tsv"
OUTPUT_DIR: str = "/kaggle/working/images_v1"
OUTPUT_PARQUET_PATH: str = "/kaggle/working/table.parquet"

# Candidate count used in the corrected Kaggle run.
SAMPLE_SIZE: Optional[int] = 100000

TIMEOUT_SECONDS: int = 8
NUM_WORKERS: int = 32
DHASH_SIZE: int = 8
HAMMING_MAX_DISTANCE: int = 4
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
RANDOM_SEED: int = 42


# ==============================================================================
# 1. Setup & Filtering
# ==============================================================================
def load_and_filter_tsv(
    tsv_path: str,
    sample_size: Optional[int] = None,
) -> Tuple[pd.DataFrame, int]:
    """Load Fakeddit TSV file, drop rows with missing id/image_url, and optionally sample."""
    # Fallback to standard path if custom path not found
    if not os.path.exists(tsv_path):
        alt_path = "/kaggle/input/fakeddit-multimodal-fake-news-classification/multimodal_only_samples/multimodal_train.tsv"
        if os.path.exists(alt_path):
            tsv_path = alt_path
        else:
            raise FileNotFoundError(f"Input TSV dataset not found at: {tsv_path}")

    print(f"Loading raw dataset from {tsv_path}...")
    df = pd.read_csv(tsv_path, sep="\t")
    total_raw_rows = len(df)

    df = df.dropna(subset=["id", "image_url"]).copy()
    df["id"] = df["id"].astype(str)

    if sample_size is not None and sample_size > 0:
        print(f"Applying SAMPLE_SIZE slice: taking first {sample_size:,} rows.")
        df = df.head(sample_size).copy()

    return df.reset_index(drop=True), total_raw_rows


# ==============================================================================
# 2. Resilient Threaded Downloader with Deep Image Verification
# ==============================================================================
def create_retry_session(
    retries: int = 3,
    backoff_factor: float = 0.5,
    status_forcelist: Tuple[int, ...] = (429, 500, 502, 503, 504),
    pool_connections: int = 32,
    pool_maxsize: int = 32,
) -> requests.Session:
    """Create a thread-safe requests session with matching pool size and fast-fail dead links."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        connect=1,  # Don't waste time retrying non-existent hostnames (e.g. tinypic)
        read=1,
        status=retries,
        backoff_factor=backoff_factor,
        status_forcelist=list(status_forcelist),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MultimodalFakeNews/1.0"}
    )
    return session


def fetch(
    url: str,
    out_path: str,
    session: requests.Session,
    timeout: int = 8,
) -> bool:
    """Download image, resize, save, and perform deep verification (verify + load)."""
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()

        im = Image.open(io.BytesIO(r.content)).convert("RGB")
        im.thumbnail((256, 256))
        im.save(out_path, "JPEG", quality=90)

        # Deep Image Verification: both verify() and load() to catch truncation
        with Image.open(out_path) as img_verify:
            img_verify.verify()
        with Image.open(out_path) as img_load:
            img_load.load()

        return True
    except Exception:
        # Catch network, timeouts, and PIL corruption errors; purge incomplete files
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        return False


def download_all(
    rows: List[Dict[str, Any]],
    workers: int = 32,
    timeout: int = 8,
) -> List[bool]:
    """Download images concurrently with smooth tqdm progress bar and thread pool matching."""
    session = create_retry_session(pool_connections=workers, pool_maxsize=workers)
    results = [False] * len(rows)

    with session:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(fetch, r["image_url"], r["path"], session, timeout): idx
                for idx, r in enumerate(rows)
            }
            for future in tqdm(
                as_completed(future_to_idx),
                total=len(rows),
                desc="Downloading images",
            ):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = False

    return results


def execute_image_downloads(
    df: pd.DataFrame,
    out_dir: str,
    workers: int = 32,
) -> pd.DataFrame:
    """Run threaded downloader on all valid rows and filter out failed downloads."""
    os.makedirs(out_dir, exist_ok=True)
    df["image_path"] = df["id"].apply(lambda img_id: os.path.join(out_dir, f"{img_id}.jpg"))

    tasks = [{"image_url": u, "path": p} for u, p in zip(df["image_url"], df["image_path"])]
    print(f"Starting resilient download of {len(tasks):,} images ({workers} workers)...")
    success_flags = download_all(tasks, workers=workers, timeout=TIMEOUT_SECONDS)

    df_success = df[success_flags].copy().reset_index(drop=True)
    return df_success


# ==============================================================================
# 3. O(log N) Perceptual Hashing & Fast Deduplication
# ==============================================================================
def dhash(path: str, size: int = 8) -> int:
    """64-bit difference hash: identical images share it, near-duplicates differ by few bits."""
    resample = getattr(Image, "Resampling", Image).BILINEAR
    im = Image.open(path).convert("L").resize((size + 1, size), resample)
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    """Count differing bits between two 64-bit integers."""
    return bin(a ^ b).count("1")


def _hash_chunks(h: int) -> Tuple[int, ...]:
    """Split 64-bit hash into 5 disjoint chunks (pigeonhole guarantee for distance <= 4)."""
    return (
        h & 0x1FFF,          # bits 0-12 (13 bits)
        (h >> 13) & 0x1FFF,  # bits 13-25 (13 bits)
        (h >> 26) & 0x1FFF,  # bits 26-38 (13 bits)
        (h >> 39) & 0x1FFF,  # bits 39-51 (13 bits)
        (h >> 52) & 0xFFF,   # bits 52-63 (12 bits)
    )


class DisjointSet:
    """Disjoint Set Union (Union-Find) with path compression."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, i: int) -> int:
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> None:
        root_i, root_j = self.find(i), self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j


def cluster_hashes(hashes: List[int], max_dist: int = 4) -> List[int]:
    """Find near-duplicates in sub-linear lookup time and assign group_id."""
    n = len(hashes)
    dsu = DisjointSet(n)

    # Multi-Index Hashing / Pigeonhole blocking: guarantees finding all pairs <= max_dist
    buckets: List[Dict[int, List[int]]] = [{} for _ in range(5)]
    seen_pairs: Set[Tuple[int, int]] = set()

    for idx in tqdm(range(n), desc="Clustering duplicate groups"):
        h = hashes[idx]
        for c_idx, chunk in enumerate(_hash_chunks(h)):
            for match_idx in buckets[c_idx].setdefault(chunk, []):
                pair = (match_idx, idx)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    if hamming(h, hashes[match_idx]) <= max_dist:
                        dsu.union(match_idx, idx)
            buckets[c_idx][chunk].append(idx)

    root_map: Dict[int, int] = {}
    return [root_map.setdefault(dsu.find(i), len(root_map)) for i in range(n)]


def compute_duplicate_groups(df: pd.DataFrame, max_dist: int = 4) -> pd.DataFrame:
    """Compute perceptual hashes with tqdm and cluster duplicates into group_id."""
    print("Computing perceptual difference hashes (dhash 64-bit)...")
    hashes = [
        dhash(p, size=DHASH_SIZE)
        for p in tqdm(df["image_path"], desc="Computing dhashes")
    ]
    df["group_id"] = cluster_hashes(hashes, max_dist=max_dist)
    return df


# ==============================================================================
# 4. Group-Aware Splitting & Memory-Efficient Formatting
# ==============================================================================
def group_aware_split(
    df: pd.DataFrame,
    group_col: str = "group_id",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> pd.Series:
    """Split data into train, validate, and test respecting group boundaries."""
    rng = np.random.default_rng(seed)
    unique_groups = df[group_col].unique()
    rng.shuffle(unique_groups)

    group_sizes = df[group_col].value_counts().to_dict()
    total_samples = len(df)
    train_target = total_samples * train_ratio
    val_target = total_samples * val_ratio

    split_map: Dict[int, str] = {}
    curr_train = 0
    curr_val = 0

    for gid in unique_groups:
        size = group_sizes[gid]
        if curr_train < train_target:
            split_map[gid] = "train"
            curr_train += size
        elif curr_val < val_target:
            split_map[gid] = "validate"
            curr_val += size
        else:
            split_map[gid] = "test"

    return df[group_col].map(split_map)


def validate_splits(
    df: pd.DataFrame,
    group_col: str = "group_id",
    split_col: str = "split",
) -> None:
    """Assert zero group overlap between train, validate, and test partitions."""
    train_g = set(df.loc[df[split_col] == "train", group_col])
    val_g = set(df.loc[df[split_col] == "validate", group_col])
    test_g = set(df.loc[df[split_col] == "test", group_col])

    assert len(train_g & val_g) == 0, f"Leakage: {len(train_g & val_g)} groups shared train/val!"
    assert len(train_g & test_g) == 0, f"Leakage: {len(train_g & test_g)} groups shared train/test!"
    assert len(val_g & test_g) == 0, f"Leakage: {len(val_g & test_g)} groups shared val/test!"


def format_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, cast to memory-efficient types (category, int8, int32, bool)."""
    rename_cols = {
        "clean_title": "text",
        "2_way_label": "label2",
        "6_way_label": "label6",
    }
    df = df.rename(columns=rename_cols)

    if "text" not in df.columns and "title" in df.columns:
        df["text"] = df["title"]

    df["is_mined_mismatch"] = False

    contract_columns = [
        "id",
        "text",
        "image_path",
        "label2",
        "label6",
        "subreddit",
        "group_id",
        "split",
        "is_mined_mismatch",
    ]

    df_filtered = df[contract_columns].copy()

    # Memory-efficient schema casting
    df_filtered["id"] = df_filtered["id"].astype("string")
    df_filtered["text"] = df_filtered["text"].astype("string")
    df_filtered["image_path"] = df_filtered["image_path"].astype("string")
    df_filtered["label2"] = 1 - df_filtered["label2"].astype(np.int8)
    assert df_filtered["label2"].eq(df_filtered["label6"].astype(np.int8).ne(0)).all()
    df_filtered["label6"] = df_filtered["label6"].astype(np.int8)
    df_filtered["subreddit"] = df_filtered["subreddit"].astype("category")
    df_filtered["group_id"] = df_filtered["group_id"].astype(np.int32)
    df_filtered["split"] = df_filtered["split"].astype("category")
    df_filtered["is_mined_mismatch"] = df_filtered["is_mined_mismatch"].astype(bool)

    return df_filtered


# ==============================================================================
# 5. Export & Summary
# ==============================================================================
def print_summary(
    total_raw: int,
    total_valid: int,
    df: pd.DataFrame,
    out_parquet: str,
) -> None:
    """Print download metrics, attrition rate, and memory-efficient split statistics."""
    downloaded = len(df)
    failed = total_valid - downloaded
    attrition_rate = (failed / total_valid * 100.0) if total_valid > 0 else 0.0
    file_size_mb = os.path.getsize(out_parquet) / (1024 * 1024) if os.path.exists(out_parquet) else 0.0

    print("\n" + "=" * 55)
    print("PHASE A1 DATA INGESTION & SPLITTING SUMMARY")
    print("=" * 55)
    print(f"Total raw rows in TSV:        {total_raw:,}")
    print(f"Valid candidates (id + URL):  {total_valid:,}")
    print(f"Successfully downloaded:      {downloaded:,}")
    print(f"Failed downloads (purged):    {failed:,}")
    print(f"Image attrition rate:         {attrition_rate:.2f}%")
    print(f"Unique duplicate groups:      {df['group_id'].nunique():,}")
    print(f"Output contract parquet:      {out_parquet} ({file_size_mb:.2f} MB)")
    print("\nSplit Distribution:")
    counts = df["split"].value_counts()
    props = df["split"].value_counts(normalize=True) * 100
    for split_name in ["train", "validate", "test"]:
        if split_name in counts:
            print(f"  - {split_name:<10}: {counts[split_name]:>7,} rows ({props[split_name]:>5.2f}%)")
    print("\nMemory Footprint:")
    print(df.info(memory_usage="deep"))
    print("=" * 55)


def run_pipeline() -> None:
    """Execute end-to-end Phase A1 data engineering pipeline."""
    # Step 1: Load and filter
    df, total_raw = load_and_filter_tsv(INPUT_TSV_PATH, sample_size=SAMPLE_SIZE)
    total_valid = len(df)

    # Step 2: Threaded download with deep image verification & network backoff
    df_downloaded = execute_image_downloads(df, OUTPUT_DIR, workers=NUM_WORKERS)

    # Step 3: Fast perceptual hashing & deduplication
    df_deduped = compute_duplicate_groups(df_downloaded, max_dist=HAMMING_MAX_DISTANCE)

    # Step 4: Group-aware split
    df_deduped["split"] = group_aware_split(
        df_deduped,
        group_col="group_id",
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=RANDOM_SEED,
    )
    validate_splits(df_deduped, group_col="group_id", split_col="split")

    # Step 5: Format contract with memory-efficient dtypes and export
    final_df = format_contract(df_deduped)
    final_df.to_parquet(OUTPUT_PARQUET_PATH, index=False, engine="pyarrow")

    # Summary
    print_summary(total_raw, total_valid, final_df, OUTPUT_PARQUET_PATH)


if __name__ == "__main__":
    run_pipeline()
