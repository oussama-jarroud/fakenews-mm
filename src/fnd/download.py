"""Image download utilities with network backoff and deep verification."""

import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple
from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
import urllib3
from urllib3.util import Retry

# Silence urllib3 connection pool & dead-link warnings in notebook output
urllib3.disable_warnings()
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)


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
        connect=1,  # Don't stall repeatedly on permanently dead DNS hostnames
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

        # Deep Image Verification: both verify() and load()
        with Image.open(out_path) as img_verify:
            img_verify.verify()
        with Image.open(out_path) as img_load:
            img_load.load()

        return True
    except Exception:
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
