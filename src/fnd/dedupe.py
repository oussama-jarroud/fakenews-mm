"""Perceptual difference hashing (dhash) and sub-linear deduplication clustering."""

import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def dhash(path: str, size: int = 8) -> Optional[int]:
    """Compute 64-bit difference hash: identical images share it, near-duplicates differ by few bits."""
    try:
        with Image.open(path) as im:
            # Resize to (size + 1, size) grayscale
            im = im.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR)
            pixels = np.array(im, dtype=np.int16)
            # Compare adjacent horizontal pixels
            diff = pixels[:, 1:] > pixels[:, :-1]
            # Convert boolean array to 64-bit unsigned integer
            h = 0
            for bit in diff.flatten():
                h = (h << 1) | int(bit)
            return h
    except Exception:
        return None


def hamming(a: int, b: int) -> int:
    """Calculate Hamming distance between two 64-bit integer hashes."""
    return bin(a ^ b).count("1")


class DisjointSet:
    """Union-Find data structure for sub-linear connected components clustering."""

    def __init__(self, elements: List[int]):
        self.parent = {x: x for x in elements}

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_x] = root_y


def cluster_hashes(
    hashes: List[int],
    max_dist: int = 4,
    max_distance: Optional[int] = None,
    num_blocks: int = 4,
) -> List[int]:
    """Group near-identical hashes into clusters using pigeonhole block indexing and Union-Find."""
    if max_distance is not None:
        max_dist = max_distance
    n = len(hashes)
    if n == 0:
        return []

    dsu = DisjointSet(list(range(n)))
    block_width = 16  # 64 bits / 4 blocks = 16 bits per block
    block_index: Dict[Tuple[int, int], List[int]] = {}

    for idx, h in enumerate(tqdm(hashes, desc="Clustering duplicate groups")):
        candidates: Set[int] = set()
        # Partition 64-bit hash into 4 chunks of 16 bits
        for b in range(num_blocks):
            chunk = (h >> (b * block_width)) & 0xFFFF
            key = (b, chunk)
            if key in block_index:
                candidates.update(block_index[key])
            else:
                block_index[key] = []
            block_index[key].append(idx)

        # Check candidate collisions within max Hamming distance
        for cand_idx in candidates:
            if cand_idx < idx:
                if hamming(h, hashes[cand_idx]) <= max_dist:
                    dsu.union(idx, cand_idx)

    # Remap roots to canonical group IDs (0, 1, 2, ...)
    root_to_group: Dict[int, int] = {}
    group_ids = []
    for idx in range(n):
        r = dsu.find(idx)
        if r not in root_to_group:
            root_to_group[r] = len(root_to_group)
        group_ids.append(root_to_group[r])

    return group_ids
