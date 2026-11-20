"""Unit tests for group-aware splitting and perceptual hash deduplication."""

import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image

import pytest
from src.fnd.dedupe import cluster_hashes, dhash, hamming
from src.fnd.splits import group_aware_split, validate_splits


def test_dhash_and_hamming(tmp_path):
    # Create synthetic images
    img1_path = str(tmp_path / "img1.jpg")
    img2_path = str(tmp_path / "img2.jpg")
    img3_path = str(tmp_path / "img3.jpg")

    # Image 1: horizontal gradient
    arr1 = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    Image.fromarray(arr1).save(img1_path, "JPEG", quality=90)

    # Image 2: slightly modified Image 1 (identical difference hash)
    arr2 = arr1.copy()
    arr2[0:2, 0:2] = 128
    Image.fromarray(arr2).save(img2_path, "JPEG", quality=90)

    # Image 3: vertical gradient (inverted / perpendicular)
    arr3 = np.tile(np.linspace(255, 0, 64, dtype=np.uint8).reshape(-1, 1), (1, 64))
    Image.fromarray(arr3).save(img3_path, "JPEG", quality=90)

    h1 = dhash(img1_path)
    h2 = dhash(img2_path)
    h3 = dhash(img3_path)

    assert hamming(h1, h2) <= 4, f"Near duplicates should have hamming <= 4, got {hamming(h1, h2)}"
    assert hamming(h1, h3) > 10, "Different images should have large hamming distance"

    groups = cluster_hashes([h1, h2, h3], max_dist=4)
    assert groups[0] == groups[1], "img1 and img2 must share group_id"
    assert groups[0] != groups[2], "img1 and img3 must have different group_id"


def test_group_aware_split_no_leakage():
    np.random.seed(42)
    # 500 rows across 100 groups with varied group sizes
    group_ids = np.random.choice(100, size=500)
    df = pd.DataFrame({
        "id": [f"id_{i}" for i in range(500)],
        "group_id": group_ids,
        "label2": np.random.choice([0, 1], size=500),
    })

    df["split"] = group_aware_split(df, group_col="group_id", train_ratio=0.70, val_ratio=0.15, seed=42)

    # Strict leakage validation
    validate_splits(df, group_col="group_id", split_col="split")

    train_groups = set(df.loc[df["split"] == "train", "group_id"])
    val_groups = set(df.loc[df["split"] == "validate", "group_id"])
    test_groups = set(df.loc[df["split"] == "test", "group_id"])

    assert len(train_groups & val_groups) == 0
    assert len(train_groups & test_groups) == 0
    assert len(val_groups & test_groups) == 0

    proportions = df["split"].value_counts(normalize=True)
    assert 0.65 <= proportions["train"] <= 0.75
    assert 0.10 <= proportions["validate"] <= 0.20
    assert 0.10 <= proportions["test"] <= 0.20


def test_leakage_detector_fails_on_leakage():
    df_leaky = pd.DataFrame({
        "id": ["1", "2", "3"],
        "group_id": [10, 10, 20],
        "split": ["train", "test", "test"],
    })
    with pytest.raises(AssertionError, match="Leakage"):
        validate_splits(df_leaky, group_col="group_id", split_col="split")

