"""Group-aware dataset splitting to guarantee zero visual leakage across splits."""

from typing import Tuple
import numpy as np
import pandas as pd


def group_aware_split(
    df: pd.DataFrame,
    group_col: str = "group_id",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> pd.Series:
    """Assign entire duplicate groups to train, validate, or test splits."""
    np.random.seed(seed)
    unique_groups = df[group_col].unique()
    np.random.shuffle(unique_groups)

    n_groups = len(unique_groups)
    n_train = int(n_groups * train_ratio)
    n_val = int(n_groups * val_ratio)

    train_groups = set(unique_groups[:n_train])
    val_groups = set(unique_groups[n_train : n_train + n_val])
    test_groups = set(unique_groups[n_train + n_val :])

    def assign_split(g):
        if g in train_groups:
            return "train"
        elif g in val_groups:
            return "validate"
        else:
            return "test"

    return df[group_col].apply(assign_split)


def validate_splits(
    df: pd.DataFrame,
    group_col: str = "group_id",
    split_col: str = "split",
) -> None:
    """Strict assertion check ensuring zero group overlap between train, val, and test."""
    train_groups = set(df[df[split_col] == "train"][group_col])
    val_groups = set(df[df[split_col] == "validate"][group_col])
    test_groups = set(df[df[split_col] == "test"][group_col])

    overlap_tv = train_groups & val_groups
    overlap_tt = train_groups & test_groups
    overlap_vt = val_groups & test_groups

    assert len(overlap_tv) == 0, f"Leakage detected: {len(overlap_tv)} groups shared between train and validate!"
    assert len(overlap_tt) == 0, f"Leakage detected: {len(overlap_tt)} groups shared between train and test!"
    assert len(overlap_vt) == 0, f"Leakage detected: {len(overlap_vt)} groups shared between validate and test!"
