"""Validation and data-splitting utilities (grouped, chronological and purged splits)."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def split_sets_by_player(
    features,
    target,
    groups,
    n_splits=5,
    random_state=42,
):
    """Create a leakage-aware train-validation split grouped by player."""

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    train_index, val_index = next(
        splitter.split(
            features,
            target,
            groups=groups,
        )
    )

    X_train = features.iloc[
        train_index
    ].copy()

    X_val = features.iloc[
        val_index
    ].copy()

    y_train = target.iloc[
        train_index
    ].copy()

    y_val = target.iloc[
        val_index
    ].copy()

    return (
        X_train,
        y_train,
        X_val,
        y_val,
    )




def split_sets_by_date(
    features,
    target,
    val_ratio=0.15,
    test_ratio=0.15,
):
    """Create a chronological train-validation-test split.

    Assumes `features` and `target` are already sorted oldest-to-newest.
    Splitting is done by position, not shuffling, so validation and test
    always come strictly after training in time — this avoids leaking
    future information into the model.

    Parameters
    ----------
    features : pandas.DataFrame
    target : pandas.Series
    val_ratio, test_ratio : float
        Proportion of rows to allocate to validation and test sets.

    Returns
    -------
    tuple
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    n = len(features)

    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    train_size = n - val_size - test_size

    X_train = features.iloc[:train_size].copy()
    y_train = target.iloc[:train_size].copy()

    X_val = features.iloc[train_size:train_size + val_size].copy()
    y_val = target.iloc[train_size:train_size + val_size].copy()

    X_test = features.iloc[train_size + val_size:].copy()
    y_test = target.iloc[train_size + val_size:].copy()

    return X_train, y_train, X_val, y_val, X_test, y_test

def chronological_split_with_gap(dates, val_ratio=0.15, test_ratio=0.15, gap_days=7):
    """Purged chronological train/validation/test split for (panel) time series.

    Splits on the sorted *unique* dates, so every token in a long panel is cut at the
    same calendar boundary. ``gap_days`` dates are dropped before the validation and
    before the test period, because a target looking ``gap_days`` ahead from the last
    training rows would otherwise overlap the next period (label leakage).

    Parameters
    ----------
    dates : array-like of datetime
        Date of every row (repeated values allowed for panels).
    val_ratio, test_ratio : float
        Share of unique dates assigned to validation and test.
    gap_days : int, default 7
        Number of unique dates purged before each later period (the forecast horizon).

    Returns
    -------
    tuple of numpy.ndarray
        Boolean masks ``(train_mask, val_mask, test_mask)`` aligned with ``dates``.
    """
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    unique = np.sort(dates.unique())
    n = len(unique)
    n_test = round(n * test_ratio)
    n_val = round(n * val_ratio)
    if n - n_test - n_val - 2 * gap_days <= 0:
        raise ValueError("Not enough dates for the requested split and gap")

    test_start = n - n_test
    val_start = test_start - n_val
    test_dates = unique[test_start:]
    val_dates = unique[val_start : test_start - gap_days]
    train_dates = unique[: val_start - gap_days]

    return (
        dates.isin(train_dates).to_numpy(),
        dates.isin(val_dates).to_numpy(),
        dates.isin(test_dates).to_numpy(),
    )
