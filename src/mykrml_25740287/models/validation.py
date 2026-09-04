"""Validation utilities for player-season classification."""

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