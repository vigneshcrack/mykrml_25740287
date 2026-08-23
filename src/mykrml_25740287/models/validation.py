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