"""Reusable dataset utilities (AT1-AT3): loading, target separation and saving sets."""

from pathlib import Path

import pandas as pd


def load_competition_data(path):
    """Load the four Kaggle competition datasets.

    Parameters
    ----------
    path : str or pathlib.Path
        Directory containing train.csv, test.csv, metadata.csv and
        sample_submission.csv.

    Returns
    -------
    tuple
        Training data, test data, metadata and sample submission.
    """
    path = Path(path)

    train = pd.read_csv(
        path / "train.csv",
        low_memory=False,
    )

    test = pd.read_csv(
        path / "test.csv",
        low_memory=False,
    )

    metadata = pd.read_csv(
        path / "metadata.csv",
        encoding="cp1252",
    )

    sample_submission = pd.read_csv(
        path / "sample_submission.csv",
    )

    return (
        train,
        test,
        metadata,
        sample_submission,
    )


def pop_target(df, target_col="label"):
    """Separate predictors and target without modifying the input dataframe."""
    df_copy = df.copy()

    target = df_copy.pop(
        target_col
    )

    return df_copy, target


def save_sets(
    X_train=None,
    y_train=None,
    X_val=None,
    y_val=None,
    X_test=None,
    y_test=None,
    path="../data/processed",
):
    """Save prepared modelling datasets."""
    path = Path(path)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    datasets = {
        "X_train.csv": X_train,
        "y_train.csv": y_train,
        "X_val.csv": X_val,
        "y_val.csv": y_val,
        "X_test.csv": X_test,
        "y_test.csv": y_test,
    }

    for filename, dataset in datasets.items():
        if dataset is not None:
            if not isinstance(dataset, (pd.DataFrame, pd.Series)):
                dataset = pd.DataFrame(dataset)  # also accept numpy arrays
            dataset.to_csv(
                path / filename,
                index=False,
            )


def load_sets(path="../data/processed"):
    """Load the sets written by ``save_sets``.

    Returns
    -------
    tuple
        ``X_train, y_train, X_val, y_val, X_test, y_test`` as DataFrames (y as Series);
        a set whose file does not exist is returned as ``None``.
    """
    path = Path(path)
    names = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    loaded = []
    for name in names:
        file = path / f"{name}.csv"
        if not file.exists():
            loaded.append(None)
            continue
        df = pd.read_csv(file)
        loaded.append(df.iloc[:, 0] if name.startswith("y_") else df)
    return tuple(loaded)