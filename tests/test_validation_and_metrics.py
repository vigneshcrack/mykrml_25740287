import numpy as np
import pandas as pd
import pytest

from mykrml_25740287.data.sets import load_sets, save_sets
from mykrml_25740287.models.performance import (
    multiclass_classification_scores,
    regression_scores,
    skill_vs_baseline,
)
from mykrml_25740287.models.validation import chronological_split_with_gap


def test_chronological_split_with_gap_is_ordered_and_purged():
    days = pd.date_range("2025-01-01", periods=100, freq="D")
    dates = np.repeat(days, 2)  # a panel with two tokens per day
    train, val, test = chronological_split_with_gap(dates, 0.15, 0.15, gap_days=7)
    d = pd.Series(dates)

    assert not (train & val).any() and not (val & test).any()
    assert d[train].max() < d[val].min() < d[test].min()
    assert (d[val].min() - d[train].max()).days == 8  # 7 purged days in between
    assert (d[test].min() - d[val].max()).days == 8
    assert d[test].nunique() == 15
    # every token is cut at the same calendar boundary
    assert train.sum() % 2 == 0 and val.sum() % 2 == 0 and test.sum() % 2 == 0


def test_chronological_split_rejects_too_little_data():
    with pytest.raises(ValueError):
        chronological_split_with_gap(pd.date_range("2025-01-01", periods=10), gap_days=7)


def test_regression_scores_include_r2():
    scores = regression_scores(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert scores["rmse"] == pytest.approx(0.0)
    assert scores["mae"] == pytest.approx(0.0)
    assert scores["r2"] == pytest.approx(1.0)


def test_multiclass_scores_include_macro_f1_and_balanced_accuracy():
    y_true = ["a", "a", "a", "b"]
    y_pred = ["a", "a", "a", "a"]
    scores = multiclass_classification_scores(y_true, y_pred)
    assert scores["accuracy"] == pytest.approx(0.75)
    assert scores["balanced_accuracy"] == pytest.approx(0.5)
    assert scores["f1_macro"] < scores["f1_weighted"]


def test_skill_vs_baseline():
    assert skill_vs_baseline(8.0, 10.0) == pytest.approx(0.2)
    assert skill_vs_baseline(12.0, 10.0) == pytest.approx(-0.2)


def test_save_and_load_sets_round_trip_with_numpy(tmp_path):
    X_train = np.array([[1, 2], [3, 4]])
    y_train = pd.Series([0.5, 0.7], name="target")
    save_sets(X_train=X_train, y_train=y_train, path=tmp_path)
    X, y, X_val, *_ = load_sets(tmp_path)
    assert X.shape == (2, 2)
    assert y.tolist() == [0.5, 0.7]
    assert X_val is None
