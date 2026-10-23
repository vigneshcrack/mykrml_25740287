import numpy as np
import pandas as pd
import pytest

from mykrml_25740287.features.crypto import (
    FEATURE_NAMES,
    OPPORTUNITY_LABELS,
    TREND_LABELS,
    classify_opportunity,
    classify_trend,
    compute_asset_performance_index,
    compute_features,
    compute_features_panel,
    compute_market_risk_components,
    compute_market_risk_index,
    compute_opportunity_score,
    compute_trend_score,
    features_owned_by,
    forward_target,
)


def _candles(n=40, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.02, n))
    open_ = np.r_[close[0], close[:-1]]  # crypto-like: open == previous close
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = rng.uniform(1_000, 2_000, n)
    index = pd.date_range("2025-01-01", periods=n, freq="D", name="date")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index
    )


def test_compute_features_returns_all_contract_columns():
    feats = compute_features(_candles())
    assert list(feats.columns) == FEATURE_NAMES


def test_features_match_independent_calculation_at_last_row():
    df = _candles()
    feats = compute_features(df).iloc[-1]
    c = df["close"].to_numpy()
    returns = c[1:] / c[:-1] - 1

    assert feats["DailyReturn"] == pytest.approx(c[-1] / c[-2] - 1)
    assert feats["Return5"] == pytest.approx(c[-1] / c[-6] - 1)
    assert feats["Return20"] == pytest.approx(c[-1] / c[-21] - 1)
    assert feats["Momentum"] == pytest.approx(c[-1] - c[-21])
    assert feats["MA5"] == pytest.approx(c[-5:].mean())
    assert feats["MA20"] == pytest.approx(c[-20:].mean())
    assert feats["TrendStrength"] == pytest.approx(c[-5:].mean() / c[-20:].mean() - 1)
    assert feats["Volatility20"] == pytest.approx(np.std(returns[-20:], ddof=1))
    assert feats["MaxPrice20"] == pytest.approx(df["high"].to_numpy()[-20:].max())
    assert feats["MaxDrawdown"] == pytest.approx(
        (df["high"].to_numpy()[-20:].max() - c[-1]) / df["high"].to_numpy()[-20:].max()
    )
    assert feats["VolumeSMA20"] == pytest.approx(df["volume"].to_numpy()[-20:].mean())
    assert feats["RelativeVolume"] == pytest.approx(
        df["volume"].to_numpy()[-1] / df["volume"].to_numpy()[-20:].mean()
    )


def test_gap_risk_is_zero_when_open_equals_previous_close():
    feats = compute_features(_candles())
    assert feats["GapRisk"].dropna().abs().max() == pytest.approx(0.0)


def test_rolling_features_are_nan_until_enough_history():
    feats = compute_features(_candles())
    assert feats["MA20"].iloc[:19].isna().all()
    assert feats["MA20"].iloc[19:].notna().all()
    assert feats["Volatility20"].iloc[:20].isna().all()  # needs 20 daily returns


def test_panel_features_do_not_mix_tokens():
    a = _candles(seed=1).reset_index().assign(token="BTC")
    b = _candles(seed=2).reset_index().assign(token="ETH")
    panel = compute_features_panel(pd.concat([b, a], ignore_index=True))
    first_eth = panel[panel["token"] == "ETH"].iloc[0]
    assert np.isnan(first_eth["DailyReturn"])  # no carry-over from BTC rows
    btc_only = compute_features(_candles(seed=1))
    assert panel[panel["token"] == "BTC"]["MA20"].to_numpy() == pytest.approx(
        btc_only["MA20"].to_numpy(), nan_ok=True
    )


def test_features_owned_by_student():
    assert features_owned_by("B") == ["MA5", "MA20", "TrendStrength"]
    assert features_owned_by("c") == ["Volatility20", "MaxPrice20", "MaxDrawdown", "GapRisk"]


def test_market_risk_index_matches_hand_calculation():
    # VRN = 0.05/0.10 = 0.5, DRN = 0.2/0.4 = 0.5, RRN = 0.15/0.30 = 0.5, GRN = 0.05/0.10 = 0.5
    feats = pd.DataFrame(
        {"Volatility20": [0.05], "MaxPrice20": [100.0], "Return20": [-0.15], "GapRisk": [0.05]}
    )
    mri = compute_market_risk_index(feats, close=pd.Series([80.0]))
    assert mri.iloc[0] == pytest.approx(50.0)
    assert mri.name == "MRI"


def test_market_risk_index_bounds():
    calm = pd.DataFrame(
        {"Volatility20": [0.0], "MaxPrice20": [100.0], "Return20": [0.3], "GapRisk": [0.0]}
    )
    crash = pd.DataFrame(
        {"Volatility20": [0.5], "MaxPrice20": [100.0], "Return20": [-0.9], "GapRisk": [0.5]}
    )
    assert compute_market_risk_index(calm, pd.Series([100.0])).iloc[0] == pytest.approx(0.0)
    assert compute_market_risk_index(crash, pd.Series([10.0])).iloc[0] == pytest.approx(100.0)


def test_market_risk_components_sum_to_index():
    feats = pd.DataFrame(
        {"Volatility20": [0.03], "MaxPrice20": [120.0], "Return20": [-0.05], "GapRisk": [0.001]}
    )
    close = pd.Series([100.0])
    comp = compute_market_risk_components(feats, close).iloc[0]
    weighted = 100 * (0.4 * comp.VRN + 0.3 * comp.DRN + 0.2 * comp.RRN + 0.1 * comp.GRN)
    assert weighted == pytest.approx(compute_market_risk_index(feats, close).iloc[0])


def test_asset_performance_index_matches_hand_calculation():
    # RS = 0.5, TS = 0.5, RV = 0.5, PS = 0.98 -> 100*(0.2+0.15+0.1+0.098) = 54.8
    feats = pd.DataFrame(
        {"Return20": [0.0], "TrendStrength": [0.0], "RelativeVolume": [1.0], "Volatility20": [0.02]}
    )
    assert compute_asset_performance_index(feats).iloc[0] == pytest.approx(54.8)


def test_trend_category_boundaries():
    scores = pd.Series([-0.2, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.2, np.nan])
    expected = [
        "Strong Downtrend", "Downtrend", "Downtrend", "Sideways", "Sideways",
        "Sideways", "Uptrend", "Uptrend", "Strong Uptrend",
    ]
    labels = classify_trend(scores)
    assert labels.iloc[:-1].tolist() == expected
    assert pd.isna(labels.iloc[-1])
    assert set(expected) <= set(TREND_LABELS)


def test_opportunity_rating_boundaries():
    scores = pd.Series([0.1, 0.2, 0.39, 0.4, 0.6, 0.79, 0.8, 1.2])
    expected = ["Strong Sell", "Sell", "Sell", "Hold", "Buy", "Buy", "Strong Buy", "Strong Buy"]
    assert classify_opportunity(scores).tolist() == expected
    assert set(expected) <= set(OPPORTUNITY_LABELS)


def test_trend_and_opportunity_scores():
    feats = pd.DataFrame(
        {"MA5": [110.0], "MA20": [100.0], "Return20": [0.2], "RelativeVolume": [1.0],
         "Volatility20": [0.05]}
    )
    assert compute_trend_score(feats).iloc[0] == pytest.approx(0.7 * 0.1 + 0.3 * 0.2)
    # M = 1.0, T = 1.1, V = 0.5, R = 0.5 -> 0.4 + 0.33 + 0.1 - 0.05 = 0.78
    assert compute_opportunity_score(feats).iloc[0] == pytest.approx(0.78)


def test_forward_target_shifts_within_each_token():
    values = pd.Series([1, 2, 3, 10, 20, 30], dtype=float)
    groups = pd.Series(["BTC", "BTC", "BTC", "ETH", "ETH", "ETH"])
    shifted = forward_target(values, horizon=1, groups=groups)
    assert shifted.tolist()[:2] == [2, 3]
    assert np.isnan(shifted.iloc[2])  # never takes ETH's value
    assert shifted.tolist()[3:5] == [20, 30]
