"""AT3 engineered features and target variables (Group 08 contract, sections 4 and 5).

Every service and notebook in the group calls these functions, so features are
computed by one code path in both training and serving (no training-serving skew).

All rolling windows use ``min_periods`` equal to the window, so the first rows of
each token are NaN until enough history exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FORECAST_HORIZON_DAYS = 7
WINDOW_SHORT = 5
WINDOW_LONG = 20

# Feature name -> owning student (exact JSON keys used by every API).
FEATURE_OWNERS: dict[str, str] = {
    "DailyReturn": "A",
    "Return5": "A",
    "Return20": "A",
    "Momentum": "A",
    "MA5": "B",
    "MA20": "B",
    "TrendStrength": "B",
    "Volatility20": "C",
    "MaxPrice20": "C",
    "MaxDrawdown": "C",
    "GapRisk": "C",
    "VolumeSMA20": "D",
    "RelativeVolume": "D",
}
FEATURE_NAMES: list[str] = list(FEATURE_OWNERS)

TREND_LABELS: list[str] = ["Strong Downtrend", "Downtrend", "Sideways", "Uptrend", "Strong Uptrend"]
OPPORTUNITY_LABELS: list[str] = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]


def features_owned_by(student: str) -> list[str]:
    """Return the feature names owned by a student (``"A"``, ``"B"``, ``"C"`` or ``"D"``)."""
    return [name for name, owner in FEATURE_OWNERS.items() if owner == student.upper()]


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def compute_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute all 13 contract features for ONE token.

    Parameters
    ----------
    ohlcv : pandas.DataFrame
        Daily candles for a single token, one row per day, with columns
        ``open, high, low, close, volume``. Rows are sorted by date first.

    Returns
    -------
    pandas.DataFrame
        Same index as the (sorted) input, one column per name in ``FEATURE_NAMES``.
    """
    df = ohlcv.sort_index()
    close, high, open_, volume = df["close"], df["high"], df["open"], df["volume"]

    out = pd.DataFrame(index=df.index)
    # Student A - returns and momentum
    out["DailyReturn"] = close.pct_change(1, fill_method=None)
    out["Return5"] = close.pct_change(WINDOW_SHORT, fill_method=None)
    out["Return20"] = close.pct_change(WINDOW_LONG, fill_method=None)
    out["Momentum"] = close - close.shift(WINDOW_LONG)
    # Student B - moving averages and trend strength
    out["MA5"] = close.rolling(WINDOW_SHORT, min_periods=WINDOW_SHORT).mean()
    out["MA20"] = close.rolling(WINDOW_LONG, min_periods=WINDOW_LONG).mean()
    out["TrendStrength"] = out["MA5"] / out["MA20"] - 1
    # Student C - volatility, drawdown and gap risk
    out["Volatility20"] = (
        out["DailyReturn"].rolling(WINDOW_LONG, min_periods=WINDOW_LONG).std(ddof=1)
    )
    out["MaxPrice20"] = high.rolling(WINDOW_LONG, min_periods=WINDOW_LONG).max()
    out["MaxDrawdown"] = (out["MaxPrice20"] - close) / out["MaxPrice20"]
    previous_close = close.shift(1)
    gap = (open_ - previous_close).abs() / previous_close
    out["GapRisk"] = gap.rolling(WINDOW_LONG, min_periods=WINDOW_LONG).mean()
    # Student D - volume
    out["VolumeSMA20"] = volume.rolling(WINDOW_LONG, min_periods=WINDOW_LONG).mean()
    out["RelativeVolume"] = volume / out["VolumeSMA20"]

    return out.replace([np.inf, -np.inf], np.nan)


def compute_features_panel(
    panel: pd.DataFrame, token_col: str = "token", date_col: str = "date"
) -> pd.DataFrame:
    """Compute the features for every token in a long (panel) DataFrame.

    Parameters
    ----------
    panel : pandas.DataFrame
        Columns ``date, token, open, high, low, close, volume`` (other columns kept).
    token_col, date_col : str
        Names of the token and date columns.

    Returns
    -------
    pandas.DataFrame
        The input columns plus the 13 feature columns, sorted by token then date.
    """
    panel = panel.sort_values([token_col, date_col]).reset_index(drop=True)
    parts = []
    for _, group in panel.groupby(token_col, sort=False):
        feats = compute_features(group.set_index(date_col))
        feats.index = group.index  # back to the panel's row index
        parts.append(feats)
    features = pd.concat(parts).loc[panel.index]
    return pd.concat([panel, features], axis=1)


# ---------------------------------------------------------------------------
# Targets (index values on the same day; shift with forward_target for t+7)
# ---------------------------------------------------------------------------


def compute_asset_performance_index(features: pd.DataFrame) -> pd.Series:
    """Student A target: Asset Performance Index (0-100)."""
    rs = ((features["Return20"] + 0.20) / 0.40).clip(lower=0, upper=1)
    ts = ((features["TrendStrength"] + 0.10) / 0.20).clip(lower=0, upper=1)
    rv = features["RelativeVolume"].clip(upper=2) / 2
    ps = 1 - features["Volatility20"]
    index = 100 * (0.40 * rs + 0.30 * ts + 0.20 * rv + 0.10 * ps)
    return index.rename("API")


def compute_market_risk_index(features: pd.DataFrame, close: pd.Series) -> pd.Series:
    """Student B target: Market Risk Index (0-100).

    Parameters
    ----------
    features : pandas.DataFrame
        Must contain ``Return20, Volatility20, MaxPrice20, GapRisk``.
    close : pandas.Series
        Closing price aligned with ``features`` (needed for the drawdown risk DR).
    """
    dr = (features["MaxPrice20"] - close) / features["MaxPrice20"]
    rr = features["Return20"].clip(upper=0).abs()
    vrn = (features["Volatility20"] / 0.10).clip(upper=1)
    drn = (dr / 0.40).clip(upper=1)
    rrn = (rr / 0.30).clip(upper=1)
    grn = (features["GapRisk"] / 0.10).clip(upper=1)
    index = 100 * (0.40 * vrn + 0.30 * drn + 0.20 * rrn + 0.10 * grn)
    return index.rename("MRI")


def compute_market_risk_components(features: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """Return the four normalised MRI components (useful for EDA and explanations)."""
    dr = (features["MaxPrice20"] - close) / features["MaxPrice20"]
    rr = features["Return20"].clip(upper=0).abs()
    return pd.DataFrame(
        {
            "VRN": (features["Volatility20"] / 0.10).clip(upper=1),
            "DRN": (dr / 0.40).clip(upper=1),
            "RRN": (rr / 0.30).clip(upper=1),
            "GRN": (features["GapRisk"] / 0.10).clip(upper=1),
        },
        index=features.index,
    )


def compute_trend_score(features: pd.DataFrame) -> pd.Series:
    """Student C score: TrendScore = 0.7 * (MA5/MA20 - 1) + 0.3 * Return20."""
    score = 0.7 * (features["MA5"] / features["MA20"] - 1) + 0.3 * features["Return20"]
    return score.rename("TrendScore")


def classify_trend(score: pd.Series) -> pd.Series:
    """Map TrendScore to the five Trend Category labels (NaN stays NaN)."""
    conditions = [
        score < -0.10,
        (score >= -0.10) & (score < -0.02),
        (score >= -0.02) & (score <= 0.02),
        (score > 0.02) & (score <= 0.10),
        score > 0.10,
    ]
    return _select_labels(score, conditions, TREND_LABELS, "TC")


def compute_opportunity_score(features: pd.DataFrame) -> pd.Series:
    """Student D score: OS = 0.4*M + 0.3*T + 0.2*V - 0.1*R."""
    m = (features["Return20"] + 0.20) / 0.40
    t = features["MA5"] / features["MA20"]
    v = features["RelativeVolume"].clip(upper=2) / 2
    r = (features["Volatility20"] / 0.10).clip(upper=1)
    return (0.40 * m + 0.30 * t + 0.20 * v - 0.10 * r).rename("OS")


def classify_opportunity(score: pd.Series) -> pd.Series:
    """Map the opportunity score to the five Trading Opportunity Rating labels."""
    conditions = [
        score < 0.20,
        (score >= 0.20) & (score < 0.40),
        (score >= 0.40) & (score < 0.60),
        (score >= 0.60) & (score < 0.80),
        score >= 0.80,
    ]
    return _select_labels(score, conditions, OPPORTUNITY_LABELS, "TOR")


def _select_labels(score: pd.Series, conditions, labels, name: str) -> pd.Series:
    values = np.select(conditions, labels, default="")
    result = pd.Series(values, index=score.index, dtype=object).where(score.notna(), np.nan)
    return result.rename(name)


def forward_target(
    series: pd.Series, horizon: int = FORECAST_HORIZON_DAYS, groups: pd.Series | None = None
) -> pd.Series:
    """Shift a target so the row at date t holds the value at t + ``horizon``.

    Parameters
    ----------
    series : pandas.Series
        Same-day index values (rows sorted by date within each group).
    horizon : int, default 7
        Days ahead to forecast.
    groups : pandas.Series, optional
        Token labels; when given, the shift never crosses from one token to another.

    Returns
    -------
    pandas.Series
        Shifted series; the last ``horizon`` rows of each group are NaN.
    """
    if groups is None:
        return series.shift(-horizon)
    return series.groupby(groups, sort=False).shift(-horizon)
