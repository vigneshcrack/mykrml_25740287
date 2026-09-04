"""Target variable formulas for AT2: Climate Comfort Index and Weather Hazard Category."""

import numpy as np
import pandas as pd


def compute_climate_comfort_index(daily_df):
    """Compute the Climate Comfort Index (CCI) from daily aggregated weather data.

    Expects columns: temperature_2m_mean, relative_humidity_2m_mean,
    wind_speed_10m_mean, cloud_cover_mean, precipitation_sum.

    Returns
    -------
    pandas.Series
        CCI values (0-100), indexed the same as daily_df.
    """
    temp_score = (1 - (daily_df["temperature_2m_mean"] - 22).abs() / 20).clip(lower=0)
    humidity_score = (1 - (daily_df["relative_humidity_2m_mean"] - 50).abs() / 50).clip(lower=0)
    wind_score = (1 - (daily_df["wind_speed_10m_mean"] - 10).abs() / 40).clip(lower=0)
    cloud_score = (1 - daily_df["cloud_cover_mean"] / 100).clip(lower=0)
    rain_score = (1 - daily_df["precipitation_sum"] / 20).clip(lower=0)

    cci = 100 * (
        0.35 * temp_score
        + 0.20 * humidity_score
        + 0.15 * wind_score
        + 0.15 * cloud_score
        + 0.15 * rain_score
    )

    cci.name = "cci"

    return cci


def compute_weather_hazard_index(daily_df):
    """Compute the continuous Weather Hazard Index (WHI) from daily aggregated weather data.

    Expects columns: precipitation_sum, wind_gusts_10m_max, cloud_cover_mean,
    snowfall_sum, temperature_2m_mean.

    Returns
    -------
    pandas.Series
        WHI values (0-100), indexed the same as daily_df.
    """
    rain_hazard = (daily_df["precipitation_sum"] / 30).clip(upper=1)
    wind_hazard = (daily_df["wind_gusts_10m_max"] / 100).clip(upper=1)
    cloud_hazard = (daily_df["cloud_cover_mean"] / 100).clip(upper=1)
    snow_hazard = (daily_df["snowfall_sum"] / 15).clip(upper=1)
    temp_hazard = ((daily_df["temperature_2m_mean"] - 22).abs() / 25).clip(upper=1)

    whi = 100 * (
        0.30 * rain_hazard
        + 0.30 * wind_hazard
        + 0.20 * cloud_hazard
        + 0.10 * snow_hazard
        + 0.10 * temp_hazard
    )

    whi.name = "whi"

    return whi


def classify_weather_hazard(whi):
    """Convert a continuous Weather Hazard Index into the 4 AT2 risk categories.

    Parameters
    ----------
    whi : pandas.Series
        Weather Hazard Index values (0-100).

    Returns
    -------
    pandas.Series
        Category labels: 'Low Risk', 'Moderate Risk', 'High Risk', 'Extreme Risk'.
    """
    bins = [-np.inf, 25, 50, 75, np.inf]
    labels = ["Low Risk", "Moderate Risk", "High Risk", "Extreme Risk"]

    whc = pd.cut(whi, bins=bins, labels=labels, right=False)
    whc.name = "whc"

    return whc


def shift_target(series, horizon_days):
    """Align a target series so the value at time t becomes the value at t + horizon_days.

    Turns same-day weather indices into forward-looking prediction targets
    (CCI 3 days ahead, WHC 7 days ahead) without leaking future data into
    the features. The last `horizon_days` rows become NaN and should be
    dropped before modelling.

    Parameters
    ----------
    series : pandas.Series
        Daily-indexed series (e.g. output of compute_climate_comfort_index).
    horizon_days : int
        Number of days ahead the target should be shifted.

    Returns
    -------
    pandas.Series
        Shifted series, same index as the input.
    """
    return series.shift(-horizon_days)