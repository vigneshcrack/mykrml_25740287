"""Utilities for fetching and aggregating historical weather data from Open-Meteo, for AT2."""

import pandas as pd
import requests

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "precipitation",
    "snowfall",
]


def fetch_historical_weather(
    start_date,
    end_date,
    latitude=-33.8688,
    longitude=151.2093,
    timezone="Australia/Sydney",
):
    """Fetch hourly historical weather data from the Open-Meteo Archive API.

    Parameters
    ----------
    start_date, end_date : str
        Dates in YYYY-MM-DD format.
    latitude, longitude : float
        Location to fetch weather for (defaults to Sydney, NSW).
    timezone : str
        IANA timezone name so daily boundaries align with local time.

    Returns
    -------
    pandas.DataFrame
        Hourly weather data indexed by timestamp.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": timezone,
    }

    response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=60)
    response.raise_for_status()

    payload = response.json()

    if payload.get("error"):
        raise ValueError(f"Open-Meteo API error: {payload.get('reason')}")

    hourly = payload["hourly"]

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")

    return df


def aggregate_daily_weather(hourly_df):
    """Aggregate hourly weather data into daily values, per the AT2 assignment rules.

    Applies: mean for temperature/humidity/wind_speed/cloud_cover, sum for
    precipitation/snowfall, max for wind_gusts.

    Parameters
    ----------
    hourly_df : pandas.DataFrame
        Hourly weather data as returned by fetch_historical_weather().

    Returns
    -------
    pandas.DataFrame
        Daily aggregated weather data, indexed by date.
    """
    daily = pd.DataFrame({
        "temperature_2m_mean": hourly_df["temperature_2m"].resample("D").mean(),
        "relative_humidity_2m_mean": hourly_df["relative_humidity_2m"].resample("D").mean(),
        "wind_speed_10m_mean": hourly_df["wind_speed_10m"].resample("D").mean(),
        "wind_gusts_10m_max": hourly_df["wind_gusts_10m"].resample("D").max(),
        "cloud_cover_mean": hourly_df["cloud_cover"].resample("D").mean(),
        "precipitation_sum": hourly_df["precipitation"].resample("D").sum(),
        "snowfall_sum": hourly_df["snowfall"].resample("D").sum(),
    })

    daily.index.name = "date"

    return daily