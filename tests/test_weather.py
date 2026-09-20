from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mykrml_25740287.data.weather import aggregate_daily_weather, fetch_historical_weather


def test_aggregate_daily_weather_applies_correct_aggregation_per_variable():
    hours = pd.date_range("2024-01-01", periods=48, freq="h")

    hourly = pd.DataFrame(
        {
            "temperature_2m": [20.0] * 24 + [25.0] * 24,
            "relative_humidity_2m": [60.0] * 24 + [70.0] * 24,
            "wind_speed_10m": [5.0] * 24 + [8.0] * 24,
            "wind_gusts_10m": list(range(24)) + [i + 10 for i in range(24)],
            "cloud_cover": [30.0] * 24 + [90.0] * 24,
            "precipitation": [0.5] * 24 + [1.0] * 24,
            "snowfall": [0.0] * 24 + [0.2] * 24,
        },
        index=hours,
    )

    daily = aggregate_daily_weather(hourly)

    assert list(daily.columns) == [
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "wind_speed_10m_mean",
        "wind_gusts_10m_max",
        "cloud_cover_mean",
        "precipitation_sum",
        "snowfall_sum",
    ]
    assert daily.index.name == "date"
    assert len(daily) == 2

    day1 = daily.iloc[0]
    assert day1["temperature_2m_mean"] == pytest.approx(20.0)
    assert day1["relative_humidity_2m_mean"] == pytest.approx(60.0)
    assert day1["wind_speed_10m_mean"] == pytest.approx(5.0)
    assert day1["wind_gusts_10m_max"] == pytest.approx(23.0)
    assert day1["cloud_cover_mean"] == pytest.approx(30.0)
    assert day1["precipitation_sum"] == pytest.approx(12.0)
    assert day1["snowfall_sum"] == pytest.approx(0.0)

    day2 = daily.iloc[1]
    assert day2["temperature_2m_mean"] == pytest.approx(25.0)
    assert day2["relative_humidity_2m_mean"] == pytest.approx(70.0)
    assert day2["wind_speed_10m_mean"] == pytest.approx(8.0)
    assert day2["wind_gusts_10m_max"] == pytest.approx(33.0)
    assert day2["cloud_cover_mean"] == pytest.approx(90.0)
    assert day2["precipitation_sum"] == pytest.approx(24.0)
    assert day2["snowfall_sum"] == pytest.approx(4.8)


def _mock_response(payload):
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


@patch("mykrml_25740287.data.weather.requests.get")
def test_fetch_historical_weather_parses_hourly_payload(mock_get):
    payload = {
        "hourly": {
            "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
            "temperature_2m": [20.0, 21.0],
            "relative_humidity_2m": [60, 61],
            "wind_speed_10m": [5, 6],
            "wind_gusts_10m": [10, 11],
            "cloud_cover": [30, 31],
            "precipitation": [0.0, 0.1],
            "snowfall": [0.0, 0.0],
        }
    }
    mock_get.return_value = _mock_response(payload)

    df = fetch_historical_weather("2024-01-01", "2024-01-01")

    assert list(df.index) == list(pd.to_datetime(payload["hourly"]["time"]))
    assert df["temperature_2m"].tolist() == [20.0, 21.0]
    assert df["wind_gusts_10m"].tolist() == [10, 11]


@patch("mykrml_25740287.data.weather.requests.get")
def test_fetch_historical_weather_uses_sydney_defaults(mock_get):
    payload = {
        "hourly": {
            "time": ["2024-01-01T00:00"],
            "temperature_2m": [20.0],
            "relative_humidity_2m": [60],
            "wind_speed_10m": [5],
            "wind_gusts_10m": [10],
            "cloud_cover": [30],
            "precipitation": [0.0],
            "snowfall": [0.0],
        }
    }
    mock_get.return_value = _mock_response(payload)

    fetch_historical_weather("2024-01-01", "2024-01-01")

    called_params = mock_get.call_args.kwargs["params"]
    assert called_params["latitude"] == pytest.approx(-33.8688)
    assert called_params["longitude"] == pytest.approx(151.2093)
    assert called_params["timezone"] == "Australia/Sydney"
    assert "wind_gusts_10m" in called_params["hourly"]


@patch("mykrml_25740287.data.weather.requests.get")
def test_fetch_historical_weather_raises_on_api_error(mock_get):
    payload = {"error": True, "reason": "Invalid date range"}
    mock_get.return_value = _mock_response(payload)

    with pytest.raises(ValueError, match="Invalid date range"):
        fetch_historical_weather("2024-01-01", "2024-01-01")
