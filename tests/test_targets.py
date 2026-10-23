import pandas as pd
import pytest

from mykrml_25740287.features.targets import (
    classify_weather_hazard,
    compute_climate_comfort_index,
    compute_weather_hazard_index,
    shift_target,
)


def test_cci_perfect_conditions_scores_100():
    df = pd.DataFrame(
        {
            "temperature_2m_mean": [22.0],
            "relative_humidity_2m_mean": [50.0],
            "wind_speed_10m_mean": [10.0],
            "cloud_cover_mean": [0.0],
            "precipitation_sum": [0.0],
        }
    )

    cci = compute_climate_comfort_index(df)

    assert cci.iloc[0] == pytest.approx(100.0)
    assert cci.name == "cci"


def test_cci_extreme_conditions_clip_to_0():
    df = pd.DataFrame(
        {
            "temperature_2m_mean": [100.0],
            "relative_humidity_2m_mean": [200.0],
            "wind_speed_10m_mean": [100.0],
            "cloud_cover_mean": [200.0],
            "precipitation_sum": [100.0],
        }
    )

    cci = compute_climate_comfort_index(df)

    assert cci.iloc[0] == pytest.approx(0.0)


def test_cci_matches_hand_calculated_value():
    df = pd.DataFrame(
        {
            "temperature_2m_mean": [32.0],
            "relative_humidity_2m_mean": [70.0],
            "wind_speed_10m_mean": [30.0],
            "cloud_cover_mean": [40.0],
            "precipitation_sum": [5.0],
        }
    )

    cci = compute_climate_comfort_index(df)

    assert cci.iloc[0] == pytest.approx(57.25)


def test_whi_calm_day_scores_0():
    df = pd.DataFrame(
        {
            "precipitation_sum": [0.0],
            "wind_gusts_10m_max": [0.0],
            "cloud_cover_mean": [0.0],
            "snowfall_sum": [0.0],
            "temperature_2m_mean": [22.0],
        }
    )

    whi = compute_weather_hazard_index(df)

    assert whi.iloc[0] == pytest.approx(0.0)
    assert whi.name == "whi"


def test_whi_extreme_conditions_clip_to_100():
    df = pd.DataFrame(
        {
            "precipitation_sum": [100.0],
            "wind_gusts_10m_max": [200.0],
            "cloud_cover_mean": [150.0],
            "snowfall_sum": [50.0],
            "temperature_2m_mean": [100.0],
        }
    )

    whi = compute_weather_hazard_index(df)

    assert whi.iloc[0] == pytest.approx(100.0)


def test_whi_matches_hand_calculated_value():
    df = pd.DataFrame(
        {
            "precipitation_sum": [15.0],
            "wind_gusts_10m_max": [50.0],
            "cloud_cover_mean": [50.0],
            "snowfall_sum": [7.5],
            "temperature_2m_mean": [34.5],
        }
    )

    whi = compute_weather_hazard_index(df)

    assert whi.iloc[0] == pytest.approx(50.0)


def test_classify_weather_hazard_category_boundaries():
    whi = pd.Series([0.0, 24.9, 25.0, 49.9, 50.0, 74.9, 75.0, 100.0])

    whc = classify_weather_hazard(whi)

    expected = [
        "Low Risk",
        "Low Risk",
        "Moderate Risk",
        "Moderate Risk",
        "High Risk",
        "High Risk",
        "Extreme Risk",
        "Extreme Risk",
    ]

    assert whc.astype(str).tolist() == expected
    assert whc.name == "whc"


def test_shift_target_aligns_future_value_to_current_row():
    series = pd.Series([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

    shifted = shift_target(series, horizon_days=3)

    assert shifted.iloc[0] == 30
    assert shifted.iloc[6] == 90


def test_shift_target_leaves_trailing_rows_as_nan():
    series = pd.Series([0, 10, 20, 30, 40])

    shifted = shift_target(series, horizon_days=2)

    assert shifted.iloc[3:].isna().all()
    assert shifted.iloc[:3].notna().all()
