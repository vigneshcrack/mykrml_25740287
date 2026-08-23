import numpy as np

from mykrml_25740287.features.basketball import (
    college_year_to_numeric,
    height_to_inches,
)


def test_height_to_inches():
    assert height_to_inches("6-7") == 79
    assert height_to_inches("5-11") == 71


def test_invalid_height_returns_nan():
    assert np.isnan(
        height_to_inches("invalid")
    )


def test_college_year_to_numeric():
    assert college_year_to_numeric("Fr") == 1
    assert college_year_to_numeric("Sr") == 4


def test_invalid_college_year_returns_nan():
    assert np.isnan(
        college_year_to_numeric("57.1")
    )