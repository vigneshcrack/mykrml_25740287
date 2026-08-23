"""Basketball-specific feature engineering utilities."""

import numpy as np
import pandas as pd


def height_to_inches(value):
    """Convert height such as '6-7' into total inches."""
    if pd.isna(value):
        return np.nan

    try:
        feet, inches = str(value).split("-")

        feet = int(feet)
        inches = int(inches)

        if not 0 <= inches <= 11:
            return np.nan

        return feet * 12 + inches

    except (ValueError, TypeError):
        return np.nan


def college_year_to_numeric(value):
    """Convert college year category to an ordinal value."""
    mapping = {
        "Fr": 1,
        "So": 2,
        "Jr": 3,
        "Sr": 4,
    }

    return mapping.get(value, np.nan)


def add_basketball_features(df):
    """Create reusable engineered basketball features."""
    df_copy = df.copy()

    if "ht" in df_copy.columns:
        df_copy["height_inches"] = (
            df_copy["ht"]
            .apply(height_to_inches)
        )

    if "yr" in df_copy.columns:
        df_copy["college_year"] = (
            df_copy["yr"]
            .apply(college_year_to_numeric)
        )

    return df_copy