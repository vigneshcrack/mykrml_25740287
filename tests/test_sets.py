import pandas as pd

from mykrml_25740287.data.sets import pop_target


def test_pop_target():
    df = pd.DataFrame(
        {
            "feature": [10, 20],
            "label": [0, 1],
        }
    )

    X, y = pop_target(
        df,
        target_col="label",
    )

    assert "label" not in X.columns
    assert y.tolist() == [0, 1]