import numpy as np

from mykrml_25740287.models.performance import (
    classification_scores,
)


def test_classification_scores():
    y_actual = np.array(
        [0, 0, 1, 1]
    )

    y_probability = np.array(
        [0.1, 0.2, 0.8, 0.9]
    )

    scores = classification_scores(
        y_actual,
        y_probability,
    )

    assert (
        0
        <= scores["average_precision"]
        <= 1
    )

    assert (
        0
        <= scores["roc_auc"]
        <= 1
    )