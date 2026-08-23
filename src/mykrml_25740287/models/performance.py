"""Classification performance utilities for AT1."""

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_scores(
    y_actual,
    y_probability,
    threshold=0.5,
):
    """Calculate probability-based and threshold-based classification metrics."""

    y_prediction = (
        y_probability >= threshold
    ).astype(int)

    return {
        "average_precision":
            average_precision_score(
                y_actual,
                y_probability,
            ),

        "roc_auc":
            roc_auc_score(
                y_actual,
                y_probability,
            ),

        "precision":
            precision_score(
                y_actual,
                y_prediction,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_actual,
                y_prediction,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y_actual,
                y_prediction,
                zero_division=0,
            ),
    }


def print_classifier_scores(
    y_actual,
    y_probability,
    set_name=None,
):
    """Print classification performance in a concise experiment format."""

    scores = classification_scores(
        y_actual,
        y_probability,
    )

    prefix = (
        f"{set_name} - "
        if set_name
        else ""
    )

    print(
        f"{prefix}Average Precision: "
        f"{scores['average_precision']:.4f}"
    )

    print(
        f"{prefix}ROC-AUC: "
        f"{scores['roc_auc']:.4f}"
    )

    print(
        f"{prefix}Precision: "
        f"{scores['precision']:.4f}"
    )

    print(
        f"{prefix}Recall: "
        f"{scores['recall']:.4f}"
    )

    print(
        f"{prefix}F1: "
        f"{scores['f1']:.4f}"
    )