"""Classification performance utilities for AT1."""

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
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



def regression_scores(y_actual, y_predicted):
    """Calculate regression performance metrics."""
    return {
        "rmse": root_mean_squared_error(y_actual, y_predicted),
        "mae": mean_absolute_error(y_actual, y_predicted),
    }


def print_regressor_scores(y_actual, y_predicted, set_name=None):
    """Print regression performance in a concise experiment format."""
    scores = regression_scores(y_actual, y_predicted)

    prefix = f"{set_name} - " if set_name else ""

    print(f"{prefix}RMSE: {scores['rmse']:.4f}")
    print(f"{prefix}MAE: {scores['mae']:.4f}")


def multiclass_classification_scores(y_actual, y_predicted):
    """Calculate multiclass classification performance metrics."""
    return {
        "accuracy": accuracy_score(y_actual, y_predicted),
        "f1_weighted": f1_score(y_actual, y_predicted, average="weighted", zero_division=0),
        "precision_weighted": precision_score(y_actual, y_predicted, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_actual, y_predicted, average="weighted", zero_division=0),
    }


def print_multiclass_classifier_scores(y_actual, y_predicted, set_name=None):
    """Print multiclass classification performance in a concise experiment format."""
    scores = multiclass_classification_scores(y_actual, y_predicted)

    prefix = f"{set_name} - " if set_name else ""

    print(f"{prefix}Accuracy: {scores['accuracy']:.4f}")
    print(f"{prefix}F1 (weighted): {scores['f1_weighted']:.4f}")
    print(f"{prefix}Precision (weighted): {scores['precision_weighted']:.4f}")
    print(f"{prefix}Recall (weighted): {scores['recall_weighted']:.4f}")