import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes = None) -> np.ndarray:
    """"
    Computes the confusion matrix from labels (y_true) and predictions (y_pred).
    The matrix columns represent the prediction labels and the rows represent the ground truth labels.
    The confusion matrix is always a 2-D array of shape `[num_classes, num_classes]`,
    where `num_classes` is the number of valid labels for a given classification task.
    The arguments y_true and y_pred must have the same shapes in order for this function to work

    num_classes represents the number of classes for the classification problem. If this is not provided,
    it will be computed from both y_true and y_pred
    """
    conf_mat = None
    # even here try to use vectorization, so NO for loops

    # 0. if the number of classes is not provided, compute it based on the y_true and y_pred arrays
    num_classes = max(y_pred.max(),y_true.max()) + 1
    # 1. create a confusion matrix of shape (num_classes, num_classes) and initialize it to 0
    conf_mat = np.zeros((num_classes,num_classes))
    # 2. use argmax to get the maximal prediction for each sample
    # hint: you might find np.add.at useful: https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    # predicted_classes = np.argmax(y_pred, axis=1)
    np.add.at(conf_mat, (y_pred, y_true), 1)
    return conf_mat


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes=None) -> float:
    """"
    Computes the precision score.
    For binary classification, the precision score is defined as the ratio tp / (tp + fp)
    where tp is the number of true positives and fp the number of false positives.

    For multiclass classification, the precision and recall scores are obtained by summing over the rows / columns
    of the confusion matrix.

    num_classes represents the number of classes for the classification problem. If this is not provided,
    it will be computed from both y_true and y_pred
    """
    precision = 0
    # remember, use vectorization, so no for loops
    conf_mat = confusion_matrix(y_true, y_pred, num_classes)
    precision = np.diag(conf_mat) / np.sum(conf_mat, axis=1)
    return np.mean(precision)


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes=None)  -> float:
    """"
    Computes the recall score.
    For binary classification, the recall score is defined as the ratio tp / (tp + fn)
    where tp is the number of true positives and fn the number of false negatives

    For multiclass classification, the precision and recall scores are obtained by summing over the rows / columns
    of the confusion matrix.

    num_classes represents the number of classes for the classification problem. If this is not provided,
    it will be computed from both y_true and y_pred
    """
    recall = 0
    # remember, use vectorization, so no for loops\
    conf_mat = confusion_matrix(y_true, y_pred, num_classes)
    recall = np.diag(conf_mat) / np.sum(conf_mat, axis=0)
    return np.mean(recall)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    acc_score = 0
    # remember, use vectorization, so no for loops
    # hint: you might find np.trace useful here https://numpy.org/doc/stable/reference/generated/numpy.trace.html
    true_pred = np.sum(y_true == y_pred)
    false_pred = np.sum(y_true != y_pred)
    acc_score = true_pred / (true_pred + false_pred)

    return acc_score

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1 = 0
    # remember, use vectorization, so no for loops
    # hint: you might find np.trace useful here https://numpy.org/doc/stable/reference/generated/numpy.trace.html
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

if __name__ == '__main__':
    # make some tests for your code
    y_true = np.array([0, 1, 2, 2, 2])
    y_pred = np.array([0, 0, 2, 2, 1])
    print("confusion matrix")
    print(confusion_matrix(y_true, y_pred, num_classes=3))
    print("precision")
    print(precision_score(y_true, y_pred, num_classes=3))
    print("recall")
    print(recall_score(y_true, y_pred, num_classes=3))
    print("accuracy")
    print(accuracy_score(y_true, y_pred))
    # you could use the sklean.metrics module (with macro averaging to check your results)