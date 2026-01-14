import numpy as np
import pytest

from knn_from_nothing.knn import KNNClassifier


def test_simple_case_majority_vote():
    X = np.array([[0, 0], [0, 1], [1, 0], [10, 10]], dtype=float)
    y = np.array(["A", "A", "A", "B"], dtype=object)

    model = KNNClassifier(k=3).fit(X, y)
    pred = model.predict([[0.2, 0.1], [9, 9]])

    assert pred[0] == "A"
    assert pred[1] == "B"


def test_invalid_k_raises():
    X = np.array([[0, 0], [1, 1]], dtype=float)
    y = np.array([0, 1], dtype=object)
    with pytest.raises(ValueError):
        KNNClassifier(k=0).fit(X, y)


def test_feature_mismatch_raises():
    X = np.array([[0, 0], [1, 1]], dtype=float)
    y = np.array([0, 1], dtype=object)
    model = KNNClassifier(k=1).fit(X, y)

    with pytest.raises(ValueError):
        model.predict([[0, 0, 0]])


def test_compare_with_sklearn_iris():
    # scikit-learn is a project requirement, but we keep the test skippable
    # for environments where it's not installed.
    sklearn = pytest.importorskip("sklearn")

    from sklearn.datasets import load_iris
    from sklearn.neighbors import KNeighborsClassifier

    iris = load_iris()
    X = iris.data
    y = iris.target

    # deterministic split
    X_train, X_test = X[:120], X[120:]
    y_train, y_test = y[:120], y[120:]

    k = 5
    my = KNNClassifier(k=k).fit(X_train, y_train)
    sk = KNeighborsClassifier(n_neighbors=k, metric="minkowski", p=2).fit(X_train, y_train)

    my_pred = my.predict(X_test)
    sk_pred = sk.predict(X_test)

    agreement = float(np.mean(my_pred == sk_pred))
    assert agreement >= 0.9

    my_acc = float(np.mean(my_pred == y_test))
    sk_acc = float(np.mean(sk_pred == y_test))

    assert my_acc >= 0.7
    assert sk_acc >= 0.7
