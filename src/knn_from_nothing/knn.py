from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence

import numpy as np


def _to_2d_array(X: Any, name: str) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D array-like, got shape={X.shape}")
    if X.shape[0] == 0:
        raise ValueError(f"{name} must have at least one sample")
    return X


def _to_1d_array(y: Any, name: str) -> np.ndarray:
    y = np.asarray(y, dtype=object)
    if y.ndim != 1:
        raise ValueError(f"{name} must be 1D array-like, got shape={y.shape}")
    if y.shape[0] == 0:
        raise ValueError(f"{name} must have at least one label")
    return y


def _majority_vote(labels: np.ndarray) -> Any:
    # Deterministyczny tie-break: pierwszy label w kolejności najbliższych
    counts = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    m = max(counts.values())
    for lab in labels:
        if counts[lab] == m:
            return lab
    return labels[0]


@dataclass
class KNNClassifier:
    k: int = 3

    _X_train: np.ndarray | None = None
    _y_train: np.ndarray | None = None

    def fit(self, X: Any, y: Any) -> "KNNClassifier":
        X = _to_2d_array(X, "X")
        y = _to_1d_array(y, "y")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y lengths do not match")
        if not isinstance(self.k, (int, np.integer)) or self.k <= 0:
            raise ValueError("k must be a positive integer")

        self._X_train = X
        self._y_train = y
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("Call fit(X, y) first.")

        Xq = _to_2d_array(X, "X")
        if Xq.shape[1] != self._X_train.shape[1]:
            raise ValueError("Feature count mismatch vs training data")

        preds = np.empty((Xq.shape[0],), dtype=object)

        # Prosty, czytelny wariant (bez optymalizacji): liczymy dystanse w pętli
        for i, x in enumerate(Xq):
            dists = []
            for j, xt in enumerate(self._X_train):
                # dystans euklidesowy dla N wymiarów
                s = 0.0
                for a, b in zip(x, xt):
                    diff = a - b
                    s += diff * diff
                d = math.sqrt(s)
                dists.append((d, j))

            dists.sort(key=lambda t: t[0])
            k_eff = min(self.k, len(dists))
            nn_idx = [idx for _, idx in dists[:k_eff]]
            nn_labels = self._y_train[nn_idx]
            preds[i] = _majority_vote(nn_labels)

        return preds

    def score(self, X: Any, y: Any) -> float:
        y_true = _to_1d_array(y, "y")
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y_true))


# Zostawiamy proste demo, ale nie jest ono “rdzeniem” projektu
if __name__ == "__main__":
    punkty = [
        [1, 3], [2, 2], [1, 1],  # F
        [5, 5], [6, 4], [4, 6]   # V
    ]
    nazwy = ["F", "F", "F", "V", "V", "V"]

    model = KNNClassifier(k=3).fit(punkty, nazwy)

    test_punkt = [3, 3]
    wynik = model.predict([test_punkt])[0]

    print("KNN demo")
    print(f"Punkt {test_punkt} to: {wynik} (F=owoc, V=warzywo)")
