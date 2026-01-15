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

        # calculating distance in loop
        for i, x in enumerate(Xq):
            dists = []
            for j, xt in enumerate(self._X_train):
                # euclides distance for n points
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


    # points over x=y line are blue, under are red
if __name__ == "__main__":
    punkty = [
        [1, 3], [2, 3], [1, 2], [1, 8], [12, 32], [10, 22], [41, 73], [12, 33], [51, 62], [15, 33], [32, 53], [17, 22], [18, 33], [20, 30], [14,18],   # B
        [5, 8], [5, 9], [3, 6], [1, 4], [12, 54], [34, 42], [11, 13], [12, 23], [14, 22], [17, 53], [20, 30], [71, 82], [41, 53], [62, 73], [81, 92],  # B
        [2, 1], [3, 2], [5, 2], [7, 3], [25, 23], [17, 12], [16, 13], [62, 43], [81, 72], [61, 23], [42, 33], [91, 82], [81, 73], [72, 63], [61, 52],  # R
        [1, 3], [2, 3], [1, 2], [1, 3], [27, 17], [19, 12], [91, 33], [29, 23], [41, 12], [66, 48], [58, 33], [21, 12], [31, 23], [42, 33], [51, 42],  # R
    ]
    nazwy = [
        "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", 
        "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R"
    ]

    model = KNNClassifier(k=3).fit(punkty, nazwy)

    test_punkt = [3, 3]
    wynik = model.predict([test_punkt])[0]

    print("KNN demo")
    print(f"Punkt {test_punkt} to: {wynik} (B=niebieski, R=czerwony)")
