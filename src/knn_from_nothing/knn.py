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
                [0,10],[1,11],[2,12],[3,13],[4,14],[5,15],[6,16],[7,17],[8,18],[9,19],[99,100] ,
                [10,20],[11,21],[12,22],[13,23],[14,24],[15,25],[16,26],[17,27],[18,28],[19,29],
                [20,30],[21,31],[22,32],[23,33],[24,34],[25,35],[26,36],[27,37],[28,38],[29,39],
                [30,40],[31,41],[32,42],[33,43],[34,44],[35,45],[36,46],[37,47],[38,48],[39,49],
                [40,50],[41,51],[42,52],[43,53],[44,54],[45,55],[46,56],[47,57],[48,58],[49,59],
                [50,60],[51,61],[52,62],[53,63],[54,64],[55,65],[56,66],[57,67],[58,68],[59,69],
                [60,70],[61,71],[62,72],[63,73],[64,74],[65,75],[66,76],[67,77],[68,78],[69,79],
                [70,80],[71,81],[72,82],[73,83],[74,84],[75,85],[76,86],[77,87],[78,88],[79,89],
                [80,90],[81,91],[82,92],[83,93],[84,94],[85,95],[86,96],[87,97],[88,98],[89,99],
                [90,100],[91,100],[92,100],[93,100],[94,100],[95,100],[96,100],[97,100],[98,100], # B



                [98, 81],[95, 14],[83, 3],[74, 35],[96, 31],[63, 28],[77, 17],[97, 86],[94, 69],[64, 11],
                [90, 54],[79, 4],[85, 27],[92, 58],[61, 12],[80, 43],[78, 25],[57, 6],[89, 50],[66, 19],
                [88, 71],[59, 21],[70, 9],[84, 39],[60, 7],[99, 65],[87, 46],[68, 33],[82, 23],[93, 48],
                [52, 5],[91, 62],[73, 15],[86, 29],[76, 40],[95, 60],[67, 10],[81, 55],[75, 34],[56, 1],
                [100, 72],[65, 22],[83, 37],[69, 18],[90, 45],[62, 8],[88, 53],[79, 30],[71, 16],[84, 41],
                [96, 68],[73, 24],[85, 32],[92, 57],[63, 20],[89, 49],[66, 13],[80, 44],[77, 26],[91, 36],
                [98, 70],[87, 47],[54, 2],[95, 61],[82, 38],[86, 51],[99, 66],[78, 34],[72, 28],[88, 42],
                [97, 73],[68, 15],[94, 59],[55, 6],[83, 52],[90, 31],[76, 23],[99, 64],[60, 11],[98, 74],
                [81, 27],[70, 17],[93, 56],[58, 4],[85, 48],[77, 35],[66, 21],[96, 67],[62, 9],[100, 75]   # R
]   
    ]
    nazwy = [
        "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", 
        "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", 
        "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", 
        "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B", 
     
        "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R",
        "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R",
        "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R",
        "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R",
    ]

    model = KNNClassifier(k=3).fit(punkty, nazwy)

    test_punkt = [3, 3]
    wynik = model.predict([test_punkt])[0]

    print("KNN demo")
    print(f"Punkt {test_punkt} to: {wynik} (B=niebieski, R=czerwony)")
