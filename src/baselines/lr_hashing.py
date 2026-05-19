from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OnlineLogisticRegression:
    hash_size: int
    learning_rate: float = 0.05
    l2: float = 1e-6

    def __post_init__(self) -> None:
        self.weights = np.zeros(self.hash_size, dtype=np.float32)
        self.bias = np.float32(0.0)

    def predict_one(self, indices: np.ndarray, values: np.ndarray) -> float:
        logit = float(self.bias + np.dot(self.weights[indices], values))
        return sigmoid(logit)

    def update_one(self, indices: np.ndarray, values: np.ndarray, label: int) -> float:
        prediction = self.predict_one(indices, values)
        error = prediction - float(label)
        self.weights[indices] -= self.learning_rate * (
            error * values + self.l2 * self.weights[indices]
        )
        self.bias -= np.float32(self.learning_rate * error)
        return prediction


def sigmoid(value: float) -> float:
    if value >= 0:
        z = np.exp(-value)
        return float(1.0 / (1.0 + z))
    z = np.exp(value)
    return float(z / (1.0 + z))

