"""Stable, serializable one-vs-rest event model wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from scripts.extract_anomaly_features import CLASS_NAMES


class IndependentEventForests:
    """Fit one fixed RandomForest classifier per controlled-event class."""

    def __init__(self, spec: dict[str, Any], seed: int) -> None:
        self.spec = dict(spec)
        self.seed = seed
        self.classes_ = tuple(CLASS_NAMES[1:])
        self.estimators_: list[RandomForestClassifier] = []

    def fit(self, features: np.ndarray, labels: list[str]) -> "IndependentEventForests":
        self.estimators_ = []
        for index, event_class in enumerate(self.classes_):
            binary_labels = np.asarray([label == event_class for label in labels], dtype=np.uint8)
            model = RandomForestClassifier(
                **self.spec,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=self.seed + index,
            )
            model.fit(features, binary_labels)
            self.estimators_.append(model)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise ValueError("independent event forests must be fitted before prediction")
        columns = []
        for model in self.estimators_:
            positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
            columns.append(model.predict_proba(features)[:, positive_index])
        return np.column_stack(columns)
