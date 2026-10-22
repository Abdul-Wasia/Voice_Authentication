"""Metrics for evaluating speaker verification performance."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np


def accuracy_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def precision_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    if tp + fp == 0:
        return 0.0
    return float(tp / (tp + fp))


def recall_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp + fn == 0:
        return 0.0
    return float(tp / (tp + fn))


def far_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)
    negatives = np.sum(y_true == 0)
    if negatives == 0:
        return 0.0
    false_accepts = np.sum((y_true == 0) & (y_pred == 1))
    return float(false_accepts / negatives)


def frr_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)
    positives = np.sum(y_true == 1)
    if positives == 0:
        return 0.0
    false_rejections = np.sum((y_true == 1) & (y_pred == 0))
    return float(false_rejections / positives)


def equal_error_rate(scores: Iterable[float], labels: Iterable[int]) -> float:
    scores = np.asarray(list(scores), dtype=float)
    labels = np.asarray(list(labels), dtype=int)
    thresholds = np.linspace(scores.min(), scores.max(), 500)
    best_gap = float("inf")
    eer = 0.0
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        far = far_score(labels, predictions)
        frr = frr_score(labels, predictions)
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap = gap
            eer = (far + frr) / 2.0
    return float(eer)


def compute_metrics(y_true: Iterable[int], y_pred: Iterable[int], scores: Iterable[float] | None = None) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "far": far_score(y_true, y_pred),
        "frr": frr_score(y_true, y_pred),
        "eer": equal_error_rate(scores if scores is not None else [0.0] * len(list(y_true)), list(y_true)) if scores is not None else 0.0,
    }
