"""Exact retrieval using cached unit vectors and a fold's frozen window settings."""

import json
from pathlib import Path

import numpy as np


def best_window_pair(query_vectors, reference_vectors, query_bounds, reference_bounds):
    scores = query_vectors @ reference_vectors.T
    i, j = np.unravel_index(scores.argmax(), scores.shape)
    return float(scores[i, j]), query_bounds[i].tolist(), reference_bounds[j].tolist()


def weighted_window_pair(query_vectors, reference_vectors, query_bounds, reference_bounds, *, weight):
    scores = query_vectors @ reference_vectors.T
    i, j = np.unravel_index(scores.argmax(), scores.shape)
    peak = float(scores[i, j])
    mean = float(scores.max(axis=1).mean(dtype=np.float64))
    return (1 - weight) * peak + weight * mean, query_bounds[i].tolist(), reference_bounds[j].tolist()


def window_offset_interval(query_window, reference_window, duration):
    """Project the 30-second query at unit speed from the winning audio window pair."""
    length = min(30.0, duration)
    start = float(np.clip(reference_window[0] - query_window[0], 0, duration - length))
    return [start, start + length]


def selected_recall(path, fold, folds_sha256):
    data = json.loads(Path(path).read_text())
    if data["folds_sha256"] != folds_sha256:
        raise ValueError(f"{path}: incompatible shared/folds.json")
    record = data["folds"][str(fold)]
    window, overlap = record["window_seconds"], record["overlap_percent"]
    if window not in (10, 15, 20) or overlap not in (25, 50, 75):
        raise ValueError(f"{path}: invalid window for fold {fold}: {window}, {overlap}")
    weight = record.get("lambda", 0)
    if weight not in (0, 0.25, 0.5, 1):
        raise ValueError(f"{path}: invalid recall weight for fold {fold}: {weight}")
    return window, overlap, weight


def selected_window(path, fold, folds_sha256):
    window, overlap, _ = selected_recall(path, fold, folds_sha256)
    return window, overlap
