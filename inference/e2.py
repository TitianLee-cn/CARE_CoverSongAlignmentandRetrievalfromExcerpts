"""Path-specific localization evidence and confidence model."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from align_cover_pairs import span_iou

LEGACY_RULE = {"threshold": 0.60, "margin": 0.15}


def select(options, threshold, margin):
    audio_center = sum(options[0]["interval"]) / 2
    eligible = sorted(
        (p for p in options if p["confidence"] >= threshold),
        key=lambda p: (-p["confidence"], abs(sum(p["interval"]) / 2 - audio_center), *p["interval"]),
    )
    agreeing = [
        (a, b)
        for i, a in enumerate(eligible)
        for b in eligible[i + 1 :]
        if a["source"] != b["source"] and span_iou(a["interval"], b["interval"]) >= 0.5
    ]
    if agreeing:
        a, b = max(agreeing, key=lambda pair: (pair[1]["confidence"], pair[0]["confidence"]))
        return a["interval"]
    if eligible:
        first = eligible[0]
        conflicts = [p for p in eligible[1:] if span_iou(first["interval"], p["interval"]) < 0.5]
        if not conflicts or first["confidence"] - conflicts[0]["confidence"] >= margin:
            return first["interval"]
    return None


SOURCES = ("audio", "harmony", "lyric")


def sources(use_lyrics):
    return SOURCES if use_lyrics else SOURCES[:2]


def feature_names(use_lyrics=True):
    shared = ["other_source_" + source + "_iou" for source in sources(use_lyrics)]
    shared += ["mean_local_audio", "mean_audio_deficit"]
    if use_lyrics:
        shared.append("fraction_aligned_lyric_lines")
    names = {
        "harmony": [
            "match_quality",
            "longest_reliable_fraction",
            "left_extension_sec",
            "right_extension_sec",
            "path_index",
        ],
        "lyric": ["match_score", "anchor_span_fraction", "extension_fraction"],
        "audio": ["window_score", "window_fraction"],
    }
    return {source: names[source] + shared for source in sources(use_lyrics)}


def add_features(options, candidate, audio_scores, query_bounds, reference_bounds, *, use_lyrics=True):
    """Evaluate window/line agreement at each proposed position on the full song."""
    q_center = np.mean(query_bounds, axis=1)
    r_center = np.mean(reference_bounds, axis=1)
    maxima = audio_scores.max(axis=1)
    lines = candidate["lyrics"]["matches"] if use_lyrics else []
    for p in options:
        start, end = p["interval"]
        rate = (end - start) / 30
        nearest = np.abs(r_center[None, :] - (start + rate * q_center[:, None])).argmin(axis=1)
        local = audio_scores[np.arange(len(nearest)), nearest]
        supported = sum(
            abs(np.mean(line["reference_sec"]) - start - rate * np.mean(line["query_sec"])) <= 5
            for line in lines
        )
        agreement = [
            max(
                (
                    span_iou(p["interval"], other["interval"])
                    for other in options
                    if other["source"] == source and source != p["source"]
                ),
                default=0.0,
            )
            for source in sources(use_lyrics)
        ]
        if p["source"] == "harmony":
            base = [
                p["match_quality"],
                p["longest_reliable_sec"] / 30,
                *p["query_extrapolation_sec"],
                p["path_index"],
            ]
        elif p["source"] == "audio":
            base = p["legacy_features"][:2]
        else:
            base = p["legacy_features"]
        p["path_features"] = [
            *base,
            *agreement,
            float(local.mean()),
            float((maxima - local).mean()),
        ]
        if use_lyrics:
            p["path_features"].append(supported / len(lines) if lines else 0.0)


def train(examples, *, use_lyrics=True):
    models = {}
    for source in sources(use_lyrics):
        rows = [(p["path_features"], y) for p, y in examples if p["source"] == source]
        if len({y for _, y in rows}) < 2:
            models[source] = dict(constant=(1 + sum(y for _, y in rows)) / (2 + len(rows)))
            continue
        x, y = zip(*rows)
        scale = StandardScaler().fit(x)
        lr = LogisticRegression(C=0.25).fit(scale.transform(x), y)
        coef = lr.coef_[0] / scale.scale_
        models[source] = dict(coef=coef.tolist(), intercept=float(lr.intercept_[0] - coef @ scale.mean_))
    return models
