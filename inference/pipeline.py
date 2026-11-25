"""Fold-specific Covers80 inference from A's acoustic caches and B's lyric caches."""

import argparse
from collections import Counter
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

CARE = Path(__file__).resolve().parents[1]
DATA = CARE / "precomputed_data"
os.environ.setdefault("CARE_ROOT", str(CARE / "data_weights"))
os.environ.setdefault("CARE_OUTPUT", str(DATA / "group_b"))
os.environ.setdefault("CARE_MANIFESTS", str(Path(os.environ["CARE_OUTPUT"]) / "manifests"))
sys.path[:0] = [str(CARE / "preprocessing"), str(CARE / "b_lyrics")]
from common import shared_inputs, read_jsonl, write_json, write_jsonl, file_sha256
from audio_retrieval import (
    best_window_pair,
    selected_recall,
    weighted_window_pair,
    window_offset_interval,
)
from align_cover_pairs import align, span_iou
import e2
import e2_validation
from group_b.cover_alignment import reliable_blocks
from group_b.pairs import Corpus, detector, fold_sets, predict


class CachedLyrics(Corpus):
    def evidence(self, work, side):
        if (work, side) in self.cache:
            return self.cache[work, side]
        try:
            evidence = super().evidence(work, side)
            duration = 30 if side == "query" else self.refs[work]["duration_sec"]
            for line in evidence[0]:
                bounded([line["start_sec"], line["end_sec"]], duration)
            return evidence
        except FileNotFoundError as exc:
            # Missing per-recording evidence is allowed. Model/manifest errors are fatal.
            if Path(exc.filename).parent not in {
                self.output / side / "text",
                self.output / side / "vocals",
            }:
                raise
            gate = dict(enabled=False, reason="missing_lyric_or_vocal_cache", seconds=0.0, line_ids=[])
            self.cache[work, side] = ([], None, gate if side == "query" else None)
            return self.cache[work, side]


def bounded(span, duration):
    if not (len(span) == 2 and 0 <= span[0] < span[1] <= duration + 1e-7):
        raise ValueError(f"Invalid time interval {span} for duration {duration}")
    return list(map(float, span))


def extrapolate(q0, r0, q1, r1, duration):
    if q1 - q0 < 15 or q0 > 7.5 or 30 - q1 > 7.5 or r1 <= r0:
        return None
    slope = (r1 - r0) / (q1 - q0)
    return bounded([max(0.0, r0 - slope * q0), min(duration, r1 + slope * (30 - q1))], duration)


class Engine:
    def __init__(self, outer, window, overlap, recall_weight=0):
        self.recall_weight = recall_weight
        self.outer = outer
        self.window = window
        self.overlap = overlap
        self.corpus = CachedLyrics({})
        detector()
        recordings, queries = shared_inputs(DATA)
        by_id = {r["recording_id"]: r for r in recordings}
        self.queries = {q["work_id"]: q for q in queries}
        if set(self.queries) != set(self.corpus.queries) or set(self.queries) != set(self.corpus.refs):
            raise ValueError("A/B work sets differ")
        folds = json.loads((DATA / "shared/folds.json").read_text())["folds"]
        self.fold = next(f for f in folds if f["outer_fold"] == outer)
        self.dev, self.test = fold_sets(outer, None)
        if (self.dev, self.test) != (self.fold["development_work_ids"], self.fold["test_work_ids"]):
            raise ValueError("A/B outer folds differ")
        self.vectors, self.bounds, self.beats, self.modes = {}, {}, {}, {}
        roots = [DATA / "cover_embeddings/coverhunter", DATA / "beat_chroma"]
        indexes = [{row["id"]: row for row in read_jsonl(root / "index.jsonl")} for root in roots]
        for work, query in self.queries.items():
            for side, field in (("query", "query_recording_id"), ("reference", "reference_recording_id")):
                rid = query[field]
                b = (self.corpus.queries if side == "query" else self.corpus.refs)[work]
                if rid != b["recording_id"] or b["sha256"] != by_id[rid]["sha256"]:
                    raise ValueError(f"A/B identity mismatch: {rid}")
                if side == "query" and any(
                    b[k] != query[k]
                    for k in ("start_sample", "end_sample", "start_sec", "end_sec", "query_id")
                ):
                    raise ValueError(f"A/B query offsets differ: {rid}")
                offset = query["start_sec"] if side == "query" else 0.0
                duration = 30.0 if side == "query" else by_id[rid]["duration_sec"]
                for root, index in zip(roots, indexes):
                    row = index[rid]
                    if (
                        row["work_id"],
                        row["source_sha256"],
                        row["start_sec"],
                        row["end_sec"],
                        row["source_start_sec"],
                    ) != (work, b["sha256"], 0, duration, offset):
                        raise ValueError(f"Acoustic cache mismatch: {rid}")
                    path = root / row["path"]
                    with np.load(path, allow_pickle=False) as data:
                        if json.loads(data["metadata"].item()) != {
                            k: v for k, v in row.items() if k != "path"
                        }:
                            raise ValueError(f"Shard differs from index: {path}")
                        if root == roots[0]:
                            config = row["configs"].index(
                                dict(window_seconds=window, overlap_percent=overlap)
                            )
                            lo, hi = data["config_offsets"][config : config + 2]
                            ids = data["config_window_rows"][lo:hi]
                            self.vectors[rid] = data["embeddings"][ids]
                            self.bounds[rid] = data["bounds_sec"][ids]
                        else:
                            self.beats[rid] = path
                            self.modes[rid] = data["four_beat_mode"]
        self.bg_data = json.loads(
            (self.corpus.output / "scores/covers80-lyric-evidence.json").read_text()
        )
        if self.bg_data["config"] != self.corpus.cfg:
            raise ValueError("B calibration does not match gate configuration")

    def background(self, inner, fit):
        fold = next(row for row in self.bg_data["folds"] if row["outer_fold"] == self.outer)
        row = fold if inner is None else next(r for r in fold["inner"] if r["inner_fold"] == inner)
        if set(row["fit_work_ids"]) != set(fit):
            raise ValueError("B background was fitted on different works")
        return row["background"]

    @lru_cache(None)
    def audio(self, work, other):
        q, r = self.queries[work]["query_recording_id"], self.queries[other]["reference_recording_id"]
        args = (self.vectors[q], self.vectors[r], self.bounds[q], self.bounds[r])
        if self.recall_weight:
            return weighted_window_pair(*args, weight=self.recall_weight)
        return best_window_pair(*args)

    @lru_cache(None)
    def geometry(self, q, r):
        return align((self.beats[q], self.beats[r]), 0.15, 3, 0.5)

    @lru_cache(None)
    def frequencies(self, gallery):
        counts = Counter()
        for work in gallery:
            modes = self.modes[self.queries[work]["reference_recording_id"]]
            counts.update(set(map(tuple, modes[modes[:, 0] >= 0])))
        return counts

    @lru_cache(None)
    def lyrics(self, work, other, semantic, same):
        result = self.corpus.score_candidate(work, other, dict(semantic=semantic, same_language=same))
        if result["matches"]:
            qlines = {r["line_id"]: r for r in self.corpus.evidence(work, "query")[0]}
            rlines = {r["line_id"]: r for r in self.corpus.evidence(other, "reference")[0]}
            for match in result["matches"]:
                q = qlines[match["query_lines"][0]]
                r = rlines[match["reference_lines"][0]]
                for edge in ("partial_left", "partial_right"):
                    match[edge] = bool(q.get(edge) or r.get(edge))
        return result

    def candidates(self, work, gallery, bg, detailed=False, *, use_lyrics=True):
        gallery = tuple(sorted(gallery))
        ranking = sorted(gallery, key=lambda w: (-self.audio(work, w)[0], w))[:10]
        df = self.frequencies(gallery)
        out = []
        for rank, other in enumerate(ranking, 1):
            q = self.queries[work]["query_recording_id"]
            r = self.queries[other]["reference_recording_id"]
            duration = self.corpus.refs[other]["duration_sec"]
            audio, hit_q, hit_r = self.audio(work, other)
            arrays, meta = self.geometry(q, r)
            paths = []
            harmony = None
            for k, path in enumerate(meta["paths"]):
                lo, hi = arrays["path_offsets"][k : k + 2]
                sl = slice(lo, hi)
                bounds, cosine = arrays["match_bounds_sec"][sl], arrays["cosine"][sl]
                valid, indices = arrays["match_valid"][sl], arrays["beat_indices"][sl]
                blocks = reliable_blocks(bounds, valid & (cosine >= 0.60))
                if k == 0:
                    # Choose one reference beat per query beat; ties use the earliest position.
                    best = {}
                    for n, (i, j) in enumerate(indices):
                        value = float(cosine[n]) if valid[n] else 0.0
                        if i not in best or value > best[i][0]:
                            best[i] = (value, j)
                    total = 0.0
                    for i, (value, j) in best.items():
                        mode = self.modes[q][i]
                        weight = 1.0
                        if mode[0] >= 0 and np.array_equal(mode, self.modes[r][j]):
                            weight += (
                                2
                                * math.log((len(gallery) + 1) / (df[tuple(mode)] + 1))
                                / math.log1p(len(gallery))
                            )
                        total += weight * value
                    harmony = total / (3 * len(self.modes[q]))
                item = dict(
                    path,
                    blocks=blocks,
                    match_quality=1.0 - path["cost"] / len(self.modes[q]),
                    longest_reliable_sec=max((b[-1][0] - b[0][0] for b in blocks), default=0.0),
                    query_extrapolation_sec=[blocks[0][0][0], 30 - blocks[-1][-1][0]] if blocks else None,
                )
                if detailed:
                    item.update(
                        beat_indices=indices.tolist(),
                        match_bounds_sec=bounds.tolist(),
                        cosine=cosine.tolist(),
                        match_valid=valid.tolist(),
                    )
                paths.append(item)
            lyric = (
                self.lyrics(work, other, bg["semantic"], bg["same_language"])
                if use_lyrics
                else {
                    "matches": [],
                    "score": None,
                    "gate": dict(enabled=False, reason="typing_not_requested"),
                }
            )
            for m in lyric["matches"]:
                bounded(m["query_sec"], 30)
                bounded(m["reference_sec"], duration)
            coverage = (
                max((b[-1][0] - b[0][0] for b in paths[0]["blocks"]), default=0.0) / 30 if paths else 0.0
            )
            out.append(
                dict(
                    recall_rank=rank,
                    reference_work_id=other,
                    reference_recording_id=r,
                    audio=audio,
                    query_window=bounded(hit_q, 30),
                    reference_window=bounded(hit_r, duration),
                    harmony=harmony,
                    coverage=coverage,
                    paths=paths,
                    lyrics=lyric,
                    lyric=lyric["score"] if lyric["matches"] and lyric["score"] > 0 else None,
                )
            )
        return out


def curves(samples, *, use_lyrics=True):
    result = {}
    for key in e2.sources(use_lyrics):
        xy = [
            (c[key], int(c["reference_work_id"] == w))
            for w, cs in samples
            for c in cs
            if c[key] is not None
        ]
        if not xy:
            result[key] = [[0.0], [0.0]]
            continue
        x, y = zip(*xy)
        model = IsotonicRegression(out_of_bounds="clip").fit(x, y)
        result[key] = [model.X_thresholds_.tolist(), model.y_thresholds_.tolist()]
    return result


def rerank(rows, calibration, *, harmony_weight=0.3, use_lyrics=True):
    result = []
    for c in rows:
        mapped = {
            k: float(np.interp(c[k], *curve)) if c[k] is not None else None
            for k, curve in calibration.items()
            if k != "lyric" or use_lyrics
        }
        score = (
            mapped["audio"]
            if mapped["harmony"] is None
            else (1 - harmony_weight) * mapped["audio"] + harmony_weight * mapped["harmony"]
        )
        if use_lyrics:
            score += 0.15 * (mapped["lyric"] or 0.0)
        result.append(dict(c, calibrated=mapped, score=score))
    return sorted(result, key=lambda c: (-c["score"], c["recall_rank"]))


def proposals(candidate, duration, *, use_lyrics=True):
    q, r = candidate["query_window"], candidate["reference_window"]
    result = [
        dict(
            source="audio",
            interval=window_offset_interval(q, r, duration),
            legacy_features=[candidate["audio"], (q[1] - q[0]) / 30, 0.0],
        )
    ]

    def anchored(source, first, last, legacy_strength, **evidence):
        q0, r0 = first
        q1, r1 = last
        interval = extrapolate(q0, r0, q1, r1, duration)
        if interval and not any(
            p["source"] == source and span_iou(p["interval"], interval) >= 0.8 for p in result
        ):
            result.append(
                dict(
                    source=source,
                    interval=interval,
                    query_extrapolation_sec=[q0, 30 - q1],
                    reference_extrapolation_sec=[r0 - interval[0], interval[1] - r1],
                    **evidence,
                    legacy_features=[legacy_strength, (q1 - q0) / 30, (q0 + 30 - q1) / 30],
                )
            )

    for index, path in enumerate(candidate["paths"]):
        blocks = path["blocks"]
        if blocks:
            anchored(
                "harmony",
                blocks[0][0],
                blocks[-1][-1],
                candidate["harmony"],
                path_index=index,
                match_quality=path["match_quality"],
                longest_reliable_sec=path["longest_reliable_sec"],
            )
    matches = candidate["lyrics"]["matches"] if use_lyrics else []
    if matches:
        # Discard window-cut line boundaries for extrapolation.
        first = next((m for m in matches if not m["partial_left"]), None)
        last = next((m for m in reversed(matches) if not m["partial_right"]), None)
        if first and last:
            anchored(
                "lyric",
                (first["query_sec"][0], first["reference_sec"][0]),
                (last["query_sec"][1], last["reference_sec"][1]),
                candidate["lyric"] or 0.0,
            )
    return result


def interval_models(examples):
    models = {}
    for source in ("audio", "harmony", "lyric"):
        rows = [(p["legacy_features"], y) for p, y in examples if p["source"] == source]
        if len(set(y for _, y in rows)) < 2:
            models[source] = dict(constant=(1 + sum(y for _, y in rows)) / (2 + len(rows)))
        else:
            x, y = zip(*rows)
            lr = LogisticRegression().fit(x, y)
            models[source] = dict(coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]))
    return models


def score_options(options, models, feature_key):
    for p in options:
        p["features"] = p[feature_key]
        p.pop("legacy_features", None)
        p.pop("path_features", None)
        m = models[p["source"]]
        p["confidence"] = (
            m["constant"]
            if "constant" in m
            else float(1 / (1 + np.exp(-np.dot(m["coef"], p["features"]) - m["intercept"])))
        )
    return options


def position_options(engine, work, candidate, path_model, *, use_lyrics=True):
    other = candidate["reference_work_id"]
    options = proposals(candidate, engine.corpus.refs[other]["duration_sec"], use_lyrics=use_lyrics)
    if path_model:
        q = engine.queries[work]["query_recording_id"]
        r = candidate["reference_recording_id"]
        e2.add_features(
            options,
            candidate,
            engine.vectors[q] @ engine.vectors[r].T,
            engine.bounds[q],
            engine.bounds[r],
            use_lyrics=use_lyrics,
        )
    return options


def decision_features(ranking, *, harmony_weight=0.3):
    top = ranking[0]
    return dict(
        score=top["score"],
        margin=top["score"] - ranking[1]["score"],
        harmony=harmony_weight > 0 and top["harmony"] is not None,
        coverage=top["coverage"] if harmony_weight > 0 else 0.0,
    )


def passes(obs, rule):
    return (
        obs["score"] >= rule["score"]
        and obs["margin"] >= rule["margin"]
        and (not obs["harmony"] or obs["coverage"] >= rule.get("min_harmony_coverage", 0.50))
    )


def confirmation_rules(validation, *, min_harmony_coverage=0.50):
    rules = {}
    for key, has_harmony in (("music", True), ("audio_only", False)):
        group = [
            (obs, known, correct) for obs, known, correct in validation if obs["harmony"] == has_harmony
        ]
        negatives = [obs for obs, known, _ in group if not known]
        positives = [obs for obs, known, correct in group if known and correct]
        best = (-1, -1.0, -1.0, -1.0)
        coverage = dict(min_harmony_coverage=min_harmony_coverage) if has_harmony else {}
        chosen = dict(score=1.16, margin=0.0, **coverage)
        if negatives:
            for s in sorted({0.0, *(v["score"] for v in positives)}):
                for d in sorted({0.0, *(v["margin"] for v in positives)}):
                    rule = dict(score=s, margin=d, **coverage)
                    fp = sum(passes(v, rule) for v in negatives)
                    tp = sum(passes(v, rule) for v in positives)
                    quality = (tp, -fp, d, s)
                    if fp / len(negatives) <= 0.05 and quality > best:
                        best, chosen = quality, rule
        rules[key] = dict(
            chosen,
            validation_open_count=len(negatives),
            validation_false_accepts=sum(passes(v, chosen) for v in negatives),
        )
    return rules


def position_examples(engine, rankings, labels, *, use_lyrics=True):
    examples = []
    for work, ranking in rankings:
        correct = next((c for c in ranking if c["reference_work_id"] == work), None)
        if correct and labels.get(work):
            for option in position_options(engine, work, correct, True, use_lyrics=use_lyrics):
                iou = e2_validation.label_iou(option["interval"], labels[work])
                examples.append((option, int(iou >= 0.5)))
    return examples


def inner_ranked_splits(engine, *, harmony_weight=0.3, lyric_rerank=True, load_lyrics=True):
    """Yield fit data and closed/open rankings from each inner held-out split."""

    def rankings(val, gallery, bg, calibrator):
        for work in val:
            closed = engine.candidates(work, gallery, bg, use_lyrics=load_lyrics)
            without_match = [ref for ref in gallery if ref != work]
            opened = engine.candidates(work, without_match, bg, use_lyrics=load_lyrics)
            yield (
                work,
                rerank(closed, calibrator, harmony_weight=harmony_weight, use_lyrics=lyric_rerank),
                rerank(opened, calibrator, harmony_weight=harmony_weight, use_lyrics=lyric_rerank),
            )

    for sub in engine.fold["inner_folds"]:
        inner = sub["inner_fold"]
        training, val = fold_sets(engine.outer, inner)
        if (
            training != sub["fit_work_ids"]
            or val != sub["validation_work_ids"]
            or set(sub["gallery_work_ids"]) != set(engine.dev)
        ):
            raise ValueError("A/B inner partitions differ")
        bg = engine.background(inner, training)
        fit_rows = [(w, engine.candidates(w, training, bg, use_lyrics=load_lyrics)) for w in training]
        calibrator = curves(fit_rows, use_lyrics=lyric_rerank)
        yield inner, bg, fit_rows, rankings(val, sub["gallery_work_ids"], bg, calibrator)


def fit(
    engine, audit_dir=None, *, harmony_weight=0.3, lyric_rerank=True, e2_lyrics=True, paired_e2=False
):
    # Reference intervals are consulted only for judging proposals on inner validation works.
    labels = {
        r["work_id"]: r["reference_intervals"]
        for r in read_jsonl(DATA / "reference_labels/covers80/reference_intervals.jsonl")
        if r["work_id"] in engine.dev
    }
    observations, positions, held_out = [], [], []
    load_lyrics = lyric_rerank or e2_lyrics
    for inner, bg, fit_rows, pairs in inner_ranked_splits(
        engine, harmony_weight=harmony_weight, lyric_rerank=lyric_rerank, load_lyrics=load_lyrics
    ):
        if audit_dir:
            examples = position_examples(engine, fit_rows, labels, use_lyrics=e2_lyrics)
            models = {"old": interval_models(examples), "new": e2.train(examples, use_lyrics=e2_lyrics)}
        for w, closed, opened in pairs:
            observations.extend(
                [
                    (
                        decision_features(closed, harmony_weight=harmony_weight),
                        True,
                        closed[0]["reference_work_id"] == w,
                    ),
                    (decision_features(opened, harmony_weight=harmony_weight), False, False),
                ]
            )
            if audit_dir:
                top = closed[0]
                opts = position_options(engine, w, top, True, use_lyrics=e2_lyrics)
                held_out.append(
                    dict(
                        work_id=w,
                        query_id=engine.queries[w]["query_id"],
                        inner_fold=inner,
                        top10=[c["reference_recording_id"] for c in closed],
                        winner=top["reference_work_id"],
                        decision=decision_features(closed, harmony_weight=harmony_weight),
                        lyrics_enabled=top["lyrics"]["gate"]["enabled"],
                        truth=labels[w],
                        old=score_options([dict(p) for p in opts], models["old"], "legacy_features"),
                        new=score_options(opts, models["new"], "path_features"),
                    )
                )
            correct = next((c for c in closed if c["reference_work_id"] == w), None)
            if correct and labels.get(w):
                truth = [[x["start_sec"], x["end_sec"]] for x in labels[w]]
                for p in proposals(correct, engine.corpus.refs[w]["duration_sec"], use_lyrics=e2_lyrics):
                    positions.append((p, int(max(span_iou(p["interval"], t) for t in truth) >= 0.5)))
        print(f"Validated inner fold {inner}/4", flush=True)
    bg = engine.background(None, engine.dev)
    final_rows = [(w, engine.candidates(w, engine.dev, bg, use_lyrics=load_lyrics)) for w in engine.dev]
    calibration = curves(final_rows, use_lyrics=lyric_rerank)
    rules = confirmation_rules(observations)
    extra = {}
    if audit_dir:
        for row in held_out:
            obs = row["decision"]
            row["confirmed"] = passes(obs, rules["music" if obs["harmony"] else "audio_only"])
        selection = e2_validation.development(held_out, audit_dir, tune_old=paired_e2)
        full_examples = position_examples(engine, final_rows, labels, use_lyrics=e2_lyrics)
        extra = dict(
            e2_version="path",
            e2_selection=selection,
            feature_names=e2.feature_names(e2_lyrics),
            fit_position_proposals=len(full_examples),
        )
        position_models = e2.train(full_examples, use_lyrics=e2_lyrics)
        if paired_e2:
            extra["e2_selection"], extra["legacy_e2_selection"] = selection
            extra["legacy_position_models"] = interval_models(full_examples)
    else:
        position_models = interval_models(positions)
    return dict(
        **extra,
        harmony_weight=harmony_weight,
        lyric_rerank=lyric_rerank,
        e2_lyrics=e2_lyrics,
        outer_fold=engine.outer,
        recall_lambda=engine.recall_weight,
        window=engine.window,
        overlap=engine.overlap,
        fit_work_ids=engine.dev,
        background=bg,
        curves=calibration,
        confirmation=rules,
        position_models=position_models,
        validation_position_proposals=len(positions),
    )


def ranking_decision(candidates, params):
    weight = params.get("harmony_weight", 0.3)
    ranking = rerank(
        candidates, params["curves"], harmony_weight=weight, use_lyrics=params.get("lyric_rerank", True)
    )
    features = decision_features(ranking, harmony_weight=weight)
    rule = params["confirmation"]["music" if features["harmony"] else "audio_only"]
    return ranking, features, rule, passes(features, rule)


def infer(engine, params, work, *, candidates=None):
    e2_lyrics = params.get("e2_lyrics", True)
    type_only = not e2_lyrics and not params.get("lyric_rerank", True)
    if candidates is None:
        candidates = engine.candidates(
            work, list(engine.queries), params["background"], True, use_lyrics=not type_only
        )
    ranking, features, rule, confirmed = ranking_decision(candidates, params)
    for final_rank, candidate in enumerate(ranking, 1):
        candidate["rank"] = final_rank
    if len({c["reference_recording_id"] for c in ranking}) != 10:
        raise ValueError("Expected ten distinct reference recordings")
    top = ranking[0]
    other = top["reference_work_id"]
    interval, options = None, []
    revised = params.get("e2_version") == "path"
    if confirmed:
        options = position_options(engine, work, top, revised, use_lyrics=e2_lyrics)
        score_options(
            options, params["position_models"], "path_features" if revised else "legacy_features"
        )
        interval = e2.select(options, **params.get("e2_selection", e2.LEGACY_RULE))
    paths = top["paths"]
    chosen = next((p for p in options if p["interval"] == interval and p["source"] == "harmony"), None)
    if revised and chosen:
        selected = paths[chosen["path_index"]]
    elif interval and paths:
        selected = max(paths, key=lambda p: span_iou(p["reference_span_sec"], interval))
    else:
        selected = next(iter(paths), None)
    request = dict(
        work_id=work,
        reference_work_id=other,
        context="end_to_end",
        confirmed=confirmed,
        reference_interval=interval,
        blocks=selected["blocks"] if selected else [],
    )
    type_invoked = not type_only or confirmed
    if type_invoked:
        result = predict(engine.corpus, request, params["background"], mode="fusion")
    else:
        result = dict(
            reference_work_id=other,
            gate=dict(enabled=False, reason="typing_not_requested"),
            type_status="skipped",
            display=[],
        )
    if type_only:
        # The classifier has already consumed the music-aligned lines internally.
        result.pop("matches", None)
        result.pop("score", None)
        result["display"] = []
        for c in ranking:
            c.pop("lyrics", None)
            c.pop("lyric", None)
    for pair in result["display"]:
        bounded(pair["query_sec"], 30)
        bounded(pair["reference_sec"], engine.corpus.refs[other]["duration_sec"])
    return dict(
        outer_fold=engine.outer,
        recall_lambda=engine.recall_weight,
        window=engine.window,
        overlap=engine.overlap,
        query_id=engine.queries[work]["query_id"],
        work_id=work,
        query_recording_id=engine.queries[work]["query_recording_id"],
        query_source_start_sec=engine.queries[work]["start_sec"],
        reference_recording_id=top["reference_recording_id"],
        confirmed=confirmed,
        decision=features,
        threshold=rule,
        reference_interval=interval,
        interval_status="confirmed" if interval else "pending" if confirmed else "unconfirmed_song",
        interval_proposals=options,
        e2_version=params.get("e2_version", "legacy"),
        interval_rule=params.get("e2_selection", e2.LEGACY_RULE),
        music_path_index=paths.index(selected) if selected else None,
        gate=result["gate"],
        prediction=result,
        type_invoked=type_invoked,
        top10=ranking,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit", "predict", "run"))
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--window", type=int, choices=(10, 15, 20))
    parser.add_argument("--overlap", type=int, choices=(25, 50, 75))
    parser.add_argument(
        "--query-id", action="append", default=[], help="Repeatable; default: all held-out queries"
    )
    parser.add_argument(
        "--e2",
        choices=("path", "legacy"),
        default="path",
        help="Path-specific localization (default) or original calibration",
    )
    parser.add_argument("--window-config", type=Path, help="Frozen audio baseline selected_windows.json")
    parser.add_argument("--output-root", type=Path, default=DATA / "inference")
    parser.add_argument("--lyric-rerank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--e2-lyrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--harmony-weight", type=float, default=0.3)
    parser.add_argument("--parameters", type=Path, help="Explicit frozen parameter file for predict")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not 0 <= args.harmony_weight <= 1:
        parser.error("--harmony-weight must be between 0 and 1")
    if args.parameters and args.action != "predict":
        parser.error("--parameters is only supported for predict")
    recall_weight = 0
    if args.window_config:
        if args.window or args.overlap:
            parser.error("Use either --window-config or explicit window settings")
        args.window, args.overlap, recall_weight = selected_recall(
            args.window_config, args.fold, file_sha256(DATA / "shared/folds.json")
        )
    else:
        args.window = args.window or 15
        args.overlap = args.overlap or 50
    engine = Engine(args.fold, args.window, args.overlap, recall_weight)
    ids = {engine.queries[w]["query_id"]: w for w in engine.test}
    if any(q not in ids for q in args.query_id):
        parser.error(f"Select held-out query IDs from {list(ids)}")
    dest = args.output_root / f"fold{args.fold}_w{args.window}_o{args.overlap}"
    if recall_weight:
        dest = dest / f"lambda{recall_weight:g}"
    if args.e2 == "path":
        dest = dest / "e2_path"
    settings = dict(
        harmony_weight=args.harmony_weight, lyric_rerank=args.lyric_rerank, e2_lyrics=args.e2_lyrics
    )
    defaults = dict(harmony_weight=0.3, lyric_rerank=True, e2_lyrics=True)
    if settings != defaults:
        dest /= f"h{args.harmony_weight:g}_lyric{int(args.lyric_rerank)}_e2lyric{int(args.e2_lyrics)}"
    param_path = args.parameters or dest / "parameters.json"
    if args.action in ("fit", "run"):
        params = fit(engine, dest if args.e2 == "path" else None, **settings)
        write_json(param_path, params)
    else:
        params = json.loads(param_path.read_text())
    if not args.parameters and any(params.get(k, defaults[k]) != v for k, v in settings.items()):
        raise ValueError("Parameters do not belong to these scoring switches")
    if params.get("recall_lambda", 0) != recall_weight:
        raise ValueError("Parameters do not belong to this recall weight")
    if (params["outer_fold"], params["window"], params["overlap"], params["fit_work_ids"]) != (
        args.fold,
        args.window,
        args.overlap,
        engine.dev,
    ):
        raise ValueError("Parameters do not belong to this fold/configuration")
    if args.action != "fit":
        works = [ids[q] for q in dict.fromkeys(args.query_id)] or engine.test
        path = args.out or param_path.parent / "predictions.jsonl"
        write_jsonl(path, (infer(engine, params, w) for w in works))
        print(f"Wrote {len(works)} queries to {path}")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
