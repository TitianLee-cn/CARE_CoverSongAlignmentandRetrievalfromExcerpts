"""Refit and evaluate complete CARE after swapping the Unified and CoverHunter roles."""

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import sys
from time import perf_counter_ns

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path.cwd() / "inference"))
import pipeline
from audio_retrieval import selected_recall, window_offset_interval
from confirmation_coverage import counts as confirmation_counts, search as search_confirmation
from coverhunter_tuning import check_rankings
from e2_validation import label_iou
from frozen_reevaluation import care_metrics
from lyrics_ablation import open_prediction
from verify import verify

DATA = pipeline.DATA
DEST = DATA / "experiments/unified_role_swap"
CONFIG = DEST / "unified_recall_config.json"
HISTORICAL = DATA / "reference_labels/covers80/reference_intervals_before_23_review.jsonl"
LABELS = DATA / "reference_labels/covers80/reference_intervals.jsonl"
REVIEW = DATA / "review/e2_no_abstention_23_listening/manual_review.jsonl"
PAIR_ROOT = DATA / "cover_pair_alignment"
original_align = pipeline.align
cached_pairs = {}


def load_pair_cache():
    rows = pipeline.read_jsonl(PAIR_ROOT / "index.jsonl")
    assert len(rows) == 80 and not pipeline.read_jsonl(PAIR_ROOT / "failures.jsonl")
    for row in rows:
        ids = (row["query_recording_id"], row["reference_recording_id"])
        for side, recording in zip(("query", "reference"), ids, strict=True):
            feature = DATA / "beat_chroma/items" / f"{recording}.npz"
            assert pipeline.file_sha256(feature) == row[f"{side}_beat_sha256"], recording
        with np.load(PAIR_ROOT / row["path"], allow_pickle=False) as shard:
            metadata = json.loads(shard["metadata"].item())
            assert metadata == {k: v for k, v in row.items() if k != "path"}, ids
            arrays = {key: shard[key] for key in shard.files if key != "metadata"}
        cached_pairs[ids] = arrays, metadata
    assert len(cached_pairs) == 80


@lru_cache(maxsize=2400)
def use_a_paths(source, penalty, keep, alternate_iou):
    ids = tuple(path.stem for path in source)
    if (penalty, keep, alternate_iou) == (0.15, 3, 0.5) and ids in cached_pairs:
        return cached_pairs[ids]
    return original_align(source, penalty, keep, alternate_iou)


def engine_for(fold):
    number = fold["outer_fold"]
    digest = pipeline.file_sha256(DATA / "shared/folds.json")
    return pipeline.Engine(number, *selected_recall(CONFIG, number, digest), audio_model="unified")


def fit_fold(fold):
    number = fold["outer_fold"]
    root = DEST / "care" / f"fold{number}"
    with threadpool_limits(limits=1):
        engine = engine_for(fold)
        params = pipeline.fit(engine, root, labels_path=HISTORICAL)
    params.update(
        folds_sha256=pipeline.file_sha256(DATA / "shared/folds.json"),
        recall_config_sha256=pipeline.file_sha256(CONFIG),
        interval_fit_labels_sha256=pipeline.file_sha256(HISTORICAL),
        audio_index_sha256=pipeline.file_sha256(DATA / "cover_embeddings/unified/index.jsonl"),
        beat_index_sha256=pipeline.file_sha256(DATA / "beat_chroma/index.jsonl"),
    )
    pipeline.write_json(root / "parameters.json", params)
    print(f"Fitted Unified CARE fold {number}", flush=True)


def tune_confirmation(fold):
    number = fold["outer_fold"]
    root = DEST / "care" / f"fold{number}"
    params = json.loads((root / "parameters.json").read_text())
    observations = []
    with threadpool_limits(limits=1):
        engine = engine_for(fold)
        for _, _, _, pairs in pipeline.inner_ranked_splits(engine):
            for work, closed, opened in pairs:
                observations.append((
                    pipeline.decision_features(closed), True, closed[0]["reference_work_id"] == work,
                ))
                observations.append((pipeline.decision_features(opened), False, False))
        control = pipeline.confirmation_rules(observations)
        assert control == params["confirmation"] and len(observations) == 128
        tuned, best_by_coverage, sweep = search_confirmation(observations, control)
    pipeline.write_json(root / "tuned_parameters.json", dict(params, confirmation=tuned))
    pipeline.write_json(root / "confirmation_selection.json", {
        "control": confirmation_counts(observations, control),
        "tuned": confirmation_counts(observations, tuned),
        "best_by_coverage": best_by_coverage,
        "settings_tested": len(sweep),
    })
    print(f"Retuned confirmation fold {number}: {tuned['music']['min_harmony_coverage']:.0%}", flush=True)


def predict_fold(fold, audio_rows):
    number = fold["outer_fold"]
    root = DEST / "care" / f"fold{number}"
    param_path = root / "tuned_parameters.json"
    params = json.loads((param_path if param_path.is_file() else root / "parameters.json").read_text())
    with threadpool_limits(limits=1):
        engine = engine_for(fold)
        assert params["audio_model"] == engine.audio_model
        assert params["fit_work_ids"] == engine.dev and params["outer_fold"] == number
        assert params["recall_config_sha256"] == pipeline.file_sha256(CONFIG)
        # Warm the DTW compiler on a development pair which has no precomputed path.
        warm = engine.queries[engine.dev[0]]["query_recording_id"]
        other = engine.queries[engine.dev[1]]["reference_recording_id"]
        engine.geometry(warm, other)
        forced_params = deepcopy(params)
        for rule in forced_params["confirmation"].values():
            rule.update(score=0.0, margin=0.0, min_harmony_coverage=0.0)
        forced_params["e2_selection"] = {"threshold": 0.0, "margin": 0.0}
        closed, opened, forced, forced_opened = [], [], [], []
        for work in engine.test:
            start = perf_counter_ns()
            row = pipeline.infer(engine, params, work)
            row["inference_ns"] = perf_counter_ns() - start
            raw_order = [
                dict(rank=c["recall_rank"], reference_recording_id=c["reference_recording_id"],
                     score=c["audio"], query_window=c["query_window"],
                     reference_window=c["reference_window"])
                for c in sorted(row["top10"], key=lambda item: item["recall_rank"])
            ]
            check_rankings(raw_order, audio_rows[row["query_id"]]["top10"])
            closed.append(row)
            candidates = engine.candidates(
                work, [w for w in fold["test_gallery_work_ids"] if w != work],
                params["background"], detailed=True,
            )
            opened.append(open_prediction(engine, candidates, params, work))
            forced_row = pipeline.infer(engine, forced_params, work)
            forced_row["inference_ns"] = row["inference_ns"]
            forced.append(forced_row)
            forced_open = pipeline.infer(engine, forced_params, work, candidates=candidates)
            forced_opened.append(forced_open)
            assert all(r["confirmed"] and r["reference_interval"] for r in (forced_row, forced_open))

    for name, rows in (
        ("predictions", closed), ("open_predictions", opened),
        ("forced_predictions", forced), ("forced_open_predictions", forced_opened),
    ):
        pipeline.write_jsonl(root / f"{name}.jsonl", rows)
    verify(root / "predictions.jsonl", score=False)
    verify(root / "forced_predictions.jsonl", score=False)
    print(f"Verified held-out predictions for fold {number}", flush=True)
    return closed, opened, forced, forced_opened


def interval_report(rows, labels, durations, *, audio_only=False):
    scored = []
    for row in rows:
        label = labels[row["work_id"]]
        if audio_only:
            winner = row["top10"][0]
            recording = winner["reference_recording_id"]
            interval = window_offset_interval(
                winner["query_window"], winner["reference_window"], durations[recording],
            )
        else:
            recording, interval = row["reference_recording_id"], row["reference_interval"]
        correct = recording == label["reference_recording_id"]
        iou = label_iou(interval, label["reference_intervals"]) if correct and interval else None
        scored.append(dict(
            query_id=row["query_id"], outer_fold=row["outer_fold"], work_id=row["work_id"],
            reference_recording_id=recording, reference_interval=interval,
            annotation_status=label["annotation_status"], correct_song=correct,
            iou=iou, hit=iou is not None and iou >= 0.5,
        ))
    return scored, dict(
        queries=len(scored),
        correct_song=sum(row["correct_song"] for row in scored),
        interval_outputs=sum(row["reference_interval"] is not None for row in scored),
        correct_song_and_interval=sum(row["hit"] for row in scored),
        manual_subset=sum(row["annotation_status"] == "manual" for row in scored),
        manual_subset_hits=sum(row["hit"] and row["annotation_status"] == "manual" for row in scored),
    )


def report(batches):
    closed, opened, forced, forced_opened = [
        [row for result in batches for row in result[i]] for i in range(4)
    ]
    assert all(len(rows) == len({row["query_id"] for row in rows}) == 80
               for rows in (closed, opened, forced, forced_opened))
    labels = {row["work_id"]: row for row in pipeline.read_jsonl(LABELS)}
    excluded = {
        row["query_id"] for row in pipeline.read_jsonl(REVIEW)
        if row["verdict"] == "reannotation_needed_both_intervals_incorrect"
    }
    for row in labels.values():
        if row["query_id"] in excluded:
            row["reference_intervals"] = []
    intervals = {work: row["reference_intervals"] for work, row in labels.items()}
    recordings, _ = pipeline.shared_inputs(DATA)
    durations = {row["recording_id"]: row["duration_sec"] for row in recordings}
    summary = dict(
        interval_fit_labels_sha256=pipeline.file_sha256(HISTORICAL),
        interval_scoring_labels_sha256=pipeline.file_sha256(LABELS),
        pending_interval_queries=sorted(excluded),
        care_confirmation=care_metrics(closed, opened, intervals),
        care_by_fold={
            fold: care_metrics(result[0], result[1], intervals)
            for fold, result in enumerate(batches, 1)
        },
        forced_open_intervals=sum(row["reference_interval"] is not None for row in forced_opened),
        interval_comparison={},
    )
    for model in ("unified_care", "unified", "coverhunter", "mert"):
        rows = forced if model == "unified_care" else pipeline.read_jsonl(DEST / f"{model}_top10.jsonl")
        scored, metrics = interval_report(rows, labels, durations, audio_only=model != "unified_care")
        pipeline.write_jsonl(DEST / f"{model}_e2_scores.jsonl", scored)
        summary["interval_comparison"][model] = metrics
    pipeline.write_json(DEST / "full_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    folds = json.loads((DATA / "shared/folds.json").read_text())["folds"]
    load_pair_cache()
    pipeline.align = use_a_paths
    if sys.argv[1:] == ["fit"]:
        with ProcessPoolExecutor(max_workers=4) as workers:
            list(workers.map(fit_fold, folds))
    elif sys.argv[1:] == ["retune"]:
        with ProcessPoolExecutor(max_workers=4) as workers:
            list(workers.map(tune_confirmation, folds))
    elif sys.argv[1:] == ["predict"]:
        audio_rows = {row["query_id"]: row for row in pipeline.read_jsonl(DEST / "unified_top10.jsonl")}
        report([predict_fold(fold, audio_rows) for fold in folds])
    else:
        raise SystemExit("Use fit, retune, or predict.")
