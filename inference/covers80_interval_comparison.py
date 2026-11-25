"""Compare CARE localization with the winning MERT and Unified window pairs."""

import argparse
from pathlib import Path

from audio_baselines import (
    OUTPUT,
    audio_cache_entry,
    check,
    check_members,
    file_sha256,
    load_model,
    read_jsonl,
    retrieve,
    shared_inputs,
    threadpool_limits,
    write_json,
    write_jsonl,
)
from audio_retrieval import selected_recall, window_offset_interval
from pipeline import bounded
from frozen_reevaluation import RECALL, write_csv
from kiki_lyric_experiment import DEST as KIKI, LABELS, baseline_audit, load_labels, read, score

MODELS = ("mert", "unified")
METHODS = ("care", *MODELS)
DEST = OUTPUT / "experiments" / "covers80_interval_comparison"
SAME_WINDOW = OUTPUT / "experiments/frozen_reevaluation/audio/same_window"
EXPECTED = {"mert": (5, 27), "unified": (74, 76)}


def protocol():
    recordings, rows = shared_inputs(OUTPUT)
    recordings = {r["recording_id"]: r for r in recordings}
    # Build the reference library by recording role. Do not use the paired answer
    # from the query manifest to construct retrieval inputs.
    works = {q["work_id"]: {k: v for k, v in q.items() if k != "reference_recording_id"} for q in rows}
    query_ids = [q["query_id"] for q in rows]
    check(len(works) == len(set(query_ids)) == len(rows) == 80, "Expected 80 distinct queries")
    references = [
        r for r in recordings.values() if r["dataset"] == "covers80" and r["role"] == "reference"
    ]
    check_members([r["work_id"] for r in references], works, "Covers80 full B-side library")
    gallery = {r["work_id"]: r for r in references}
    expected = {
        r["recording_id"]: audio_cache_entry(dict(recording=r, query=None)) for r in gallery.values()
    }
    for query in works.values():
        recording = recordings[query["query_recording_id"]]
        check(
            recording["work_id"] == query["work_id"]
            and recording["role"] == "query_source"
            and query["duration_sec"] == 30,
            f"Invalid query manifest entry: {query['query_id']}",
        )
        expected[recording["recording_id"]] = audio_cache_entry(dict(recording=recording, query=query))
    path = OUTPUT / "shared/folds.json"
    folds = read(path)["folds"]
    digest = file_sha256(path)
    check_members([f["outer_fold"] for f in folds], range(1, 6), path)
    check_members([w for f in folds for w in f["test_work_ids"]], works, path)
    configurations = {}
    for fold in folds:
        n = fold["outer_fold"]
        config = selected_recall(RECALL, n, digest)
        check(
            len(fold["test_work_ids"]) == 16 and config == (20, 50 if n == 2 else 75, 0),
            f"Unexpected fold size or frozen window for fold {n}",
        )
        check_members(fold["test_gallery_work_ids"], gallery, path)
        configurations[n] = config[:2]
    return works, gallery, expected, folds, configurations, digest


def top1_prediction(query, fold, configuration, hits, interval, reported):
    winner = hits[0]
    return dict(
        work_id=query["work_id"],
        query_id=query["query_id"],
        query_recording_id=query["query_recording_id"],
        query_source_start_sec=query["start_sec"],
        outer_fold=fold,
        window_seconds=configuration[0],
        overlap_percent=configuration[1],
        gallery_size=80,
        top10=hits,
        reference_recording_id=winner["reference_recording_id"],
        reported_recording_id=winner["reference_recording_id"] if reported else None,
        score=winner["score"],
        query_window=winner["query_window"],
        reference_window=winner["reference_window"],
        reference_interval=interval,
    )


def care_predictions(works, folds, configurations):
    for fold in folds:
        n = fold["outer_fold"]
        rows = [r for r in read_jsonl(KIKI / f"fold{n}/A.jsonl") if r["mode"] == "closed"]
        check_members([r["work_id"] for r in rows], fold["test_work_ids"], "Kiki A")
        for row in rows:
            query = works[row["work_id"]]
            check(
                row["outer_fold"] == n
                and row["query_id"] == query["query_id"]
                and row["reference_recording_id"] == row["top10"][0]["reference_recording_id"]
                and (not row["decision"]["harmony"] or row["threshold"].get("min_harmony_coverage") == 0)
                and row["localization"]["final_interval"] == row["original_e2"],
                f"Kiki A does not match the frozen E2 baseline: {row['query_id']}",
            )
            hits = [dict(c, rank=i) for i, c in enumerate(row["top10"], 1)]
            yield top1_prediction(query, n, configurations[n], hits, row["original_e2"], row["confirmed"])


def run_predictions(works, gallery, expected, folds, configurations, dest):
    write_jsonl(dest / "care_predictions.jsonl", care_predictions(works, folds, configurations))
    durations = {r["recording_id"]: r["duration_sec"] for r in gallery.values()}
    with threadpool_limits(limits=1):
        for model in MODELS:
            bank = load_model(model, expected)
            rows = []
            for fold in folds:
                n = fold["outer_fold"]
                config = configurations[n]
                refs = sorted(gallery[w]["recording_id"] for w in fold["test_gallery_work_ids"])
                for w in fold["test_work_ids"]:
                    q = works[w]
                    hits = retrieve(bank[config], q["query_recording_id"], refs)
                    winner = hits[0]
                    interval = window_offset_interval(
                        winner["query_window"],
                        winner["reference_window"],
                        durations[winner["reference_recording_id"]],
                    )
                    rows.append(top1_prediction(q, n, config, hits, interval, True))
            write_jsonl(dest / f"{model}_predictions.jsonl", rows)
            print(f"{model}: reran {len(rows)} queries against 80 references", flush=True)


def audit(rows, model, works, gallery, folds, configurations):
    check_members([r["query_id"] for r in rows], [q["query_id"] for q in works.values()], model)
    durations = {r["recording_id"]: r["duration_sec"] for r in gallery.values()}
    partition = {w: f["outer_fold"] for f in folds for w in f["test_work_ids"]}
    for r in rows:
        w = r["work_id"]
        q = works[w]
        n = partition[w]
        hits = r["top10"]
        check(
            r["query_id"] == q["query_id"]
            and r["outer_fold"] == n
            and r["query_recording_id"] == q["query_recording_id"]
            and r["query_source_start_sec"] == q["start_sec"]
            and r["gallery_size"] == 80
            and (r["window_seconds"], r["overlap_percent"]) == configurations[n],
            "Query or fold changed",
        )
        ids = [h["reference_recording_id"] for h in hits]
        check(
            len(set(ids)) == len(ids) == 10
            and set(ids) <= durations.keys()
            and [h["rank"] for h in hits] == list(range(1, 11)),
            "Invalid Top-10 reference IDs",
        )
        check(
            all(
                r[k] == hits[0][k]
                for k in ("reference_recording_id", "score", "query_window", "reference_window")
            ),
            "Saved winner differs from rank 1",
        )
        for h in hits:
            bounded(h["query_window"], 30)
            bounded(h["reference_window"], durations[h["reference_recording_id"]])
        span = r["reference_interval"]
        winner = r["reference_recording_id"]
        check(r["reported_recording_id"] in (None, winner), "Reported recording differs from winner")
        if span is not None:
            check(r["reported_recording_id"] == winner, "Interval on unreported recording")
            bounded(span, durations[winner])
        if model != "care":
            check(
                r["reported_recording_id"] == winner
                and span
                == window_offset_interval(r["query_window"], r["reference_window"], durations[winner]),
                "Audio interval differs from highest-scoring winner windows",
            )


def scores(rows, labels):
    for row in rows:
        label = labels[row["work_id"]]
        truth = label["reference_recording_id"]
        ids = [r["reference_recording_id"] for r in row["top10"]]
        rank = ids.index(truth) + 1 if truth in ids else None
        interval = row["reference_interval"]
        iou = (
            score(dict(row, mode="closed"), interval, labels)
            if rank == 1 and interval is not None
            else None
        )
        query_reported = row["reported_recording_id"] is not None
        yield dict(
            query_id=row["query_id"],
            outer_fold=row["outer_fold"],
            work_id=row["work_id"],
            correct_reference_recording_id=truth,
            actual_reference_recording_id=row["reference_recording_id"],
            reported_recording_id=row["reported_recording_id"],
            correct_rank=rank,
            top1_correct=rank == 1,
            song_reported=query_reported,
            correct_song_reported=row["reported_recording_id"] == truth,
            interval=interval,
            max_iou=iou,
            interval_hit=iou is not None and iou >= 0.5,
            wrong_song_with_interval=interval is not None and rank != 1,
            correct_song_iou_below_threshold=iou is not None and iou < 0.5,
            annotation_status=label["annotation_status"],
        )


def method_summary(rows):
    n = len(rows)
    outputs = sum(r["interval"] is not None for r in rows)
    hits = sum(r["interval_hit"] for r in rows)
    wrong_song = sum(r["wrong_song_with_interval"] for r in rows)
    wrong_position = sum(r["correct_song_iou_below_threshold"] for r in rows)
    check(outputs == hits + wrong_song + wrong_position, "Interval outcome accounting differs")
    return dict(
        queries=n,
        top1_correct=sum(r["top1_correct"] for r in rows),
        songs_reported=sum(r["song_reported"] for r in rows),
        correctly_reported=sum(r["correct_song_reported"] for r in rows),
        intervals_output=outputs,
        interval_hits=hits,
        hit_rate=hits / n,
        hit_rate_when_output=hits / outputs if outputs else None,
        wrong_song_with_interval=wrong_song,
        correct_song_iou_below_threshold=wrong_position,
    )


def rank_comparison(model, predictions, scored):
    source = SAME_WINDOW / f"{model}.jsonl"
    previous = read_jsonl(source)
    by_query = {r["query_id"]: r for r in previous}
    check_members([r["query_id"] for r in previous], [r["query_id"] for r in predictions], source)
    ranks = {r["query_id"]: r["correct_rank"] for r in scored}
    changes = []
    for row in predictions:
        query = row["query_id"]
        old = by_query[query]
        fields = ("outer_fold", "window_seconds", "overlap_percent", "query_recording_id")
        changed = [k for k in fields if row[k] != old[k]]
        if old["correct_rank"] != ranks[query]:
            changed.append("correct_rank")
        hit_diffs = []
        for a, b in zip(row["top10"], old["top10"], strict=True):
            fields = ("rank", "reference_recording_id", "query_window", "reference_window")
            differing = [k for k in fields if a[k] != b[k]]
            if abs(a["score"] - b["score"]) > 1e-6:
                differing.append("score")
            if differing:
                hit_diffs.append(dict(fields=differing, current=a, previous=b))
        if changed or hit_diffs:
            changes.append(dict(query_id=query, fields=changed, candidates=hit_diffs))
    hit1 = sum(r == 1 for r in ranks.values())
    hit10 = sum(r is not None for r in ranks.values())
    return dict(
        queries=len(predictions),
        hits_at_1=hit1,
        hits_at_10=hit10,
        expected_hit1=EXPECTED[model][0],
        expected_hit10=EXPECTED[model][1],
        source=str(source.relative_to(OUTPUT)),
        source_sha256=file_sha256(source),
        cache_index_sha256=file_sha256(OUTPUT / "cover_embeddings" / model / "index.jsonl"),
        differences=changes,
    )


def common_subsets(by_model):
    by_id = {model: {r["query_id"]: r for r in rows} for model, rows in by_model.items()}
    trials = (("care", "unified"), ("care", "mert"), METHODS)
    subsets = {}
    table = []
    for models in trials:
        name = "_".join(models)
        ids = sorted(q for q in by_id["care"] if all(by_id[m][q]["top1_correct"] for m in models))
        entries = [
            dict(
                query_id=q,
                outer_fold=by_id["care"][q]["outer_fold"],
                methods={
                    m: {k: by_id[m][q][k] for k in ("interval", "max_iou", "interval_hit")}
                    for m in models
                },
            )
            for q in ids
        ]
        subsets[name] = dict(methods=models, query_ids=ids, queries=entries)
        for model in models:
            local = [by_id[model][q] for q in ids]
            emitted = [r["max_iou"] for r in local if r["interval"] is not None]
            table.append(
                dict(
                    subset=name,
                    model=model,
                    queries=len(ids),
                    intervals_output=len(emitted),
                    pending=len(ids) - len(emitted),
                    interval_hits=sum(r["interval_hit"] for r in local),
                    mean_iou_over_outputs=sum(emitted) / len(emitted) if emitted else None,
                )
            )
    return table, subsets


def bundle(dest, works, gallery, folds, configurations, digest, *, persist):
    import csv
    from io import StringIO

    # Ground truth comes into scope only after prediction files have been written.
    _, truth_queries = shared_inputs(OUTPUT)
    truth_works = {r["work_id"]: r for r in truth_queries}
    labels = load_labels(truth_works)
    baseline_audit(KIKI, digest)
    for row in labels.values():
        duration = gallery[row["work_id"]]["duration_sec"]
        for interval in row["reference_intervals"]:
            bounded([interval["start_sec"], interval["end_sec"]], duration)
    by_model = {}
    audio_audit = {}
    end_table = []
    for model in METHODS:
        predictions = read_jsonl(dest / f"{model}_predictions.jsonl")
        audit(predictions, model, works, gallery, folds, configurations)
        scored = list(scores(predictions, labels))
        by_model[model] = scored
        overall = method_summary(scored)
        end_table.append(dict(model=model, fold="all", **overall))
        for f in folds:
            n = f["outer_fold"]
            end_table.append(
                dict(model=model, fold=n, **method_summary([r for r in scored if r["outer_fold"] == n]))
            )
        if model == "care":
            check(
                predictions == list(care_predictions(works, folds, configurations)),
                "CARE differs from Kiki A",
            )
            observed = tuple(
                overall[k]
                for k in (
                    "top1_correct",
                    "correctly_reported",
                    "intervals_output",
                    "interval_hits",
                    "correct_song_iou_below_threshold",
                )
            )
            check(
                observed == (67, 59, 43, 40, 3) and overall["wrong_song_with_interval"] == 0,
                f"CARE baseline failed: {observed}",
            )
        else:
            audio_audit[model] = rank_comparison(model, predictions, scored)
        output = dest / f"{model}_scores.jsonl"
        if persist:
            write_jsonl(output, scored)
        else:
            check(scored == read_jsonl(output), f"Persisted scoring differs for {model}")

    rank_audit = dict(folds_sha256=digest, frozen_config_sha256=file_sha256(RECALL), models=audio_audit)
    if persist:
        write_json(dest / "ranking_audit.json", rank_audit)
    else:
        check(read(dest / "ranking_audit.json") == rank_audit, "Persisted ranking audit differs")
    for model, r in audio_audit.items():
        check(
            not r["differences"] and (r["hits_at_1"], r["hits_at_10"]) == EXPECTED[model],
            f"{model} retrieval differs; see {dest / 'ranking_audit.json'}",
        )

    common_table, members = common_subsets(by_model)
    summary = dict(
        reference_labels_sha256=file_sha256(LABELS),
        folds_sha256=digest,
        annotations=dict(manual=9, automatic=71),
        expected_iou=0.5,
        folds={str(k): list(v) for k, v in configurations.items()},
        end_to_end=end_table,
        conditional=common_table,
        sources=dict(
            care="kiki_lyric_localization/fold*/A.jsonl",
            audio="recomputed from cache",
            frozen_reevaluation=str(SAME_WINDOW.relative_to(OUTPUT)),
        ),
    )
    for path, value in (("summary.json", summary), ("common_subsets.json", members)):
        if persist:
            write_json(dest / path, value)
        else:
            # Compare JSON equivalents, including tuples generated before serialization.
            import json

            check(read(dest / path) == json.loads(json.dumps(value)), f"Persisted {path} differs")
    for filename, rows in (("end_to_end.csv", end_table), ("same_correct_song.csv", common_table)):
        output = dest / filename
        if persist:
            write_csv(output, rows)
        else:
            stream = StringIO()
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            check(output.read_bytes() == stream.getvalue().encode(), f"Persisted {filename} differs")
    print(
        "Recomputed 3 x 80 scored results and all common subsets from persisted predictions.", flush=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "verify"))
    parser.add_argument("--output", type=Path, default=DEST)
    args = parser.parse_args()
    works, gallery, expected, folds, configurations, digest = protocol()
    if args.stage == "run":
        run_predictions(works, gallery, expected, folds, configurations, args.output)
    bundle(args.output, works, gallery, folds, configurations, digest, persist=args.stage == "run")


if __name__ == "__main__":
    main()
