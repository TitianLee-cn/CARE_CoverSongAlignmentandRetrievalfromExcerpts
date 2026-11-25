"""Five-fold exact audio retrieval on Covers80 using the three frozen embedding caches."""

import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter_ns
from zipfile import BadZipFile

import numpy as np
from threadpoolctl import threadpool_limits

from audio_retrieval import best_window_pair, selected_window

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from common import (
    OUTPUT,
    audio_cache_entry,
    file_sha256,
    read_jsonl,
    shared_inputs,
    write_atomic,
    write_json,
    write_jsonl,
)
from extract_cover_embeddings import CONFIGS, window_bounds

MODELS = {"coverhunter": 128, "mert": 768, "unified": 1024}
CONFIGURATIONS = [(c["window_seconds"], c["overlap_percent"]) for c in CONFIGS]
TIMING = (
    "One CPU BLAS thread. Each query is searched three times; median wall time includes "
    "window dot products, recording maxima, sorting and Top-10 construction. Excludes "
    "NPZ loading/validation, indexing, scoring metrics and disk output."
)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def check_members(ids, expected, context):
    duplicates = [k for k, n in Counter(ids).items() if n > 1]
    missing = set(expected) - set(ids)
    extra = set(ids) - set(expected)
    check(
        not (duplicates or missing or extra),
        f"{context}: duplicate={duplicates}, missing={sorted(missing)}, extra={sorted(extra)}",
    )


def inputs():
    recordings, queries = shared_inputs(OUTPUT)
    recordings = {r["recording_id"]: r for r in recordings}
    works = {q["work_id"]: q for q in queries}
    check(
        len(works) == len(queries) == len({q["query_id"] for q in queries}) == 80,
        "shared/queries.jsonl: expected 80 unique queries and works",
    )
    expected = {}
    for w, q in works.items():
        for side in ("query", "reference"):
            rid = q[side + "_recording_id"]
            r = recordings[rid]
            check(
                rid not in expected
                and r["role"] == ("query_source" if side == "query" else "reference")
                and r["work_id"] == w,
                f"shared/recordings.jsonl: invalid {side} recording {rid}",
            )
            expected[rid] = audio_cache_entry(dict(recording=r, query=q if side == "query" else None))
    path = OUTPUT / "shared/folds.json"
    folds = json.loads(path.read_text())["folds"]
    check_members([f["outer_fold"] for f in folds], range(1, 6), path)
    check_members([w for f in folds for w in f["test_work_ids"]], works, path)
    for f in folds:
        test, dev = f["test_work_ids"], f["development_work_ids"]
        context = f"{path}, outer fold {f['outer_fold']}"
        check(len(test) == 16 and len(dev) == 64, context)
        check_members(dev, set(works) - set(test), context)
        check_members(f["test_gallery_work_ids"], works, context)
        check_members([i["inner_fold"] for i in f["inner_folds"]], range(1, 5), context)
        check_members([w for i in f["inner_folds"] for w in i["validation_work_ids"]], dev, context)
        for i in f["inner_folds"]:
            val = i["validation_work_ids"]
            check(len(val) == 16, context)
            check_members(i["gallery_work_ids"], dev, context)
            check_members(i["fit_work_ids"], set(dev) - set(val), context)
    return works, expected, folds, file_sha256(path)


def load_model(model, expected):
    root = OUTPUT / "cover_embeddings" / model
    index = root / "index.jsonl"
    entries = read_jsonl(index)
    check_members([r["id"] for r in entries], expected, index)
    failures = read_jsonl(root / "failures.jsonl")
    check(not failures, f"{root}/failures.jsonl: {failures[:1]}")
    bank = {c: {} for c in CONFIGURATIONS}
    for row in entries:
        rid = row["id"]
        path = root / row["path"]
        try:
            with np.load(path, allow_pickle=False) as shard:
                metadata = json.loads(shard["metadata"].item())
                check(
                    metadata == {k: v for k, v in row.items() if k != "path"}, "index metadata mismatch"
                )
                for key, value in dict(
                    expected[rid],
                    model=model,
                    stage="cover_embeddings",
                    embedding_dimension=MODELS[model],
                ).items():
                    check(metadata[key] == value, f"unexpected {key}={metadata[key]!r}")
                vectors = shard["embeddings"]
                times, samples = shard["bounds_sec"], shard["bounds_samples"]
                offsets, idx = shard["config_offsets"], shard["config_window_rows"]
                configs = [(c["window_seconds"], c["overlap_percent"]) for c in metadata["configs"]]
                check_members(configs, CONFIGURATIONS, "configs")
                n = row["window_count"]
                check(
                    vectors.shape == (n, MODELS[model]) and times.shape == samples.shape == (n, 2),
                    "invalid embedding or time bounds shape",
                )
                check(
                    np.isfinite(vectors).all()
                    and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5),
                    "invalid or non-unit cached vector",
                )
                check(
                    offsets.shape == (len(configs) + 1,)
                    and offsets.dtype.kind in "iu"
                    and offsets[0] == 0
                    and offsets[-1] == len(idx)
                    and idx.dtype.kind in "iu"
                    and np.array_equal(np.diff(offsets), row["config_window_counts"])
                    and np.all((0 <= idx) & (idx < n)),
                    "invalid configuration offsets/indices",
                )
                rate = row["sample_rate"]
                end = int(samples[:, 1].max())
                check(
                    abs(end / rate - row["end_sec"]) <= 1 / rate
                    and np.array_equal(times, np.minimum(samples / rate, row["end_sec"])),
                    "wrong end time or seconds/sample mapping",
                )
                for i, (window, overlap) in enumerate(configs):
                    length = window * rate
                    ids = idx[offsets[i] : offsets[i + 1]]
                    check(
                        np.array_equal(
                            samples[ids], window_bounds(end, length, length * (100 - overlap) // 100)
                        ),
                        f"incomplete or mismatched windows: {window}s / {overlap}%",
                    )
                    bank[window, overlap][rid] = vectors[ids], times[ids]
        except (KeyError, IndexError, ValueError, OSError, BadZipFile) as exc:
            raise ValueError(f"{path} ({rid}): {exc}") from exc
    return bank


def retrieve(bank, query_id, refs, pair_score=best_window_pair):
    vectors, times = bank[query_id]
    hits = []
    for rid in refs:
        rvec, rtime = bank[rid]
        score, qwin, rwin = pair_score(vectors, rvec, times, rtime)
        hits.append((score, rid, qwin, rwin))
    hits.sort(key=lambda h: (-h[0], h[1]))
    return [
        dict(rank=k, reference_recording_id=rid, score=score, query_window=qwin, reference_window=rwin)
        for k, (score, rid, qwin, rwin) in enumerate(hits[:10], 1)
    ]


def search_queries(
    bank, works, queries, gallery, *, outer_fold, inner_fold, model, config, pair_score=best_window_pair
):
    refs = sorted(works[w]["reference_recording_id"] for w in gallery)
    rows = []
    for w in queries:
        q = works[w]
        timings = []
        for _ in range(3):
            start = perf_counter_ns()
            hits = retrieve(bank, q["query_recording_id"], refs, pair_score)
            timings.append(perf_counter_ns() - start)
        ids = [h["reference_recording_id"] for h in hits]
        check(len(set(ids)) == 10, f"{q['query_id']}: Top-10 contains duplicate IDs")
        correct = q["reference_recording_id"]
        rows.append(
            dict(
                model=model,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                work_id=w,
                query_id=q["query_id"],
                query_recording_id=q["query_recording_id"],
                query_source_start_sec=q["start_sec"],
                correct_reference_recording_id=correct,
                window_seconds=config[0],
                overlap_percent=config[1],
                gallery_size=len(refs),
                correct_rank=ids.index(correct) + 1 if correct in ids else None,
                retrieval_ns=median(timings),
                top10=hits,
            )
        )
    return rows


def metrics(rows):
    ranks = [r["correct_rank"] for r in rows if r["correct_rank"] is not None]
    counts = {f"hits_at_{k}": sum(rank <= k for rank in ranks) for k in (1, 5, 10)}
    ns = sum(r["retrieval_ns"] for r in rows)
    n = len(rows)
    return dict(
        queries=n,
        **counts,
        **{f"hit_at_{k}": counts[f"hits_at_{k}"] / n for k in (1, 5, 10)},
        mrr_at_10=sum(1 / r for r in ranks) / n,
        retrieval_ns=ns,
        ms_per_query=ns / n / 1e6,
    )


def smoke(banks, works, fold):
    """Independent exhaustive float64 cosine reference on two development queries."""
    val = fold["inner_folds"][0]["validation_work_ids"][:2]
    refs = [works[w]["reference_recording_id"] for w in fold["development_work_ids"]]
    for model, configs in banks.items():
        for config in ((10, 25), (15, 50), (20, 75)):
            bank = configs[config]
            for w in val:
                rid = works[w]["query_recording_id"]
                hits = retrieve(bank, rid, refs)
                q, qt = bank[rid]
                q = q.astype(np.float64)
                q /= np.linalg.norm(q, axis=1, keepdims=True)
                expected = []
                for ref in refs:
                    r, rt = bank[ref]
                    r = r.astype(np.float64)
                    r /= np.linalg.norm(r, axis=1, keepdims=True)
                    scores = np.einsum("id,jd->ij", q, r)
                    i, j = np.unravel_index(scores.argmax(), scores.shape)
                    expected.append((-scores[i, j], ref, qt[i].tolist(), rt[j].tolist()))
                for hit, (neg, ref, qw, rw) in zip(hits, sorted(expected)[:10], strict=True):
                    check(
                        hit["reference_recording_id"] == ref
                        and abs(hit["score"] + neg) < 2e-6
                        and hit["query_window"] == qw
                        and hit["reference_window"] == rw,
                        f"float64 smoke mismatch for {model}/{config}/{rid}: {hit} vs {ref}, {qw}, {rw}",
                    )
    print(
        "Smoke passed: 2 development queries x 3 encoders x 3 configurations; 64 references.", flush=True
    )


def select(banks, works, folds, digest, dest):
    result = dict(
        folds_sha256=digest,
        rule="maximize total validation hits_at_10 across 3 x 64 queries; "
        "tie: minimize summed retrieval_ns; exact tie: smaller window, then smaller overlap",
        timing=TIMING,
        folds={},
    )
    for fold in folds:
        outer = fold["outer_fold"]
        records, scores = [], []
        for config in CONFIGURATIONS:
            by_model = {}
            for model in MODELS:
                holdout = []
                inner_metrics = {}
                for inner in fold["inner_folds"]:
                    rows = search_queries(
                        banks[model][config],
                        works,
                        inner["validation_work_ids"],
                        inner["gallery_work_ids"],
                        outer_fold=outer,
                        inner_fold=inner["inner_fold"],
                        model=model,
                        config=config,
                    )
                    holdout.extend(rows)
                    inner_metrics[str(inner["inner_fold"])] = metrics(rows)
                check_members([r["work_id"] for r in holdout], fold["development_work_ids"], "inner OOF")
                by_model[model] = dict(metrics(holdout), inner=inner_metrics)
                records.extend(holdout)
            hits = sum(m["hits_at_10"] for m in by_model.values())
            ns = sum(m["retrieval_ns"] for m in by_model.values())
            scores.append(
                dict(
                    window_seconds=config[0],
                    overlap_percent=config[1],
                    hits_at_10=hits,
                    denominator=3 * 64,
                    mean_hit_at_10=hits / (3 * 64),
                    retrieval_ns=ns,
                    models=by_model,
                )
            )
        best = min(
            scores,
            key=lambda r: (
                -r["hits_at_10"],
                r["retrieval_ns"],
                r["window_seconds"],
                r["overlap_percent"],
            ),
        )
        tied = [
            [r["window_seconds"], r["overlap_percent"]]
            for r in scores
            if r["hits_at_10"] == best["hits_at_10"]
        ]
        result["folds"][str(outer)] = {
            k: best[k]
            for k in (
                "window_seconds",
                "overlap_percent",
                "hits_at_10",
                "denominator",
                "mean_hit_at_10",
                "retrieval_ns",
            )
        }
        result["folds"][str(outer)].update(tied_by_hits=tied, configurations=scores)
        write_jsonl(dest / "validation" / f"fold{outer}.jsonl", records)
        print(
            f"Fold {outer}: {best['window_seconds']}s / {best['overlap_percent']}%, {best['hits_at_10']}/192",
            flush=True,
        )
    write_json(dest / "selected_windows.json", result)


def test_selected(banks, works, folds, digest, dest):
    path = dest / "selected_windows.json"
    configs = {f["outer_fold"]: selected_window(path, f["outer_fold"], digest) for f in folds}
    summary = []
    for model in MODELS:
        rows = []
        for fold in folds:
            outer = fold["outer_fold"]
            config = configs[outer]
            rows.extend(
                search_queries(
                    banks[model][config],
                    works,
                    fold["test_work_ids"],
                    fold["test_gallery_work_ids"],
                    outer_fold=outer,
                    inner_fold=None,
                    model=model,
                    config=config,
                )
            )
        check_members([r["query_id"] for r in rows], [q["query_id"] for q in works.values()], model)
        write_jsonl(dest / f"{model}.jsonl", rows)
        summary.append(dict(model=model, **metrics(rows)))
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
    write_atomic(dest / "summary.csv", lambda stream: stream.write(buf.getvalue().encode()))
    write_json(
        dest / "summary.json",
        dict(
            models=summary, timing=TIMING, folds_sha256=digest, selected_windows_sha256=file_sha256(path)
        ),
    )
    print(buf.getvalue(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "run", "test"))
    parser.add_argument("--output", type=Path, default=OUTPUT / "experiments/audio_fivefold")
    args = parser.parse_args()
    works, expected, folds, digest = inputs()
    banks = {model: load_model(model, expected) for model in MODELS}
    print("Validated 480 recording caches, each with 9 configurations.", flush=True)
    if args.action in ("smoke", "run"):
        smoke(banks, works, folds[0])
    if args.action == "run":
        select(banks, works, folds, digest, args.output)
    if args.action != "smoke":
        test_selected(banks, works, folds, digest, args.output)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
