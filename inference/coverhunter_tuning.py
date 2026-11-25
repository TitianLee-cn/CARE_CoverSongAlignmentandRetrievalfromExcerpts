"""Exploratory, paired five-fold CoverHunter retrieval experiment."""

import argparse
import json
from functools import partial

import numpy as np

from audio_baselines import (
    CONFIGURATIONS,
    OUTPUT,
    TIMING,
    check,
    check_members,
    file_sha256,
    inputs,
    load_model,
    metrics,
    read_jsonl,
    retrieve,
    search_queries,
    selected_window,
    threadpool_limits,
    write_json,
    write_jsonl,
)
from audio_retrieval import best_window_pair, weighted_window_pair

BASE = OUTPUT / "experiments/audio_fivefold"
DEST = OUTPUT / "experiments/coverhunter_tuning"
WEIGHTS = (0, 0.25, 0.5, 1)
RULE = "Maximize Hit@10, then exact MRR@10, then minimize lambda, summed median retrieval ns, window, overlap."


def check_rankings(first, second):
    for a, b in zip(first, second, strict=True):
        check(
            all(
                a[k] == b[k]
                for k in ("rank", "reference_recording_id", "query_window", "reference_window")
            )
            and abs(a["score"] - b["score"]) <= 1e-6,
            f"Inconsistent zero-weight ranking: {a} vs {b}",
        )


def audit(rows, works, folds, config):
    check_members([r["query_id"] for r in rows], [q["query_id"] for q in works.values()], "outer results")
    for r in rows:
        q = works[r["work_id"]]
        f = next(f for f in folds if f["outer_fold"] == r["outer_fold"])
        ids = [h["reference_recording_id"] for h in r["top10"]]
        order = sorted(r["top10"], key=lambda h: (-h["score"], h["reference_recording_id"]))
        gallery = {works[w]["reference_recording_id"] for w in f["test_gallery_work_ids"]}
        check(
            len(ids) == len(set(ids)) == 10
            and set(ids) <= gallery
            and order == r["top10"]
            and [h["rank"] for h in order] == list(range(1, 11)),
            r["query_id"],
        )
        check(
            r["work_id"] in f["test_work_ids"]
            and r["query_recording_id"] == q["query_recording_id"]
            and r["query_id"] == q["query_id"]
            and r["model"] == "coverhunter"
            and r["gallery_size"] == 80
            and q["reference_recording_id"] == r["correct_reference_recording_id"],
            r["query_id"],
        )
        settings = (r["window_seconds"], r["overlap_percent"], r.get("lambda", 0))
        check(settings == config[r["outer_fold"]], f"Wrong frozen settings: {r['query_id']}")
        rank = next((i for i, rid in enumerate(ids, 1) if rid == q["reference_recording_id"]), None)
        check(rank == r["correct_rank"], f"Incorrect rank for {q['query_id']}")


def baseline(works, folds, bank, digest):
    rows = read_jsonl(BASE / "coverhunter.jsonl")
    configs = {
        f["outer_fold"]: (*selected_window(BASE / "selected_windows.json", f["outer_fold"], digest), 0)
        for f in folds
    }
    audit(rows, works, folds, configs)
    score = metrics(rows)
    check((score["hits_at_1"], score["hits_at_10"]) == (52, 68), "Unexpected A baseline")
    refs = sorted(q["reference_recording_id"] for q in works.values())
    zero = partial(weighted_window_pair, weight=0)
    for r in rows:
        data = bank[configs[r["outer_fold"]][:2]]
        check_rankings(retrieve(data, r["query_recording_id"], refs, zero), r["top10"])
    write_jsonl(DEST / "A.jsonl", rows)
    print("Verified all 800 saved A candidates against the lambda=0 formula.", flush=True)
    return configs


def evaluate(bank, works, queries, gallery, outer, inner, cfg, weight):
    scorer = partial(weighted_window_pair, weight=weight) if weight else best_window_pair
    rows = search_queries(
        bank[cfg],
        works,
        queries,
        gallery,
        outer_fold=outer,
        inner_fold=inner,
        model="coverhunter",
        config=cfg,
        pair_score=scorer,
    )
    for r in rows:
        r["lambda"] = weight
    return rows


def choose(candidates):
    # 2520 is divisible by every possible nonzero rank. This keeps mathematical ties exact.
    return min(
        candidates,
        key=lambda x: (
            -x["hits_at_10"],
            -x["reciprocal_rank_units"],
            x["lambda"],
            x["retrieval_ns"],
            x["window_seconds"],
            x["overlap_percent"],
        ),
    )


def freeze(works, folds, bank, digest):
    locks = {
        g: dict(group=g, exploratory=True, folds_sha256=digest, timing=TIMING, rule=RULE, folds={})
        for g in ("B", "C")
    }
    for f in folds:
        outer = f["outer_fold"]
        sweep = []
        for cfg in CONFIGURATIONS:
            for weight in WEIGHTS:
                rows = []
                for sub in f["inner_folds"]:
                    rows += evaluate(
                        bank,
                        works,
                        sub["validation_work_ids"],
                        sub["gallery_work_ids"],
                        outer,
                        sub["inner_fold"],
                        cfg,
                        weight,
                    )
                check_members([r["work_id"] for r in rows], f["development_work_ids"], f"fold {outer}")
                sweep.append(
                    dict(
                        metrics(rows),
                        window_seconds=cfg[0],
                        overlap_percent=cfg[1],
                        reciprocal_rank_units=sum(
                            2520 // r["correct_rank"] for r in rows if r["correct_rank"]
                        ),
                        validation=[
                            (r["query_id"], r["inner_fold"], r["correct_rank"], r["retrieval_ns"])
                            for r in rows
                        ],
                        **{"lambda": weight},
                    )
                )
        for g, lock in locks.items():
            allowed = [r for r in sweep if g == "C" or not r["lambda"]]
            best = choose(allowed)
            lock["folds"][str(outer)] = {k: v for k, v in best.items() if k != "validation"}
            lock["folds"][str(outer)]["configurations"] = allowed
        print(f"Frozen choices for fold {outer}", flush=True)
    for g, lock in locks.items():
        write_json(DEST / f"{g}_config.json", lock)


def load_choices(group, folds, digest):
    path = DEST / f"{group}_config.json"
    lock = json.loads(path.read_text())
    result = {}
    for f in folds:
        n = f["outer_fold"]
        win, overlap = selected_window(path, n, digest)
        rec = lock["folds"][str(n)]
        check(rec["lambda"] in (WEIGHTS if group == "C" else (0,)), "Invalid lambda")
        best = choose(rec["configurations"])
        check(
            all(v == best[k] for k, v in rec.items() if k != "configurations"), "Frozen choice mismatch"
        )
        result[n] = win, overlap, rec["lambda"]
    return result


def transitions(rows1, rows2):
    def hits(rows):
        return {r["query_id"] for r in rows if r["correct_rank"] is not None}

    a, b = hits(rows1), hits(rows2)
    return dict(entered=sorted(b - a), dropped=sorted(a - b), net=len(b) - len(a))


def verify_files(works, folds, bank, digest):
    """Recompute each recorded Top-10 against all 80 references with independent row-wise scoring."""
    check(
        file_sha256(BASE / "coverhunter.jsonl") == file_sha256(DEST / "A.jsonl"),
        "A baseline copy differs",
    )
    summary = json.loads((DEST / "summary.json").read_text())
    refs = sorted(q["reference_recording_id"] for q in works.values())
    for group in ("A", "B", "C"):
        if group == "A":
            choices = {
                f["outer_fold"]: (
                    *selected_window(BASE / "selected_windows.json", f["outer_fold"], digest),
                    0,
                )
                for f in folds
            }
        else:
            choices = load_choices(group, folds, digest)
        rows = read_jsonl(DEST / f"{group}.jsonl")
        audit(rows, works, folds, choices)
        check(metrics(rows) == summary["groups"][group], f"{group}: saved metrics differ")
        for row in rows:
            w, o, weight = choices[row["outer_fold"]]
            data = bank[w, o]
            query, qt = data[row["query_recording_id"]]
            candidates = []
            for rid in refs:
                reference, rt = data[rid]
                # Direct float64 dot products avoid reusing the production reduction.
                matrix = np.einsum("id,jd->ij", query, reference, dtype=np.float64)
                maxima = [max(line) for line in matrix]
                peak = max(maxima)
                mean = sum(maxima) / len(maxima)
                candidates.append(((1 - weight) * peak + weight * mean, rid, matrix, qt, rt))
            candidates.sort(key=lambda c: (-c[0], c[1]))
            for hit, (score, rid, matrix, qt, rt) in zip(row["top10"], candidates[:10], strict=True):
                check(
                    hit["reference_recording_id"] == rid and abs(hit["score"] - score) < 1e-6,
                    row["query_id"],
                )
                # The winning float32 cosine can tie within roundoff. Confirm the recorded
                # window pair attains the global peak as well as having valid boundaries.
                (qi,) = np.flatnonzero(np.all(qt == hit["query_window"], axis=1))
                (ri,) = np.flatnonzero(np.all(rt == hit["reference_window"], axis=1))
                check(abs(matrix[qi, ri] - matrix.max()) < 1e-6, f"{rid}: wrong winning window")
    for f in folds:
        n = f["outer_fold"]
        old = read_jsonl(BASE / "validation" / f"fold{n}.jsonl")
        known = {
            (r["query_id"], r["window_seconds"], r["overlap_percent"]): r["correct_rank"]
            for r in old
            if r["model"] == "coverhunter"
        }
        settings = json.loads((DEST / "C_config.json").read_text())["folds"][str(n)]["configurations"]
        check_members(
            [(c["window_seconds"], c["overlap_percent"], c["lambda"]) for c in settings],
            [(w, o, a) for w, o in CONFIGURATIONS for a in WEIGHTS],
            f"fold {n} sweep",
        )
        for c in settings:
            val = c["validation"]
            for sub in f["inner_folds"]:
                check_members(
                    [q for q, j, _, _ in val if j == sub["inner_fold"]],
                    [works[w]["query_id"] for w in sub["validation_work_ids"]],
                    f"fold {n} OOF",
                )
            if not c["lambda"]:
                check(
                    all(
                        known[q, c["window_seconds"], c["overlap_percent"]] == rank
                        for q, _, rank, _ in val
                    ),
                    f"fold {n}: legacy validation ranks differ",
                )
            check(
                sum(r is not None for _, _, r, _ in val) == c["hits_at_10"]
                and sum(2520 // r for _, _, r, _ in val if r) == c["reciprocal_rank_units"]
                and sum(ns for _, _, _, ns in val) == c["retrieval_ns"],
                f"fold {n}: selection evidence differs",
            )
    print(
        "Verified persisted ranks, peak windows and scores for A/B/C; checked all five development sweeps."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    works, expected, folds, digest = inputs()
    bank = load_model("coverhunter", expected)
    if args.verify_only:
        verify_files(works, folds, bank, digest)
        return
    configs = {"A": baseline(works, folds, bank, digest)}
    freeze(works, folds, bank, digest)
    for group in ("B", "C"):
        configs[group] = load_choices(group, folds, digest)
        rows = []
        for f in folds:
            n = f["outer_fold"]
            w, o, weight = configs[group][n]
            rows.extend(
                evaluate(
                    bank, works, f["test_work_ids"], f["test_gallery_work_ids"], n, None, (w, o), weight
                )
            )
        write_jsonl(DEST / f"{group}.jsonl", rows)
    groups = {g: read_jsonl(DEST / f"{g}.jsonl") for g in configs}
    for g, rows in groups.items():
        audit(rows, works, folds, configs[g])
    summary = dict(
        exploratory=True,
        baseline_sha256=file_sha256(BASE / "coverhunter.jsonl"),
        timing=TIMING,
        groups={g: metrics(rows) for g, rows in groups.items()},
        by_fold={
            f["outer_fold"]: {
                g: metrics([r for r in rows if r["outer_fold"] == f["outer_fold"]])
                for g, rows in groups.items()
            }
            for f in folds
        },
        transitions={
            a + "_to_" + b: transitions(groups[a], groups[b]) for a, b in (("A", "B"), ("B", "C"))
        },
    )
    write_json(DEST / "summary.json", summary)
    verify_files(works, folds, bank, digest)
    print(json.dumps(summary["groups"]), json.dumps(summary["transitions"]), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
