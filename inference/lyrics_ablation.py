"""Exploratory paired five-fold evaluation of lyric reranking and E2 evidence."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import csv
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from unittest.mock import patch

from pipeline import (
    DATA,
    Engine,
    confirmation_rules,
    curves,
    decision_features,
    e2,
    e2_validation,
    file_sha256,
    fold_sets,
    infer,
    passes,
    position_examples,
    position_options,
    ranking_decision,
    read_jsonl,
    rerank,
    score_options,
    selected_recall,
    threadpool_limits,
    write_json,
    write_jsonl,
)
from common import write_atomic
from verify import verify

RECALL = DATA / "experiments/coverhunter_tuning/B_config.json"
DEST = DATA / "experiments/lyrics_ablation"
LABELS = DATA / "reference_labels/covers80/reference_intervals.jsonl"
WEIGHTS = (0, 0.1, 0.2, 0.3, 0.4, 0.5)
GROUPS = ("A", "B", "C", "D")
GROUP_PAIRS = tuple(zip(GROUPS, GROUPS[1:]))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rank(rows, calibration, scoring):
    return rerank(
        rows, calibration, harmony_weight=scoring["harmony_weight"], use_lyrics=scoring["lyric_rerank"]
    )


def validate_scheme(splits, scoring):
    observations = []
    reciprocal = 0
    for split in splits:
        for work, closed, opened in split["pairs"]:
            ranked = rank(closed, split["curves"], scoring)
            open_ranked = rank(opened, split["curves"], scoring)
            actual = next((i for i, c in enumerate(ranked, 1) if work == c["reference_work_id"]), None)
            if actual:
                reciprocal += 2520 // actual
            features = decision_features(ranked, harmony_weight=scoring["harmony_weight"])
            open_features = decision_features(open_ranked, harmony_weight=scoring["harmony_weight"])
            observations.extend([(features, True, actual == 1), (open_features, False, False)])
    rules = confirmation_rules(observations)

    def accepted(features):
        return passes(features, rules["music" if features["harmony"] else "audio_only"])

    n = sum(known for _, known, _ in observations)
    false_accepts = sum(accepted(obs) for obs, known, _ in observations if not known)
    return dict(
        scoring,
        confirmation=rules,
        validation_queries=n,
        correct_confirmations=sum(
            accepted(obs) and correct for obs, known, correct in observations if known
        ),
        hit1=sum(correct for _, known, correct in observations if known),
        mrr_at_10=reciprocal / (2520 * n),
        reciprocal_rank_units=reciprocal,
        open_false_accepts=false_accepts,
        open_queries=n,
    )


def fit_positions(engine, splits, final_rows, labels, scheme, use_lyrics):
    heldout = []
    for split in splits:
        examples = position_examples(engine, split["fit_rows"], labels, use_lyrics=use_lyrics)
        models = e2.train(examples, use_lyrics=use_lyrics)
        params = dict(scheme, curves=split["curves"])
        for work, closed, _ in split["pairs"]:
            ranking, _, _, confirmed = ranking_decision(closed, params)
            winner = ranking[0]
            if confirmed and winner["reference_work_id"] == work and labels.get(work):
                options = position_options(engine, work, winner, True, use_lyrics=use_lyrics)
                heldout.append(
                    dict(new=score_options(options, models, "path_features"), truth=labels[work])
                )
    best, sweep = e2_validation.select_rule(heldout) if heldout else (dict(rule=e2.LEGACY_RULE), [])
    examples = position_examples(engine, final_rows, labels, use_lyrics=use_lyrics)
    return dict(
        e2_version="path",
        e2_lyrics=use_lyrics,
        e2_selection=best["rule"],
        feature_names=e2.feature_names(use_lyrics),
        position_models=e2.train(examples, use_lyrics=use_lyrics),
        e2_validation=dict(queries=len(heldout), selection=best, sweep=sweep),
        fit_position_proposals=len(examples),
    )


def best_weight(sweep):
    return max(
        (s for s in sweep if s["open_false_accepts"] * 20 <= s["open_queries"]),
        key=lambda s: (
            s["correct_confirmations"],
            s["hit1"],
            s["reciprocal_rank_units"],
            -s["harmony_weight"],
        ),
    )


def fit_fold(engine, labels):
    splits = []
    for sub in engine.fold["inner_folds"]:
        inner = sub["inner_fold"]
        training, val = fold_sets(engine.outer, inner)
        require(
            (training, val) == (sub["fit_work_ids"], sub["validation_work_ids"]), "A/B inner folds differ"
        )
        bg = engine.background(inner, training)
        fit_rows = [(work, engine.candidates(work, training, bg)) for work in training]
        gallery = sub["gallery_work_ids"]
        pairs = [
            (
                work,
                engine.candidates(work, gallery, bg),
                engine.candidates(work, [ref for ref in gallery if ref != work], bg),
            )
            for work in val
        ]
        splits.append(dict(fit_rows=fit_rows, curves=curves(fit_rows), pairs=pairs))
    a = validate_scheme(splits, dict(harmony_weight=0.3, lyric_rerank=True))
    sweep = [validate_scheme(splits, dict(harmony_weight=w, lyric_rerank=False)) for w in WEIGHTS]
    b = next(s for s in sweep if s["harmony_weight"] == 0.3)
    c = best_weight(sweep)
    bg = engine.background(None, engine.dev)
    final_rows = [(w, engine.candidates(w, engine.dev, bg)) for w in engine.dev]
    calibration = curves(final_rows)
    common = dict(
        background=bg,
        outer_fold=engine.outer,
        window=engine.window,
        overlap=engine.overlap,
        recall_lambda=engine.recall_weight,
        fit_work_ids=engine.dev,
        recall_config_sha256=file_sha256(RECALL),
        exploratory=True,
    )
    # Freeze identical lyric-enabled E2 models and decision thresholds for A/B/C.
    full = fit_positions(engine, splits, final_rows, labels, a, True)
    strict = fit_positions(engine, splits, final_rows, labels, c, False)
    params = {}
    for group, scheme in zip(GROUPS, (a, b, c, c)):
        enabled_curves = {k: v for k, v in calibration.items() if k != "lyric" or scheme["lyric_rerank"]}
        params[group] = dict(
            common, **scheme, curves=enabled_curves, **(strict if group == "D" else full)
        )
    return params, sweep


def signature(candidates):
    fields = (
        "recall_rank",
        "reference_recording_id",
        "audio",
        "query_window",
        "reference_window",
        "harmony",
        "paths",
    )
    payload = [[c[key] for key in fields] for c in sorted(candidates, key=lambda c: c["recall_rank"])]
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def open_prediction(engine, candidates, params, work):
    ranked, features, rule, confirmed = ranking_decision(candidates, params)
    return dict(
        outer_fold=engine.outer,
        work_id=work,
        query_id=engine.queries[work]["query_id"],
        reference_recording_id=ranked[0]["reference_recording_id"],
        confirmed=confirmed,
        decision=features,
        threshold=rule,
        gallery_size=len(engine.queries) - 1,
        acoustic_sha256=signature(candidates),
        top10=[
            {
                k: c[k]
                for k in (
                    "reference_recording_id",
                    "recall_rank",
                    "audio",
                    "query_window",
                    "reference_window",
                    "score",
                )
            }
            for c in ranked
        ],
    )


def smoke_checks(engine, params, work, candidates, confirmed):
    altered = deepcopy(candidates)
    for i, item in enumerate(altered):
        item["lyric"] = 99.0 if i % 2 else -99.0
        item["lyrics"]["matches"] = [
            dict(query_sec=[0, 25], reference_sec=[0, 25], partial_left=False, partial_right=False)
        ]
    strict = params["D"]
    for group in ("B", "C", "D"):
        original = ranking_decision(candidates, params[group])
        changed = ranking_decision(altered, params[group])
        require(
            original[1:] == changed[1:]
            and [(r["reference_work_id"], r["score"]) for r in original[0]]
            == [(r["reference_work_id"], r["score"]) for r in changed[0]],
            "Lyrics affect disabled reranking",
        )
    original = position_options(engine, work, candidates[0], True, use_lyrics=False)
    changed = position_options(engine, work, altered[0], True, use_lyrics=False)
    require(original == changed, "Lyrics affect disabled E2")
    if confirmed:
        from pipeline import predict

        with patch("pipeline.predict", wraps=predict) as classifier:
            pred = infer(engine, strict, work, candidates=candidates)
        require(
            classifier.call_count == 1 and pred["type_invoked"] and not pred["prediction"]["display"],
            "Strict configuration failed to invoke internal typing",
        )


def run_fold(number, dest, smoke=False):
    with threadpool_limits(limits=1):
        root = dest / f"fold{number}"
        setup = selected_recall(RECALL, number, file_sha256(DATA / "shared/folds.json"))
        engine = Engine(number, *setup)
        labels = {
            r["work_id"]: r["reference_intervals"]
            for r in read_jsonl(LABELS)
            if r["work_id"] in engine.dev
        }
        params, sweep = fit_fold(engine, labels)
        write_json(root / "weight_sweep.json", sweep)
        for g in GROUPS:
            write_json(root / g / "parameters.json", params[g])
        params = {g: json.loads((root / g / "parameters.json").read_text()) for g in GROUPS}
        gallery = engine.fold["test_gallery_work_ids"]
        closed, opened = {g: [] for g in GROUPS}, {g: [] for g in GROUPS}
        smoke_checked = False
        type_checked = False
        for work in engine.test:
            candidates = engine.candidates(work, gallery, params["A"]["background"], True)
            # Independently remove the whole matching work before audio retrieval.
            open_candidates = engine.candidates(
                work, [w for w in gallery if w != work], params["A"]["background"], True
            )
            for g in GROUPS:
                closed[g].append(infer(engine, params[g], work, candidates=candidates))
                opened[g].append(open_prediction(engine, open_candidates, params[g], work))
            if smoke and (not smoke_checked or not type_checked):
                confirmed = closed["D"][-1]["confirmed"] and closed["D"][-1]["gate"]["enabled"]
                smoke_checks(engine, params, work, candidates, confirmed)
                smoke_checked = True
                type_checked |= confirmed
        if smoke:
            require(type_checked, "Smoke fold must exercise strict typing with valid lyrics")
        for g in GROUPS:
            write_jsonl(root / g / "predictions.jsonl", closed[g])
            write_jsonl(root / g / "open_predictions.jsonl", opened[g])
        print(f"Fold {number} written, w={params['C']['harmony_weight']}", flush=True)


def measure(closed, opened, labels):
    n = len(closed)
    ranks = [
        next((c["rank"] for c in row["top10"] if c["reference_work_id"] == row["work_id"]), None)
        for row in closed
    ]
    correct = [r for r, rank in zip(closed, ranks) if r["confirmed"] and rank == 1]
    emitted = [r for r in closed if r["reference_interval"] is not None]
    output_correct = [r for r in correct if r["reference_interval"] is not None and labels[r["work_id"]]]
    hits = sum(
        e2_validation.label_iou(r["reference_interval"], labels[r["work_id"]]) >= 0.5
        for r in output_correct
    )
    confirmations = sum(r["confirmed"] for r in closed)
    false_accepts = sum(r["confirmed"] for r in opened)
    top1 = sum(r == 1 for r in ranks)
    return dict(
        queries=n,
        top1=top1,
        hit_at_1=top1 / n,
        mrr_at_10=sum(1 / r for r in ranks if r) / n,
        correct_confirmations=len(correct),
        incorrect_confirmations=confirmations - len(correct),
        confirmation_coverage=confirmations / n,
        open_false_accepts=false_accepts,
        open_queries=len(opened),
        open_false_accept_rate=false_accepts / len(opened),
        interval_outputs=len(emitted),
        interval_hits=hits,
        interval_total=n,
        interval_eligible=sum(bool(labels[r["work_id"]]) for r in correct),
        interval_on_correct=len(output_correct),
        type_answered=sum(r["prediction"]["type_status"] == "answered" for r in closed),
    )


def ordered(rows):
    return sorted(rows, key=lambda r: r["recall_rank"])


def validate_outer(number, root, groups, labels):
    fold = json.loads((DATA / "shared/folds.json").read_text())["folds"][number - 1]
    ratios = json.loads((root / "weight_sweep.json").read_text())
    require([r["harmony_weight"] for r in ratios] == list(WEIGHTS), "Incomplete development sweep")
    params = {g: json.loads((root / g / "parameters.json").read_text()) for g in GROUPS}
    recall = selected_recall(RECALL, number, file_sha256(DATA / "shared/folds.json"))
    closed = {}
    opened = {}
    for g in GROUPS:
        p = params[g]
        require(
            p["fit_work_ids"] == fold["development_work_ids"]
            and (p["window"], p["overlap"], p["recall_lambda"]) == recall
            and p["open_false_accepts"] * 20 <= p["open_queries"]
            and p["open_queries"] == 64,
            "Invalid frozen fit",
        )
        require(p["lyric_rerank"] == (g == "A") and p["e2_lyrics"] == (g != "D"), "Wrong switch settings")
        verify(root / g / "predictions.jsonl", False)
        closed[g], opened[g] = groups[g]
        for mode in (closed[g], opened[g]):
            require(
                len(mode) == 16 and {r["work_id"] for r in mode} == set(fold["test_work_ids"]),
                "Outer fold is incomplete",
            )
    require(
        params["D"]["position_models"].keys() == {"audio", "harmony"}
        and all("lyric" not in name for names in params["D"]["feature_names"].values() for name in names),
        "D trained with lyrics",
    )
    selected = best_weight(ratios)
    require(
        all(params[g][k] == v for g in ("C", "D") for k, v in selected.items()),
        "Frozen C/D scheme differs from development selection",
    )
    require(
        all(
            params[g][key] == params["A"][key]
            for g in ("B", "C")
            for key in ("position_models", "e2_selection", "feature_names")
        ),
        "A/B/C must share E2",
    )
    for g in GROUPS:
        for row in (*closed[g], *opened[g]):
            f = row["decision"]
            rule = params[g]["confirmation"]["music" if f["harmony"] else "audio_only"]
            require(
                row["threshold"] == rule and row["confirmed"] == passes(f, rule),
                "Saved confirmation differs from frozen rule",
            )
            if not params[g]["harmony_weight"]:
                require(not f["harmony"] and f["coverage"] == 0, "w=0 used harmony for confirmation")
    pure = {r["query_id"]: r for r in read_jsonl(RECALL.parent / "B.jsonl") if r["outer_fold"] == number}
    expected_weights = (0.3, 0.3, params["C"]["harmony_weight"], params["C"]["harmony_weight"])
    require(
        tuple(params[g]["harmony_weight"] for g in GROUPS) == expected_weights,
        "Inconsistent fused weights",
    )
    for paired in zip(*(closed[g] for g in GROUPS), strict=True):
        a, b, c, d = paired
        require(
            len({r["query_id"] for r in paired}) == 1
            and len({signature(r["top10"]) for r in paired}) == 1,
            "Closed-set acoustic candidates differ",
        )
        require(
            [
                (t["reference_recording_id"], t["query_window"], t["reference_window"], t["audio"])
                for t in ordered(a["top10"])
            ]
            == [
                (t["reference_recording_id"], t["query_window"], t["reference_window"], t["score"])
                for t in pure[a["query_id"]]["top10"]
            ],
            "Closed-set recall differs from frozen B",
        )
        require(
            all(
                c[key] == d[key]
                for key in ("reference_recording_id", "confirmed", "decision", "threshold")
            )
            and [t["reference_recording_id"] for t in c["top10"]]
            == [t["reference_recording_id"] for t in d["top10"]],
            "C and D must have identical song decisions",
        )
        require(
            d["type_invoked"] == d["confirmed"]
            and all(p["source"] != "lyric" for p in d["interval_proposals"]),
            "Strict typing or E2 isolation failed",
        )
    for paired, a in zip(zip(*(opened[g] for g in GROUPS), strict=True), closed["A"], strict=True):
        correct = pure[a["query_id"]]["correct_reference_recording_id"]
        for r in paired:
            recalled = [t["reference_recording_id"] for t in ordered(r["top10"])]
            orig = [t["reference_recording_id"] for t in ordered(a["top10"])]
            retained = [rid for rid in orig if rid != correct]
            require(
                r["query_id"] == a["query_id"]
                and r["gallery_size"] == 79
                and correct not in recalled
                and len(set(recalled)) == 10
                and recalled[: len(retained)] == retained,
                "Invalid open-set retrieval",
            )
        require(
            len({r["acoustic_sha256"] for r in paired}) == 1 and paired[2] == paired[3],
            "Open-set candidates changed",
        )
    return {g: measure(closed[g], opened[g], labels) for g in GROUPS}


def query_changes(groups, labels):
    by_group = {g: {r["query_id"]: r for r in pairs[0]} for g, pairs in groups.items()}
    rows = []
    for before, after in GROUP_PAIRS:
        for query, old in by_group[before].items():
            new = by_group[after][query]

            def state(row):
                iou = (
                    e2_validation.label_iou(row["reference_interval"], labels[row["work_id"]])
                    if row["top10"][0]["reference_work_id"] == row["work_id"]
                    else 0
                )
                return dict(
                    winner=row["reference_recording_id"],
                    confirmed=row["confirmed"],
                    top1_correct=row["top10"][0]["reference_work_id"] == row["work_id"],
                    interval=row["reference_interval"],
                    iou=iou,
                )

            a, b = state(old), state(new)
            if a != b:
                rows.append(
                    dict(
                        comparison=before + "_to_" + after,
                        mode="closed",
                        fold=old["outer_fold"],
                        query_id=query,
                        before=a,
                        after=b,
                    )
                )
    by_group = {g: {r["query_id"]: r for r in pairs[1]} for g, pairs in groups.items()}
    for before, after in GROUP_PAIRS:
        for query, old in by_group[before].items():
            new = by_group[after][query]
            if old["confirmed"] != new["confirmed"]:
                rows.append(
                    dict(
                        comparison=before + "_to_" + after,
                        fold=old["outer_fold"],
                        query_id=query,
                        mode="open",
                        before=dict(confirmed=old["confirmed"], winner=old["reference_recording_id"]),
                        after=dict(confirmed=new["confirmed"], winner=new["reference_recording_id"]),
                    )
                )
    return rows


def summarize(dest, folds, *, save):
    labels = {r["work_id"]: r["reference_intervals"] for r in read_jsonl(LABELS)}
    require(len(labels) == 80, "Expected 80 automatic references")
    by_fold = {}
    combined = {g: ([], []) for g in GROUPS}
    for n in folds:
        root = dest / f"fold{n}"
        groups = {
            g: tuple(
                read_jsonl(root / g / f"{name}.jsonl") for name in ("predictions", "open_predictions")
            )
            for g in GROUPS
        }
        by_fold[str(n)] = validate_outer(n, root, groups, labels)
        for g in GROUPS:
            for total, partition in zip(combined[g], groups[g], strict=True):
                total.extend(partition)
    metrics = {g: measure(*combined[g], labels) for g in GROUPS}
    changes = query_changes(combined, labels)
    report = dict(exploratory=True, folds=folds, groups=metrics, by_fold=by_fold)
    if save:
        write_json(dest / "summary.json", report)
        write_jsonl(dest / "changes.jsonl", changes)
        table = StringIO()
        writer = csv.DictWriter(table, fieldnames=["group", "fold", *metrics["A"]])
        writer.writeheader()
        writer.writerows(
            dict(group=g, fold=fold, **row)
            for fold, results in [("all", metrics), *by_fold.items()]
            for g, row in results.items()
        )
        write_atomic(dest / "summary.csv", lambda stream: stream.write(table.getvalue().encode()))
    else:
        require(
            report == json.loads((dest / "summary.json").read_text())
            and changes == read_jsonl(dest / "changes.jsonl"),
            "Saved reports differ from predictions",
        )
    print(json.dumps(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEST)
    parser.add_argument(
        "--smoke", action="store_true", help="Run fold 3 and check both switches on real queries"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument(
        "--report-only", action="store_true", help="Rebuild reports from existing prediction files"
    )
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    dest = args.output / "smoke" if args.smoke else args.output
    folds = [3] if args.smoke else list(range(1, 6))
    if not (args.verify_only or args.report_only):
        if args.smoke:
            run_fold(3, dest, True)
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                for completed in [pool.submit(run_fold, n, dest) for n in folds]:
                    completed.result()
    summarize(dest, folds, save=not args.verify_only)


if __name__ == "__main__":
    main()
