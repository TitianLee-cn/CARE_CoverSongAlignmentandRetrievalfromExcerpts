"""Five-fold reevaluation of the frozen, lyric-enabled CARE configuration."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from io import StringIO
import json
from pathlib import Path
from time import perf_counter_ns

from audio_baselines import MODELS, TIMING, check_members, inputs, load_model, metrics, search_queries
from audio_retrieval import selected_recall
from coverhunter_tuning import check_rankings
from pipeline import DATA, Engine, e2_validation, file_sha256, fit, infer, read_jsonl
from common import write_atomic, write_json, write_jsonl
from lyrics_ablation import measure, open_prediction, require
from threadpoolctl import threadpool_limits
from verify import verify

EXPERIMENTS = DATA / "experiments"
OFFICIAL = EXPERIMENTS / "audio_fivefold"
TUNING = EXPERIMENTS / "coverhunter_tuning"
PREVIOUS = EXPERIMENTS / "lyrics_ablation"
RECALL = TUNING / "B_config.json"
DEST = EXPERIMENTS / "frozen_reevaluation"
LABELS = DATA / "reference_labels/covers80/reference_intervals.jsonl"


def fit_fold(fold, dest, digest):
    number = fold["outer_fold"]
    window, overlap, weight = selected_recall(RECALL, number, digest)
    require(weight == 0, "Frozen CoverHunter B configuration must use lambda=0")
    root = dest / "care" / f"fold{number}"
    with threadpool_limits(limits=1):
        engine = Engine(number, window, overlap, weight)
        params = fit(engine, root, paired_e2=True)
    params.update(folds_sha256=digest, recall_config_sha256=file_sha256(RECALL))
    write_json(root / "parameters.json", params)
    print(f"Fitted fold {number} from development caches", flush=True)


def configs(folds, digest, config_path):
    return {
        fold["outer_fold"]: selected_recall(config_path, fold["outer_fold"], digest) for fold in folds
    }


def audit_audio(rows, works, folds, choices):
    check_members([r["query_id"] for r in rows], [q["query_id"] for q in works.values()], "audio queries")
    by_fold = {f["outer_fold"]: f for f in folds}
    for row in rows:
        q = works[row["work_id"]]
        fold = row["outer_fold"]
        by_id = {x["reference_recording_id"] for x in works.values()}
        hits = row["top10"]
        ids = [h["reference_recording_id"] for h in hits]
        found = next((i for i, rid in enumerate(ids, 1) if rid == q["reference_recording_id"]), None)
        require(
            row["work_id"] in by_fold[fold]["test_work_ids"]
            and row["query_id"] == q["query_id"]
            and row["correct_reference_recording_id"] == q["reference_recording_id"]
            and row["query_recording_id"] == q["query_recording_id"]
            and row["gallery_size"] == 80
            and (row["window_seconds"], row["overlap_percent"], row.get("lambda", 0)) == choices[fold]
            and found == row["correct_rank"]
            and len(ids) == len(set(ids)) == 10
            and set(ids) <= by_id
            and hits == sorted(hits, key=lambda h: (-h["score"], h["reference_recording_id"])),
            f"Invalid audio retrieval row for {q['query_id']}",
        )


def run_audio(works, expected, folds, digest, dest):
    from audio_baselines import retrieve

    official = configs(folds, digest, OFFICIAL / "selected_windows.json")
    matched = configs(folds, digest, RECALL)
    original = json.loads((OFFICIAL / "summary.json").read_text())
    require(
        original["folds_sha256"] == digest
        and original["timing"] == TIMING
        and original["selected_windows_sha256"] == file_sha256(OFFICIAL / "selected_windows.json"),
        "Formal audio provenance or timing differs",
    )
    refs = sorted(q["reference_recording_id"] for q in works.values())
    for model in MODELS:
        bank = load_model(model, expected)
        old = read_jsonl(OFFICIAL / f"{model}.jsonl")
        audit_audio(old, works, folds, official)
        require(
            next(r for r in original["models"] if r["model"] == model)
            == dict(model=model, **metrics(old)),
            f"Formal {model} summary differs",
        )
        for row in old:
            config = official[row["outer_fold"]][:2]
            check_rankings(retrieve(bank[config], row["query_recording_id"], refs), row["top10"])
        write_jsonl(dest / "audio" / "formal" / f"{model}.jsonl", old)
        fresh = []
        for fold in folds:
            outer = fold["outer_fold"]
            config = matched[outer][:2]
            fresh.extend(
                search_queries(
                    bank[config],
                    works,
                    fold["test_work_ids"],
                    fold["test_gallery_work_ids"],
                    outer_fold=outer,
                    inner_fold=None,
                    model=model,
                    config=config,
                )
            )
        audit_audio(fresh, works, folds, matched)
        if model == "coverhunter":
            anchor = {r["query_id"]: r for r in read_jsonl(TUNING / "B.jsonl")}
            for row in fresh:
                check_rankings(row["top10"], anchor[row["query_id"]]["top10"])
            counts = metrics(fresh)
            require(
                (counts["hits_at_1"], counts["hits_at_10"]) == (62, 70), "CoverHunter B did not reproduce"
            )
        write_jsonl(dest / "audio" / "same_window" / f"{model}.jsonl", fresh)
        print(f"Verified formal and timed same-window {model}", flush=True)


def run_care(fold, dest, digest):
    from align_cover_pairs import align

    number = fold["outer_fold"]
    root = dest / "care" / f"fold{number}"
    params = json.loads((root / "parameters.json").read_text())
    start = perf_counter_ns()
    engine = Engine(number, *selected_recall(RECALL, number, digest))
    # Exclude first-use JIT compilation. No held-out query is used for startup.
    warm = engine.queries[engine.dev[0]]
    align(
        (engine.beats[warm["query_recording_id"]], engine.beats[warm["reference_recording_id"]]),
        0.15,
        3,
        0.5,
    )
    init_ns = perf_counter_ns() - start
    new, old, opened = [], [], []
    for work in engine.test:
        start = perf_counter_ns()
        prediction = infer(engine, params, work)
        prediction["inference_ns"] = perf_counter_ns() - start
        new.append(prediction)
    legacy = dict(
        params,
        e2_version="legacy",
        e2_selection=params["legacy_e2_selection"],
        position_models=params["legacy_position_models"],
    )
    gallery = fold["test_gallery_work_ids"]
    for work in engine.test:
        old.append(infer(engine, legacy, work))
        candidates = engine.candidates(
            work, [other for other in gallery if other != work], params["background"], True
        )
        opened.append(open_prediction(engine, candidates, params, work))
    write_jsonl(root / "predictions.jsonl", new)
    write_jsonl(root / "legacy_predictions.jsonl", old)
    write_jsonl(root / "open_predictions.jsonl", opened)
    write_json(root / "timing.json", dict(setup_ns=init_ns))
    print(f"Predicted fold {number} with both E2 versions and open-set evaluation", flush=True)


def audit_development(fold, root, params, digest, works):
    number = fold["outer_fold"]
    require(
        params["fit_work_ids"] == fold["development_work_ids"]
        and params["outer_fold"] == number
        and params["folds_sha256"] == digest
        and params["recall_config_sha256"] == file_sha256(RECALL)
        and (params["window"], params["overlap"], params["recall_lambda"])
        == selected_recall(RECALL, number, digest)
        and params["harmony_weight"] == 0.3
        and params["lyric_rerank"]
        and params["e2_lyrics"]
        and params["e2_version"] == "path",
        "Frozen parameters do not match the experiment",
    )
    rows = read_jsonl(root / "development_oof.jsonl")
    check_members([r["work_id"] for r in rows], fold["development_work_ids"], "development OOF")
    for inner in fold["inner_folds"]:
        check_members(
            [r["work_id"] for r in rows if r["inner_fold"] == inner["inner_fold"]],
            inner["validation_work_ids"],
            "inner validation",
        )
    references = {works[w]["reference_recording_id"] for w in fold["development_work_ids"]}
    for row in rows:
        require(
            row["query_id"] == works[row["work_id"]]["query_id"]
            and len(set(row["top10"])) == 10
            and set(row["top10"]) <= references,
            "Non-development query or gallery used during E2 tuning",
        )
    eligible = [r for r in rows if r["confirmed"] and r["winner"] == r["work_id"] and r["truth"]]
    for version, key in (("old", "legacy_e2_selection"), ("new", "e2_selection")):
        require(
            e2_validation.select_rule(eligible, options_key=version)[0]["rule"] == params[key],
            f"Fold {number}: {version} threshold was not selected on development",
        )


def audit_care(fold, root, labels, frozen_audio):
    new = read_jsonl(root / "predictions.jsonl")
    old = read_jsonl(root / "legacy_predictions.jsonl")
    opened = read_jsonl(root / "open_predictions.jsonl")
    for name in ("predictions", "legacy_predictions"):
        require(verify(root / f"{name}.jsonl", False)["queries"] == 16, f"{root}: invalid {name} count")
    for rows in (new, old, opened):
        check_members([r["work_id"] for r in rows], fold["test_work_ids"], root)
    details = []
    paired_fields = (
        "query_id",
        "top10",
        "reference_recording_id",
        "gate",
        "confirmed",
        "decision",
        "threshold",
    )
    for revision, legacy, opened_row in zip(new, old, opened, strict=True):
        query = revision["query_id"]
        require(
            all(revision[k] == legacy[k] for k in paired_fields), f"Upstream E2 evidence changed: {query}"
        )
        recall = sorted(revision["top10"], key=lambda c: c["recall_rank"])
        audio = frozen_audio[query]
        check_rankings(
            [
                dict(
                    rank=c["recall_rank"],
                    score=c["audio"],
                    reference_recording_id=c["reference_recording_id"],
                    query_window=c["query_window"],
                    reference_window=c["reference_window"],
                )
                for c in recall
            ],
            audio["top10"],
        )
        correct = audio["correct_reference_recording_id"]
        open_ids = [
            c["reference_recording_id"]
            for c in sorted(opened_row["top10"], key=lambda c: c["recall_rank"])
        ]
        retained = [
            c["reference_recording_id"] for c in audio["top10"] if c["reference_recording_id"] != correct
        ]
        require(
            opened_row["query_id"] == query
            and opened_row["gallery_size"] == 79
            and len(set(open_ids)) == 10
            and correct not in open_ids
            and open_ids[: len(retained)] == retained,
            f"Invalid open retrieval: {query}",
        )
        old_options, new_options = legacy["interval_proposals"], revision["interval_proposals"]
        positions = lambda options: [(p["source"], p.get("path_index"), p["interval"]) for p in options]
        require(positions(old_options) == positions(new_options), f"Proposal locations changed: {query}")
        truth = labels[revision["work_id"]]
        eligible = revision["confirmed"] and revision["reference_recording_id"] == correct and bool(truth)
        row = dict(
            query_id=query,
            outer_fold=revision["outer_fold"],
            eligible=eligible,
            lyrics_enabled=revision["gate"]["enabled"],
            confirmed=revision["confirmed"],
            reference_recording_id=revision["reference_recording_id"],
            reference_intervals=truth,
        )
        if eligible:
            row.update(
                e2_validation.diagnostic(
                    query,
                    old_options,
                    new_options,
                    legacy["reference_interval"],
                    revision["reference_interval"],
                    truth,
                    row["lyrics_enabled"],
                    revision["interval_rule"],
                    old_rule=legacy["interval_rule"],
                )
            )
            row["proposal_count"] = len(new_options)
            row["qualifying_proposal_count"] = sum(
                e2_validation.label_iou(p["interval"], truth) >= 0.5 for p in new_options
            )
        else:
            row.update(
                old_interval=legacy["reference_interval"], new_interval=revision["reference_interval"]
            )
        details.append(row)
    return new, opened, details


def care_metrics(rows, opened, labels):
    result = measure(rows, opened, labels)
    ranks = [c["rank"] for row in rows for c in row["top10"] if c["reference_work_id"] == row["work_id"]]
    ns = sum(r["inference_ns"] for r in rows)
    result.update(
        hit5=sum(rank <= 5 for rank in ranks),
        hit10=len(ranks),
        inference_ns=ns,
        ms_per_query=ns / len(rows) / 1e6,
        total_ms=ns / 1e6,
    )
    return result


def previous_differences(fold, closed, opened):
    root = PREVIOUS / f"fold{fold}" / "A"
    expected = {
        name: {r["query_id"]: r for r in read_jsonl(root / f"{name}.jsonl")}
        for name in ("predictions", "open_predictions")
    }
    differences = []
    for name, rows in (("predictions", closed), ("open_predictions", opened)):
        check_members([r["query_id"] for r in rows], expected[name], root)
        for row in rows:
            source = expected[name][row["query_id"]]
            fields = ["reference_recording_id", "confirmed", "decision", "threshold", "top10"]
            if name == "predictions":
                fields += ["reference_interval", "interval_proposals"]

            def comparable(prediction, key):
                if key == "threshold" and prediction["decision"]["harmony"]:
                    return {"min_harmony_coverage": 0.5, **prediction[key]}
                return prediction[key]

            changes = [key for key in fields if comparable(source, key) != comparable(row, key)]
            if changes:
                differences.append(
                    dict(
                        query_id=row["query_id"],
                        mode=name,
                        changed_fields=changes,
                        previous={k: source[k] for k in changes},
                        current={k: row[k] for k in changes},
                    )
                )
    return differences


def e2_metrics(rows):
    eligible = [r for r in rows if r["eligible"]]
    groups = e2_validation.summarize(eligible)
    for name, members in (
        ("all", eligible),
        ("no_lyrics", [r for r in eligible if not r["lyrics_enabled"]]),
    ):
        groups[name]["proposal_count"] = sum(r["proposal_count"] for r in members)
        groups[name]["qualifying_proposal_count"] = sum(r["qualifying_proposal_count"] for r in members)
        groups[name]["all_queries"] = (
            len(rows) if name == "all" else sum(not r["lyrics_enabled"] for r in rows)
        )
    return {k: groups[k] for k in ("all", "no_lyrics")}


def write_csv(path, rows):
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    write_atomic(path, lambda out: out.write(stream.getvalue().encode()))


def summarize(dest, works, folds, digest):
    official = configs(folds, digest, OFFICIAL / "selected_windows.json")
    matched = configs(folds, digest, RECALL)
    labels = {r["work_id"]: r["reference_intervals"] for r in read_jsonl(LABELS)}
    require(
        set(labels) == set(works) and all(labels.values()), "Incomplete automatic interval references"
    )
    audio_rows = []
    for family, settings in (("formal", official), ("same_window", matched)):
        for model in MODELS:
            rows = read_jsonl(dest / "audio" / family / f"{model}.jsonl")
            audit_audio(rows, works, folds, settings)
            score = metrics(rows)
            audio_rows.append(
                dict(group=family, model=model, **score, total_ms=score["retrieval_ns"] / 1e6)
            )
    audio_b = {r["query_id"]: r for r in read_jsonl(dest / "audio/same_window/coverhunter.jsonl")}
    new, opened, details, differences = [], [], [], []
    by_fold = {}
    for fold in folds:
        number = fold["outer_fold"]
        root = dest / "care" / f"fold{number}"
        params = json.loads((root / "parameters.json").read_text())
        audit_development(fold, root, params, digest, works)
        prediction, open_prediction_rows, paired = audit_care(fold, root, labels, audio_b)
        new.extend(prediction)
        opened.extend(open_prediction_rows)
        details.extend(paired)
        differences.extend(previous_differences(number, prediction, open_prediction_rows))
        by_fold[str(number)] = dict(
            care=care_metrics(prediction, open_prediction_rows, labels),
            e2=e2_metrics(paired),
            legacy_rule=params["legacy_e2_selection"],
            path_rule=params["e2_selection"],
            setup_ns=json.loads((root / "timing.json").read_text())["setup_ns"],
        )
    for rows in (new, opened, details):
        check_members([r["query_id"] for r in rows], audio_b, "all five outer folds")
    totals = care_metrics(new, opened, labels)
    localization = e2_metrics(details)
    expected = (67, 39, 3)
    observed = tuple(totals[k] for k in ("top1", "correct_confirmations", "open_false_accepts"))
    report = dict(
        evaluation="frozen_configuration_reevaluation",
        folds_sha256=digest,
        audio=audio_rows,
        care=totals,
        e2=localization,
        by_fold=by_fold,
        recall_configurations=dict(formal=official, same_window=matched),
        sources=dict(
            formal_audio=str(OFFICIAL.relative_to(DATA)),
            prior_care_A=str(PREVIOUS.relative_to(DATA)),
            coverhunter_b_config=str(RECALL.relative_to(DATA)),
            references=str(LABELS.relative_to(DATA)),
        ),
        reproduction=dict(expected=expected, observed=observed, differing_rows=len(differences)),
        timing=dict(
            audio=TIMING,
            full_care="One serial pass, one BLAS thread; starts at in-memory embedding "
            "retrieval and ends after E2 and lyric typing. Includes lazy beat chroma and lyric cache reads and "
            "DTW. Excludes Engine startup/read of embedding caches, JIT warmup, fitting, open-set evaluation, "
            "JSON serialization and scoring. Per-fold setup_ns is recorded separately.",
        ),
    )
    write_json(dest / "summary.json", report)
    write_jsonl(dest / "e2_comparison.jsonl", details)
    write_jsonl(dest / "prior_care_differences.jsonl", differences)
    write_csv(dest / "audio_summary.csv", audio_rows)
    write_csv(
        dest / "care_summary.csv",
        [dict(fold="all", **totals), *[dict(fold=n, **r["care"]) for n, r in by_fold.items()]],
    )
    table = []
    for number, groups in [("all", localization), *((n, m["e2"]) for n, m in by_fold.items())]:
        for group, measures in groups.items():
            for version in ("old", "new", "old_at_new_rule", "new_at_old_rule"):
                table.append(
                    dict(
                        fold=number,
                        group=group,
                        version=version,
                        all_queries=measures["all_queries"],
                        proposal_count=measures["proposal_count"],
                        qualifying_proposal_count=measures["qualifying_proposal_count"],
                        **measures[version],
                    )
                )
    write_csv(dest / "e2_summary.csv", table)
    print(json.dumps(dict(reproduction=report["reproduction"], care=totals, e2=localization)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "audio", "fit", "predict", "report"))
    parser.add_argument("--output", type=Path, default=DEST)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    works, expected, folds, digest = inputs()
    if args.stage in ("run", "audio"):
        with threadpool_limits(limits=1):
            run_audio(works, expected, folds, digest, args.output)
    if args.stage in ("run", "fit"):
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for task in [pool.submit(fit_fold, f, args.output, digest) for f in folds]:
                task.result()
    if args.stage in ("run", "predict"):
        with threadpool_limits(limits=1):
            for fold in folds:
                run_care(fold, args.output, digest)
    if args.stage in ("run", "report"):
        summarize(args.output, works, folds, digest)


if __name__ == "__main__":
    main()
