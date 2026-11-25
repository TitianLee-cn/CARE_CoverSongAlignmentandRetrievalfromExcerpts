"""Exploratory five-fold reevaluation of the harmony coverage confirmation threshold."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from pipeline import (
    Engine,
    confirmation_rules,
    decision_features,
    file_sha256,
    inner_ranked_splits,
    passes,
    read_jsonl,
    selected_recall,
    threadpool_limits,
    write_json,
    write_jsonl,
)
from frozen_reevaluation import DEST as FROZEN, RECALL, inputs, require, write_csv

DEST = FROZEN.parent / "confirmation_coverage"
COVERAGES = (0.0, 0.2, 0.3, 0.4, 0.5)
GROUPS = ("control", "tuned")


def read(path):
    return json.loads(path.read_text())


def rule_for(obs, rules):
    return rules["music" if obs["harmony"] else "audio_only"]


def accepted(obs, rules):
    return passes(obs, rule_for(obs, rules))


def fit_validation(fold, dest, digest):
    number = fold["outer_fold"]
    source = FROZEN / "care" / f"fold{number}"
    root = dest / f"fold{number}"
    params = read(source / "parameters.json")
    config = selected_recall(RECALL, number, digest)
    require(
        params["fit_work_ids"] == fold["development_work_ids"]
        and (params["window"], params["overlap"], params["recall_lambda"]) == config
        and params["folds_sha256"] == digest
        and params["recall_config_sha256"] == file_sha256(RECALL)
        and params["harmony_weight"] == 0.3
        and params["lyric_rerank"],
        "Frozen CARE inputs do not match",
    )
    rows = []
    with threadpool_limits(limits=1):
        engine = Engine(number, *config)
        historical = {r["work_id"]: r for r in read_jsonl(source / "development_oof.jsonl")}
        for inner, _, _, pairs in inner_ranked_splits(engine):
            for work, closed, opened in pairs:
                old = historical[work]
                closed_obs = decision_features(closed)
                closed_ids = [r["reference_recording_id"] for r in closed]
                correct_recording = engine.queries[work]["reference_recording_id"]
                require(
                    old["inner_fold"] == inner
                    and closed_obs == old["decision"]
                    and closed_ids == old["top10"],
                    f"Frozen inner results differ for {work}",
                )
                rows.append(
                    dict(
                        work_id=work,
                        query_id=old["query_id"],
                        inner_fold=inner,
                        lyrics_enabled=old["lyrics_enabled"],
                        correct_reference_recording_id=correct_recording,
                        closed=dict(
                            top10=closed_ids, decision=closed_obs, recall_top10=recall_ids(closed)
                        ),
                        opened=dict(
                            top10=[r["reference_recording_id"] for r in opened],
                            decision=decision_features(opened),
                            recall_top10=recall_ids(opened),
                        ),
                    )
                )
    write_jsonl(root / "validation_oof.jsonl", rows)
    print(
        f"Fold {number}: reproduced frozen validation rankings and regenerated open-set decisions",
        flush=True,
    )


def recall_ids(candidates):
    return [r["reference_recording_id"] for r in sorted(candidates, key=lambda r: r["recall_rank"])]


def validation_observations(rows, fold, works):
    gallery = {works[w]["reference_recording_id"] for w in fold["development_work_ids"]}
    require(
        len(rows) == len({r["work_id"] for r in rows}) == 64, "Expected 64 unique development queries"
    )
    for split in fold["inner_folds"]:
        require(
            {r["work_id"] for r in rows if r["inner_fold"] == split["inner_fold"]}
            == set(split["validation_work_ids"]),
            "Validation row outside inner held-out split",
        )
    result = []
    for row in rows:
        truth = works[row["work_id"]]["reference_recording_id"]
        require(
            row["query_id"] == works[row["work_id"]]["query_id"]
            and row["correct_reference_recording_id"] == truth,
            "Validation query identity changed",
        )
        for name in ("closed", "opened"):
            item = row[name]
            ids = item["top10"]
            require(
                len(ids) == len(set(ids)) == 10 and set(ids) <= gallery, "Invalid validation retrieval"
            )
            if name == "opened":
                kept = [rec for rec in row["closed"]["recall_top10"] if rec != truth]
                require(
                    truth not in ids and item["recall_top10"][: len(kept)] == kept,
                    "Open gallery did not exclude the work",
                )
            result.append((item["decision"], name == "closed", ids[0] == truth))
    return result


def counts(observations, rules):
    return dict(
        correct_confirmations=sum(
            known and correct and accepted(obs, rules) for obs, known, correct in observations
        ),
        incorrect_confirmations=sum(
            known and not correct and accepted(obs, rules) for obs, known, correct in observations
        ),
        open_false_accepts=sum(not known and accepted(obs, rules) for obs, known, _ in observations),
        open_queries=sum(not known for _, known, _ in observations),
    )


def search(observations, control):
    audio_rule = control["audio_only"]
    music = [(obs, known, correct) for obs, known, correct in observations if obs["harmony"]]
    fixed = counts(
        [(obs, known, correct) for obs, known, correct in observations if not obs["harmony"]], control
    )
    total_open = sum(not known for _, known, _ in observations)
    sweep, best_by_coverage = [], []
    for coverage in COVERAGES:
        # In an optimum that confirms a correct song, thresholds can be raised to the
        # smallest score and margin of the accepted correct songs. Keep abstention too.
        positives = [
            obs for obs, known, correct in music if known and correct and obs["coverage"] >= coverage
        ]
        scores = sorted({0.0, 1.16, *(obs["score"] for obs in positives)})
        margins = sorted({0.0, *(obs["margin"] for obs in positives)})
        tested = []
        for score in scores:
            for margin in margins:
                rule = dict(score=score, margin=margin, min_harmony_coverage=coverage)
                tp = fp = false_accepts = 0
                for obs, known, correct in music:
                    if passes(obs, rule):
                        if not known:
                            false_accepts += 1
                        elif correct:
                            tp += 1
                        else:
                            fp += 1
                validation = dict(
                    correct_confirmations=tp + fixed["correct_confirmations"],
                    incorrect_confirmations=fp + fixed["incorrect_confirmations"],
                    open_false_accepts=false_accepts + fixed["open_false_accepts"],
                    open_queries=total_open,
                )
                tested.append(
                    dict(
                        rule=rule,
                        validation=validation,
                        eligible=validation["open_false_accepts"] * 20 <= total_open,
                    )
                )
        eligible = [r for r in tested if r["eligible"]]
        require(eligible, "No rule satisfied the open-set false-acceptance cap")
        best_by_coverage.append(max(eligible, key=objective))
        sweep.extend(tested)
    selected = max(best_by_coverage, key=objective)["rule"]
    negative_music = [obs for obs, known, _ in music if not known]
    selected = dict(
        selected,
        validation_open_count=len(negative_music),
        validation_false_accepts=sum(passes(obs, selected) for obs in negative_music),
    )
    return dict(music=selected, audio_only=audio_rule), best_by_coverage, sweep


def objective(row):
    data, rule = row["validation"], row["rule"]
    return (
        data["correct_confirmations"],
        -data["incorrect_confirmations"],
        -data["open_false_accepts"],
        rule["min_harmony_coverage"],
        rule["margin"],
        rule["score"],
    )


def audit_switch():
    obs = dict(score=0.8, margin=0.2, coverage=0.0, harmony=True)
    default = dict(score=0.7, margin=0.1)
    off = dict(default, min_harmony_coverage=0.0)
    require(not passes(obs, default) and passes(obs, off), "Coverage zero or legacy default failed")
    require(
        not passes(dict(obs, margin=0), off) and not passes(dict(obs, score=0), off),
        "Disabling coverage bypassed score or margin",
    )
    require(
        passes(dict(obs, harmony=False), dict(default, min_harmony_coverage=1.0)),
        "Harmony gate used for audio-only decision",
    )


def freeze_rules(fold, root, works):
    number = fold["outer_fold"]
    source = FROZEN / "care" / f"fold{number}"
    frozen_params = read(source / "parameters.json")
    validation = validation_observations(read_jsonl(root / "validation_oof.jsonl"), fold, works)
    control = confirmation_rules(validation)
    historical = {k: dict(rule) for k, rule in frozen_params["confirmation"].items()}
    historical["music"].setdefault("min_harmony_coverage", 0.5)
    require(control == historical, f"Fold {number}: the refit control thresholds differ")
    tuned, best_by_coverage, sweep = search(validation, control)
    write_jsonl(root / "search.jsonl", sweep)
    write_json(
        root / "selection.json",
        dict(
            control=dict(rules=control, validation=counts(validation, control)),
            tuned=dict(rules=tuned, validation=counts(validation, tuned)),
            best_by_coverage=best_by_coverage,
            source_parameters_sha256=file_sha256(source / "parameters.json"),
        ),
    )
    for group, rules in zip(GROUPS, (control, tuned), strict=True):
        write_json(root / group / "parameters.json", dict(frozen_params, confirmation=rules))
    print(f"Fold {number}: froze coverage {tuned['music']['min_harmony_coverage']:.0%}", flush=True)


def source_rows(fold, works):
    source = FROZEN / "care" / f"fold{fold['outer_fold']}"
    all_refs = {works[w]["reference_recording_id"] for w in fold["test_gallery_work_ids"]}
    closed, opened = (
        read_jsonl(source / name) for name in ("predictions.jsonl", "open_predictions.jsonl")
    )
    require(
        len(closed) == len(opened) == 16 and {r["work_id"] for r in closed} == set(fold["test_work_ids"]),
        "Outer query set differs from frozen fold",
    )
    for c, o in zip(closed, opened, strict=True):
        work = works[c["work_id"]]
        truth = work["reference_recording_id"]
        require(
            c["outer_fold"] == o["outer_fold"] == fold["outer_fold"]
            and c["query_id"] == o["query_id"] == work["query_id"]
            and c["work_id"] == o["work_id"],
            "Frozen outer identity mismatch",
        )
        closed_refs, open_refs = recall_ids(c["top10"]), recall_ids(o["top10"])
        kept = [rid for rid in closed_refs if rid != truth]
        require(
            len(set(closed_refs)) == len(set(open_refs)) == 10
            and set(closed_refs + open_refs) <= all_refs
            and o["gallery_size"] == 79
            and truth not in open_refs
            and open_refs[: len(kept)] == kept,
            f"{work['query_id']}: frozen open retrieval mismatch",
        )
    return closed, opened


def replay_row(row, rules, truth, lyrics_enabled, mode):
    candidates = [
        {
            k: c[k]
            for k in (
                "reference_recording_id",
                "recall_rank",
                "score",
                "audio",
                "query_window",
                "reference_window",
            )
        }
        for c in row["top10"]
    ]
    decision = row["decision"]
    require(
        row["reference_recording_id"] == candidates[0]["reference_recording_id"]
        and decision["score"] == candidates[0]["score"]
        and decision["margin"] == candidates[0]["score"] - candidates[1]["score"],
        "Frozen scores differ",
    )
    rule = rule_for(decision, rules)
    return dict(
        outer_fold=row["outer_fold"],
        work_id=row["work_id"],
        query_id=row["query_id"],
        mode=mode,
        lyrics_enabled=lyrics_enabled,
        confirmed=passes(decision, rule),
        reference_recording_id=row["reference_recording_id"],
        correct_reference_recording_id=truth,
        decision=decision,
        threshold=rule,
        top10=candidates,
    )


def run_outer(fold, root, works):
    closed, opened = source_rows(fold, works)
    for group in GROUPS:
        rules = read(root / group / "parameters.json")["confirmation"]
        rows = []
        for c, o in zip(closed, opened, strict=True):
            truth = works[c["work_id"]]["reference_recording_id"]
            for mode, source in (("closed", c), ("open", o)):
                output = replay_row(source, rules, truth, c["gate"]["enabled"], mode)
                if group == "control":
                    require(
                        output["confirmed"] == source["confirmed"],
                        f"Control did not reproduce {source['query_id']}",
                    )
                rows.append(output)
        write_jsonl(root / group / "predictions.jsonl", rows)


def metric(rows):
    closed = [r for r in rows if r["mode"] == "closed"]
    opened = [r for r in rows if r["mode"] == "open"]
    correct = [r for r in closed if r["reference_recording_id"] == r["correct_reference_recording_id"]]
    tp = sum(r["confirmed"] for r in correct)
    total = sum(r["confirmed"] for r in closed)
    false_accepts = sum(r["confirmed"] for r in opened)
    return dict(
        queries=len(closed),
        correct_confirmations=tp,
        incorrect_confirmations=total - tp,
        confirmation_coverage=total / len(closed) if closed else None,
        correct_top1_held=len(correct) - tp,
        open_queries=len(opened),
        open_false_accepts=false_accepts,
        open_false_accept_rate=false_accepts / len(opened) if opened else None,
    )


def audit_fold(fold, root, works):
    closed, opened = source_rows(fold, works)
    source = FROZEN / "care" / f"fold{fold['outer_fold']}" / "parameters.json"
    frozen = read(source)
    search_rows = read_jsonl(root / "search.jsonl")
    validation = validation_observations(read_jsonl(root / "validation_oof.jsonl"), fold, works)
    selection = read(root / "selection.json")
    control = confirmation_rules(validation)
    tuned, best_by_coverage, sweep = search(validation, control)
    require(
        search_rows == sweep
        and selection["best_by_coverage"] == best_by_coverage
        and selection["source_parameters_sha256"] == file_sha256(source),
        "Persisted search differs",
    )
    groups = {}
    for group, rules in zip(GROUPS, (control, tuned), strict=True):
        params = read(root / group / "parameters.json")
        require(
            params.pop("confirmation") == rules
            and selection[group] == dict(rules=rules, validation=counts(validation, rules)),
            "Saved selection differs",
        )
        require(
            params == {k: v for k, v in frozen.items() if k != "confirmation"},
            "Non-confirmation parameters changed",
        )
        results = read_jsonl(root / group / "predictions.jsonl")
        require(len(results) == 32, "Missing outer replay rows")
        for c, o, cr, orr in zip(closed, opened, results[::2], results[1::2], strict=True):
            for mode, row, new in (("closed", c, cr), ("open", o, orr)):
                truth = works[row["work_id"]]["reference_recording_id"]
                require(
                    new == replay_row(row, rules, truth, c["gate"]["enabled"], mode),
                    "Replay did not preserve candidate evidence",
                )
                if group == "control":
                    require(new["confirmed"] == row["confirmed"], "Historical confirmation changed")
        groups[group] = results
    return groups, selection


def summarize(dest, folds, works):
    combined = {g: [] for g in GROUPS}
    by_fold, parameter_rows = {}, []
    for fold in folds:
        number = fold["outer_fold"]
        groups, selection = audit_fold(fold, dest / f"fold{number}", works)
        for g in GROUPS:
            combined[g].extend(groups[g])
            music = selection[g]["rules"]["music"]
            parameter_rows.append(
                dict(
                    fold=number,
                    group=g,
                    coverage=music["min_harmony_coverage"],
                    coverage_seconds=30 * music["min_harmony_coverage"],
                    min_score=music["score"],
                    min_margin=music["margin"],
                    **selection[g]["validation"],
                )
            )
        by_fold[str(number)] = {g: metric(rows) for g, rows in groups.items()}
    metrics = {g: metric(rows) for g, rows in combined.items()}
    if len(folds) == 5:
        base = metrics["control"]
        require(
            (base["correct_confirmations"], base["incorrect_confirmations"], base["open_false_accepts"])
            == (39, 0, 3),
            "Control failed the published five-fold check",
        )
    for group, rows in combined.items():
        require(
            len({(r["query_id"], r["mode"]) for r in rows}) == len(rows) == 32 * len(folds),
            f"{group}: duplicate queries",
        )
    changes = []
    fields = (
        "outer_fold",
        "query_id",
        "mode",
        "lyrics_enabled",
        "reference_recording_id",
        "correct_reference_recording_id",
        "decision",
        "top10",
    )
    for a, b in zip(combined["control"], combined["tuned"], strict=True):
        require(all(a[k] == b[k] for k in fields), "Outer ranking or input scores changed")
        if a["confirmed"] != b["confirmed"]:
            changes.append(
                {k: a[k] for k in fields if k != "top10"}
                | {
                    "change": "added" if b["confirmed"] else "removed",
                    "control_rule": a["threshold"],
                    "tuned_rule": b["threshold"],
                }
            )
    lyrics = {}
    for name, enabled in (("lyrics_enabled", True), ("no_effective_lyrics", False)):
        lyrics[name] = {
            g: metric([r for r in rows if r["lyrics_enabled"] == enabled]) for g, rows in combined.items()
        }
        lyrics[name]["changes"] = [r for r in changes if r["lyrics_enabled"] == enabled]
    write_json(
        dest / "summary.json",
        dict(
            evaluation="exploratory_reevaluation",
            source=str(FROZEN),
            groups=metrics,
            by_fold=by_fold,
            lyrics=lyrics,
        ),
    )
    write_jsonl(dest / "changes.jsonl", changes)
    write_csv(dest / "parameters.csv", parameter_rows)
    write_csv(
        dest / "summary.csv",
        [
            dict(group=g, fold=f, **s)
            for f, groups in [("all", metrics), *by_fold.items()]
            for g, s in groups.items()
        ],
    )
    print(json.dumps(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "run", "verify"))
    parser.add_argument("--output", type=Path, default=DEST)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    audit_switch()
    works, _, folds, digest = inputs()
    dest = args.output / "smoke" if args.stage == "smoke" else args.output
    if args.stage == "smoke":
        folds = [folds[2]]
        fit_validation(folds[0], dest, digest)
    if args.stage == "run":
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for future in [pool.submit(fit_validation, fold, dest, digest) for fold in folds]:
                future.result()
    if args.stage != "verify":
        for f in folds:
            freeze_rules(f, dest / f"fold{f['outer_fold']}", works)
        for f in folds:
            run_outer(f, dest / f"fold{f['outer_fold']}", works)
    summarize(dest, folds, works)


if __name__ == "__main__":
    main()
