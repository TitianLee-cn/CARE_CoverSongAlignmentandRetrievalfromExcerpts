"""Exploratory five-fold Kiki lexical localization after frozen CARE song confirmation."""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from pipeline import (
    DATA,
    Engine,
    e2,
    e2_validation,
    file_sha256,
    infer,
    passes,
    read_jsonl,
    threadpool_limits,
    write_json,
    write_jsonl,
)
from confirmation_coverage import FROZEN, inputs, require, rule_for, source_rows

EXPERIMENTS = DATA / "experiments"
DEST = EXPERIMENTS / "kiki_lyric_localization"
ZERO = EXPERIMENTS / "confirmation_coverage_zero"
REFERENCES = DATA / "reference_labels/covers80"
HISTORICAL_LABELS = REFERENCES / "reference_intervals_before_23_review.jsonl"
LABELS = Path(os.environ.get(
    "CARE_KIKI_REFERENCE_INTERVALS",
    str(HISTORICAL_LABELS if HISTORICAL_LABELS.is_file() else REFERENCES / "reference_intervals.jsonl"),
)).resolve()
EXPECTED = (67, 59, 0, 1, 43, 40, 16)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def paths_for(fold):
    root = FROZEN / "care" / f"fold{fold}"
    params = (
        ZERO / "fold1_parameters.json"
        if fold == 1
        else EXPERIMENTS / "confirmation_coverage" / f"fold{fold}" / "tuned" / "parameters.json"
    )
    return root, params


def load_labels(works):
    labels = {r["work_id"]: r for r in read_jsonl(LABELS)}
    require(
        set(labels) == set(works)
        and Counter(r["annotation_status"] for r in labels.values())
        == {"manual": 9, "automatic_reference": 71},
        f"Unexpected reference version: {LABELS}",
    )
    for w, row in labels.items():
        require(
            all(
                row[key] == works[w][key]
                for key in ("query_id", "query_recording_id", "reference_recording_id")
            )
            and row["reference_intervals"]
            and all(0 <= i["start_sec"] < i["end_sec"] for i in row["reference_intervals"]),
            f"Invalid reference for {w}",
        )
    return labels


def score(row, interval, labels):
    if interval is None:
        return None
    ref = labels[row["work_id"]]
    if row["mode"] == "open" or row["reference_recording_id"] != ref["reference_recording_id"]:
        return 0.0
    return e2_validation.label_iou(interval, ref["reference_intervals"])


def ranking(row):
    keys = ("reference_recording_id", "recall_rank", "score", "audio", "query_window", "reference_window")
    return [{k: c[k] for k in keys} for c in row["top10"]]


def build_baseline(folds, works, digest, labels, dest):
    replay = {(r["query_id"], r["mode"]): r for r in read_jsonl(ZERO / "predictions.jsonl")}
    require(len(replay) == 160, "The zero-coverage confirmation replay must contain 160 rows")
    rows, rebuilt = [], []
    for fold in folds:
        n = fold["outer_fold"]
        root, param_path = paths_for(n)
        params = read(param_path)
        previous = read(root / "parameters.json")
        require(
            {k: v for k, v in params.items() if k != "confirmation"}
            == {k: v for k, v in previous.items() if k != "confirmation"}
            and params["fit_work_ids"] == fold["development_work_ids"]
            and params["folds_sha256"] == digest
            and params["outer_fold"] == n
            and params["recall_lambda"] == 0
            and params["harmony_weight"] == 0.3
            and params["lyric_rerank"]
            and params["e2_lyrics"]
            and params["e2_version"] == "path"
            and params["confirmation"]["music"]["min_harmony_coverage"] == 0,
            f"Fold {n} E2 or retrieval settings differ from the frozen reevaluation",
        )
        closed, opened = source_rows(fold, works)
        engine = None
        for mode, predictions in (("closed", closed), ("open", opened)):
            for original in predictions:
                work, query = original["work_id"], original["query_id"]
                ref = original["reference_recording_id"]
                rule = rule_for(original["decision"], params["confirmation"])
                confirmed = passes(original["decision"], rule)
                anchor = replay[query, mode]
                require(
                    (confirmed, ref) == (anchor["confirmed"], anchor["reference_recording_id"]),
                    f"Confirmation mismatch for {query}/{mode}: {param_path}",
                )
                options = original.get("interval_proposals", [])
                if confirmed and not options:
                    if engine is None:
                        engine = Engine(n, params["window"], params["overlap"], params["recall_lambda"])
                    candidates = (
                        original["top10"]
                        if mode == "closed"
                        else engine.candidates(
                            work,
                            [w for w in fold["test_gallery_work_ids"] if w != work],
                            params["background"],
                            True,
                        )
                    )
                    computed = infer(engine, params, work, candidates=candidates)
                    require(
                        computed["confirmed"]
                        and computed["decision"] == original["decision"]
                        and ranking(computed) == ranking(original),
                        f"Rebuilt E2 changed ranking for {query}/{mode}",
                    )
                    options = computed["interval_proposals"]
                    rebuilt.append(dict(query_id=query, mode=mode))
                interval = e2.select(options, **params["e2_selection"]) if confirmed else None
                if confirmed and original.get("interval_proposals"):
                    require(
                        interval == original["reference_interval"], f"Frozen E2 replay changed {query}"
                    )
                rows.append(
                    dict(
                        mode=mode,
                        outer_fold=n,
                        query_id=query,
                        work_id=work,
                        confirmed=confirmed,
                        reference_recording_id=ref,
                        decision=original["decision"],
                        threshold=rule,
                        top10=ranking(original),
                        e2_proposals=options if confirmed else [],
                        e2_rule=params["e2_selection"],
                        original_e2=interval,
                    )
                )
    closed = [r for r in rows if r["mode"] == "closed"]
    opened = [r for r in rows if r["mode"] == "open"]
    correct = [
        r for r in closed if r["reference_recording_id"] == labels[r["work_id"]]["reference_recording_id"]
    ]
    ious = [score(r, r["original_e2"], labels) for r in closed if r["original_e2"] is not None]
    observed = (
        len(correct),
        sum(r["confirmed"] for r in correct),
        sum(r["confirmed"] for r in closed) - sum(r["confirmed"] for r in correct),
        sum(r["confirmed"] for r in opened),
        len(ious),
        sum(v >= 0.5 for v in ious),
        sum(r["confirmed"] and r["original_e2"] is None for r in closed),
    )
    history = read(ZERO / "summary.json")["current"]
    require(
        observed == EXPECTED and abs(sum(ious) - history["interval_iou_sum"]) < 1e-8,
        f"Baseline mismatch: {observed}; IoU sum {sum(ious)}; {LABELS}:{file_sha256(LABELS)}",
    )
    write_jsonl(dest / "baseline.jsonl", rows)
    write_json(
        dest / "baseline_audit.json",
        dict(
            observed=observed,
            iou_sum=sum(ious),
            rebuilt=rebuilt,
            baseline_sha256=file_sha256(dest / "baseline.jsonl"),
            labels_sha256=file_sha256(LABELS),
            folds_sha256=digest,
            sources={
                str(p.relative_to(DATA)): file_sha256(p)
                for f in folds
                for p in (
                    *[
                        paths_for(f["outer_fold"])[0] / name
                        for name in (
                            "predictions.jsonl",
                            "open_predictions.jsonl",
                            "development_oof.jsonl",
                        )
                    ],
                    paths_for(f["outer_fold"])[1],
                )
            },
            confirmation_replay_sha256=file_sha256(ZERO / "predictions.jsonl"),
        ),
    )
    print(f"Reproduced baseline {observed}; rebuilt {rebuilt}", flush=True)
    return rows


def case(option, row, labels):
    value = score(row, option, labels)
    return dict(
        interval=option,
        iou=value,
        hit=value is not None and value >= 0.5,
        error=value is not None and value < 0.5,
    )


def rules(near, tolerance=1.5, margin=1.0):
    return dict(near_threshold=near, tolerance=tolerance, position_margin=margin)


B_RULE = rules(None)
C_RULE = rules(0.70)
GROUPS = ("A", "B", "C", "D")


def load_for_row(corpus, row, ref_by_id):
    from kiki_lyric_localization import read_evidence

    return read_evidence(corpus, row["work_id"], ref_by_id[row["reference_recording_id"]])


def locate(row, evidence, config, *, baseline_output=False, generated=None):
    from time import perf_counter_ns
    from kiki_lyric_localization import fuse, generate

    old = row["original_e2"]
    if evidence is None:
        return dict(
            final_interval=old,
            interval_source="original_e2" if old else "pending",
            fusion_decision="baseline" if row["confirmed"] else "unconfirmed_song",
            selected=None,
            candidates=[],
            selection_reason=None,
            timing=None,
        )
    timing = {"cache_ns": evidence["cache_ns"], "gate_ns": evidence["gate_ns"]}
    if generated is None:
        start = perf_counter_ns()
        generated = generate(
            evidence, near_threshold=config["near_threshold"], tolerance=config["tolerance"]
        )
        timing["match_ns"] = perf_counter_ns() - start if evidence["reason"] is None else None
    start = perf_counter_ns()
    result = fuse(
        generated, old, row["e2_proposals"], row["e2_rule"]["threshold"], config["position_margin"]
    )
    if baseline_output:
        result.update(
            final_interval=old,
            interval_source="original_e2" if old else "pending",
            fusion_decision="development_selected_baseline",
        )
    timing["fusion_ns"] = perf_counter_ns() - start
    return dict(result, **generated, timing=timing, gate=evidence.get("gate"))


def development_rows(fold, works):
    root, param_path = paths_for(fold["outer_fold"])
    params = read(param_path)
    rows = read_jsonl(root / "development_oof.jsonl")
    require(len(rows) == len({r["work_id"] for r in rows}) == 64, "Incomplete inner validation rows")
    gallery = {works[w]["reference_recording_id"] for w in fold["development_work_ids"]}
    missing = set()
    for split in fold["inner_folds"]:
        inner = split["inner_fold"]
        require(
            {r["work_id"] for r in rows if r["inner_fold"] == inner} == set(split["validation_work_ids"]),
            "Incorrect development held-out split",
        )
    for r in rows:
        require(
            r["query_id"] == works[r["work_id"]]["query_id"]
            and len(r["top10"]) == len(set(r["top10"])) == 10
            and set(r["top10"]) <= gallery
            and r["top10"][0] == works[r["winner"]]["reference_recording_id"],
            "Invalid inner winner",
        )
        if not r.get("new"):
            missing.add(r["work_id"])
    if missing:
        return repair_development(rows, missing, params)
    return rows, []


def repair_development(rows, missing, params):
    # Only an inner split's own 48 works can supply confidence fitting examples.
    from pipeline import inner_ranked_splits, position_examples, position_options, score_options

    engine = Engine(params["outer_fold"], params["window"], params["overlap"], params["recall_lambda"])
    snapshots = {r["work_id"]: r for r in rows}
    training_labels = {w: r["truth"] for w, r in snapshots.items()}
    repaired = []
    for inner, _, fit_rows, pairs in inner_ranked_splits(engine):
        models = None
        for w, closed, _ in pairs:
            if w not in missing:
                continue
            original = snapshots[w]
            require(
                original["inner_fold"] == inner
                and original["top10"] == [c["reference_recording_id"] for c in closed],
                f"Rebuilt inner ranking differed for {w}",
            )
            if models is None:
                models = e2.train(position_examples(engine, fit_rows, training_labels))
            options = position_options(engine, w, closed[0], True)
            original["new"] = score_options(options, models, "path_features")
            repaired.append(w)
    require(set(repaired) == missing, "Inner E2 repair incomplete")
    return rows, repaired


def select_development(fold, works, labels, corpus, ref_by_id, dest):
    from itertools import product
    from kiki_lyric_localization import generate

    number = fold["outer_fold"]
    root = dest / f"fold{number}"
    rows, repaired = development_rows(fold, works)
    params = read(paths_for(number)[1])
    configs = [
        rules(near, tolerance, margin)
        for near, tolerance, margin in product((None, 0.50, 0.70), (1.5, 3.0), (0.5, 1.0))
    ]
    validation = [[] for _ in configs]
    baseline = []
    for row in rows:
        work = row["work_id"]
        require(work in labels, "Development scorer received a non-development query")
        if not passes(row["decision"], rule_for(row["decision"], params["confirmation"])):
            continue
        old = e2.select(row["new"], **params["e2_selection"])
        query = dict(
            work_id=work,
            query_id=row["query_id"],
            mode="closed",
            original_e2=old,
            reference_recording_id=row["top10"][0],
            e2_proposals=row["new"],
            e2_rule=params["e2_selection"],
        )
        cached = load_for_row(corpus, query, ref_by_id)
        old_score = case(old, query, labels)
        baseline.append(dict(query_id=row["query_id"], inner_fold=row["inner_fold"], **old_score))
        generated = {}
        for config, output in zip(configs, validation, strict=True):
            key = config["near_threshold"], config["tolerance"]
            if key not in generated:
                generated[key] = generate(cached, near_threshold=key[0], tolerance=key[1])
            choice = locate(query, cached, config, generated=generated[key])
            output.append(
                dict(
                    query_id=row["query_id"],
                    inner_fold=row["inner_fold"],
                    reason=choice["selection_reason"],
                    fusion=choice["fusion_decision"],
                    **case(choice["final_interval"], query, labels),
                )
            )
    a_hits = sum(r["hit"] for r in baseline)
    a_errors = sum(r["error"] for r in baseline)
    sweep = []
    for config, entries in zip(configs, validation, strict=True):
        hits = sum(r["hit"] for r in entries)
        errors = sum(r["error"] for r in entries)
        regressions = sum(a["hit"] and b["error"] for a, b in zip(baseline, entries, strict=True))
        sweep.append(
            dict(
                config=config,
                hits=hits,
                errors=errors,
                baseline_hits_changed_to_error=regressions,
                acceptable=errors <= a_errors,
                improves=hits > a_hits,
            )
        )
    permitted = [row for row in sweep if row["acceptable"] and row["improves"]]
    selected = max(permitted, key=selection_key)["config"] if permitted else None
    write_json(
        root / "configuration.json",
        dict(
            outer_fold=number,
            exploratory=True,
            selected=selected,
            fallback_diagnostic_configuration=C_RULE if selected is None else None,
            fit_works=fold["development_work_ids"],
            baseline=dict(hits=a_hits, errors=a_errors, confirmed=len(baseline)),
            trials=sweep,
            repaired_inner_works=repaired,
            search_policy="Increase hits without increasing errors. Ties: fewer "
            "A hits changed to errors, fewer errors, exact-only, higher Jaccard, shorter tolerance, larger margin.",
        ),
    )
    write_jsonl(
        root / "development_trials.jsonl",
        [
            {"config": None, "predictions": baseline},
            *[
                dict(config=config, predictions=entries)
                for config, entries in zip(configs, validation, strict=True)
            ],
        ],
    )
    print(f"Fold {number} development: A {a_hits} hits / {a_errors} errors, D {selected}", flush=True)


def selection_key(row):
    c = row["config"]
    near = c["near_threshold"]
    return (
        row["hits"],
        -row["baseline_hits_changed_to_error"],
        -row["errors"],
        near is None,
        near or 0,
        -c["tolerance"],
        c["position_margin"],
    )


def baseline_audit(dest, digest):
    audit = read(dest / "baseline_audit.json")
    require(
        tuple(audit["observed"]) == EXPECTED
        and audit["baseline_sha256"] == file_sha256(dest / "baseline.jsonl")
        and audit["labels_sha256"] == file_sha256(LABELS)
        and audit["folds_sha256"] == digest
        and audit["confirmation_replay_sha256"] == file_sha256(ZERO / "predictions.jsonl"),
        "The verified baseline has a different label or confirmation version",
    )
    for path, digest in audit["sources"].items():
        require(file_sha256(DATA / path) == digest, f"Baseline source changed: {path}")
    return audit


def apply_fold(fold, rows, labels, corpus, ref_by_id, dest):
    number = fold["outer_fold"]
    root = dest / f"fold{number}"
    frozen = read(root / "configuration.json")
    output = {g: [] for g in GROUPS}
    for row in rows:
        data = load_for_row(corpus, row, ref_by_id) if row["confirmed"] else None
        for group, config in (("A", None), ("B", B_RULE), ("C", C_RULE), ("D", frozen["selected"])):
            evidence = data if group != "A" else None
            result = locate(
                row, evidence, config or C_RULE, baseline_output=group == "D" and config is None
            )
            output[group].append(
                dict(
                    row,
                    group=group,
                    method_config=config,
                    correct_reference_recording_id=labels[row["work_id"]]["reference_recording_id"],
                    localization=result,
                    evaluation=case(result["final_interval"], row, labels),
                    best_candidate_iou=max(
                        (score(row, p["interval"], labels) for p in result["candidates"]), default=None
                    ),
                )
            )
    for group, predictions in output.items():
        write_jsonl(root / f"{group}.jsonl", predictions)
    print(f"Fold {number} outer: wrote {len(rows)} queries to each group", flush=True)


def smoke_cases():
    from kiki_lyric_localization import fuse, generate

    def line(i, a, b, text, **kwargs):
        return dict(line_id=i, start_sec=a, end_sec=b, text=text, **kwargs)

    def run(query, ref, near=None):
        return generate(
            dict(reason=None, query_lines=query, reference_lines=ref, duration=200),
            near_threshold=near,
            tolerance=1.5,
        )

    query = [
        line(1, 2, 8, "Oh, first line exactly here"),
        line(2, 20, 26, "different words near the ending"),
    ]
    cut = [
        line(1, 54.4, 58, "first line"),
        line(2, 58, 61.6, "exactly here"),
        line(3, 76, 83.2, "different words near the ending"),
    ]
    exact = run(query, cut)
    (candidate,) = exact["candidates"]
    require(
        candidate["exact_count"] == 2
        and abs(candidate["slope"] - 1.2) < 1e-9
        and candidate["matches"][0]["reference_line_ids"] == (1, 2),
        "Exact, merge or tempo check failed",
    )
    swapped = run(
        [line(1, 2, 5, "first line"), line(2, 5, 8, "exactly here"), query[1] | {"line_id": 3}],
        [
            line(1, 52.4, 59.6, "first line exactly here"),
            line(2, 74, 81.2, "different words near the ending"),
        ],
    )
    require(
        swapped["candidates"][0]["matches"][0]["query_line_ids"] == (1, 2), "Split-query match failed"
    )
    partial = run([query[0] | {"partial_left": True}, query[1]], cut)
    require(partial["reason"] == "no_two_effective_matches", "Partial line entered matching")
    repeat = [query[0], line(2, 12, 16, query[0]["text"]), query[1] | {"line_id": 3}]
    repeats_ref = [
        line(i, line_["start_sec"] + 50, line_["end_sec"] + 50, line_["text"])
        for i, line_ in enumerate(repeat, 1)
    ]
    repeats = run(repeat, repeats_ref)
    require(repeats["candidates"][0]["score"] == 2, "Repeated text scored twice")
    two = run(
        repeat,
        repeats_ref
        + [
            line(i + 3, l["start_sec"] + 65, l["end_sec"] + 65, l["text"])
            for i, l in enumerate(repeats_ref, 1)
        ],
    )
    require(
        len(two["candidates"]) == 2
        and fuse(two, None, [], 0.5, 1.0)["selection_reason"] == "ambiguous_positions",
        "Repeated chorus was not kept ambiguous",
    )
    musical = [dict(source="audio", confidence=0.6, interval=[50, 80])]
    require(
        fuse(two, None, musical, 0.5, 1.0)["selection_reason"] == "music_tiebreak",
        "Music did not break tie",
    )
    require(
        fuse(two, [115, 145], musical, 0.5, 1.0)["fusion_decision"] == "keep_original_conflict",
        "Music-tiebroken lyrics replaced a conflicting E2 output",
    )
    require(
        fuse(exact, [50, 84], [], 0.5, 1.0)["fusion_decision"] == "adjust_boundary"
        and fuse(exact, [100, 130], [], 0.5, 1.0)["fusion_decision"] == "replace_conflict",
        "Confirmed E2 output could not be adjusted or conservatively replaced",
    )
    near_query = [line(1, 2, 8, "one two three four five six seven eight nine"), query[1]]
    near_ref = [
        line(1, 52, 58, "one two three four five six seven eight zero"),
        line(2, 70, 76, query[1]["text"]),
    ]
    require(
        not run(near_query, near_ref)["candidates"]
        and run(near_query, near_ref, 0.70)["candidates"][0]["score"] == 1.5,
        "Near-hit score failed",
    )
    none = generate(dict(reason="no_effective_lyrics"), near_threshold=None, tolerance=1.5)
    require(fuse(none, [2, 32], [], 0.5, 1.0)["final_interval"] == [2, 32], "No-lyrics fallback failed")
    print("Lexical and fusion checks passed", flush=True)


def verify_selection(root, fold, labels):
    frozen = read(root / "configuration.json")
    rows = read_jsonl(root / "development_trials.jsonl")
    require(
        len(rows) == 13
        and frozen["fit_works"] == fold["development_work_ids"]
        and rows[0]["config"] is None,
        "Incomplete development grid",
    )
    configs = [rules(n, t, m) for n in (None, 0.50, 0.70) for t in (1.5, 3.0) for m in (0.5, 1.0)]
    require([r["config"] for r in rows[1:]] == configs, "Unexpected development parameters")
    allowed = [{labels[w]["query_id"] for w in s["validation_work_ids"]} for s in fold["inner_folds"]]
    base = rows[0]["predictions"]
    ids = [r["query_id"] for r in base]
    require(
        len(ids) == len(set(ids)) and all(r["query_id"] in allowed[r["inner_fold"] - 1] for r in base),
        "D selection included an invalid holdout",
    )
    source, parameters = paths_for(fold["outer_fold"])
    params = read(parameters)
    oof_by_query = {
        r["query_id"]: r
        for r in read_jsonl(source / "development_oof.jsonl")
        if passes(r["decision"], rule_for(r["decision"], params["confirmation"]))
    }
    require(set(oof_by_query) == set(ids), "Development confirmation set changed")
    for trial in rows:
        for result in trial["predictions"]:
            oof = oof_by_query[result["query_id"]]
            query = dict(work_id=oof["work_id"], mode="closed", reference_recording_id=oof["top10"][0])
            expected = case(result["interval"], query, labels)
            require(all(result[k] == v for k, v in expected.items()), "Saved development score differs")
            if trial["config"] is None and oof.get("new"):
                require(
                    result["interval"] == e2.select(oof["new"], **params["e2_selection"]),
                    "Development baseline used a different E2 interval",
                )
    a = dict(hits=sum(r["hit"] for r in base), errors=sum(r["error"] for r in base), confirmed=len(base))
    require(frozen["baseline"] == a, "Saved development baseline differs")
    sweep = []
    for row in rows[1:]:
        pred = row["predictions"]
        require([r["query_id"] for r in pred] == ids, "Development option used different queries")
        hits, errors = sum(r["hit"] for r in pred), sum(r["error"] for r in pred)
        sweep.append(
            dict(
                config=row["config"],
                hits=hits,
                errors=errors,
                baseline_hits_changed_to_error=sum(
                    x["hit"] and y["error"] for x, y in zip(base, pred, strict=True)
                ),
                acceptable=errors <= a["errors"],
                improves=hits > a["hits"],
            )
        )
    pool = [r for r in sweep if r["acceptable"] and r["improves"]]
    selected = max(pool, key=selection_key)["config"] if pool else None
    require(sweep == frozen["trials"] and selected == frozen["selected"], "Development selection differs")
    return selected


def metrics(rows):
    from numpy import percentile

    seen = [r for r in rows if r["confirmed"]]
    emitted = [r["evaluation"] for r in seen if r["evaluation"]["interval"] is not None]
    hits = sum(r["hit"] for r in emitted)
    timers = [r["localization"]["timing"] for r in seen if r["localization"]["timing"] is not None]
    status = Counter(
        r["localization"]["selection_reason"] for r in seen if r["localization"]["timing"] is not None
    )
    result = dict(
        queries=len(rows),
        confirmed=len(seen),
        correct_confirmed=sum(
            r["mode"] == "closed" and r["reference_recording_id"] == r["correct_reference_recording_id"]
            for r in seen
        ),
        outputs=len(emitted),
        hits=hits,
        errors=len(emitted) - hits,
        pending=len(seen) - len(emitted),
        mean_iou=sum(r["iou"] for r in emitted) / len(emitted) if emitted else None,
        hit_per_output=hits / len(emitted) if emitted else None,
        cache_calls=len(timers),
        candidate_reasons=dict(status),
        fusion_decisions=dict(Counter(r["localization"]["fusion_decision"] for r in seen)),
    )
    for stage in ("cache", "gate", "match", "fusion"):
        samples = [t[stage + "_ns"] / 1e6 for t in timers if t[stage + "_ns"] is not None]
        result.update(
            {
                stage + "_calls": len(samples),
                stage + "_median_ms": float(percentile(samples, 50)) if samples else None,
                stage + "_p95_ms": float(percentile(samples, 95)) if samples else None,
            }
        )
    for name, members in (
        ("all", seen),
        ("original_output", [r for r in seen if r["original_e2"] is not None]),
        ("original_pending", [r for r in seen if r["original_e2"] is None]),
    ):
        result["candidate_" + name] = dict(
            queries=len(members), with_hit=sum((r["best_candidate_iou"] or 0) >= 0.5 for r in members)
        )
    return result


def read_groups(dest, folds, baseline, labels):
    from kiki_lyric_localization import fuse
    from group_b.lyrics import iou

    by_group = {g: [] for g in GROUPS}
    for fold in folds:
        n = fold["outer_fold"]
        root = dest / f"fold{n}"
        selected = verify_selection(root, fold, labels)
        expected = {(r["query_id"], r["mode"]): r for r in baseline if r["outer_fold"] == n}
        for group, config in (("A", None), ("B", B_RULE), ("C", C_RULE), ("D", selected)):
            rows = read_jsonl(root / f"{group}.jsonl")
            require(
                len(rows) == 32 and {(r["query_id"], r["mode"]) for r in rows} == set(expected),
                f"Fold {n} {group}: duplicate or missing query",
            )
            for r in rows:
                a = expected[r["query_id"], r["mode"]]
                require(
                    all(r[k] == v for k, v in a.items()) and r["method_config"] == config,
                    f"{group} changed frozen rankings or settings for {r['query_id']}",
                )
                loc = r["localization"]
                active = group != "A" and r["confirmed"]
                require(
                    active == (loc["timing"] is not None),
                    f"Wrong matching eligibility for {r['query_id']}",
                )
                if active:
                    recomputed = fuse(
                        loc,
                        r["original_e2"],
                        r["e2_proposals"],
                        r["e2_rule"]["threshold"],
                        (config or C_RULE)["position_margin"],
                    )
                    if group == "D" and config is None:
                        require(
                            loc["fusion_decision"] == "development_selected_baseline"
                            and loc["final_interval"] == r["original_e2"],
                            "D fallback altered the original",
                        )
                    else:
                        require(all(loc[k] == v for k, v in recomputed.items()), "Saved fusion differs")
                    for p in loc["candidates"]:
                        pairs = p["matches"]
                        require(
                            len(pairs) >= 2
                            and 0.67 <= p["slope"] <= 1.50
                            and p["exact_count"] == sum(m["score"] == 1 for m in pairs) >= 1
                            and p["score"] == sum(m["score"] for m in pairs) >= 1.5
                            and all(
                                m["residual_sec"] <= (config or C_RULE)["tolerance"] + 1e-9 for m in pairs
                            )
                            and all(
                                x[k][-1] < y[k][0]
                                for x, y in zip(pairs, pairs[1:])
                                for k in ("query_line_ids", "reference_line_ids")
                            )
                            and all(
                                x[k][1] <= y[k][0]
                                for x, y in zip(pairs, pairs[1:])
                                for k in ("query_sec", "reference_sec")
                            ),
                            "Invalid lexical path",
                        )
                    require(
                        all(
                            iou(p["interval"], q["interval"]) < 0.8
                            for i, p in enumerate(loc["candidates"])
                            for q in loc["candidates"][:i]
                        ),
                        "Duplicate lexical interval",
                    )
                else:
                    require(
                        loc["final_interval"] == r["original_e2"], "Inactive localization output changed"
                    )
                require(
                    r["correct_reference_recording_id"] == labels[r["work_id"]]["reference_recording_id"]
                    and r["evaluation"] == case(loc["final_interval"], r, labels)
                    and r["best_candidate_iou"]
                    == max((score(r, p["interval"], labels) for p in loc["candidates"]), default=None),
                    "Wrong maximum-IoU scoring",
                )
            by_group[group].extend(rows)
    for group, rows in by_group.items():
        require(
            len({(r["mode"], r["query_id"]) for r in rows}) == len(rows) == 32 * len(folds),
            f"Duplicate outer results for {group}",
        )
    return by_group


def compare(groups):
    baseline = {(r["mode"], r["query_id"]): r for r in groups["A"]}
    for group in ("B", "C", "D"):
        for row in groups[group]:
            old = baseline[row["mode"], row["query_id"]]["evaluation"]
            new = row["evaluation"]
            if old == new:
                continue
            state = lambda x: "hit" if x["hit"] else "error" if x["error"] else "pending"
            old_span, new_span = old["interval"], new["interval"]
            evidence = row["localization"]
            yield dict(
                query_id=row["query_id"],
                group=group,
                mode=row["mode"],
                outer_fold=row["outer_fold"],
                transition=state(old) + "_to_" + state(new),
                before=old,
                after=new,
                boundary_delta_sec=[b - a for a, b in zip(old_span, new_span)]
                if old_span and new_span
                else None,
                iou_delta=new["iou"] - old["iou"] if old_span and new_span else None,
                selected=evidence["selected"],
                selection_reason=evidence["selection_reason"],
                fusion_decision=evidence["fusion_decision"],
            )


def report(dest, folds, baseline, labels, *, persist):
    from frozen_reevaluation import write_csv

    groups = read_groups(dest, folds, baseline, labels)
    changes = list(compare(groups))
    overall = {}
    summary_rows = []
    for mode in ("closed", "open"):
        overall[mode] = {}
        for fold in ("all", *[f["outer_fold"] for f in folds]):
            overall[mode][str(fold)] = {}
            for group in GROUPS:
                rows = [
                    r
                    for r in groups[group]
                    if r["mode"] == mode and (fold == "all" or r["outer_fold"] == fold)
                ]
                measurements = metrics(rows)
                overall[mode][str(fold)][group] = measurements
                summary_rows.append(
                    dict(
                        mode=mode,
                        fold=fold,
                        group=group,
                        **{k: v for k, v in measurements.items() if not isinstance(v, dict)},
                    )
                )
    deltas = {}
    for group in ("B", "C", "D"):
        a, b = [overall["closed"]["all"][g] for g in ("A", group)]
        deltas[group] = dict(
            net_hits=b["hits"] - a["hits"],
            net_errors=b["errors"] - a["errors"],
            transitions=dict(
                Counter(r["transition"] for r in changes if r["group"] == group and r["mode"] == "closed")
            ),
        )
    summary = dict(
        assessment="exploratory",
        label_version=dict(manual=9, automatic=71, sha256=file_sha256(LABELS)),
        results=overall,
        changes_vs_a=deltas,
    )
    if persist:
        write_json(dest / "summary.json", summary)
        write_jsonl(dest / "changes.jsonl", changes)
        write_csv(dest / "summary.csv", summary_rows)
    else:
        require(
            summary == read(dest / "summary.json") and changes == read_jsonl(dest / "changes.jsonl"),
            "Saved summaries differ from paired predictions",
        )
    for g, v in overall["closed"]["all"].items():
        print(
            g,
            {k: v[k] for k in ("confirmed", "outputs", "hits", "errors", "pending", "match_calls")},
            flush=True,
        )


def main():
    from group_b.pairs import Corpus

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("baseline", "smoke", "run", "verify"))
    parser.add_argument("--output", type=Path, default=DEST)
    parser.add_argument("--smoke-only", action="store_true", help="Verify the fold 3 smoke directory")
    args = parser.parse_args()
    works, _, folds, digest = inputs()
    labels = load_labels(works)
    with threadpool_limits(limits=1):
        if args.stage == "baseline" or (
            args.stage in {"smoke", "run"} and not (args.output / "baseline_audit.json").exists()
        ):
            build_baseline(folds, works, digest, labels, args.output)
        baseline_audit(args.output, digest)
        if args.stage == "baseline":
            return
        baseline = read_jsonl(args.output / "baseline.jsonl")
        require(
            len({(r["query_id"], r["mode"]) for r in baseline}) == len(baseline) == 160,
            "Incomplete baseline replay",
        )
        dest = args.output
        if args.stage == "smoke" or args.smoke_only:
            smoke_cases()
            dest = dest / "smoke"
            folds = [folds[2]]
        if args.stage != "verify":
            corpus = Corpus({})
            references = {r["recording_id"]: w for w, r in corpus.refs.items()}
            for fold in folds:
                dev_labels = {w: labels[w] for w in fold["development_work_ids"]}
                select_development(fold, works, dev_labels, corpus, references, dest)
            for fold in folds:
                apply_fold(
                    fold,
                    [r for r in baseline if r["outer_fold"] == fold["outer_fold"]],
                    labels,
                    corpus,
                    references,
                    dest,
                )
        report(dest, folds, baseline, labels, persist=args.stage != "verify")


if __name__ == "__main__":
    main()
