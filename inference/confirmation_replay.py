"""Replay frozen CARE decisions and score the forced-output interval comparison."""

import argparse
from itertools import chain
import json
from pathlib import Path


DATA = Path(__file__).resolve().parents[1] / "precomputed_data"
EXPERIMENTS = DATA / "experiments"
FROZEN = EXPERIMENTS / "frozen_reevaluation" / "care"
TUNED = EXPERIMENTS / "confirmation_coverage"
ZERO = EXPERIMENTS / "confirmation_coverage_zero"
FORCED = EXPERIMENTS / "confirmation_no_abstention"
LABELS = DATA / "reference_labels/covers80/reference_intervals.jsonl"
REVIEW = DATA / "review/e2_no_abstention_23_listening/manual_review.jsonl"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(json.dumps(row) + "\n" for row in values).encode("utf-8"))


def label_index(path):
    result = {r["query_id"]: r for r in rows(path)}
    if len(result) != 80:
        raise ValueError(f"Expected 80 query labels in {path}")
    return result


def position_iou(prediction, label):
    if prediction is None:
        return None
    start, end = prediction
    if not start < end:
        raise ValueError(f"Invalid predicted interval: {prediction}")
    values = []
    for ref in label["reference_intervals"]:
        left, right = ref["start_sec"], ref["end_sec"]
        overlap = max(0.0, min(end, right) - max(start, left))
        values.append(overlap / (end - start + right - left - overlap))
    return max(values)


def replay(labels, destination, *, forced):
    from pipeline import Engine, e2, passes, position_options, score_options

    folds = read_json(DATA / "shared/folds.json")["folds"]
    result = []
    for fold in folds:
        n = fold["outer_fold"]
        source = FROZEN / f"fold{n}"
        params = read_json(TUNED / f"fold{n}" / "tuned" / "parameters.json")
        params["confirmation"]["music"]["min_harmony_coverage"] = 0.0
        if forced:
            for rule in params["confirmation"].values():
                rule.update(score=0.0, margin=0.0, min_harmony_coverage=0.0)
            params["e2_selection"] = {"threshold": 0.0, "margin": 0.0}
        engine = None
        batch = []
        for mode, filename in (("closed", "predictions.jsonl"), ("open", "open_predictions.jsonl")):
            predictions = rows(source / filename)
            if sorted(r["work_id"] for r in predictions) != sorted(fold["test_work_ids"]):
                raise ValueError(f"Fold mismatch in {source / filename}")
            for original in predictions:
                query, work = original["query_id"], original["work_id"]
                ref = original["reference_recording_id"]
                obs = original["decision"]
                rule = params["confirmation"]["music" if obs["harmony"] else "audio_only"]
                confirmed = passes(obs, rule)
                if forced and not confirmed:
                    raise ValueError(f"Forced confirmation failed for {query}/{mode}")
                interval = None
                if confirmed:
                    proposals = original.get("interval_proposals", [])
                    if not proposals:
                        if engine is None:
                            engine = Engine(n, params["window"], params["overlap"], params["recall_lambda"])
                        winner = original["top10"][0]
                        candidate = winner
                        if mode == "open":
                            library = [w for w in fold["test_gallery_work_ids"] if w != work]
                            candidates = engine.candidates(work, library, params["background"], detailed=True)
                            candidate = next(c for c in candidates if c["reference_recording_id"] == ref)
                        if any(candidate[k] != winner[k] for k in ("reference_recording_id", "query_window", "reference_window")):
                            raise ValueError(f"Frozen winning windows changed for {query}/{mode}")
                        revised = params["e2_version"] == "path"
                        proposals = position_options(engine, work, candidate, revised, use_lyrics=params["e2_lyrics"])
                        score_options(
                            proposals, params["position_models"], "path_features" if revised else "legacy_features",
                        )
                    interval = e2.select(proposals, **params["e2_selection"])
                    if forced and interval is None:
                        raise ValueError(f"Forced interval failed for {query}/{mode}")
                correct = ref == labels[query]["reference_recording_id"] if mode == "closed" else None
                batch.append(dict(
                    query_id=query, work_id=work, mode=mode, reference_recording_id=ref,
                    title_correct=correct, confirmed=confirmed, reference_interval=interval,
                    iou=position_iou(interval, labels[query]) if correct else None,
                    song_score=obs["score"], previous_confirmed=original["confirmed"],
                ))
        write_rows(destination / f"fold{n}" / "predictions.jsonl", batch)
        write_json(destination / f"fold{n}" / "parameters.json", params)
        if not forced and n == 1:
            write_json(destination / "fold1_parameters.json", params)
        result.extend(batch)
    if not forced:
        write_rows(destination / "predictions.jsonl", [
            {k: r[k] for k in ("query_id", "mode", "confirmed", "reference_recording_id", "previous_confirmed")}
            for r in result
        ])
    return result


def counts(queries):
    return dict(
        queries=len(queries),
        song_outputs=sum(r["confirmed"] for r in queries),
        songs_correct=sum(r["confirmed"] and r["title_correct"] is True for r in queries),
        interval_outputs=sum(r["reference_interval"] is not None for r in queries),
        interval_hits=sum(r["iou"] is not None and r["iou"] >= 0.50 for r in queries),
        interval_iou_sum=sum(r["iou"] for r in queries if r["iou"] is not None),
    )


def write_replay_summary(destination, predictions, *, forced):
    closed = counts([r for r in predictions if r["mode"] == "closed"])
    opened = counts([r for r in predictions if r["mode"] == "open"])
    if forced:
        summary = {"closed": closed, "open": dict(opened, false_accepts=opened["song_outputs"])}
    else:
        summary = {"current": dict(
            correct_confirmations=closed["songs_correct"],
            incorrect_confirmations=closed["song_outputs"] - closed["songs_correct"],
            open_false_accepts=opened["song_outputs"],
            **closed,
        )}
    write_json(destination / "summary.json", summary)


def interval_comparison(labels, pending, care_root, audio_root):
    by_model = {}
    sources = {
        "care": chain.from_iterable(
            (dict(p, outer_fold=i) for p in rows(care_root / f"fold{i}" / "predictions.jsonl"))
            for i in range(1, 6)
        ),
        **{model: rows(audio_root / f"{model}_predictions.jsonl") for model in ("unified", "mert")},
    }
    for model, predictions in sources.items():
        scored = {}
        for p in predictions:
            if p.get("mode") == "open":
                continue
            q = p["query_id"]
            ref = p["reference_recording_id"] if model == "care" else p["reported_recording_id"]
            if q in scored or not ref or p["reference_interval"] is None or not p.get("confirmed", True):
                raise ValueError(f"Expected one forced song and interval per query: {model}/{q}")
            correct = ref == labels[q]["reference_recording_id"]
            scored[q] = dict(query_id=q, outer_fold=p["outer_fold"], song_correct=correct, pending=q in pending, iou=(
                position_iou(p["reference_interval"], labels[q]) if correct and q not in pending else None
            ))
        if scored.keys() != labels.keys():
            raise ValueError(f"Prediction and label IDs differ for {model}")
        by_model[model] = scored

    def metrics(scored):
        correct = [r for r in scored if r["song_correct"]]
        ious = [r["iou"] for r in correct if r["iou"] is not None]
        return dict(
            queries=len(scored), songs_correct=len(correct),
            pending=sum(r["pending"] for r in correct),
            interval_hits=sum(iou >= 0.50 for iou in ious),
            evaluated_intervals=len(ious), mean_iou=sum(ious) / len(ious) if ious else None,
        )

    overall = {m: metrics(list(scored.values())) for m, scored in by_model.items()}
    subsets = {}
    for models in (("care", "unified"), ("care", "mert"), tuple(by_model)):
        common = [q for q in labels if all(by_model[m][q]["song_correct"] for m in models)]
        subsets["-".join(models)] = {m: metrics([by_model[m][q] for q in common]) for m in models}
    by_fold = {
        m: {i: metrics([r for r in scored.values() if r["outer_fold"] == i]) for i in range(1, 6)}
        for m, scored in by_model.items()
    }
    return dict(
        overall=overall, by_fold=by_fold, common_subsets=subsets,
        care_hits_with_pending_as_hits=overall["care"]["interval_hits"] + overall["care"]["pending"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("restore-labels", "zero-coverage", "run", "report"))
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--review", type=Path, default=REVIEW)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--care-predictions", type=Path, default=FORCED)
    parser.add_argument("--audio-predictions", type=Path, default=EXPERIMENTS / "covers80_interval_comparison")
    args = parser.parse_args()
    if args.action == "restore-labels":
        if args.output is None or args.output.resolve() == args.labels.resolve():
            parser.error("restore-labels needs a separate --output file")
        original = {r["query_id"]: r["previous_reference"] for r in rows(args.review)}
        write_rows(args.output, [original.get(r["query_id"], r) for r in rows(args.labels)])
        return
    labels = label_index(args.labels)
    if args.action in ("zero-coverage", "run"):
        forced = args.action == "run"
        destination = args.output or (FORCED if forced else ZERO)
        prediction_files = [destination / "summary.json", destination / "fold1/predictions.jsonl"]
        if any(path.exists() for path in prediction_files):
            parser.error("choose a new --output directory to preserve existing predictions")
        write_replay_summary(destination, replay(labels, destination, forced=forced), forced=forced)
        return
    pending = {
        r["query_id"] for r in rows(args.review)
        if r["verdict"] == "reannotation_needed_both_intervals_incorrect"
    }
    report = interval_comparison(labels, pending, args.care_predictions, args.audio_predictions)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
