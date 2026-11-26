"""Tune Unified recall on development folds and compare both frozen encoders."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / "inference"))
from audio_baselines import (
    CONFIGURATIONS, OUTPUT, TIMING, check_members, file_sha256, inputs, load_model,
    metrics, search_queries, threadpool_limits, write_json, write_jsonl,
)
from coverhunter_tuning import choose
from frozen_reevaluation import audit_audio


def main():
    works, expected, folds, digest = inputs()
    dest = OUTPUT / "experiments" / "unified_role_swap"
    lock = dict(
        model="unified",
        folds_sha256=digest,
        rule="Maximize Hit@10, then exact MRR@10, then choose smaller window and overlap.",
        folds={},
    )
    models = ("unified", "coverhunter", "mert")
    with threadpool_limits(limits=1):
        banks = {model: load_model(model, expected) for model in models}
        for fold in folds:
            outer = fold["outer_fold"]
            sweep = []
            for window, overlap in CONFIGURATIONS:
                config = (window, overlap)
                rows = [
                    row
                    for inner in fold["inner_folds"]
                    for row in search_queries(
                        banks["unified"][config], works,
                        inner["validation_work_ids"], inner["gallery_work_ids"],
                        outer_fold=outer, inner_fold=inner["inner_fold"],
                        model="unified", config=config,
                    )
                ]
                check_members(
                    [row["work_id"] for row in rows], fold["development_work_ids"],
                    f"Unified fold {outer}, config {config}",
                )
                sweep.append({
                    **metrics(rows),
                    "window_seconds": window,
                    "overlap_percent": overlap,
                    "lambda": 0,
                    "reciprocal_rank_units": sum(
                        2520 // row["correct_rank"] for row in rows if row["correct_rank"]
                    ),
                    "validation": [
                        [row["query_id"], row["inner_fold"], row["correct_rank"], row["retrieval_ns"]]
                        for row in rows
                    ],
                })
            # Retain the original metric priorities and avoid using machine timing to break exact ties.
            selected = choose([dict(row, retrieval_ns=0) for row in sweep])
            settings = {k: selected[k] for k in ("window_seconds", "overlap_percent", "lambda")}
            best = next(row for row in sweep if all(row[k] == v for k, v in settings.items()))
            lock["folds"][str(outer)] = {
                **{k: v for k, v in best.items() if k != "validation"},
                "configurations": sweep,
            }
            print(f"Fold {outer}: {settings}", flush=True)

        write_json(dest / "unified_recall_config.json", lock)
        choices = {
            int(fold): (row["window_seconds"], row["overlap_percent"], row["lambda"])
            for fold, row in lock["folds"].items()
        }
        summary = {}
        for model, bank in banks.items():
            rows = []
            per_fold = {}
            for fold in folds:
                outer = fold["outer_fold"]
                config = choices[outer][:2]
                found = search_queries(
                    bank[config], works, fold["test_work_ids"], fold["test_gallery_work_ids"],
                    outer_fold=outer, inner_fold=None, model=model, config=config,
                )
                rows.extend(found)
                per_fold[outer] = metrics(found)
            audit_audio(rows, works, folds, choices)
            write_jsonl(dest / f"{model}_top10.jsonl", rows)
            summary[model] = dict(
                total=metrics(rows),
                by_fold=per_fold,
                cache_index_sha256=file_sha256(OUTPUT / "cover_embeddings" / model / "index.jsonl"),
            )
            print(model, summary[model]["total"], flush=True)
    write_json(dest / "e1_summary.json", dict(
        configuration=choices, folds_sha256=digest, timing=TIMING, models=summary,
    ))


if __name__ == "__main__":
    main()
