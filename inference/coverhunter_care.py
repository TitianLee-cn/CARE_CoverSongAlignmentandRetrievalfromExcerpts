"""Run the paired, refitted full CARE comparison if the exploratory retrieval gate passes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
from pathlib import Path

from coverhunter_tuning import DEST, check, check_rankings, inputs, load_choices, read_jsonl, write_json
from verify import verify


def run_fold(task):
    group, fold, config, run = task
    win, overlap, weight = config
    number = fold["outer_fold"]
    root = DEST / "full_care" / group
    path = root / f"fold{number}_w{win}_o{overlap}"
    if weight:
        path /= f"lambda{weight:g}"
    path /= "e2_path"
    if run:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("pipeline.py")),
                "run",
                "--e2",
                "path",
                "--fold",
                str(number),
                "--window-config",
                str(DEST / f"{group}_config.json"),
                "--output-root",
                str(root),
            ],
            check=True,
        )
    params = json.loads((path / "parameters.json").read_text())
    check(
        params["fit_work_ids"] == fold["development_work_ids"]
        and (params["outer_fold"], params["window"], params["overlap"], params["recall_lambda"])
        == (number, win, overlap, weight),
        "Wrong fitted parameters",
    )
    pure = {r["query_id"]: r for r in read_jsonl(DEST / f"{group}.jsonl") if r["outer_fold"] == number}
    preds = read_jsonl(path / "predictions.jsonl")
    check(
        len(pure) == len(preds) == 16 and set(pure) == {p["query_id"] for p in preds},
        "Fold query mismatch",
    )
    for p in preds:
        candidates = sorted(p["top10"], key=lambda c: c["recall_rank"])
        restored = [
            dict(
                rank=c["recall_rank"],
                reference_recording_id=c["reference_recording_id"],
                query_window=c["query_window"],
                reference_window=c["reference_window"],
                score=c["audio"],
            )
            for c in candidates
        ]
        check_rankings(restored, pure[p["query_id"]]["top10"])
    counts = verify(path / "predictions.jsonl", False)
    return (
        group,
        number,
        {k: counts.get(k, 0) for k in ("queries", "top1_correct", "confirmed", "confirmed_correct")},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    s = json.loads((DEST / "summary.json").read_text())
    if s["groups"]["C"]["hits_at_10"] <= s["groups"]["B"]["hits_at_10"]:
        print("No Hit@10 improvement. Full CARE comparison skipped.")
        return
    _, _, folds, digest = inputs()
    tasks = []
    for group in ("B", "C"):
        choices = load_choices(group, folds, digest)
        tasks.extend((group, f, choices[f["outer_fold"]], not args.verify_only) for f in folds)
    results = {"B": {}, "C": {}}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for g, n, counts in pool.map(run_fold, tasks):
            results[g][n] = counts
    total = {
        g: {
            k: sum(r[k] for r in group.values())
            for k in ("queries", "top1_correct", "confirmed", "confirmed_correct")
        }
        for g, group in results.items()
    }
    write_json(DEST / "full_care" / "summary.json", dict(exploratory=True, totals=total, by_fold=results))
    print(total)


if __name__ == "__main__":
    main()
