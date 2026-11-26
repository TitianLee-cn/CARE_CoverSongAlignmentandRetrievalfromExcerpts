"""Re-score the recorded five-fold predictions using the latest human marks."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from statistics import mean
import json


REVIEW = Path(__file__).resolve().parents[3] / "precomputed_data/review/unified_care"


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def iou(first, second):
    overlap = max(0, min(first[1], second[1]) - max(first[0], second[0]))
    return overlap / (max(first[1], second[1]) - min(first[0], second[0]))


def score(root):
    data = root / "precomputed_data"
    experiment = data / "experiments/unified_role_swap"
    predictions = {
        name: [
            row
            for fold in range(1, 6)
            for row in rows(experiment / "care" / f"fold{fold}" / f"{name}.jsonl")
        ]
        for name in ("predictions", "forced_predictions")
    }
    labels = {
        row["query_id"]: row
        for row in rows(data / "reference_labels/covers80/reference_intervals.jsonl")
    }
    intervals = {
        qid: [(r["start_sec"], r["end_sec"]) for r in row["reference_intervals"]]
        for qid, row in labels.items()
    }
    marks = {row["query_id"]: row for row in rows(REVIEW / "human_annotations.jsonl")}
    forced = {row["query_id"]: row for row in predictions["forced_predictions"]}
    if any(len(batch) != 80 or {r["query_id"] for r in batch} != labels.keys()
           for batch in predictions.values()):
        raise ValueError("Both prediction sets must cover the same 80 labeled queries")
    for qid, mark in marks.items():
        old = intervals[qid]
        start, end = mark["human_start_sec"], mark["human_end_sec"]
        if mark["human_judgement"] == "aligned":
            row = forced[qid]
            if row["reference_recording_id"] != labels[qid]["reference_recording_id"]:
                raise ValueError(f"Accepted prediction has the wrong recording: {qid}")
            intervals[qid] = old + [row["reference_interval"]]
        elif start is not None and end is not None:
            new = (start, end)
            intervals[qid] = [r for r in old if len(old) > 1 and iou(r, new) < 0.5] + [new]
    # These two reference locations were unresolved in the reviewed run.
    for qid in ("covers80_036_q", "covers80_046_q"):
        intervals[qid] = []

    def correct(row):
        return row["reference_recording_id"] == labels[row["query_id"]]["reference_recording_id"]

    def best(qid, span):
        return max((iou(span, r) for r in intervals[qid]), default=0.0)

    confirmed = [r for r in predictions["predictions"] if r["confirmed"] and correct(r)]
    emitted = [r for r in confirmed if r["reference_interval"]]
    ious = [best(r["query_id"], r["reference_interval"]) for r in emitted]
    hits = Counter()
    for qid, row in forced.items():
        if correct(row) and best(qid, row["reference_interval"]) >= 0.5:
            if marks.get(qid, {}).get("human_judgement") not in {"not_aligned", "partial", "unsure"}:
                hits[row["outer_fold"]] += 1
    return {
        "gated_correct_confirmations": len(confirmed),
        "gated_qualifying_proposals": sum(
            any(best(r["query_id"], p["interval"]) >= 0.5 for p in r["interval_proposals"])
            for r in confirmed
        ),
        "gated_outputs": len(emitted),
        "gated_joint_hits": sum(x >= 0.5 for x in ious),
        "gated_mean_iou": mean(ious) if ious else None,
        "forced_query_count": len(forced),
        "forced_correct_song": sum(correct(r) for r in forced.values()),
        "forced_joint_hits": hits.total(),
        "forced_joint_hits_by_fold": {fold: hits[fold] for fold in range(1, 6)},
        "human_annotation_count": len(marks),
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("main_checkout", type=Path)
    args = parser.parse_args()
    print(json.dumps(score(args.main_checkout), indent=2))
