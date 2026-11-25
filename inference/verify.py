"""Check end-to-end output and optionally score already-frozen interval predictions."""

import argparse
from collections import Counter
import json
from pathlib import Path

from pipeline import DATA, bounded, read_jsonl, shared_inputs, span_iou, write_json, write_jsonl
from e2_validation import diagnostic, summarize


def verify(path, score):
    recordings, queries = shared_inputs(DATA)
    recordings = {row["recording_id"]: row for row in recordings}
    queries = {row["query_id"]: row for row in queries}
    folds = json.loads((DATA / "shared/folds.json").read_text())["folds"]
    labels = {}
    if score:
        labels = {
            row["work_id"]: row["reference_intervals"]
            for row in read_jsonl(DATA / "reference_labels/covers80/reference_intervals.jsonl")
        }
    results = read_jsonl(path)
    seen = set()
    counts = Counter()
    overlaps = []
    for row in results:
        query = queries[row["query_id"]]
        fold = next(f for f in folds if f["outer_fold"] == row["outer_fold"])
        if (row["work_id"], row["query_recording_id"], row["query_source_start_sec"]) != (
            query["work_id"],
            query["query_recording_id"],
            query["start_sec"],
        ):
            raise ValueError("Query identity or time origin mismatch")
        if query["work_id"] not in fold["test_work_ids"] or row["query_id"] in seen:
            raise ValueError("Duplicate query or query outside the fold")
        seen.add(row["query_id"])
        candidates = row["top10"]
        ids = [c["reference_recording_id"] for c in candidates]
        if len(ids) != 10 or len(set(ids)) != 10 or ids[0] != row["reference_recording_id"]:
            raise ValueError("Invalid top ten or inconsistent winner")
        if sorted(c["recall_rank"] for c in candidates) != list(range(1, 11)):
            raise ValueError("Invalid recall ranks")
        if [c["rank"] for c in candidates] != list(range(1, 11)):
            raise ValueError("Invalid final ranks")
        for c in candidates:
            ref = recordings[c["reference_recording_id"]]
            duration = ref["duration_sec"]
            if ref["role"] != "reference" or ref["work_id"] != c["reference_work_id"]:
                raise ValueError("Candidate reference identity mismatch")
            bounded(c["query_window"], 30)
            bounded(c["reference_window"], duration)
            for match in c.get("lyrics", {}).get("matches", []):
                bounded(match["query_sec"], 30)
                bounded(match["reference_sec"], duration)
            for p in c["paths"]:
                bounded(p["reference_span_sec"], duration)
                if (
                    not len(p["beat_indices"])
                    == len(p["cosine"])
                    == len(p["match_valid"])
                    == len(p["match_bounds_sec"])
                ):
                    raise ValueError("Incomplete DTW path")
                for q0, q1, r0, r1 in p["match_bounds_sec"]:
                    bounded([q0, q1], 30)
                    bounded([r0, r1], duration)
        other = candidates[0]["reference_work_id"]
        if other != row["prediction"]["reference_work_id"]:
            raise ValueError("B received a different winner")
        correct = ids[0] == query["reference_recording_id"]
        counts["queries"] += 1
        counts["top10_correct"] += query["reference_recording_id"] in ids
        counts["top1_correct"] += correct
        counts["confirmed"] += row["confirmed"]
        counts["confirmed_correct"] += row["confirmed"] and correct
        counts["lyrics_enabled"] += row["gate"]["enabled"]
        if not row["gate"]["enabled"]:
            counts["music_only_confirmed"] += row["confirmed"]
            if row["prediction"]["type_status"] != "skipped" or any(
                c.get("lyric") is not None for c in candidates
            ):
                raise ValueError("Disabled lyric evidence entered the result")
        interval = row["reference_interval"]
        if interval is not None:
            if not row["confirmed"]:
                raise ValueError("Located an unconfirmed recording")
            bounded(interval, recordings[ids[0]]["duration_sec"])
            counts["located"] += 1
            if score and correct and labels.get(query["work_id"]):
                overlaps.append(
                    max(
                        span_iou(interval, [i["start_sec"], i["end_sec"]])
                        for i in labels[query["work_id"]]
                    )
                )
        for pair in row["prediction"]["display"]:
            bounded(pair["query_sec"], 30)
            bounded(pair["reference_sec"], recordings[ids[0]]["duration_sec"])
        duration = recordings[ids[0]]["duration_sec"]
        for option in row["interval_proposals"]:
            bounded(option["interval"], duration)
            if row.get("e2_version") == "path" and option["source"] == "harmony":
                path = candidates[0]["paths"][option["path_index"]]
                beats = len({i for i, _ in path["beat_indices"]})
                quality = 1 - path["cost"] / beats
                blocks = path["blocks"]
                longest = max(b[-1][0] - b[0][0] for b in blocks)
                extensions = [blocks[0][0][0], 30 - blocks[-1][-1][0]]
                ref_extensions = [
                    blocks[0][0][1] - option["interval"][0],
                    option["interval"][1] - blocks[-1][-1][1],
                ]
                if (
                    abs(option["match_quality"] - quality) > 1e-8
                    or abs(option["longest_reliable_sec"] - longest) > 1e-8
                    or option["features"][:5]
                    != [quality, longest / 30, *extensions, option["path_index"]]
                    or extensions != option["query_extrapolation_sec"]
                    or ref_extensions != option["reference_extrapolation_sec"]
                ):
                    raise ValueError("Proposal evidence differs from its DTW path")
    report = dict(counts)
    if score:
        report["scored_correct_and_located"] = len(overlaps)
        report["mean_iou_vs_supplied_intervals"] = sum(overlaps) / len(overlaps) if overlaps else None
    return report


def compare(old_path, new_path):
    if old_path.resolve() == new_path.resolve():
        raise ValueError("Select separate baseline and revised prediction files")
    old_rows = read_jsonl(old_path)
    new_rows = read_jsonl(new_path)
    old = {r["query_id"]: r for r in old_rows}
    if len(old) != len(old_rows) or set(old) != {r["query_id"] for r in new_rows}:
        raise ValueError("Baseline and revised query sets differ")
    labels = {
        r["work_id"]: r["reference_intervals"]
        for r in read_jsonl(DATA / "reference_labels/covers80/reference_intervals.jsonl")
    }
    details = []
    for r in new_rows:
        a = old[r["query_id"]]
        fields = (
            "outer_fold",
            "window",
            "overlap",
            "query_recording_id",
            "work_id",
            "reference_recording_id",
            "confirmed",
            "gate",
            "threshold",
            "decision",
        )
        rank_fields = (
            "reference_recording_id",
            "recall_rank",
            "rank",
            "audio",
            "harmony",
            "lyric",
            "score",
        )
        if any(a[k] != r[k] for k in fields) or any(
            any(x[k] != y[k] for k in rank_fields) for x, y in zip(a["top10"], r["top10"], strict=True)
        ):
            raise ValueError(f"Upstream inference changed for {r['query_id']}")
        eligible = r["confirmed"] and r["work_id"] == r["top10"][0]["reference_work_id"]
        if eligible:
            if not labels.get(r["work_id"]):
                raise ValueError(f"Missing scoring reference for {r['work_id']}")
            entry = diagnostic(
                r["query_id"],
                a["interval_proposals"],
                r["interval_proposals"],
                a["reference_interval"],
                r["reference_interval"],
                labels[r["work_id"]],
                r["gate"]["enabled"],
                r["interval_rule"],
            )
            details.append(entry)
    metrics = dict(paired_queries=len(new_rows), groups=summarize(details))
    write_jsonl(new_path.parent / "outer_comparison.jsonl", details)
    write_json(new_path.parent / "outer_metrics.json", metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--score", action="store_true")
    parser.add_argument(
        "--compare",
        type=Path,
        metavar="BASELINE",
        help="Compare frozen predictions to the baseline and score both on automatic intervals",
    )
    args = parser.parse_args()
    report = verify(args.predictions, args.score)
    if args.compare:
        report["comparison"] = compare(args.compare, args.predictions)
    print(json.dumps(report, indent=2))
