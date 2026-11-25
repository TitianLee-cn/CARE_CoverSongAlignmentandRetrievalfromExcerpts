"""Combine A/B manual annotation rows and check their shared IDs and time bounds."""

import argparse
from pathlib import Path

from common import OUTPUT, read_jsonl, shared_inputs, write_jsonl


IDENTITY = ("work_id", "query_id", "query_recording_id", "reference_recording_id", "official_index")


def check_interval(interval, duration):
    start, end = interval["start_sec"], interval["end_sec"]
    if (not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or
            isinstance(start, bool) or isinstance(end, bool) or not 0 <= start < end <= duration):
        raise ValueError(f"Invalid interval [{start}, {end}) for {duration}s audio")


def merge(output, annotation_paths):
    shared = output / "shared"
    recordings, queries = shared_inputs(output)
    duration = {row["recording_id"]: row["duration_sec"] for row in recordings}
    queries = {row["work_id"]: row for row in queries}
    template = read_jsonl(shared / "annotation_template.jsonl")
    filled = {}
    for path in annotation_paths:
        for row in read_jsonl(path):
            work_id = row["work_id"]
            query = queries[work_id]
            if any(row[key] != query[key] for key in IDENTITY):
                raise ValueError(f"{path}: annotation IDs changed for {work_id}")
            if work_id in filled:
                raise ValueError(f"{path}: duplicate annotations for {work_id}; resolve before merging")
            reference_seconds = duration[row["reference_recording_id"]]
            for interval in row.get("reference_intervals", []):
                check_interval(interval, reference_seconds)
            for pair in row.get("line_pairs", []):
                for field, seconds in (("query_lines", query["duration_sec"]),
                                       ("reference_lines", reference_seconds)):
                    if not pair[field]:
                        raise ValueError(f"{work_id}: empty {field}")
                    for line in pair[field]:
                        check_interval(line, seconds)
                        if not isinstance(line["text"], str) or not line["text"].strip():
                            raise ValueError(f"{work_id}: missing lyric text in {field}")
            filled[work_id] = row
    merged = [dict(row, **filled.get(row["work_id"], {})) for row in template]
    write_jsonl(shared / "annotations.jsonl", merged)
    print(f"Merged {len(filled)}/{len(merged)} supplied annotation rows into {shared / 'annotations.jsonl'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path, nargs="+", help="One JSONL file per annotator, with disjoint work IDs")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    merge(args.output, args.annotations)


if __name__ == "__main__":
    main()
