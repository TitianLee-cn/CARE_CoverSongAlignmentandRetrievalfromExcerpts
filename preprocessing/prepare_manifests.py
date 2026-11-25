"""Validate the local audio and create the authoritative shared manifests."""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from common import (
    CARE, DATA, OUTPUT, QUERY_SECONDS, SEED, file_sha256, inspect_audio,
    read_jsonl, write_json, write_jsonl,
)


def covers80_inputs():
    root = DATA / "datasets" / "covers80"
    official_lists = [root / "splits" / f"list{i}.txt" for i in (1, 2)]
    lists = [
        [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for path in official_lists
    ]
    if any(len(rows) != 80 or len(set(rows)) != 80 for rows in lists):
        raise ValueError("Covers80 requires exactly 80 unique rows in each official list")
    for index, (a, b) in enumerate(zip(*lists), start=1):
        a_work, b_work = PurePosixPath(a).parent, PurePosixPath(b).parent
        if a_work != b_work or a == b:
            raise ValueError(f"Covers80 pair mismatch on row {index}: {a!r}, {b!r}")
        work_id = f"covers80_{index:03d}"
        for name, version, role in ((a, "a", "query_source"), (b, "b", "reference")):
            audio = root / "audio" / f"{name}.mp3"
            yield {
                "dataset": "covers80", "work_id": work_id, "official_index": index,
                "work_name": a_work.name, "recording_id": f"{work_id}_{version}",
                "version": version.upper(), "role": role,
                "audio_path": audio.relative_to(CARE).as_posix(),
            }


def jamendo_inputs():
    root = DATA / "datasets" / "jamendolyrics"
    metadata = read_jsonl(root / "metadata.jsonl")
    if len(metadata) != 79 or len({row["name"] for row in metadata}) != 79:
        raise ValueError("JamendoLyrics requires exactly 79 unique metadata names")
    for index, row in enumerate(metadata, start=1):
        audio = root / "audio" / f'{row["name"]}.mp3'
        work_id = f"jamendo_{index:03d}"
        yield {
            "dataset": "jamendolyrics", "work_id": work_id, "official_index": index,
            "recording_id": work_id, "metadata_name": row["name"],
            "role": "song", "audio_path": audio.relative_to(CARE).as_posix(),
        }


def validate_recording(recording):
    path = CARE / recording["audio_path"]
    audio = inspect_audio(path)
    if recording["dataset"] == "covers80" and audio["duration_sec"] < QUERY_SECONDS:
        raise ValueError("Covers80 recording is shorter than 30 seconds")
    return dict(recording, **audio)


def make_queries(recordings):
    rng = random.Random(SEED)
    by_id = {row["recording_id"]: row for row in recordings}
    result = []
    for row in recordings:
        if row["role"] != "query_source":
            continue
        work = row["work_id"]
        length = QUERY_SECONDS * row["sample_rate"]
        start = rng.randint(0, row["num_samples"] - length)
        result.append({
            "query_id": f"{work}_q", "work_id": work, "official_index": row["official_index"],
            "query_recording_id": row["recording_id"],
            "reference_recording_id": by_id[f"{work}_b"]["recording_id"],
            "sample_rate": row["sample_rate"], "start_sample": start,
            "end_sample": start + length, "start_sec": start / row["sample_rate"],
            "end_sec": (start + length) / row["sample_rate"], "duration_sec": QUERY_SECONDS,
        })
    return result


def make_folds(queries):
    rng = random.Random(SEED)
    works = [query["work_id"] for query in queries]
    shuffled = rng.sample(works, k=len(works))
    folds = []
    for outer in range(5):
        test = shuffled[outer * 16:(outer + 1) * 16]
        development = [work for work in works if work not in test]
        dev_shuffled = rng.sample(development, k=len(development))
        inner_folds = []
        for inner in range(4):
            validation = dev_shuffled[inner * 16:(inner + 1) * 16]
            inner_folds.append({
                "inner_fold": inner + 1, "validation_work_ids": validation,
                "fit_work_ids": [work for work in development if work not in validation],
                "gallery_work_ids": development,
            })
        folds.append({
            "outer_fold": outer + 1, "test_work_ids": test,
            "development_work_ids": development, "test_gallery_work_ids": works,
            "inner_folds": inner_folds,
        })
    return {"seed": SEED, "folds": folds}


def annotation_rows(queries):
    for query in queries:
        yield {
            "work_id": query["work_id"], "query_id": query["query_id"],
            "query_recording_id": query["query_recording_id"],
            "reference_recording_id": query["reference_recording_id"],
            "official_index": query["official_index"],
            "annotator": "A" if query["official_index"] <= 40 else "B",
            "annotation_status": "pending", "review_status": "pending",
            "query_has_effective_lyrics": None, "query_lyrics_reason": None,
            "query_language": None, "reference_language": None, "cover_type": None,
            "reference_interval_status": "pending", "reference_intervals": [],
            "line_pair_status": "pending", "line_pairs": [], "notes": "",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verify", action="store_true", help="Validate against an existing shared manifest")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    inputs = [*covers80_inputs(), *jamendo_inputs()]
    print(f"Decoding and checksumming {len(inputs)} recordings...", flush=True)
    recordings, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [(row, pool.submit(validate_recording, row)) for row in inputs]
        for row, future in pending:
            try:
                recordings.append(future.result())
            except Exception as error:
                failures.append({
                    "recording_id": row["recording_id"], "path": row["audio_path"], "error": str(error),
                })

    shared = args.output / "shared"
    if failures:
        for failure in failures:
            print(f'{failure["path"]}: {failure["error"]}')
        if not args.verify:
            write_jsonl(shared / "validation_failures.jsonl", failures)
        return 1

    queries = make_queries(recordings)
    folds = make_folds(queries)
    if args.verify:
        if recordings != read_jsonl(shared / "recordings.jsonl"):
            raise SystemExit("Local audio checksums or decoded metadata differ from the shared manifest")
        if queries != read_jsonl(shared / "queries.jsonl") or folds != json.loads(
            (shared / "folds.json").read_text(encoding="utf-8")
        ):
            raise SystemExit("Shared query starts or folds differ from the seed-42 protocol")
        print("Shared manifests match all local audio, query starts and folds.")
        return 0

    write_jsonl(shared / "recordings.jsonl", recordings)
    write_jsonl(shared / "queries.jsonl", queries)
    write_json(shared / "folds.json", folds)
    if not (shared / "annotation_template.jsonl").exists():
        write_jsonl(shared / "annotation_template.jsonl", annotation_rows(queries))
    write_jsonl(shared / "validation_failures.jsonl", [])
    write_json(shared / "validation.json", {
        "status": "passed", "covers80_recordings": 160, "jamendo_recordings": 79,
        "queries": len(queries), "duration_sec": QUERY_SECONDS, "rng": "python.random.Random",
        "seed": SEED, "query_start_unit": "native audio sample", "audio_path_base": "CARE",
        "manifest_sha256": {
            name: file_sha256(shared / name) for name in ("recordings.jsonl", "queries.jsonl", "folds.json")
        },
    })
    print(f"Validated 80 Covers80 pairs and 79 JamendoLyrics recordings. Wrote {shared}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
