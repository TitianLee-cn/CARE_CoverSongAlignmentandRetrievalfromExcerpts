"""Check coverage, dimensions and timestamps for CARE's preprocessing caches."""

import numpy as np

from common import cache_parser, read_jsonl, shared_inputs
from extract_cover_embeddings import CONFIGS, window_bounds


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_index(root):
    rows = read_jsonl(root / "index.jsonl")
    items = {row["id"]: (row, root / row["path"]) for row in rows}
    require(len(items) == len(rows), f"{root}: duplicate cache IDs")
    failures = read_jsonl(root / "failures.jsonl")
    require(not failures, f"{root}: review failures.jsonl")
    return items


def check_embeddings(root, queries, dimension):
    items = load_index(root)
    samples = {}
    for query in queries:
        first_windows = {}
        for key in ("query_recording_id", "reference_recording_id"):
            recording = query[key]
            metadata, path = items[recording]
            require(all(metadata["diagnostics"][str(length)]["valid_feature_frames"] > 0
                        for length in (10, 15, 20)), f"{recording}: missing feature-frame diagnostics")
            with np.load(path, allow_pickle=False) as cache:
                vectors, bounds = cache["embeddings"], cache["bounds_samples"]
                times, offsets = cache["bounds_sec"], cache["config_offsets"]
                indices = cache["config_window_rows"]
                require(vectors.shape == (len(bounds), dimension), f"{recording}: embedding shape")
                require(np.isfinite(vectors).all() and
                        np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5),
                        f"{recording}: invalid embedding values")
                require(len(np.unique(bounds, axis=0)) == len(bounds), f"{recording}: duplicate windows")
                require(np.isfinite(times).all() and times[0, 0] == 0 and
                        np.all(times[:, 1] > times[:, 0]) and
                        np.isclose(times[:, 1].max(), metadata["end_sec"], atol=1e-4, rtol=0),
                        f"{recording}: invalid window timestamps")
                require(offsets.shape == (len(CONFIGS) + 1,) and offsets[0] == 0 and
                        offsets[-1] == len(indices) and np.all(np.diff(offsets) > 0) and
                        np.all((indices >= 0) & (indices < len(bounds))), f"{recording}: invalid index")
                rate = metadata["sample_rate"]
                for i, config in enumerate(CONFIGS):
                    length = config["window_seconds"] * rate
                    hop = length * (100 - config["overlap_percent"]) // 100
                    config_rows = indices[offsets[i]:offsets[i + 1]]
                    expected = window_bounds(int(bounds[:, 1].max()), length, hop)
                    require(np.array_equal(bounds[config_rows], expected),
                            f"{recording}: bounds do not match {config}")
                    if config["overlap_percent"] == 50:
                        first_windows[config["window_seconds"]] = vectors[config_rows[0]]
                        if key == "query_recording_id" and length == 15 * rate:
                            require(np.array_equal(bounds[config_rows, 0], [0, 7.5 * rate, 15 * rate]),
                                    f"{recording}: unexpected default query windows")
            if key == "query_recording_id":
                samples[recording] = first_windows

    if len(samples) >= 2:
        first, second = list(samples.values())[:2]
        for seconds in first:
            cosine = float(first[seconds] @ second[seconds])
            require(cosine < 1 - 1e-6, f"{root.name}: identical {seconds}s vectors for different works")
            print(f"{root.name} {seconds}s: two query windows cosine={cosine:.5f}")


def check_beats(root, queries):
    items = load_index(root)
    for query in queries:
        for key in ("query_recording_id", "reference_recording_id"):
            recording = query[key]
            metadata, path = items[recording]
            with np.load(path, allow_pickle=False) as cache:
                beats, bounds = cache["beat_sec"], cache["beat_bounds_sec"]
                chroma, valid, modes = cache["chroma"], cache["valid"], cache["four_beat_mode"]
                require(np.isfinite(beats).all() and np.all(np.diff(beats) > 0) and
                        np.all((beats >= 0) & (beats <= metadata["end_sec"])),
                        f"{recording}: invalid beat timestamps")
                require(np.array_equal(bounds, np.column_stack((beats[:-1], beats[1:]))),
                        f"{recording}: invalid beat intervals")
                require(chroma.shape == (len(bounds), 12) and valid.shape == (len(bounds),) and
                        modes.shape == (len(bounds), 4) and np.isfinite(chroma).all() and
                        np.allclose(np.linalg.norm(chroma[valid], axis=1), 1, atol=1e-5),
                        f"{recording}: invalid chroma or mode shape")


def check_frames(root, songs):
    items = load_index(root)
    for song in songs:
        recording = song["recording_id"]
        metadata, path = items[recording]
        with np.load(path, allow_pickle=False) as cache:
            frames, times = cache["frames"], cache["frame_sec"]
            step = metadata["frame_stride_samples"] / metadata["sample_rate"]
            require(frames.shape == (len(times), 768) and np.isfinite(frames).all(),
                    f"{recording}: invalid MERT frames")
            require(len(times) > 0 and times[0] >= 0 and times[-1] <= metadata["end_sec"] and
                    np.allclose(np.diff(times), step, rtol=0, atol=1e-9),
                    f"{recording}: invalid MERT frame times")


def check_alignment(root, queries):
    items = load_index(root)
    for query in queries:
        pair = f'{query["query_recording_id"]}__{query["reference_recording_id"]}'
        metadata, path = items[pair]
        with np.load(path, allow_pickle=False) as cache:
            indices, offsets, times = cache["beat_indices"], cache["path_offsets"], cache["match_bounds_sec"]
            require(len(times) == len(indices) and len(offsets) == 1 + len(metadata["paths"]) and
                    offsets[0] == 0 and offsets[-1] == len(indices) and np.all(np.diff(offsets) > 0),
                    f"{pair}: invalid alignment path index")
            require(np.isfinite(times).all() and np.all(times[:, 0] >= 0) and
                    np.all(times[:, 1] <= metadata["query_end_sec"]) and
                    np.all(times[:, 2] >= 0) and np.all(times[:, 3] <= metadata["reference_end_sec"]),
                    f"{pair}: invalid alignment times")
            for start, end in zip(offsets, offsets[1:]):
                steps = np.diff(indices[start:end], axis=0)
                require(np.all((steps >= 0) & (steps <= 1)) and np.all(steps.sum(axis=1) >= 1),
                        f"{pair}: illegal DTW step")
                for direction in ([0, 1], [1, 0]):
                    repeated = (steps == direction).all(axis=1)
                    require(not (repeated[1:] & repeated[:-1]).any(), f"{pair}: repeated gap steps")


def main():
    parser = cache_parser(__doc__)
    parser.add_argument("--include-alignments", action="store_true")
    args = parser.parse_args()
    if args.force or args.work_id:
        parser.error("Use --limit to check the first N works, or omit it for all works")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    recordings, queries = shared_inputs(args.manifest_root or args.output)
    queries = queries[:args.limit]
    songs = [item for item in recordings if item["dataset"] == "jamendolyrics"][:args.limit]
    for model, dimension in (("coverhunter", 128), ("mert", 768), ("unified", 1024)):
        check_embeddings(args.output / "cover_embeddings" / model, queries, dimension)
    check_beats(args.output / "beat_chroma", queries)
    check_frames(args.output / "jamendo_mert_frames", songs)
    if args.include_alignments:
        check_alignment(args.output / "cover_pair_alignment", queries)
    print(f"Verified {len(queries)} Covers80 pairs and {len(songs)} JamendoLyrics frame caches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
