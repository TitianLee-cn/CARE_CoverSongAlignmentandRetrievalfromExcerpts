"""Compute unweighted subsequence DTW paths for Covers80 recording pairs."""

from pathlib import Path

import numpy as np
from numba import njit

from common import (
    audio_cache_entry, cache_available, cache_parser, cover_inputs, file_sha256, read_jsonl,
    run_cache, shared_inputs,
)


@njit
def dtw(cost, penalty):
    """Each state records the final step: diagonal, horizontal or vertical."""
    m, n = cost.shape
    scores = np.full((m + 1, n + 1, 3), np.inf)
    back = np.full((m + 1, n + 1, 3), -1, dtype=np.int8)
    scores[0, :, 0] = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            c = cost[i - 1, j - 1]
            prev = scores[i - 1, j - 1]
            state = 0 if prev[0] <= prev[1] else 1
            if prev[2] < prev[state]:
                state = 2
            scores[i, j, 0] = c + scores[i - 1, j - 1, state]
            back[i, j, 0] = state
            state = 0 if scores[i, j - 1, 0] <= scores[i, j - 1, 2] else 2
            scores[i, j, 1] = c + penalty + scores[i, j - 1, state]
            back[i, j, 1] = state
            state = 0 if scores[i - 1, j, 0] <= scores[i - 1, j, 1] else 1
            scores[i, j, 2] = c + penalty + scores[i - 1, j, state]
            back[i, j, 2] = state
    return scores[m, 1:], back


@njit
def trace_path(back, endpoint, state):
    i, j = back.shape[0] - 1, endpoint + 1
    result = np.empty((i + j, 2), dtype=np.int64)
    count = 0
    while i > 0:
        result[count] = (i - 1, j - 1)
        count += 1
        previous = back[i, j, state]
        i -= state != 1
        j -= state != 2
        state = previous
    return result[:count][::-1].copy()


def local_minima(values):
    """Use the earliest endpoint on a flat local minimum."""
    for index, value in enumerate(values):
        previous = values[index - 1] if index else np.inf
        following = values[index + 1] if index + 1 < len(values) else np.inf
        if np.isfinite(value) and value < previous and value <= following:
            end = index + 1
            while end < len(values) and values[end] == value:
                end += 1
            if end == len(values) or value < values[end]:
                yield index


def span_iou(first, second):
    overlap = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    return overlap / ((first[1] - first[0]) + (second[1] - second[0]) - overlap)


def has_four_valid(valid):
    return np.convolve(valid.astype(np.int8), np.ones(4, dtype=np.int8), mode="valid").max() == 4


def align(source, penalty, keep, alternate_iou):
    query_path, reference_path = source
    with np.load(query_path, allow_pickle=False) as query, np.load(reference_path, allow_pickle=False) as ref:
        q, r = query["chroma"], ref["chroma"]
        q_times, r_times = query["beat_bounds_sec"], ref["beat_bounds_sec"]
        q_valid, r_valid = query["valid"], ref["valid"]

    candidates = []
    match_valid = q_valid[:, None] & r_valid[None, :]
    enough = all(len(mask) >= 4 and has_four_valid(mask) for mask in (q_valid, r_valid))
    if enough:
        for shift in range(12):
            cosine = np.clip(q @ np.roll(r, shift, axis=1).T, -1, 1)
            cost = np.where(match_valid, 1 - cosine, 1.0)
            last, back = dtw(cost, penalty)
            end_costs = last.min(axis=1)
            for endpoint in local_minima(end_costs):
                state = int(last[endpoint].argmin())
                path = trace_path(back, endpoint, state)
                qi, ri = path.T
                ref_span = (float(r_times[ri[0], 0]), float(r_times[ri[-1], 1]))
                key = (float(end_costs[endpoint]), state, int(ri[0]), shift, endpoint)
                candidates.append((key, ref_span, path, cosine[qi, ri], match_valid[qi, ri]))

    paths, matches, scores, masks = [], [], [], []
    spans, saved_summaries = [], []
    for key, span, path, cosine, valid in sorted(candidates, key=lambda item: item[0]):
        if any(span_iou(span, saved) >= alternate_iou for saved in spans):
            continue
        qi, ri = path.T
        spans.append(span)
        paths.append(path)
        matches.append(np.column_stack((q_times[qi], r_times[ri])))
        scores.append(cosine)
        masks.append(valid)
        saved_summaries.append({
            "reference_span_sec": list(span), "cost": key[0], "reference_shift_semitones": key[3],
        })
        if len(paths) == keep:
            break

    arrays = {
        "beat_indices": np.concatenate(paths) if paths else np.empty((0, 2), dtype=np.int64),
        "match_bounds_sec": np.concatenate(matches) if matches else np.empty((0, 4)),
        "cosine": np.concatenate(scores) if scores else np.empty(0, dtype=np.float32),
        "match_valid": np.concatenate(masks) if masks else np.empty(0, dtype=bool),
        "path_offsets": np.cumsum([0, *(len(path) for path in paths)], dtype=np.int64),
    }
    status = "ok" if paths else "insufficient_valid_beats" if not enough else "no_path"
    return arrays, {"status": status, "paths": saved_summaries}


def make_items(manifest_root, beat_root, pairs, settings):
    _, queries = shared_inputs(manifest_root)
    official = {query["query_recording_id"]: query for query in queries}
    beat_entries = {item["recording"]["recording_id"]: audio_cache_entry(item)
                    for item in cover_inputs(manifest_root)}
    beat_versions = {}
    if pairs is None:
        pairs = queries

    seen = set()
    for pair in pairs:
        a, b = pair["query_recording_id"], pair["reference_recording_id"]
        if a not in official or b not in beat_entries or beat_entries[b]["role"] != "reference":
            raise ValueError(f"Invalid query/reference IDs: {a}, {b}")
        pair_id = f"{a}__{b}"
        if pair_id in seen:
            raise ValueError(f"Duplicate pair: {pair_id}")
        seen.add(pair_id)
        paths = [beat_root / "items" / f"{recording}.npz" for recording in (a, b)]
        for recording, path in zip((a, b), paths):
            if recording not in beat_versions:
                beat_versions[recording] = (
                    file_sha256(path) if cache_available(path, beat_entries[recording]) else None
                )
        entry = {
            "id": pair_id, "work_id": official[a]["work_id"], "query_id": official[a]["query_id"],
            "query_recording_id": a, "reference_recording_id": b,
            "reference_work_id": beat_entries[b]["work_id"],
            "query_source_start_sec": official[a]["start_sec"],
            "query_start_sec": 0.0, "query_end_sec": official[a]["duration_sec"],
            "reference_start_sec": 0.0, "reference_end_sec": beat_entries[b]["end_sec"],
            "query_beat_sha256": beat_versions[a], "reference_beat_sha256": beat_versions[b],
            **settings,
        }
        yield entry, (paths, bool(beat_versions[a] and beat_versions[b]))


def main():
    parser = cache_parser(__doc__)
    parser.add_argument("--pairs", type=Path, help="JSONL with query_recording_id and reference_recording_id")
    parser.add_argument("--beat-root", type=Path, help="Defaults to <output>/beat_chroma")
    parser.add_argument("--paths", type=int, default=3)
    parser.add_argument("--alternate-iou", type=float, default=0.5)
    args = parser.parse_args()
    if args.paths < 1 or not 0 < args.alternate_iou <= 1:
        parser.error("--paths must be positive and --alternate-iou must be in (0, 1]")
    settings = {
        "stage": "unweighted_subsequence_dtw", "cache_version": 1, "step_penalty": 0.15,
        "path_limit": args.paths, "alternate_iou": args.alternate_iou,
    }
    pairs = read_jsonl(args.pairs) if args.pairs else None
    items = make_items(args.manifest_root or args.output, args.beat_root or args.output / "beat_chroma",
                       pairs, settings)

    def compute(source):
        paths, available = source
        if not available:
            raise ValueError("A beat/chroma cache is missing or does not match the shared manifest")
        return align(paths, settings["step_penalty"], args.paths, args.alternate_iou)

    name = "candidate_alignment" if pairs is not None else "cover_pair_alignment"
    return run_cache(args.output / name, items, args, compute)


if __name__ == "__main__":
    raise SystemExit(main())
