"""Cache all nine Covers80 window configurations for one audio encoder."""

import numpy as np
import torch

from common import (
    audio_cache_entry, cache_parser, cover_inputs, read_audio, resample, run_cache, use_device,
)
from encoders import CoverHunter, Mert, Unified


CONFIGS = [
    {"window_seconds": length, "overlap_percent": overlap}
    for length in (10, 15, 20)
    for overlap in (25, 50, 75)
]


def window_bounds(num_samples, length, step):
    if num_samples < length:
        raise ValueError(f"{num_samples} samples are shorter than a {length}-sample window")
    starts = list(range(0, num_samples - length + 1, step))
    end_aligned = num_samples - length
    if starts[-1] != end_aligned:
        starts.append(end_aligned)
    return [(start, start + length) for start in starts]


def extract(item, encoder, batch_size=None):
    if batch_size is None:
        batch_size = encoder.default_batch_size
    waveform, native_rate = read_audio(item["recording"], item["query"])
    rate = encoder.sample_rate
    waveform = resample(waveform, native_rate, rate)
    num_samples = len(waveform)
    bounds_by_config = [
        window_bounds(num_samples, config["window_seconds"] * rate,
                      config["window_seconds"] * rate * (100 - config["overlap_percent"]) // 100)
        for config in CONFIGS
    ]
    unique_bounds = sorted(set().union(*bounds_by_config))
    row_by_bounds = {bounds: row for row, bounds in enumerate(unique_bounds)}
    vectors = np.empty((len(unique_bounds), encoder.dimension), dtype=np.float32)
    diagnostics = {}
    # Group only equal-length windows; keep the original ordering in the cache.
    for length in sorted({end - start for start, end in unique_bounds}):
        bounds = [(start, end) for start, end in unique_bounds if end - start == length]
        for offset in range(0, len(bounds), batch_size):
            batch = bounds[offset:offset + batch_size]
            audio = np.stack([waveform[start:end] for start, end in batch])
            encoded, frame_count = encoder.encode_batch(audio)
            norms = np.linalg.norm(encoded, axis=1, keepdims=True)
            if (encoded.shape != (len(batch), encoder.dimension) or
                    not np.isfinite(norms).all() or np.any(norms <= 0)):
                raise ValueError(f"Invalid embeddings for windows {batch}")
            normalized = encoded / norms
            vectors[[row_by_bounds[bound] for bound in batch]] = normalized
            if offset == 0:
                first = normalized[0]
                diagnostics[str(length // rate)] = {
                    "input_samples": length, "valid_feature_frames": frame_count,
                    "waveform_padding_samples": 0, "dimension": len(first),
                    "finite": True, "norm": float(np.linalg.norm(first)),
                    "first_eight": first[:8].tolist(),
                }

    rows = [row_by_bounds[bounds] for config in bounds_by_config for bounds in config]
    offsets = np.cumsum([0, *(len(config) for config in bounds_by_config)], dtype=np.int64)
    bounds = np.array(unique_bounds, dtype=np.int64)
    duration = audio_cache_entry(item)["end_sec"]
    # The resampler can round the output length up by a fraction of a sample.
    seconds = np.minimum(bounds / rate, duration)
    arrays = {
        "embeddings": vectors, "bounds_samples": bounds, "bounds_sec": seconds,
        "config_window_rows": np.array(rows, dtype=np.int64), "config_offsets": offsets,
    }
    return arrays, {
        "inference_batch_size": batch_size,
        "window_count": len(bounds),
        "config_window_counts": list(map(len, bounds_by_config)),
        "embedding_dimension": encoder.dimension,
        "diagnostics": diagnostics,
    }


def main():
    parser = cache_parser(__doc__, gpu=True)
    models = {"coverhunter": CoverHunter, "mert": Mert, "unified": Unified}
    parser.add_argument("--model", choices=models, required=True)
    parser.add_argument("--batch-size", type=int, help="Equal-length windows per forward pass (MERT: 4, others: 8)")
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    torch.set_num_threads(4)
    encoder = models[args.model](use_device(args.device))
    settings = {
        "stage": "cover_embeddings", "cache_version": 1, "model": args.model,
        "model_version": encoder.version, "sample_rate": encoder.sample_rate,
        "configs": CONFIGS, "device": args.device, "torch": str(torch.__version__),
    }
    items = [(dict(audio_cache_entry(item), **settings), item)
             for item in cover_inputs(args.manifest_root or args.output)]
    return run_cache(
        args.output / "cover_embeddings" / args.model, items, args,
        lambda item: extract(item, encoder, args.batch_size),
    )


if __name__ == "__main__":
    raise SystemExit(main())
