"""Extract a single, timestamped MERT frame grid for each JamendoLyrics song."""

import numpy as np
import torch

from common import (
    audio_cache_entry, cache_parser, read_audio, resample, run_cache, shared_inputs, use_device,
)
from encoders import Mert


def extract(recording, model, chunk_seconds, context_seconds):
    waveform, native_rate = read_audio(recording)
    waveform = resample(waveform, native_rate, model.sample_rate)
    stride, width = model.frame_geometry()
    frame_count = (len(waveform) - width) // stride + 1
    if frame_count < 1:
        raise ValueError("Audio does not contain a complete MERT frame")
    frames_per_chunk = max(1, round(chunk_seconds * model.sample_rate / stride))
    context_frames = round(context_seconds * model.sample_rate / stride)
    features = np.empty((frame_count, model.dimension), dtype=np.float32)
    for start in range(0, frame_count, frames_per_chunk):
        end = min(frame_count, start + frames_per_chunk)
        input_start = max(0, start - context_frames)
        input_end = min(frame_count, end + context_frames)
        audio = waveform[input_start * stride:(input_end - 1) * stride + width]
        values = model.frames(audio)
        if values.shape != (input_end - input_start, model.dimension):
            raise ValueError(f"Unexpected MERT frame shape {values.shape}")
        features[start:end] = values[start - input_start:end - input_start]

    if not np.isfinite(features).all():
        raise ValueError("MERT produced non-finite frame values")
    center_samples = np.arange(frame_count, dtype=np.float64) * stride + (width - 1) / 2
    frame_seconds = center_samples / model.sample_rate
    if frame_seconds[-1] > recording["duration_sec"]:
        raise ValueError("MERT frame timestamps exceed the recording duration")
    arrays = {"frames": features, "frame_sec": frame_seconds}
    return arrays, {
        "frame_count": frame_count, "frame_dimension": model.dimension,
        "frame_stride_samples": stride, "frame_receptive_field_samples": width,
    }


def main():
    parser = cache_parser(__doc__, gpu=True)
    parser.add_argument("--chunk-seconds", type=float, default=15)
    parser.add_argument("--context-seconds", type=float, default=2.5)
    args = parser.parse_args()
    if not (np.isfinite(args.chunk_seconds) and args.chunk_seconds > 0 and
            np.isfinite(args.context_seconds) and args.context_seconds >= 0):
        parser.error("Provide finite chunk and context durations (chunk > 0, context >= 0)")
    torch.set_num_threads(4)
    model = Mert(use_device(args.device))
    recordings, _ = shared_inputs(args.manifest_root or args.output)
    settings = {
        "stage": "jamendo_mert_frames", "cache_version": 1, "model": "mert",
        "model_version": model.version, "sample_rate": model.sample_rate,
        "chunk_seconds": args.chunk_seconds, "context_seconds": args.context_seconds,
        "device": args.device, "torch": str(torch.__version__),
    }
    items = [(dict(audio_cache_entry({"recording": recording, "query": None}), **settings), recording)
             for recording in recordings if recording["dataset"] == "jamendolyrics"]
    return run_cache(
        args.output / "jamendo_mert_frames", items, args,
        lambda recording: extract(recording, model, args.chunk_seconds, args.context_seconds),
    )


if __name__ == "__main__":
    raise SystemExit(main())
