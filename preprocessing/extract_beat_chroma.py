"""Cache Beat This! beat intervals and beat-synchronous chroma for Covers80."""

from importlib.metadata import version

import librosa
import numpy as np
import torch

from common import (
    DATA, audio_cache_entry, cache_parser, cover_inputs, read_audio, resample, run_cache,
    use_device, weight_versions,
)


CHROMA_RATE = 22050
HOP_LENGTH = 256
FFT_LENGTH = 4096


def four_beat_modes(chroma, valid):
    modes = np.full((len(chroma), 4), -1, dtype=np.int8)
    peaks = chroma.argmax(axis=1)
    unique_peak = (chroma == chroma.max(axis=1, keepdims=True)).sum(axis=1) == 1
    for end in range(3, len(chroma)):
        start = end - 3
        if (valid[start:end + 1] & unique_peak[start:end + 1]).all():
            modes[end] = (peaks[start:end + 1] - peaks[start]) % 12
    return modes


def extract(item, tracker):
    waveform, original_rate = read_audio(item["recording"], item["query"])
    duration = len(waveform) / original_rate
    beat_times, _ = tracker(waveform, original_rate)
    beats = np.asarray(beat_times, dtype=np.float64)
    if not np.isfinite(beats).all():
        raise ValueError("Beat This! returned non-finite timestamps")
    beats = np.unique(beats[(beats >= 0) & (beats <= duration)])
    bounds = np.column_stack((beats[:-1], beats[1:]))

    audio = resample(waveform, original_rate, CHROMA_RATE)
    spectrum = np.abs(librosa.stft(audio, n_fft=FFT_LENGTH, hop_length=HOP_LENGTH)) ** 2
    chroma_frames = librosa.feature.chroma_stft(
        S=spectrum, sr=CHROMA_RATE, n_fft=FFT_LENGTH, hop_length=HOP_LENGTH, norm=None,
    ).T
    frame_sec = np.arange(len(chroma_frames)) * HOP_LENGTH / CHROMA_RATE
    frame_indices = np.searchsorted(frame_sec, bounds)
    chroma = np.zeros((len(bounds), 12), dtype=np.float32)
    valid = np.zeros(len(bounds), dtype=bool)
    for index, (lo, hi) in enumerate(frame_indices):
        if hi > lo:
            mean = chroma_frames[lo:hi].mean(axis=0)
            norm = np.linalg.norm(mean)
            if np.isfinite(norm) and norm > 0:
                chroma[index] = mean / norm
                valid[index] = True

    modes = four_beat_modes(chroma, valid)
    unique_modes = np.unique(modes[modes[:, 0] >= 0], axis=0)
    arrays = {
        "beat_sec": beats, "beat_bounds_sec": bounds, "chroma": chroma, "valid": valid,
        "four_beat_mode": modes, "unique_modes": unique_modes,
    }
    return arrays, {
        "beat_count": len(beats), "valid_interval_count": int(valid.sum()),
        "mode_count": len(unique_modes),
    }


def main():
    parser = cache_parser(__doc__, gpu=True)
    args = parser.parse_args()
    from beat_this.inference import Audio2Beats

    torch.set_num_threads(4)
    checkpoint = DATA / "weights" / "beat-this-final0" / "final0.ckpt"
    tracker = Audio2Beats(checkpoint_path=str(checkpoint), device=use_device(args.device))
    settings = {
        "stage": "beat_chroma", "cache_version": 1, "model": "beat-this-final0",
        "model_version": {"package": version("beat-this"), "weights": weight_versions(checkpoint)},
        "chroma": {
            "method": "chroma_stft", "sample_rate": CHROMA_RATE, "hop_length": HOP_LENGTH,
            "n_fft": FFT_LENGTH, "librosa": librosa.__version__,
        },
        "device": args.device, "torch": str(torch.__version__),
    }
    items = [(dict(audio_cache_entry(item), **settings), item)
             for item in cover_inputs(args.manifest_root or args.output)]
    return run_cache(
        args.output / "beat_chroma", items, args, lambda item: extract(item, tracker),
    )


if __name__ == "__main__":
    raise SystemExit(main())
