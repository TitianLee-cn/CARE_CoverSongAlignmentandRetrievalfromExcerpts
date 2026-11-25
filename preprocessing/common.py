"""Paths, audio decoding and per-recording cache I/O for CARE preprocessing."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile


CARE = Path(__file__).resolve().parents[1]
DATA = CARE / "data_weights"
OUTPUT = CARE / "precomputed_data"
LOCAL_FFMPEG = Path("/mnt/e/AI4Titi/x-lance/DeepLearning/F5-TTS/model/.ffmpeg_runtime")
QUERY_SECONDS = 30
SEED = 42


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_atomic(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            write(stream)
            stream.flush()
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    write_atomic(path, lambda stream: stream.write((json_text(value) + "\n").encode()))


def write_jsonl(path, rows):
    def write(stream):
        for row in rows:
            stream.write((json_text(row) + "\n").encode())

    write_atomic(path, write)


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def weight_versions(*paths):
    return {str(path.relative_to(DATA)): file_sha256(path) for path in paths}


def source_version(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def executable(name):
    if prefix := os.environ.get("CARE_FFMPEG"):
        path = Path(prefix) / "bin" / name
        if path.is_file():
            return str(path)
    if path := shutil.which(name):
        return path
    path = LOCAL_FFMPEG / "bin" / name
    if path.is_file():
        return str(path)
    raise FileNotFoundError(f"{name} is required. Set CARE_FFMPEG to its install prefix.")


def ffmpeg_pcm_command(path, query=None):
    command = [
        executable("ffmpeg"), "-nostdin", "-hide_banner", "-v", "error", "-xerror",
        "-threads", "1", "-i", str(path), "-map", "0:a:0", "-ac", "1",
    ]
    if query is not None:
        start, end = query["start_sample"], query["end_sample"]
        command.extend(["-af", f"atrim=start_sample={start}:end_sample={end}"])
    return [*command, "-f", "f32le", "pipe:1"]


def inspect_audio(path):
    metadata = subprocess.check_output([
        executable("ffprobe"), "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels", "-of", "json", str(path),
    ])
    audio = json.loads(metadata)["streams"][0]
    sample_rate = int(audio["sample_rate"])
    byte_count = 0
    with subprocess.Popen(ffmpeg_pcm_command(path), stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE) as process:
        for block in iter(lambda: process.stdout.read(1 << 20), b""):
            byte_count += len(block)
        error = process.stderr.read().decode(errors="replace").strip()
        if process.wait() or error:
            raise ValueError(error or "FFmpeg could not decode the audio")
    samples, remainder = divmod(byte_count, 4)
    if remainder or not samples or sample_rate <= 0:
        raise ValueError("invalid decoded PCM")
    return {
        "sample_rate": sample_rate, "channels": int(audio["channels"]),
        "num_samples": samples, "duration_sec": samples / sample_rate,
        "sha256": file_sha256(path),
    }


def read_audio(recording, query=None):
    import numpy as np

    path = CARE / recording["audio_path"]
    pcm = subprocess.check_output(ffmpeg_pcm_command(path, query), stderr=subprocess.PIPE)
    waveform = np.frombuffer(pcm, dtype="<f4")
    sample_rate = recording["sample_rate"]
    expected = query["end_sample"] - query["start_sample"] if query else recording["num_samples"]
    if len(waveform) != expected:
        raise ValueError(f"{path}: decoded sample count changed; verify the shared manifest")
    return waveform.copy(), sample_rate


def resample(waveform, sample_rate, target_rate):
    if sample_rate == target_rate:
        return waveform
    import torch
    import torchaudio

    # Reuse the filter for tracks with the same input and output rates.
    if (sample_rate, target_rate) not in _RESAMPLERS:
        _RESAMPLERS[sample_rate, target_rate] = torchaudio.transforms.Resample(
            sample_rate, target_rate
        )
    return _RESAMPLERS[sample_rate, target_rate](torch.from_numpy(waveform)).numpy()


_RESAMPLERS = {}


def shared_inputs(root):
    shared = root / "shared"
    report = json.loads((shared / "validation.json").read_text(encoding="utf-8"))
    for name, digest in report["manifest_sha256"].items():
        if file_sha256(shared / name) != digest:
            raise ValueError(f"{shared / name}: file does not match the shared validation report")
    recordings = read_jsonl(shared / "recordings.jsonl")
    queries = read_jsonl(shared / "queries.jsonl")
    return recordings, queries


def cover_inputs(root):
    recordings, queries = shared_inputs(root)
    by_id = {row["recording_id"]: row for row in recordings}
    for query in queries:
        for id_field, clip in (("query_recording_id", query), ("reference_recording_id", None)):
            recording = by_id[query[id_field]]
            yield {"recording": recording, "query": clip}


def cache_parser(description, *, gpu=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest-root", type=Path, help="Defaults to --output")
    parser.add_argument("--limit", type=int, help="Use the first N works after filtering")
    parser.add_argument("--work-id", action="append", default=[], help="Repeat to select works")
    parser.add_argument("--force", action="store_true", help="Recompute selected caches")
    if gpu:
        parser.add_argument("--device", default="cuda")
    return parser


def select_works(items, args):
    work_ids = list(dict.fromkeys(item["work_id"] for item in items))
    unknown = set(args.work_id) - set(work_ids)
    if unknown:
        raise SystemExit(f"Unknown work IDs: {', '.join(sorted(unknown))}")
    if args.work_id:
        work_ids = [work_id for work_id in work_ids if work_id in args.work_id]
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        work_ids = work_ids[:args.limit]
    return set(work_ids)


def use_device(name):
    import torch

    device = torch.device(name)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
    return device


def audio_cache_entry(item):
    recording, query = item["recording"], item["query"]
    return {
        "id": recording["recording_id"], "work_id": recording["work_id"],
        "recording_id": recording["recording_id"],
        "query_id": query["query_id"] if query else None,
        "role": "query" if query else recording["role"],
        "source_sha256": recording["sha256"],
        "start_sec": 0.0,
        "end_sec": QUERY_SECONDS if query else recording["duration_sec"],
        "source_start_sample": query["start_sample"] if query else 0,
        "source_end_sample": query["end_sample"] if query else recording["num_samples"],
        "source_sample_rate": recording["sample_rate"],
        "source_start_sec": query["start_sec"] if query else 0.0,
    }


def cache_metadata(path):
    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as cache:
            return json.loads(cache["metadata"].item())
    except (OSError, EOFError, ValueError, KeyError, BadZipFile):
        return None


def cache_available(path, expected):
    metadata = cache_metadata(path)
    if metadata is None:
        return False
    return all(metadata.get(key) == value for key, value in expected.items())


def run_cache(directory, items, args, compute):
    """Write complete NPZ shards atomically and index all compatible shards."""
    import numpy as np

    entries = list(items)
    selected = select_works([entry for entry, _ in entries], args)
    failures = []
    for entry, source in entries:
        if entry["work_id"] not in selected:
            continue
        path = directory / "items" / f'{entry["id"]}.npz'
        if not args.force and cache_available(path, entry):
            print(f'cached  {entry["id"]}', flush=True)
            continue
        try:
            arrays, details = compute(source)
            metadata = dict(entry, **details)
            write_atomic(path, lambda stream: np.savez(stream, metadata=json_text(metadata), **arrays))
            print(f'wrote   {entry["id"]}', flush=True)
        except Exception as error:
            failure = {"id": entry["id"], "work_id": entry["work_id"], "error": str(error)}
            if isinstance(error, subprocess.CalledProcessError) and error.stderr:
                stderr = error.stderr
                failure["error"] += f": {stderr.decode(errors='replace') if isinstance(stderr, bytes) else stderr}"
            failures.append(failure)
            print(f'failed  {entry["id"]}: {failure["error"]}', flush=True)

    index = []
    for entry, _ in entries:
        path = directory / "items" / f'{entry["id"]}.npz'
        metadata = cache_metadata(path)
        if metadata is not None and all(metadata.get(key) == value for key, value in entry.items()):
            index.append(dict(metadata, path=path.relative_to(directory).as_posix()))
    write_jsonl(directory / "index.jsonl", index)
    write_jsonl(directory / "failures.jsonl", failures)
    print(f"{len(index)}/{len(entries)} cached; {len(failures)} failed in this run.", flush=True)
    return 1 if failures else 0
