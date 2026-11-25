#!/usr/bin/env python3
"""Download CARE datasets and checkpoints. Run with python3 in WSL."""

import argparse
import fcntl
import hashlib
import http.client
import json
import re
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1] / "data_weights"
BUFFER_SIZE = 1024 * 1024
ATTEMPTS = 4
DRIVE_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=1rDZ9CDInpxQUvXRLv87mr-hfDfnV7Y-j&export=download&confirm=t"
)
COVERS_URLS = (
    "https://www.ee.columbia.edu/~dpwe/LabROSA/projects/coversongs/covers80",
    "http://labrosa.ee.columbia.edu/projects/coversongs/covers80",
)
VINET_URL = (
    "https://raw.githubusercontent.com/raraz15/Discogs-VINet/"
    "3029d61c6f8e6ef9991acd2f0e4743e748134d9a/logs/checkpoints/Discogs-VINet"
)
BEAT_URL = "https://cloud.cp.jku.at/index.php/s/7ik4RrBKTS273gp/download?path=%2F&files=final0.ckpt"
HF_URL = "https://huggingface.co"
HF_MODELS = {
    "echora-kiki": ("hcxlabs/echora-kiki", "b35a2d893c8c287c6f77da7c2931ac17df0b90df"),
    "qwen3-embedding-0.6b": ("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"),
    "mert-v1-95m": ("m-a-p/MERT-v1-95M", "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"),
    "qwen3-asr-1.7b": ("Qwen/Qwen3-ASR-1.7B", "7278e1e70fe206f11671096ffdd38061171dd6e5"),
    "qwen3-forcedaligner-0.6b": ("Qwen/Qwen3-ForcedAligner-0.6B", "c7cbfc2048c462b0d63a45797104fc9db3ad62b7"),
}
JAMENDO_REPO = "jamendolyrics/jamendolyrics"
JAMENDO_REVISION = "de188c963fd4539bc769b3feb83582e5a9595e36"
DATASETS = ("covers80", "discogs-vi-yt", "jamendolyrics")
WEIGHTS = ("coverhunter", "discogs-vinet-ismir2024", "beat-this-final0", "unified-music-identification", *HF_MODELS)


def request(url, offset=0):
    headers = {"User-Agent": "CARE-data-downloader/1.0", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return urlopen(Request(url, headers=headers), timeout=60)


def retry(action):
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return action()
        except (OSError, http.client.HTTPException, ValueError) as error:
            if isinstance(error, HTTPError) and error.code not in (408, 429) and error.code < 500:
                raise
            if attempt == ATTEMPTS:
                raise
            print(f"  Retry {attempt + 1}/{ATTEMPTS}: {error}", file=sys.stderr, flush=True)
            time.sleep(2 ** (attempt - 1))


def valid_file(path, size=None, checksum=None, magic=None):
    if not path.is_file():
        return False
    actual_size = path.stat().st_size
    if not actual_size or (size is not None and actual_size != size):
        return False
    with path.open("rb") as stream:
        if magic is not None and stream.read(len(magic)) != magic:
            return False
        if checksum is not None:
            algorithm, expected = checksum.split(":", 1)
            stream.seek(0)
            if algorithm == "git-sha1":
                digest = hashlib.sha1(f"blob {actual_size}\0".encode())
            else:
                digest = hashlib.new(algorithm)
            for chunk in iter(lambda: stream.read(BUFFER_SIZE), b""):
                digest.update(chunk)
            return digest.hexdigest() == expected
    return True


def transfer(url, part, size):
    offset = part.stat().st_size if part.exists() else 0
    while True:
        try:
            response = request(url, offset)
        except HTTPError as error:
            if error.code == 416 and offset:
                offset = 0
                continue
            raise
        with response:
            if response.headers.get_content_type() in ("text/html", "application/xhtml+xml"):
                raise ValueError("the server sent a web page instead of a file")

            content_length = response.headers.get("Content-Length")
            remaining = int(content_length) if content_length is not None else None
            if response.status == 206:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if not match and offset:
                    offset = 0
                    continue
                if not match:
                    raise ValueError("the server did not send a valid Content-Range")
                start, end, total = map(int, match.groups())
                if start != offset or not start <= end < total:
                    raise ValueError("the server returned an unexpected byte range")
                if remaining is not None and remaining != end - start + 1:
                    raise ValueError("Content-Length does not match Content-Range")
                remaining = end - start + 1
            elif response.status == 200:
                # A server can ignore Range. Overwrite the partial file with the full body.
                offset = 0
                total = remaining
            else:
                raise ValueError(f"unexpected HTTP status {response.status}")

            if size is not None and total is not None and size != total:
                raise ValueError(f"server reports {total} bytes, expected {size}")
            expected_size = size if size is not None else total
            end_of_response = offset + remaining if remaining is not None else expected_size
            if offset:
                print(f"  Resuming at byte {offset:,}", flush=True)
            written = offset
            last_update = time.monotonic()
            with part.open("ab" if offset else "wb") as stream:
                while end_of_response is None or written < end_of_response:
                    read_size = BUFFER_SIZE if end_of_response is None else min(BUFFER_SIZE, end_of_response - written)
                    chunk = response.read1(read_size)
                    if not chunk:
                        break
                    written += len(chunk)
                    stream.write(chunk)
                    if time.monotonic() - last_update >= 5:
                        print(f"  {written / 1024**2:,.1f} MiB saved", flush=True)
                        last_update = time.monotonic()
            if expected_size is not None and written != expected_size:
                raise OSError(f"download stopped at {written:,} of {expected_size:,} bytes")
            return


def download(urls, destination, *, size=None, checksum=None, magic=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if valid_file(destination, size, checksum, magic):
        print(f"  Exists: {destination.name}", flush=True)
        return destination

    part = destination.with_name(destination.name + ".part")

    def finish(url):
        if part.exists() and size is not None and part.stat().st_size >= size:
            if not valid_file(part, size, checksum, magic):
                part.unlink()
        if not ((size is not None or checksum is not None) and valid_file(part, size, checksum, magic)):
            transfer(url, part, size)
            if not valid_file(part, size, checksum, magic):
                part.unlink()
                raise ValueError("the downloaded file failed validation")
        part.replace(destination)

    if isinstance(urls, str):
        urls = (urls,)
    for index, url in enumerate(urls):
        print(f"  Downloading {destination.name}", flush=True)
        try:
            retry(lambda: finish(url))
        except (OSError, http.client.HTTPException, ValueError) as error:
            if index == len(urls) - 1:
                raise
            print(f"  Source unavailable ({error}), trying the alternate URL", file=sys.stderr, flush=True)
        else:
            return destination


def archive_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"unsafe archive member path: {name}")
    return path


def extract_file(source, destination, size):
    if destination.is_file() and destination.stat().st_size == size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".extracting")
    with partial.open("wb") as stream:
        shutil.copyfileobj(source, stream, BUFFER_SIZE)
    if partial.stat().st_size != size:
        raise OSError(f"incomplete archive member: {destination.name}")
    partial.replace(destination)


def extract_tar(archive, output, *, audio_only=False):
    extracted = set()
    print(f"  Extracting {archive.name}", flush=True)
    with tarfile.open(archive, "r|gz") as bundle:
        for member in bundle:
            if not member.isfile():
                continue
            path = archive_path(member.name)
            if audio_only:
                if path.suffix.lower() != ".mp3" or len(path.parts) < 2:
                    continue
                path = PurePosixPath(*path.parts[-2:])
            with bundle.extractfile(member) as source:
                extract_file(source, output / path, member.size)
            extracted.add(path)
    return extracted


def covers80(output):
    splits = []
    for name, size in (("list1.txt", 5092), ("list2.txt", 5245)):
        splits.append(download(tuple(f"{url}/{name}" for url in COVERS_URLS), output / "splits" / name, size=size))
    archive = download(tuple(f"{url}/covers80.tgz" for url in COVERS_URLS), output / "covers80.tgz",
                       size=163630117, magic=b"\x1f\x8b")
    audio = extract_tar(archive, output / "audio", audio_only=True)
    stems = {str(path)[:-len(path.suffix)] for path in audio}
    for split in splits:
        names = split.read_text(encoding="utf-8-sig").splitlines()
        if len(names) != 80 or any(name.strip() not in stems for name in names):
            raise ValueError(f"extracted Covers80 audio does not match {split.name}")


def coverhunter(output):
    archive = download(DRIVE_URL, output / "pretrain_model.tar.gz", size=68480295, magic=b"\x1f\x8b")
    if not extract_tar(archive, output):
        raise ValueError("CoverHunter archive contains no regular files")


def zenodo(record, filename):
    return f"https://zenodo.org/records/{record}/files/{filename}?download=1"


def discogs(output):
    archive = download(zenodo(13983028, "main.zip"), output / "metadata" / "main.zip",
                       size=1407051987, checksum="md5:4bd20bc4163d1e69f5b1cd151e8eb934")
    download(zenodo(13983028, "LICENSE-metadata"), output / "metadata" / "LICENSE-metadata",
             checksum="md5:136c671dba2d2f644b882e31c3e289e8")
    download(zenodo(13983028, "README.md"), output / "metadata" / "README.md",
             checksum="md5:70c18c8b4ee18abed993858bb9060766")
    prefix = "Discogs-VI-YT-20240701"
    files = {f"{prefix}.jsonl": "metadata", f"{prefix}-light.json": "metadata"}
    files.update({f"{prefix}-light.json.{split}": "splits" for split in ("train", "val", "test")})
    print("  Extracting Discogs-VI-YT metadata and official splits", flush=True)
    with zipfile.ZipFile(archive) as bundle:
        for name, directory in files.items():
            member = bundle.getinfo(f"main/{name}")
            with bundle.open(member) as source:
                extract_file(source, output / directory / name, member.file_size)
    print("  Discogs-VI-YT audio is available separately via --discogs-audio-ids.", flush=True)


def hf_inventory(kind, repo, revision):
    def load():
        with request(f"{HF_URL}/api/{kind}/{repo}/revision/{revision}?blobs=true") as response:
            inventory = json.load(response)
        if inventory["sha"] != revision or not inventory["siblings"]:
            raise ValueError(f"unexpected Hugging Face revision or inventory for {repo}")
        return inventory["siblings"]

    return retry(load)


def hf_download(base, entry, destination):
    if entry.get("lfs"):
        checksum = f"sha256:{entry['lfs']['sha256']}"
    else:
        checksum = f"git-sha1:{entry['blobId']}"
    download(f"{base}/{quote(entry['rfilename'])}", destination, size=entry["size"], checksum=checksum)


def hf_model(output, name):
    repo, revision = HF_MODELS[name]
    base = f"{HF_URL}/{repo}/resolve/{revision}"
    for entry in hf_inventory("models", repo, revision):
        filename = entry["rfilename"]
        # CARE uses MERT through Transformers. The Fairseq export duplicates the model.
        if filename in (".gitattributes", "MERT-v1-95M_fairseq.pt"):
            continue
        hf_download(base, entry, output / archive_path(filename))


def jamendo(output):
    files = hf_inventory("datasets", JAMENDO_REPO, JAMENDO_REVISION)
    base = f"{HF_URL}/datasets/{JAMENDO_REPO}/resolve/{JAMENDO_REVISION}"
    audio_count = 0
    for entry in files:
        path = archive_path(entry["rfilename"])
        if path.match("subsets/*/mp3/*.mp3"):
            destination = output / "audio" / path.name
            audio_count += 1
        elif path.parts[0] == "annotations":
            destination = output / path
        elif path.parts[0] == "lyrics":
            destination = output / "annotations" / path
        elif str(path) in ("JamendoLyrics.csv", "metadata.jsonl", "README.md", "LICENSE"):
            destination = output / path
        else:
            continue
        hf_download(base, entry, destination)
    if audio_count != 79:
        raise ValueError(f"expected 79 JamendoLyrics audio files, found {audio_count}")


def direct_weights(output, name):
    if name == "discogs-vinet-ismir2024":
        download(f"{VINET_URL}/config.yaml", output / "config.yaml", size=716,
                 checksum="git-sha1:a3001937254608aa496fdeba30a0c4bb04dbb7cd")
        download(f"{VINET_URL}/model_checkpoint.pth", output / "model_checkpoint.pth", size=62328329,
                 checksum="git-sha1:21dcb52b539ce211ce8d4a6c929fa3a4a0080d22")
    elif name == "beat-this-final0":
        download(BEAT_URL, output / "final0.ckpt", size=81058141,
                 checksum="sha1:e1506282faf66ca10e8ab50ee26bd542b7b9ff0a")
    else:
        download(zenodo(22016080, "epoch-8-step-90000.ckpt"), output / "epoch-8-step-90000.ckpt",
                 size=2385498233, checksum="sha256:31572a9bb69f058a66c3c93e93137c3114ec477c306d62735eab7088a53d822c")
        download(zenodo(22016080, "hparams.yaml"), output / "hparams.yaml",
                 checksum="sha256:91a60ae71e98c335223d0d6a4ff17300a7373b08397472ffd29ede483c82fb48")


def discogs_audio(ids_file, output):
    if not shutil.which("yt-dlp"):
        raise OSError("yt-dlp is required for Discogs audio. Install it and rerun this command.")
    ids = [line.partition("#")[0].strip() for line in ids_file.read_text(encoding="utf-8-sig").splitlines()]
    ids = list(dict.fromkeys(video_id for video_id in ids if video_id))
    if not ids or any(not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id) for video_id in ids):
        raise ValueError("provide one 11-character YouTube video ID per line")
    output.mkdir(parents=True, exist_ok=True)
    command = [
        "yt-dlp", "--continue", "--part", "--no-overwrites", "--no-playlist",
        "--retries", "5", "--fragment-retries", "5", "-f", "bestaudio",
        "--download-archive", str(output / ".downloaded.txt"),
        "-o", str(output / "%(id)s.%(ext)s"), "--batch-file", "-",
    ]
    urls = "".join(f"https://www.youtube.com/watch?v={video_id}\n" for video_id in ids)
    subprocess.run(command, input=urls, text=True, check=True)


def run_target(root, name):
    dataset_handlers = {"covers80": covers80, "discogs-vi-yt": discogs, "jamendolyrics": jamendo}
    if name in dataset_handlers:
        dataset_handlers[name](root / "datasets" / name)
    else:
        output = root / "weights" / name
        if name == "coverhunter":
            coverhunter(output)
        elif name in HF_MODELS:
            hf_model(output, name)
        else:
            direct_weights(output, name)


def main():
    groups = {"all": (*DATASETS, *WEIGHTS), "datasets": DATASETS, "weights": WEIGHTS}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", metavar="TARGET", help="all (default), datasets, weights, or a name from --list")
    parser.add_argument("--list", action="store_true", help="show downloadable targets without making network requests")
    parser.add_argument("--root", type=Path, default=ROOT, help="output root (default: repository data_weights directory)")
    parser.add_argument("--discogs-audio-ids", type=Path, metavar="FILE",
                        help="download Discogs audio using yt-dlp and a file with one YouTube ID per line")
    args = parser.parse_intermixed_args()
    if args.list:
        print("Datasets: " + ", ".join(DATASETS))
        print("Weights:  " + ", ".join(WEIGHTS))
        return 0

    targets = args.targets or ([] if args.discogs_audio_ids else ["all"])
    selected = {}
    for target in targets:
        if target not in groups and target not in groups["all"]:
            parser.error(f"unknown target {target!r}, see --list")
        selected.update(dict.fromkeys(groups.get(target, (target,))))

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    failures = []
    with (root / ".download.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(1, f"Another download is using {root}\n")
        print(f"Output: {root}", flush=True)
        for name in selected:
            print(f"\n[{name}]", flush=True)
            try:
                run_target(root, name)
            except (OSError, ValueError, KeyError, http.client.HTTPException, tarfile.TarError, zipfile.BadZipFile) as error:
                print(f"  Failed: {error}", file=sys.stderr, flush=True)
                failures.append(name)
        if args.discogs_audio_ids:
            print("\n[discogs-vi-yt audio]", flush=True)
            try:
                discogs_audio(args.discogs_audio_ids, root / "datasets" / "discogs-vi-yt" / "audio")
            except (OSError, ValueError, subprocess.CalledProcessError) as error:
                print(f"  Failed: {error}", file=sys.stderr, flush=True)
                failures.append("Discogs audio (--discogs-audio-ids)")
    if failures:
        print("\nIncomplete: " + ", ".join(failures) + ". Rerun to resume.", file=sys.stderr)
        return 1
    print("\nSelected downloads are ready.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit("\nInterrupted. Rerun the same command to resume.")
