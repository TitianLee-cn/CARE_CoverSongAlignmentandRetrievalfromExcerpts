"""Export the five-fold forced-output CARE predictions as sample-accurate audio."""

from concurrent.futures import ThreadPoolExecutor
import csv
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import wave
from zipfile import ZIP_DEFLATED, ZipFile

root = Path.cwd()
sys.path.insert(0, str(root / "b_lyrics"))
from group_b.cache import MANIFESTS, rows
from group_b.review_pack import crop


def pack(output):
    experiments = root / "precomputed_data/experiments/unified_role_swap/care"
    queries = {row["query_id"]: row for row in rows(MANIFESTS / "covers80_queries.jsonl")}
    references = {row["recording_id"]: row for row in rows(MANIFESTS / "covers80_references.jsonl")}
    predictions = [
        row for n in range(1, 6) for row in rows(experiments / f"fold{n}" / "forced_predictions.jsonl")
    ]
    assert len(predictions) == len({row["query_id"] for row in predictions}) == len(queries) == 80
    tasks, manifest = [], []
    for row in predictions:
        query_id = row["query_id"]
        q = queries[query_id]
        r = references[row["reference_recording_id"]]
        rate = r["sample_rate"]
        start, end = (round(value * rate) for value in row["reference_interval"])
        qpath, rpath = f"{query_id}/a_query.wav", f"{query_id}/b_predicted.wav"
        tasks.extend(((q, q["start_sample"], q["end_sample"], qpath), (r, start, end, rpath)))
        manifest.append(dict(
            query_id=query_id, fold=row["outer_fold"], query_title=q["work_name"],
            query_audio=qpath, query_start_sec=q["start_sec"], query_end_sec=q["end_sec"],
            selected_recording=r["recording_id"], selected_title=r["work_name"], reference_audio=rpath,
            reference_start_sec=start / rate, reference_end_sec=end / rate,
            mode="forced_output",
        ))

    table = io.StringIO(newline="")
    writer = csv.DictWriter(table, fieldnames=list(manifest[0]))
    writer.writeheader()
    writer.writerows(manifest)
    partial = output.with_suffix(".partial.zip")
    with TemporaryDirectory() as directory:
        work = Path(directory)

        def make(task):
            recording, left, right, name = task
            target = work / name
            target.parent.mkdir(parents=True, exist_ok=True)
            crop(recording, left, right, target)
            with wave.open(str(target), "rb") as audio:
                assert audio.getnframes() == right - left and audio.getframerate() == recording["sample_rate"]
            return target, name

        with ThreadPoolExecutor(max_workers=4) as workers:
            with ZipFile(partial, "w", ZIP_DEFLATED, compresslevel=1) as archive:
                archive.writestr("predictions.csv", table.getvalue().encode("utf-8-sig"))
                for path, name in workers.map(make, tasks):
                    archive.write(path, name)
        with ZipFile(partial) as archive:
            assert len(archive.namelist()) == 161 and archive.testzip() is None
        partial.replace(output)
    print("Verified 80 query clips and 80 predicted reference clips.", flush=True)


if __name__ == "__main__":
    pack(Path(sys.argv[1]))
