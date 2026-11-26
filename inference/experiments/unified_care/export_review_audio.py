"""Combine 80 forced-output A/B clip pairs with the latest human annotations."""

from argparse import ArgumentParser
from contextlib import nullcontext
import csv
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from rescore_human_review import REVIEW, rows


def export(predictions, output, manifest, *, review_clips=None, main_checkout=None):
    marks = {r["query_id"]: r for r in rows(REVIEW / "human_annotations.jsonl")}
    partial = output.with_suffix(".partial.zip")
    if main_checkout:
        sys.path.insert(0, str(main_checkout.resolve() / "b_lyrics"))
        from group_b.cache import MANIFESTS
        from group_b.review_pack import crop

        references = {r["recording_id"]: r for r in rows(MANIFESTS / "covers80_references.jsonl")}
    clip_source = ZipFile(review_clips) if review_clips else nullcontext()
    with ZipFile(predictions) as source, clip_source as review, TemporaryDirectory() as scratch:
        data = list(csv.DictReader(io.StringIO(source.read("predictions.csv").decode("utf-8-sig"))))
        ids = {r["query_id"] for r in data}
        if len(data) != 80 or len(ids) != 80 or not marks.keys() <= ids:
            raise ValueError("Expected 80 unique query IDs that include all human annotations")
        table = io.StringIO(newline="")
        with ZipFile(partial, "w", ZIP_DEFLATED, compresslevel=1) as result:
            for row in data:
                qid = row["query_id"]
                mark = marks.get(qid, {})
                start, end = mark.get("human_start_sec"), mark.get("human_end_sec")
                judgement = mark.get("human_judgement")
                clip = ""
                if start is not None and end is not None:
                    clip = f"{qid}/b_human.wav"
                    if review:
                        result.writestr(clip, review.read(f"audio/{qid}_human.wav"))
                    else:
                        recording = references[qid.removesuffix("_q") + "_b"]
                        rate = recording["sample_rate"]
                        target = Path(scratch) / f"{qid}.wav"
                        crop(recording, round(start * rate), round(end * rate), target)
                        result.write(target, clip)
                elif judgement == "aligned":
                    clip = row["reference_audio"]
                for name in (row["query_audio"], row["reference_audio"]):
                    result.writestr(name, source.read(name))
                row.update(
                    human_judgement=judgement or ("note_only" if mark else "not_reviewed"),
                    human_start_sec=start,
                    human_end_sec=end,
                    human_note=mark.get("human_note", ""),
                    human_reference_audio=clip,
                )
            writer = csv.DictWriter(table, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
            payload = table.getvalue().encode("utf-8-sig")
            result.writestr("audio_manifest.csv", payload)
            result.write(REVIEW / "human_annotations.jsonl", "human_annotations.jsonl")
        with ZipFile(partial) as result:
            if result.testzip() is not None or len(result.namelist()) != len(set(result.namelist())):
                raise ValueError("Invalid audio archive")
    partial.replace(output)
    manifest.write_bytes(payload)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("predictions_zip", type=Path, help="The 80-query archive from export_role_swap_clips.py")
    parser.add_argument("output_zip", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--review-clips", type=Path, help="The review archive containing the three hand-cut B clips")
    source.add_argument("--main-checkout", type=Path, help="CARE main checkout with the matching audio dataset")
    parser.add_argument("--manifest", type=Path, default=REVIEW / "audio_manifest.csv")
    args = parser.parse_args()
    export(args.predictions_zip, args.output_zip, args.manifest,
           review_clips=args.review_clips, main_checkout=args.main_checkout)
