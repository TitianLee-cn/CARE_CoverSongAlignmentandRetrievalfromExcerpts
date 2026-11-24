"""Adapt the official A/B handoff to the B runner without creating splits."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from .cache import ROOT


def require(condition, message):
    if not condition:
        raise ValueError(message)


def records(data):
    return [json.loads(line) for line in data.decode("utf-8-sig").splitlines() if line.strip()]


def dump_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n"
                            for row in rows), encoding="utf-8")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def adapt(archive, root, output):
    names = ("recordings.jsonl", "queries.jsonl", "folds.json", "annotation_template.jsonl",
             "validation.json", "validation_failures.jsonl")
    with ZipFile(archive) as source:
        require(len(source.namelist()) == len(set(source.namelist())), "Duplicate archive members")
        raw = {name:source.read("shared/"+name) for name in names}
    validation = json.loads(raw["validation.json"])
    require(validation["status"] == "passed" and not records(raw["validation_failures.jsonl"]),
            "Source validation did not pass")
    for name in names[:3]:
        require(hashlib.sha256(raw[name]).hexdigest() == validation["manifest_sha256"][name],
                f"Manifest SHA-256 mismatch: {name}")

    datasets = root/"datasets"
    inputs = {}
    source_records = records(raw["recordings.jsonl"])
    roles = {"query_source":"covers80", "reference":"covers80", "song":"jamendolyrics"}
    require(Counter(row["role"] for row in source_records) ==
            {"query_source":80, "reference":80, "song":79}, "Unexpected recording counts")
    for row in source_records:
        rid = row["recording_id"]
        path = PurePosixPath(row["audio_path"])
        require(path.parts[:4] == ("data_weights","datasets",roles[row["role"]],"audio")
                and ".." not in path.parts and "\\" not in row["audio_path"] and rid not in inputs,
                f"Invalid recording entry: {rid}")
        relative = PurePosixPath(*path.parts[2:])
        require(digest(datasets/relative) == row["sha256"], f"Audio SHA-256 mismatch: {rid}")
        require(abs(row["duration_sec"] - row["num_samples"]/row["sample_rate"]) < 1e-7,
                f"Audio duration mismatch: {rid}")
        inputs[rid] = {**row, "path":relative.as_posix()}

    qs, refs = [], []
    for query in records(raw["queries.jsonl"]):
        left = inputs[query["query_recording_id"]]
        right = inputs[query["reference_recording_id"]]
        sr = query["sample_rate"]
        start, end = query["start_sample"], query["end_sample"]
        require(left["work_id"] == right["work_id"] == query["work_id"] and
                left["official_index"] == right["official_index"] == query["official_index"] and
                left["role"] == "query_source" and right["role"] == "reference", "Invalid pair identity")
        require(sr == left["sample_rate"] and 0 <= start < end <= left["num_samples"] and
                end-start == 30*sr and query["start_sec"] == start/sr and query["end_sec"] == end/sr,
                f"Invalid sample bounds: {query['query_id']}")
        qs.append({**left, "query_id":query["query_id"], "start_sample":start, "end_sample":end,
                   "start_sec":query["start_sec"], "end_sec":query["end_sec"]})
        refs.append(right)
    works = {r["work_id"] for r in qs}
    require(len(qs) == len(works) == len({r["query_id"] for r in qs}) == 80 and
            {r["official_index"] for r in qs} == set(range(1,81)), "Unexpected query coverage")

    folds = json.loads(raw["folds.json"])["folds"]
    require([f["outer_fold"] for f in folds] == list(range(1,6)), "Unexpected outer fold IDs")
    require(Counter(w for f in folds for w in f["test_work_ids"]) == dict.fromkeys(works,1),
            "Outer test folds do not partition works")
    for fold in folds:
        test, dev = set(fold["test_work_ids"]), set(fold["development_work_ids"])
        inner = fold["inner_folds"]
        require(len(test) == 16 and len(dev) == 64 and not test & dev and test | dev == works and
                set(fold["test_gallery_work_ids"]) == works, "Invalid outer fold")
        require([f["inner_fold"] for f in inner] == list(range(1,5)) and
                Counter(w for f in inner for w in f["validation_work_ids"]) == dict.fromkeys(dev,1),
                "Inner validation folds do not partition development works")
        for subfold in inner:
            held = set(subfold["validation_work_ids"])
            require(len(held) == 16 and set(subfold["fit_work_ids"]) == dev-held and
                    set(subfold["gallery_work_ids"]) == dev, "Invalid inner fold")
        fold["inner"] = inner

    annotations = records(raw["annotation_template.jsonl"])
    require(Counter(r["work_id"] for r in annotations) == dict.fromkeys(works,1) and
            Counter(r["annotator"] for r in annotations) == {"A":40,"B":40}, "Invalid annotation template")
    for ann in annotations:
        q = next(row for row in qs if row["work_id"] == ann["work_id"])
        require(ann["query_id"] == q["query_id"] and ann["official_index"] == q["official_index"] and
                ann["query_recording_id"] == q["recording_id"] and
                ann["reference_recording_id"] == next(row["recording_id"] for row in refs if row["work_id"] == ann["work_id"]),
                f"Annotation ID mismatch: {ann['work_id']}")

    metadata = datasets/"jamendolyrics/metadata.jsonl"
    meta = {PurePosixPath(r["file_name"]).name:r for r in records(metadata.read_bytes())}
    songs = [row for row in inputs.values() if row["role"] == "song"]
    require(set(meta) == {PurePosixPath(r["path"]).name for r in songs}, "Jamendo recording set differs from official metadata")
    gold = []
    for song in songs:
        track = meta[PurePosixPath(song["path"]).name]
        require(track["name"] == song["metadata_name"], f"Jamendo metadata mismatch: {song['recording_id']}")
        for i, line in enumerate(track["lines"], 1):
            require(0 <= line["start"] < line["end"] <= song["duration_sec"] and line["text"].strip(),
                    f"Invalid Jamendo reference line: {song['recording_id']} {i}")
            gold.append(dict(recording_id=song["recording_id"], line_id=i, **line, valid=True))

    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("covers80_queries",qs), ("covers80_references",refs), ("covers80_folds",folds),
                       ("jamendo_inputs",songs), ("jamendo_scoring_only",gold)):
        dump_rows(output/(name+".jsonl"), rows)
    (output/"annotation_template.jsonl").write_bytes(raw["annotation_template.jsonl"])
    provenance = dict(archive_name=archive.name, archive_sha256=digest(archive),
                      source_sha256={name:hashlib.sha256(data).hexdigest() for name,data in raw.items()},
                      jamendo_gold_source=str(metadata), jamendo_gold_source_sha256=digest(metadata),
                      verified_audio=len(inputs), queries=len(qs), outer_folds=len(folds), jamendo_gold_lines=len(gold))
    (output/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(provenance))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("archive",type=Path)
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--out",type=Path,required=True)
    args = p.parse_args()
    adapt(args.archive,args.root,args.out)
