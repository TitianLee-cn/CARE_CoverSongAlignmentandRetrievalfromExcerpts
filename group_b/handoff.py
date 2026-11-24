"""Index available feature caches and create a blank B-group annotation sheet."""
import argparse
import json
from pathlib import Path

from .cache import MANIFESTS, NAMES, OUTPUT, cache_path, load_stage, rows, write
from .lyrics import shingles


def index(output):
    entries,missing=[],[]
    for dataset,stages in (("query",("kiki","vocals","text")),("reference",("kiki","text")),("jamendo",("kiki","asr"))):
        for row in rows(MANIFESTS/(NAMES[dataset]+".jsonl")):
            for stage in stages:
                path=cache_path(dataset,stage,row,output)
                rec=dict(dataset=dataset,stage=stage,recording_id=row["recording_id"],work_id=row.get("work_id"))
                if not path.exists():
                    missing.append(rec)
                    continue
                data=load_stage(dataset,stage,row,output)
                entries.append(dict(**rec,file=path.relative_to(output).as_posix(),identity=data["identity"],
                                    versions=data["model_versions"],params=data["params"],code_hash=data["code_hash"]))
    write(output/"cache_index.json",dict(entries=entries,missing=missing,complete=not missing))


def annotation_sheet(path):
    template = [row for row in rows(MANIFESTS/"annotation_template.jsonl") if row["annotator"] == "B"]
    if len(template) != 40 or {row["official_index"] for row in template} != set(range(41,81)):
        raise ValueError("Expected 40 B-side rows from A's official template")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf8") as f:
        for row in template:
            f.write(json.dumps(row,ensure_ascii=False)+"\n")


def export_shingles(path):
    lines = []
    for role in ("query","reference"):
        for row in rows(MANIFESTS/(NAMES[role]+".jsonl")):
            for line in load_stage(role,"text",row)["lines"]:
                if line["clean"]:
                    lines.append(dict(recording_id=row["recording_id"],line_id=line["line_id"],
                                      shingles=sorted(shingles(line["clean"]))))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line,ensure_ascii=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    actions = {"index":index,"annotations":annotation_sheet,"shingles":export_shingles}
    p.add_argument("action",choices=actions)
    p.add_argument("--out",type=Path,help="Directory for index, file for annotations or shingles")
    a=p.parse_args()
    if a.out is None and a.action!="index":
        p.error(f"{a.action} requires an explicit --out JSONL path")
    actions[a.action](a.out or OUTPUT)
