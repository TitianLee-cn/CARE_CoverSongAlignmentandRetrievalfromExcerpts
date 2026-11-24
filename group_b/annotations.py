"""Convert the shared A/B schema into inputs for human-annotated experiments."""
import argparse
import json
from pathlib import Path

from .cache import MANIFESTS, rows
from .evaluate import CLASSES


def normalize(sources, destination):
    template = {row["work_id"]:row for row in rows(MANIFESTS/"annotation_template.jsonl")}
    identifiers = ("query_id","query_recording_id","reference_recording_id","official_index")
    normalized = {}
    for source in sources:
        for row in rows(source):
            work = row["work_id"]
            original = template[work]
            if work in normalized or any(row[k] != original[k] for k in identifiers):
                raise ValueError(f"Annotation identity mismatch or duplicate: {work}")
            if str(row.get("annotation_status","")).casefold().startswith("auto"):
                raise ValueError(f"Automatic draft cannot supply human labels: {work}")
            gate, cover = row["query_has_effective_lyrics"], row["cover_type"]
            if (gate is not None and type(gate) is not bool) or (cover is not None and cover not in CLASSES):
                raise ValueError(f"Invalid gate or type label: {work}")
            pairs = row.get("line_pairs")
            if not pairs and row.get("line_pair_status") in (None,"pending"):
                pairs = None
            normalized[work] = dict(work_id=work,valid_lyrics=gate,cover_type=cover,pairs=pairs,
                                    query_language=row["query_language"],reference_language=row["reference_language"],
                                    reference_intervals=row.get("reference_intervals"))
    with destination.open("x",encoding="utf-8") as stream:
        for row in normalized.values():
            stream.write(json.dumps(row,ensure_ascii=False)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sources",type=Path,nargs="+")
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    normalize(args.sources,args.out)
