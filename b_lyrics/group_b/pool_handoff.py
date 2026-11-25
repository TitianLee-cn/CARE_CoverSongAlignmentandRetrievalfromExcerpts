"""Read A's multipart Jamendo MERT archive and pool official-run predicted lines."""
import argparse
import hashlib
import io
import json
from contextlib import ExitStack
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .cache import MANIFESTS, OUTPUT, rows, write
from .jamendo import pool_recording


def pool(directory, stages):
    prefix = "jamendo_mert_frames/"
    packs = sorted(directory.glob("CARE_jamendo_mert_frames_for_B_part*.zip"))
    if len(packs) != 4:
        raise ValueError("Expected four MERT archive parts")
    recordings = {r["recording_id"]:r for r in rows(MANIFESTS/"jamendo_inputs.jsonl")}
    with ExitStack() as stack:
        entries = {}
        index = None
        for pack in packs:
            archive = stack.enter_context(ZipFile(pack))
            for file in archive.namelist():
                if file in entries:
                    raise ValueError(f"Duplicate MERT entry: {file}")
                entries[file] = archive
                if file == prefix+"index.jsonl":
                    index = [json.loads(line) for line in archive.read(file).splitlines()]
        failures = prefix+"failures.jsonl"
        if index is None or entries[failures].read(failures).strip():
            raise ValueError("Missing MERT index or failed alignments")
        if len(index)!=79 or set(recordings)!={i["recording_id"] for i in index}:
            raise ValueError("MERT manifest does not match the 79 official recording IDs")
        results = dict(stages=list(stages),recordings=[])
        for item in index:
            rid = item["recording_id"]
            row = recordings[rid]
            if (item["source_sha256"],item["source_sample_rate"],item["source_end_sample"]) != (
                    row["sha256"],row["sample_rate"],row["num_samples"]) or item["source_start_sample"]!=0:
                raise ValueError(f"MERT source mismatch: {rid}")
            member = prefix+f"items/{rid}.npz"
            if item["path"] != f"items/{rid}.npz" or item["start_sec"] != 0 or abs(item["end_sec"]-row["duration_sec"])>1e-8:
                raise ValueError(f"MERT coordinate mismatch: {rid}")
            arc = entries[member]
            binary = arc.read(member)
            source = dict(archive=Path(arc.filename).name,member=member,sha256=hashlib.sha256(binary).hexdigest())
            with np.load(io.BytesIO(binary),allow_pickle=False) as npz:
                meta = json.loads(npz["metadata"].item())
                if meta != {k:v for k,v in item.items() if k!="path"}:
                    raise ValueError(f"MERT NPZ metadata does not match index: {rid}")
                times,vectors = npz["frame_sec"],npz["frames"]
            if vectors.shape != (item["frame_count"],768) or item["frame_dimension"] != 768 or not 0<=times[0]<times[-1]<=row["duration_sec"]:
                raise ValueError(f"MERT array dimensions or bounds differ: {rid}")
            pooled = {}
            for stage in stages:
                count,empty = pool_recording(row,stage,OUTPUT,times,vectors,source)
                pooled[stage] = dict(lines=count,without_frames=empty)
            results["recordings"].append(dict(recording_id=rid,source=source,counts=pooled))
        results["model_versions"] = [json.loads(v) for v in sorted({json.dumps(i["model_version"],sort_keys=True) for i in index})]
        write(OUTPUT/"scores"/("jamendo-mert-"+"-".join(stages)+".json"),results)
        print(json.dumps({s:sum(r["counts"][s]["lines"] for r in results["recordings"]) for s in stages}))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("directory",type=Path)
    p.add_argument("--stage",choices=["kiki","asr"],action="append")
    a=p.parse_args()
    pool(a.directory,a.stage or ("kiki","asr"))
