"""Score saved predictions, or pool A's MERT frames within predicted intervals."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .cache import MANIFESTS, OUTPUT, load_stage, rows, write
from .lyrics import ordered_lines


def evaluate(records, gold, stage, threshold, output):
    counts=dict(reference_lines=0,matched_lines=0,both_within_05=0)
    starts,ends,details=[],[],[]
    for row in records:
        ref=gold[row["recording_id"]]
        if not ref or any(not l["valid"] for l in ref):
            raise ValueError(f"Missing or invalid gold: {row['recording_id']}")
        pred=load_stage("jamendo",stage,row,output)["lines"]
        pairs=ordered_lines(ref,pred,threshold)
        counts["reference_lines"]+=len(ref)
        counts["matched_lines"]+=len(pairs)
        matches=[]
        for i,j in pairs:
            a=abs(ref[i]["start"]-pred[j]["start_sec"])
            b=abs(ref[i]["end"]-pred[j]["end_sec"])
            starts.append(a)
            ends.append(b)
            counts["both_within_05"]+=int(max(a,b)<=.5)
            matches.append(dict(ref_id=ref[i]["line_id"],prediction_id=pred[j]["line_id"],start_error=a,end_error=b))
        details.append(dict(recording_id=row["recording_id"],references=len(ref),predictions=len(pred),matches=matches))
    n=counts["reference_lines"]
    return dict(**counts,recordings=len(records),text_threshold=threshold,line_recall=counts["matched_lines"]/n,
                start_mae_sec=np.mean(starts).item() if starts else None,end_mae_sec=np.mean(ends).item() if ends else None,
                both_within_05_all_references=counts["both_within_05"]/n,details=details)


def pool_frames(index, records, stage, output):
    entries={r["recording_id"]:r for r in rows(index)}
    for row in records:
        rid=row["recording_id"]
        entry=entries[rid]
        if entry.get("sha256")!=row["sha256"]:
            raise ValueError(f"MERT audio checksum mismatch: {rid}")
        with np.load(index.parent/entry["file"],allow_pickle=False) as arr:
            t,x=arr["times"],arr["vectors"]
        pool_recording(row,stage,output,t,x)


def pool_recording(row, stage, output, times, vectors, source=None):
    rid=row["recording_id"]
    if times.ndim!=1 or vectors.ndim!=2 or not len(times) or len(times)!=len(vectors) or not np.isfinite(times).all() or not np.isfinite(vectors).all() or (np.diff(times)<=0).any():
        raise ValueError(f"Invalid MERT frames: {rid}")
    lines=load_stage("jamendo",stage,row,output)["lines"]
    features=[]
    for line in lines:
        start,end=np.searchsorted(times,[line["start_sec"],line["end_sec"]])
        features.append(dict(line_id=line["line_id"],start_sec=line["start_sec"],end_sec=line["end_sec"],
                             frames=int(end-start),vector=vectors[start:end].mean(0).tolist() if end>start else None))
    write(output/"jamendo"/("mert-"+stage)/(rid+".json"),
          dict(recording_id=rid,source=source,lines=features))
    return len(features),sum(f["frames"]==0 for f in features)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["score","pool"])
    p.add_argument("--mert-index",type=Path)
    p.add_argument("--limit",type=int,choices=range(1,80),metavar="1..79")
    p.add_argument("--text-threshold",type=float,default=.60)
    p.add_argument("--out",type=Path,default=OUTPUT)
    args=p.parse_args()
    records=rows(MANIFESTS/"jamendo_inputs.jsonl")[:args.limit]
    if args.mode=="pool":
        if not args.mert_index:
            p.error("--mert-index required for pooling")
        for stage in ("kiki","asr"):
            pool_frames(args.mert_index,records,stage,args.out)
    else:
        if not 0 < args.text_threshold <= 1:
            p.error("text threshold must be in (0,1]")
        references=defaultdict(list)
        for line in rows(MANIFESTS/"jamendo_scoring_only.jsonl"):
            references[line["recording_id"]].append(line)
        write(args.out/"scores"/f"jamendo-{len(records)}.json",
              {stage:evaluate(records,references,stage,args.text_threshold,args.out) for stage in ("kiki","asr")})
