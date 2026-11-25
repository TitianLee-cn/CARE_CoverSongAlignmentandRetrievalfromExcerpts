"""Validate A's unweighted DTW handoff and compare oracle-pair displays."""
import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .cache import MANIFESTS, OUTPUT, read, rows, write
from .pairs import Corpus, predict

PREFIX = "cover_pair_alignment/"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reliable_blocks(bounds, reliable):
    """Use only uninterrupted reliable matches, with seconds on the query clip axis."""
    selected = np.flatnonzero(reliable)
    blocks = []
    for span in np.split(selected, np.flatnonzero(np.diff(selected) != 1)+1):
        if not len(span):
            continue
        anchors = defaultdict(list)
        for q0,q1,r0,r1 in bounds[span]:
            anchors[float(q0)].append(r0)
            anchors[float(q1)].append(r1)
        block = [[q,float(np.median(anchors[q]))] for q in sorted(anchors)]
        require(np.all(np.diff(np.asarray(block)[:,1]) >= -1e-9), "Nonmonotonic reliable block")
        blocks.append(block)
    return blocks


def inspect(archive, cosine_min):
    queries = {q["work_id"]:q for q in rows(MANIFESTS/"covers80_queries.jsonl")}
    references = {r["work_id"]:r for r in rows(MANIFESTS/"covers80_references.jsonl")}
    requests, diagnostics = {}, []
    with ZipFile(archive) as source:
        entries = source.namelist()
        require(len(set(entries)) == len(entries), "Duplicate archive member")
        require(not source.read(PREFIX+"failures.jsonl").strip(), "A's failure list is not empty")
        index = [json.loads(line) for line in source.read(PREFIX+"index.jsonl").splitlines() if line.strip()]
        require(Counter(r["work_id"] for r in index) == dict.fromkeys(queries,1) and len(queries)==80,
                "Alignment work IDs do not match the shared 80-query manifest")
        expected_files = {PREFIX+"index.jsonl", PREFIX+"failures.jsonl"}
        for record in index:
            work = record["work_id"]
            query, ref = queries[work], references[work]
            rid = query["recording_id"]+"__"+ref["recording_id"]
            require(record["status"]=="ok" and record["stage"]=="unweighted_subsequence_dtw" and
                    (record["reference_work_id"], record["query_id"], record["query_recording_id"],
                     record["reference_recording_id"], record["id"], record["path"]) ==
                    (work,query["query_id"],query["recording_id"],ref["recording_id"],rid,f"items/{rid}.npz"),
                    f"Alignment IDs or path differ: {work}")
            require((record["query_start_sec"], record["query_end_sec"], record["reference_start_sec"]) == (0,30,0)
                    and abs(record["query_source_start_sec"]-query["start_sec"])<1e-9
                    and abs(record["reference_end_sec"]-ref["duration_sec"])<1e-9,
                    f"Alignment time origin differs: {work}")
            filename = PREFIX+record["path"]
            expected_files.add(filename)
            with np.load(io.BytesIO(source.read(filename)), allow_pickle=False) as item:
                require(json.loads(item["metadata"].item()) == {k:v for k,v in record.items() if k!="path"},
                        f"NPZ metadata differs: {work}")
                beat, bounds = item["beat_indices"], item["match_bounds_sec"]
                cosine, valid, offsets = item["cosine"], item["match_valid"], item["path_offsets"]
            paths = record["paths"]
            n = len(cosine)
            require(beat.shape==(n,2) and bounds.shape==(n,4) and valid.shape==cosine.shape==(n,)
                    and valid.dtype.kind=="b" and beat.dtype.kind in "iu" and offsets.dtype.kind in "iu"
                    and offsets.shape==(len(paths)+1,) and 1<=len(paths)<=record["path_limit"]
                    and offsets[0]==0 and offsets[-1]==n and np.all(np.diff(offsets)>0),
                    f"Invalid DTW array layout: {work}")
            require(np.isfinite(bounds).all() and np.isfinite(cosine).all() and
                    np.all(np.abs(cosine)<=1.000001) and np.all(beat>=0) and
                    np.all(bounds[:,0]>=0) and np.all(bounds[:,1]<=30+1e-8) and
                    np.all(bounds[:,2]>=0) and np.all(bounds[:,3]<=ref["duration_sec"]+1e-8) and
                    np.all(bounds[:,0]<bounds[:,1]) and np.all(bounds[:,2]<bounds[:,3]),
                    f"Invalid beat timestamps or cosine scores: {work}")
            blocks, path_stats = [], []
            for path,start,end in zip(paths,offsets[:-1],offsets[1:],strict=True):
                span = slice(start,end)
                piece = bounds[span]
                steps = np.diff(beat[span],axis=0)
                require(np.all((0<=steps)&(steps<=1)) and np.all(steps.sum(axis=1)>0) and
                        np.all(np.diff(piece,axis=0)>=-1e-9) and
                        np.allclose(path["reference_span_sec"], [piece[0,2],piece[-1,3]], rtol=0, atol=1e-8),
                        f"Invalid path geometry: {work}")
                reliable = valid[span] & (cosine[span]>=cosine_min)
                runs = reliable_blocks(piece,reliable)
                blocks.extend(runs)
                path_stats.append(dict(**path, matches=int(end-start), valid=int(valid[span].sum()),
                                       reliable=int(reliable.sum()), reliable_blocks=len(runs),
                                       longest_reliable_sec=max((b[-1][0]-b[0][0] for b in runs),default=0.)))
            diagnostics.append(dict(work_id=work, query_id=query["query_id"], recording_pair=rid,
                                    query_source_start_sec=query["start_sec"], paths=path_stats))
            requests[work] = dict(work_id=work, reference_work_id=work, confirmed=True,
                                  query_id=query["query_id"], reference_recording_id=ref["recording_id"],
                                  blocks=blocks)
        require(set(entries)==expected_files,"Archive contains unexpected files")
    return requests, dict(archive=archive.name,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                          pairs=len(requests),failed=0,paths=sum(len(x["paths"]) for x in diagnostics),
                          cosine_min=cosine_min, pairs_with_reliable_blocks=sum(bool(r["blocks"]) for r in requests.values()),
                          matches=sum(p["matches"] for x in diagnostics for p in x["paths"]),
                          cache_versions=sorted(set(r["cache_version"] for r in index)),
                          step_penalties=sorted(set(r["step_penalty"] for r in index)),
                          alternate_ious=sorted(set(r["alternate_iou"] for r in index)),
                          items=diagnostics)


def compare(requests, backgrounds, dest, audit):
    corpus = Corpus(backgrounds["config"])
    folds = rows(MANIFESTS/"covers80_folds.jsonl")
    modes = ("semantic","music","fusion")
    predictions = {mode:[] for mode in modes}
    for fold, fitted in zip(folds,backgrounds["folds"],strict=True):
        require(fitted["outer_fold"]==fold["outer_fold"] and
                fitted["fit_work_ids"]==fold["development_work_ids"], "Fold calibration differs from shared manifest")
        for work in fold["test_work_ids"]:
            req = requests[work]
            for mode in modes:
                result = predict(corpus,req,fitted["background"],mode)
                predictions[mode].append(dict(outer_fold=fold["outer_fold"],query_id=req["query_id"],
                    reference_recording_id=req["reference_recording_id"],**result))
    summaries = {}
    for mode, results in predictions.items():
        typed = Counter(r.get("cover_type",r["type_status"]) for r in results)
        shown = sum(bool(r["display"]) for r in results)
        eligible = sum(r["gate"]["enabled"] for r in results)
        summaries[mode] = dict(queries=len(results),gate_enabled=eligible,
                               displayed_queries=shown,display_pairs=sum(len(r["display"]) for r in results),
                               displayed_fraction_all=shown/len(results),
                               displayed_fraction_gate_enabled=shown/eligible if eligible else None,
                               both_languages_reliable=sum(bool(r["query_language"] and r["reference_language"])
                                                           for r in results),
                               type_counts=dict(typed))
        write(dest/f"{mode}.json",dict(mode=mode,context="oracle_correct_pair",config=corpus.cfg,
              alignment_archive_sha256=audit["archive_sha256"],cosine_min=audit["cosine_min"],
              predictions=results))
    def displayed_ids(result):
        return [(item["query_lines"],item["reference_lines"]) for item in result["display"]]

    changed = [music["work_id"] for music,fusion in zip(predictions["music"],predictions["fusion"],strict=True)
               if displayed_ids(music)!=displayed_ids(fusion)]
    write(dest/"summary.json", dict(context="oracle_correct_pair",query_count=len(requests),
                                    scored_against_human_labels=False, modes=summaries,
                                    alignment_archive_sha256=audit["archive_sha256"],cosine_min=audit["cosine_min"],
                                    music_vs_fusion_changed_works=changed))
    print(json.dumps(summaries))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("--cosine-min", type=float, default=.60)
    p.add_argument("--audit-only",action="store_true")
    args = p.parse_args()
    if not -1 <= args.cosine_min <= 1:
        p.error("Cosine threshold must be in [-1,1]")
    read(MANIFESTS/"provenance.json")
    requests,audit = inspect(args.archive,args.cosine_min)
    dest = OUTPUT/"cover_pair_alignment"
    write(dest/"audit.json",audit)
    write(dest/"oracle_requests.json",list(requests.values()))
    if args.audit_only:
        print(json.dumps({k:v for k,v in audit.items() if k!="items"}))
    else:
        compare(requests,read(OUTPUT/"scores"/"covers80-lyric-evidence.json"),dest,audit)
