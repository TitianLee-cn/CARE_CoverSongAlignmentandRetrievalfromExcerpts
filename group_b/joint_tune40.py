"""Compare audio and lyric localisation and tune their fusion on shared nested folds."""
import argparse
import hashlib
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from statistics import mean, median
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from .cache import MANIFESTS, OUTPUT, cache_path, read, rows, write
from .listen40 import (STEP, audio_candidates, chroma, lyric_anchors, lyric_candidates,
                       lyric_evidence, page)
from .position_fusion import WEIGHT_KEYS as CONSENSUS_KEYS, score_positions
from .review_pack import crop
from .tune_listen40 import clip, edit_distance_at, iou, timed_word_streams, word_distance, word_evidence

KEYS=("audio","a_path","anchors","timed_words","window_words","far_penalty")
GRIDS={
    "audio":[(1.,a,0.,0.,0.,p) for a,p in product((0.,.05,.12,.20),(0.,.08))],
    "lyrics":[(.12,.025,l,t,w,p) for l,t,w,p in
              product((.25,.5,.9),(.18,.36,.60),(0.,.18),(0.,.08))],
    "fusion":[(1.,a,l,t,w,p) for a,l,t,w,p in
              product((.0,.04,.09),(.18,.34,.52,.72),(0.,.10,.23,.38),(0.,.12),(0.,.08,.16))],
}
CONSENSUS_GRID=list(product((.25,.40,.55,.70,.85),(.06,.12),(.0,.18,.36),(.0,.06,.12)))


def chunk_words(query, ref):
    """Keep explicit clip-relative word positions for the three ten-second comparisons."""
    return [([[word for t,word in q if i<=t<i+10] for i in (0,10,20)],times,words)
            for q,times,words in timed_word_streams(query,ref)]


def temporal_word_score(streams,left,speed):
    model_results=[]
    for thirds,times,words in streams:
        num,den=0.,0
        for i,query in enumerate(thirds):
            if len(query)<3:
                continue
            a=bisect_left(times,left+10*i*speed)
            b=bisect_left(times,left+10*(i+1)*speed)
            num+=len(query)*(1-word_distance(query,words[a:b]))
            den+=len(query)
        if den:
            model_results.append(num/den)
    return median(model_results) if model_results else 0.


def candidate_features(task):
    return task[0]["work_id"],build_position_candidates(*task)


def build_position_candidates(q,ref,request,paths,previous=None):
    """Compute scoring features for one query and one reference without supervision."""
    query,reference=chroma(q,True),chroma(ref)
    duration=ref["duration_sec"]
    anchors=lyric_anchors(q,ref)
    exact=word_evidence(q,ref)
    timed=chunk_words(q,ref)
    proposals=[(s,v,"global_audio") for s,v in audio_candidates(query,reference,duration)]
    proposals+=lyric_candidates(anchors)
    if previous is not None:
        proposals.extend((c["alignment_start_sec"],c["tempo_ratio"],"independent_prior") for c in previous["candidates"])
    pitch=cache_path("annotation","melody-v2",q)
    if pitch.is_file():
        proposals.extend((p["start_sec"],p["tempo_ratio"],"independent_pitch") for p in read(pitch)[:12])
    a_centers=[]
    for p in paths:
        a,b=p["reference_span_sec"]
        proposals.append((a,(b-a)/30,"A_path"))
        a_centers.append((a+b)/2)
    for path in request["blocks"]:
        (i,j),(k,l)=path[0],path[-1]
        if k-i>=15 and l>j:
            speed=(l-j)/(k-i)
            proposals.append((j-i*speed,speed,"A_reliable_block"))
    expanded=[(s+offset,v,origin) for s,v,origin in proposals for offset in (0.,-.5,.5)]
    bank=np.stack([np.roll(query,k,axis=2) for k in range(12)])
    ref_length=len(reference)
    models_cache,options,seen={},[],set()
    if previous is not None:
        baseline=previous["reference_source_sec"][0]
    elif proposals:
        left,speed,_=proposals[0]
        baseline=clip(left+15*speed-15,duration)
    else:
        return []
    for left,speed,origin in expanded:
        key=(round(left,2),round(speed,3))
        if key in seen or not .48<=speed<=1.85:
            continue
        seen.add(key)
        at=np.rint(left/STEP+np.arange(len(query))*speed).astype(int)
        valid=(0<=at)&(at<ref_length)
        if valid.sum()<len(query)*.85:
            continue
        q_bank=bank[:,valid]
        r=reference[at[valid]]
        scores=np.sum(q_bank*r[None],axis=(2,3))
        means=scores.mean(axis=1)
        transpose=int(means.argmax())
        thirds=[float(x.mean()) for x in np.array_split(scores[transpose],3)]
        # Let each third tolerate small tempo drift while retaining its position in the song.
        aligned_query=q_bank[transpose]
        alternatives=[]
        for shift in (-4,0,4):
            drifted=reference[np.clip(at[valid]+shift,0,ref_length-1)]
            alternatives.append([float(x.mean()) for x in np.array_split(np.sum(aligned_query*drifted,axis=(1,2)),3)])
        relaxed=float(np.max(alternatives,axis=0).min())
        acoustic=.72*float(means[transpose])+.18*min(thirds)+.10*relaxed
        ordered,model_ids=lyric_evidence(anchors,left,speed)
        span=ordered[-1]["query_sec"]-ordered[0]["query_sec"] if len(ordered)>1 else 0.
        anchor_score=min(1.,len(ordered)/4)*min(1.,span/12)
        if len(model_ids)<2:
            anchor_score*=.88
        midpoint=left+15*speed
        start=clip(midpoint-15,duration)
        if start not in models_cache:
            models_cache[start]=edit_distance_at(exact,start)
        edit=models_cache[start]
        temporal=temporal_word_score(timed,left,speed)
        far=abs(start-baseline)>15
        weak_jump=far and (anchor_score<.70 or (len(model_ids)<2 and acoustic<.6))
        options.append(dict(start_sec=start,source=origin,tempo_ratio=speed,lyric_anchors=ordered,
                      audio=acoustic,a_path=float(any(abs(midpoint-x)<=3 for x in a_centers)),
                      anchors=anchor_score,timed_words=temporal,window_words=1-edit if edit is not None else 0.,
                      far_penalty=-float(weak_jump),edit_distance=edit,lyric_models=model_ids))
    return options


def offset(truth,starts,fit):
    return round(np.clip(median(truth[w]-starts[w] for w in fit),-3,3)*2)/2


def official_fit(fold,grid,candidates,truth,durations,keys=KEYS):
    works=set(truth)
    development=works.intersection(fold["development_work_ids"])
    hold=works.intersection(fold["test_work_ids"])
    inner=[(sorted(development.intersection(s["fit_work_ids"])),
            sorted(development.intersection(s["validation_work_ids"]))) for s in fold["inner"]]
    if development & hold or any(not fit or not val or set(fit)&set(val) for fit,val in inner):
        raise ValueError("Invalid fold membership")
    if sorted(w for _,val in inner for w in val)!=sorted(development):
        raise ValueError("Inner validation must cover the development works once")
    objectives={}
    for weights in grid:
        proposed={w:r["start_sec"] for w,r in candidates[weights].items()}
        errors=[]
        for fit,val in inner:
            bias=offset(truth,proposed,fit)
            errors.extend(abs(clip(proposed[w]+bias,durations[w])-truth[w]) for w in val)
        overlap=mean(map(iou,errors))
        catastrophic=mean(e>=15 for e in errors)
        objectives[weights]=(overlap-.10*catastrophic,overlap,-mean(errors),-sum(weights[1:]))
    weights=max(grid,key=objectives.get)
    starts={w:p["start_sec"] for w,p in candidates[weights].items()}
    return dict(weights=dict(zip(keys,weights)),start_bias_sec=float(offset(truth,starts,sorted(development))),
                inner_validation_mean_iou=objectives[weights][1],development_work_ids=sorted(development),
                test_work_ids=sorted(hold))


def metrics(predictions,field):
    errors=[abs(r[field]["start_sec"]-r["target_start_sec"]) for r in predictions]
    edit=[r[field]["edit_distance"] for r in predictions if r[field]["edit_distance"] is not None]
    return dict(works=len(errors),mean_iou=mean(map(iou,errors)),start_mae_sec=mean(errors),
                within_2_sec=sum(e<=2 for e in errors),error_ge_15_sec=sum(e>=15 for e in errors),
                mean_lyric_edit_distance=mean(edit) if edit else None,lexical_works=len(edit))


def consensus_candidates(pools):
    selected={weights:{} for weights in CONSENSUS_GRID}
    for work,options in pools.items():
        for weights in CONSENSUS_GRID:
            values=score_positions(options,**dict(zip(CONSENSUS_KEYS,weights)))
            selected[weights][work]=options[int(values.argmax())]
    return selected


def run(labels_path,previous_run,folder,workers,prior_path=None):
    source=rows(labels_path)
    labels={x["work_id"]:x for x in source}
    queries={r["work_id"]:r for r in rows(MANIFESTS/"covers80_queries.jsonl") if 41<=r["official_index"]<=80}
    refs={r["work_id"]:r for r in rows(MANIFESTS/"covers80_references.jsonl") if r["work_id"] in queries}
    if len(source)!=40 or set(labels)!=set(queries) or set(refs)!=set(queries):
        raise ValueError("Reference file must contain the official 40 B works")
    truth={}
    for w,x in labels.items():
        q,r=queries[w],refs[w]
        rate=r["sample_rate"]
        a,b=x["reference_samples"]
        if (x["query_id"],x["query_samples"],x["reference_recording_id"],x["reference_sample_rate"],x["source_hashes"]) != (
                q["query_id"],[q["start_sample"],q["end_sample"]],r["recording_id"],rate,
                {"query":q["sha256"],"reference":r["sha256"]}) or not (0<=a<b<=r["num_samples"] and b-a==30*rate):
            raise ValueError(f"Label identity or sample range differs: {w}")
        truth[w]=a/rate
    pair=OUTPUT/"cover_pair_alignment"
    paths={r["work_id"]:r["paths"] for r in read(pair/"audit.json")["items"]}
    reqs={r["work_id"]:r for r in read(pair/"oracle_requests.json")}
    if prior_path is None:
        prior_path=OUTPUT/"Covers80_B40_fixed30_listening/manifest.json"
    prior={r["work_id"]:r for r in read(prior_path)["entries"]}
    prev=read(previous_run)
    prev_results={r["work_id"]:r for r in prev["predictions"]}
    if set(prev_results)!=set(queries):
        raise ValueError("Previous cross-validation did not cover the official B works")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pools=dict(pool.map(candidate_features,[(q,refs[w],reqs[w],paths[w],prior[w]) for w,q in queries.items()]))
    if any(not candidates for candidates in pools.values()):
        raise ValueError("Missing candidate features for an official work")
    print(f"Candidate proposals for 40 works, median count {median(map(len,pools.values()))}",flush=True)
    configs={w for grid in GRIDS.values() for w in grid}
    selected={}
    for weights in configs:
        selected[weights]={w:max(items,key=lambda c:sum(c[k]*v for k,v in zip(KEYS,weights)))
                           for w,items in pools.items()}
    consensus=consensus_candidates(pools)
    folds=[]
    predictions=[]
    durations={w:r["duration_sec"] for w,r in refs.items()}
    for fold in rows(MANIFESTS/"covers80_folds.jsonl"):
        fitted={method:official_fit(fold,grid,selected,truth,durations) for method,grid in GRIDS.items()}
        fitted["consensus"]=official_fit(fold,CONSENSUS_GRID,consensus,truth,durations,CONSENSUS_KEYS)
        chosen_method=max(("fusion","consensus"),key=lambda name:fitted[name]["inner_validation_mean_iou"])
        folds.append(dict(outer_fold=fold["outer_fold"],methods=fitted,development_selected=chosen_method))
        for work in fitted["audio"]["test_work_ids"]:
            streams=word_evidence(queries[work],refs[work])
            result=dict(work_id=work,outer_fold=fold["outer_fold"],target_start_sec=truth[work])
            oracle=min(pools[work],key=lambda c:abs(c["start_sec"]-truth[work]))
            old=prev_results[work]["tuned_start_sec"]
            result["pool_ceiling_start_sec"]=oracle["start_sec"]
            result["previous"]=dict(start_sec=old,edit_distance=edit_distance_at(streams,old))
            for method,param in fitted.items():
                keys=CONSENSUS_KEYS if method=="consensus" else KEYS
                weights=tuple(param["weights"][k] for k in keys)
                candidates=consensus if method=="consensus" else selected
                option=candidates[weights][work]
                start=clip(option["start_sec"]+param["start_bias_sec"],durations[work])
                result[method]=dict(start_sec=start,edit_distance=edit_distance_at(streams,start),choice=option)
            result["development_selected"]=dict(result[chosen_method],method=chosen_method)
            predictions.append(result)
    if sorted(x["work_id"] for x in predictions)!=sorted(queries):
        raise ValueError("Incomplete or duplicate out-of-fold predictions")
    ceiling=[abs(r["pool_ceiling_start_sec"]-r["target_start_sec"]) for r in predictions]
    ranges={name:{key:sorted({w[i] for w in grid}) for i,key in enumerate(KEYS)}
            for name,grid in GRIDS.items()}
    ranges["consensus"]={key:sorted({w[i] for w in CONSENSUS_GRID}) for i,key in enumerate(CONSENSUS_KEYS)}
    summary=dict(review_sha256=hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        reference_status="user-designated automatic review intervals",protocol="official 5 outer and 4 inner folds",
        parameter_grids=ranges,selection_objective="mean interval IoU minus 0.10 times the 15s error rate",
        metrics={name:metrics(predictions,name) for name in ("previous",*GRIDS,"consensus","development_selected")},
        proposal_coverage=dict(within_2_sec=sum(e<=2 for e in ceiling),mean_oracle_iou=mean(map(iou,ceiling))),
        folds=folds,predictions=predictions)
    export(folder,prev,summary,queries,refs)
    print(summary["metrics"],summary["proposal_coverage"],flush=True)


def export(folder,prev,summary,queries,refs,primary="development_selected"):
    folder.mkdir(parents=True,exist_ok=False)
    (folder/"audio").mkdir()
    result={r["work_id"]:r for r in summary["predictions"]}
    listening=[]
    for sample in prev["entries"]:
        work=sample["work_id"]
        q,ref=queries[work],refs[work]
        prediction=result[work]
        chosen=prediction[primary]["choice"]
        rate=ref["sample_rate"]
        start=round(prediction[primary]["start_sec"]*rate)
        qi=f"audio/{work}_input_30s.wav"
        ri=f"audio/{work}_selected_30s.wav"
        crop(q,q["start_sample"],q["end_sample"],folder/qi)
        crop(ref,start,start+30*rate,folder/ri)
        others=[dict(listening_start_sec=prediction[method]["start_sec"]) for method in ("audio","lyrics")]
        listening.append(dict(**prediction,official_index=q["official_index"],title=q["work_name"],
            review_status="review",competing_lyric_location=False,
            candidates=[dict(chosen,listening_start_sec=start/rate),*others],
            query_source_sec=[q["start_sec"],q["end_sec"]],reference_source_sec=[start/rate,start/rate+30],
            query_audio=qi,reference_audio=ri))
    summary["entries"]=listening
    summary["listening_method"]=primary
    write(folder/"validation.json",summary)
    page(folder/"index.html",listening,title="Covers80 · 音频与歌词融合五折试听",
         intro=("使用与上一轮相同的 10 对。左边是官方 30 秒查询，右边是同位置证据融合的结果。"
                "下方两个备选位置依次来自音频优先和歌词优先。" if primary=="consensus" else
                "使用与上一轮相同的 10 对。左边是官方 30 秒查询，右边是各折开发集选出的融合结果。"
                "下方两个备选位置依次来自音频优先和歌词优先。"),
         storage_key=f"care-covers80-joint-b40-{primary}-v1")
    with ZipFile(folder.with_suffix(".zip"),"x",compression=ZIP_DEFLATED) as z:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                z.write(path,path.relative_to(folder).as_posix())


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("labels",type=Path)
    parser.add_argument("previous",type=Path)
    parser.add_argument("folder",type=Path)
    parser.add_argument("--prior",type=Path,help="Earlier candidate manifest used for experiment reproducibility")
    parser.add_argument("--workers",type=int,default=5)
    args=parser.parse_args()
    run(args.labels,args.previous,args.folder,args.workers,args.prior)
