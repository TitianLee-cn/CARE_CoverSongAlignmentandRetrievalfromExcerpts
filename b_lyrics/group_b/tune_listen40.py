"""Fit full-window candidate ranking and time bias with official nested folds."""
import argparse
from bisect import bisect_left
import hashlib
import random
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from statistics import mean, median
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from .cache import MANIFESTS, OUTPUT, cache_path, load_stage, read, rows, write
from .listen40 import STEP, audio_candidates, chroma, lyric_candidates, lyric_evidence, lyric_phrases, page
from .lyrics import clean_line, tokens
from .review_pack import crop

SEED = 42
BASELINE = (.19,.035)
GRID = [(l,a) for l,a in product((0.,.08,.16,.24,.32,.45,.60),(0.,.04,.08,.12))]
EDIT_WEIGHTS = (0.,.10,.20,.35,.55,.80)
EDIT_LOSS_WEIGHT = .35


def timed_words(lines):
    timed=[]
    for line in lines:
        words=tokens(clean_line(line["text"]))
        start,end=line["start_sec"],line["end_sec"]
        if end<=start:
            continue
        for i,word in enumerate(words):
            timed.append((start+(i+.5)*(end-start)/len(words),word))
    return sorted(timed)


def word_distance(query, reference):
    previous=list(range(len(reference)+1))
    for i,word in enumerate(query,1):
        current=[i]
        for j,other in enumerate(reference,1):
            current.append(min(previous[j]+1,current[-1]+1,previous[j-1]+(word!=other)))
        previous=current
    return previous[-1]/max(1,len(query),len(reference))


def stable_language(record):
    choices=record["language_top2"]
    return choices[0][0] if len(choices)>1 and choices[0][1]>=.8 and choices[0][1]-choices[1][1]>=.2 else None


def timed_word_streams(query, ref):
    """Use Kiki and any additional transcript streams that are available on both sides."""
    for stage in ("kiki","independent-qwen","independent-whisper"):
        if stage!="kiki" and not (cache_path("query",stage,query).is_file()
                                  and cache_path("reference",stage,ref).is_file()):
            continue
        q=load_stage("query",stage,query)
        r=load_stage("reference",stage,ref)
        if stage=="kiki":
            q_lang=stable_language(load_stage("query","text",query))
            r_lang=stable_language(load_stage("reference","text",ref))
        else:
            q_lang=(q.get("language") or "").strip().casefold()
            r_lang=(r.get("language") or "").strip().casefold()
        if not q_lang or q_lang!=r_lang or "," in q_lang:
            continue
        q_timed=[(t,w) for t,w in timed_words(q["lines"]) if 0<=t<30]
        words=[w for _,w in q_timed]
        if len(words)<5 or len(set(words))<3:
            continue
        r_timed=timed_words(r["lines"])
        yield q_timed,[t for t,_ in r_timed],[w for _,w in r_timed]


def word_evidence(query,ref):
    return [([w for _,w in q],times,words) for q,times,words in timed_word_streams(query,ref)]


def edit_distance_at(streams,start):
    if not streams:
        return None
    values=[]
    for query,times,words in streams:
        lo,hi=bisect_left(times,start),bisect_left(times,start+30)
        values.append(word_distance(query,words[lo:hi]))
    return median(values)


def pool_for(task):
    query,ref,request,paths,earlier,with_edit=task
    q,r=chroma(query,query=True),chroma(ref)
    kp=lyric_phrases(load_stage("query","kiki",query)["lines"])
    rp=lyric_phrases(load_stage("reference","kiki",ref)["lines"])
    anchors=[(t,u,phrase,"kiki") for phrase,times in kp.items() for t in times for u in rp.get(phrase,[])]
    edit_streams=word_evidence(query,ref) if with_edit else []
    edit_cache={}
    duration=ref["duration_sec"]
    proposals=[(a,s,"audio") for a,s in audio_candidates(q,r,duration)]
    proposals+=lyric_candidates(anchors)
    proposals.extend((x["alignment_start_sec"],x["tempo_ratio"],"earlier_search") for x in earlier)
    centers=[]
    for path in paths:
        a,b=path["reference_span_sec"]
        proposals.append((a,(b-a)/30,"A_path"))
        centers.append((a+b)/2)
    for block in request["blocks"]:
        (a,b),(c,d)=block[0],block[-1]
        if c-a>=15 and d>b:
            speed=(d-b)/(c-a)
            proposals.append((b-a*speed,speed,"A_reliable_block"))

    options,seen=[],set()
    bank=np.stack([np.roll(q,shift,axis=2) for shift in range(12)])
    for left,speed,source in proposals:
        key=(round(left,2),round(speed,3))
        if key in seen or not .48<=speed<=1.85:
            continue
        seen.add(key)
        at=np.rint(left/STEP+np.arange(len(q))*speed).astype(int)
        valid=(at>=0)&(at<len(r))
        if valid.sum()<len(q)*.85:
            continue
        sim=np.sum(bank[:,valid]*r[at[valid]][None],axis=(2,3))
        avg=sim.mean(axis=1)
        shift=int(avg.argmax())
        thirds=[float(x.mean()) for x in np.array_split(sim[shift],3)]
        lyrics,_=lyric_evidence(anchors,left,speed)
        span=lyrics[-1]["query_sec"]-lyrics[0]["query_sec"] if len(lyrics)>1 else 0.
        center=left+15*speed
        start=clip(center-15,duration)
        if start not in edit_cache:
            edit_cache[start]=edit_distance_at(edit_streams,start)
        options.append(dict(start_sec=start,source=source,tempo_ratio=speed,
                            audio=.8*float(avg[shift])+.2*min(thirds),
                            lyrics=min(1.,len(lyrics)/4)*min(1.,span/12),
                            a_path=float(any(abs(center-x)<=3 for x in centers)),lyric_anchors=lyrics,
                            edit_distance=edit_cache[start]))
    if not options:
        raise ValueError(f"No candidates for {query['work_id']}")
    return query["work_id"],options


def clip(t,duration):
    return max(0.,min(duration-30,t))


def select(options,l,a,e=0.):
    return max(options,key=lambda x:x["audio"]+l*x["lyrics"]+a*x["a_path"]-e*(x.get("edit_distance") or 0.))


def iou(delta):
    return max(0.,30-delta)/(30+min(30,delta))


def fit_bias(weights,starts,targets,fit):
    error=median(targets[w]-starts[weights][w] for w in fit)
    return round(max(-3.,min(3.,error))*2)/2


def tune(fold,starts,targets,durations, baseline=BASELINE, edit_losses=None):
    eligible=set(targets)
    dev=eligible.intersection(fold["development_work_ids"])
    hold=eligible.intersection(fold["test_work_ids"])
    if dev & hold or not hold:
        raise ValueError("Invalid outer-fold partition")
    inner_splits=[]
    for inner in fold["inner"]:
        fit=sorted(dev.intersection(inner["fit_work_ids"]))
        val=sorted(dev.intersection(inner["validation_work_ids"]))
        if not fit or not val or set(fit)&set(val) or hold.intersection(fit+val):
            raise ValueError("Invalid inner-fold partition")
        inner_splits.append((fit,val))
    if sorted(w for _,val in inner_splits for w in val)!=sorted(dev):
        raise ValueError("Inner folds must validate every available development work once")
    grid=starts.keys()
    objectives={}
    for weights in grid:
        errors,lyric_errors=[],[]
        for fit,val in inner_splits:
            bias=fit_bias(weights,starts,targets,fit)
            errors.extend(abs(clip(starts[weights][w]+bias,durations[w])-targets[w]) for w in val)
            if edit_losses is not None:
                lyric_errors.extend(edit_losses[weights][w] for w in val if edit_losses[weights][w] is not None)
        interval_iou=mean(map(iou,errors))
        edit_error=mean(lyric_errors) if lyric_errors else None
        loss=1-interval_iou+EDIT_LOSS_WEIGHT*(edit_error or 0.)
        objectives[weights]=((loss,mean(errors),sum(abs(a-b) for a,b in zip(weights,baseline))),
                             interval_iou,edit_error)
    weights=min(grid,key=lambda w:objectives[w][0])
    return dict(outer_fold=fold["outer_fold"],development_work_ids=sorted(dev),test_work_ids=sorted(hold),
                lyrics_weight=weights[0],a_path_weight=weights[1],
                reference_start_bias_sec=fit_bias(weights,starts,targets,sorted(dev)),
                lyric_edit_weight=weights[2] if len(weights)>2 else 0.,
                inner_validation_mean_iou=objectives[weights][1],
                inner_validation_start_mae_sec=objectives[weights][0][1],
                inner_validation_edit_distance=objectives[weights][2])


def metrics(predictions,field):
    errors=[abs(p[field]-p["target_start_sec"]) for p in predictions]
    edit=[p.get(field+"_edit") for p in predictions if p.get(field+"_edit") is not None]
    return dict(count=len(errors),mean_iou=mean(map(iou,errors)),start_mae_sec=mean(errors),
                within_2_sec=sum(e<=2 for e in errors),
                edit_distance_count=len(edit),mean_lyric_edit_distance=mean(edit) if edit else None)


def create(label_path,folder,workers,max_shift=None,with_edit=False):
    source=rows(label_path)
    labels={r["work_id"]:r for r in source}
    queries={r["work_id"]:r for r in rows(MANIFESTS/"covers80_queries.jsonl") if 41<=r["official_index"]<=80}
    refs={r["work_id"]:r for r in rows(MANIFESTS/"covers80_references.jsonl") if r["work_id"] in queries}
    if len(source)!=40 or set(labels)!=set(queries) or set(refs)!=set(queries):
        raise ValueError("Review file must correspond to the 40 official B-side works")
    targets={}
    for work,label in labels.items():
        q,r=queries[work],refs[work]
        left,right=label["reference_samples"]
        if (label["query_id"],label["query_recording_id"],label["reference_recording_id"]) != (
                q["query_id"],q["recording_id"],r["recording_id"]) or label["source_hashes"]!={"query":q["sha256"],"reference":r["sha256"]}:
            raise ValueError(f"Review identity mismatch: {work}")
        if label["query_samples"]!=[q["start_sample"],q["end_sample"]] or label["reference_sample_rate"]!=r["sample_rate"] or not (
                0<=left<right<=r["num_samples"] and right-left==30*r["sample_rate"]):
            raise ValueError(f"Review sample range mismatch: {work}")
        targets[work]=left/r["sample_rate"]

    pair_dir=OUTPUT/"cover_pair_alignment"
    reqs={r["work_id"]:r for r in read(pair_dir/"oracle_requests.json")}
    audit={r["work_id"]:r["paths"] for r in read(pair_dir/"audit.json")["items"]}
    prior=read(OUTPUT/"Covers80_B40_fixed30_listening/manifest.json")
    earlier={r["work_id"]:r for r in prior["entries"]}
    tasks=[(q,refs[w],reqs[w],audit[w],earlier[w]["candidates"],with_edit) for w,q in queries.items()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        options=dict(pool.map(pool_for,tasks))
    print(f"Scored independent candidate pools for {len(options)} works",flush=True)
    if max_shift is not None:
        baseline={w:select(items,*BASELINE)["start_sec"] for w,items in options.items()}
        options={w:[p for p in items if abs(p["start_sec"]-baseline[w])<=max_shift]
                 for w,items in options.items()}
    durations={w:r["duration_sec"] for w,r in refs.items()}
    baseline=(*BASELINE,0.) if with_edit else BASELINE
    grid=[(*weights,e) for weights in GRID+[BASELINE] for e in EDIT_WEIGHTS] if with_edit else GRID+[BASELINE]
    candidates={weights:{w:select(items,*weights) for w,items in options.items()} for weights in grid}
    starts={weights:{w:p["start_sec"] for w,p in selected.items()} for weights,selected in candidates.items()}
    edit_losses={weights:{w:p["edit_distance"] for w,p in selected.items()}
                 for weights,selected in candidates.items()} if with_edit else None
    folds=[tune(fold,starts,targets,durations,baseline,edit_losses) for fold in rows(MANIFESTS/"covers80_folds.jsonl")]
    predictions=[]
    for fold in folds:
        weights=(fold["lyrics_weight"],fold["a_path_weight"])
        if with_edit:
            weights=(*weights,fold["lyric_edit_weight"])
        for work in fold["test_work_ids"]:
            choice=candidates[weights][work]
            corrected=clip(choice["start_sec"]+fold["reference_start_bias_sec"],durations[work])
            result=dict(work_id=work,outer_fold=fold["outer_fold"],tuned_start_sec=corrected,
                        baseline_start_sec=starts[baseline][work],
                        previous_pack_start_sec=earlier[work]["reference_source_sec"][0],
                        target_start_sec=targets[work],choice=choice)
            if with_edit:
                streams=word_evidence(queries[work],refs[work])
                for key in ("baseline_start_sec","tuned_start_sec","previous_pack_start_sec"):
                    result[key+"_edit"]=edit_distance_at(streams,result[key])
            predictions.append(result)
    if sorted(p["work_id"] for p in predictions)!=sorted(queries):
        raise ValueError("Expected exactly one out-of-fold prediction per work")

    folder.mkdir(parents=True,exist_ok=False)
    (folder/"audio").mkdir()
    written=[]
    rng=random.Random(SEED)
    for fold in folds:
        eligible=[p for p in predictions if p["outer_fold"]==fold["outer_fold"]]
        for p in sorted(rng.sample(eligible,2),key=lambda x:x["work_id"]):
            work=p["work_id"]
            q,r=queries[work],refs[work]
            rate=r["sample_rate"]
            left=round(p["tuned_start_sec"]*rate)
            qi=f"audio/{work}_input_30s.wav"
            ri=f"audio/{work}_reference_30s.wav"
            crop(q,q["start_sample"],q["end_sample"],folder/qi)
            crop(r,left,left+30*rate,folder/ri)
            written.append(dict(**p,official_index=q["official_index"],title=q["work_name"],
                review_status="review",competing_lyric_location=False,
                candidates=[dict(p["choice"],listening_start_sec=left/rate)],
                query_source_sec=[q["start_sec"],q["end_sec"]],reference_source_sec=[left/rate,left/rate+30],
                query_audio=qi,reference_audio=ri))
    summary=dict(review_file=label_path.name,review_sha256=hashlib.sha256(label_path.read_bytes()).hexdigest(),
                 reference_kind="user-designated automatic review intervals, no human confirmation",
                 protocols="official outer 5 folds and inner 4 folds, restricted to 40 B-side works",
                 candidate_pool="independent audio, Kiki lyric proposals, A alignment and earlier pre-label candidates",
                 max_candidate_shift_from_baseline_sec=max_shift,
                 exploratory_guard_after_initial_evaluation=max_shift is not None,
                 lyric_loss=dict(metric="normalized word-level Levenshtein distance, median over eligible transcript models",
                                 inner_validation_term_weight=EDIT_LOSS_WEIGHT,
                                 score_weights=EDIT_WEIGHTS) if with_edit else None,
                 folds=folds,metrics={field:metrics(predictions,field)
                     for field in ("baseline_start_sec","tuned_start_sec","previous_pack_start_sec")},
                 predictions=predictions,entries=written)
    write(folder/"validation.json",summary)
    page(folder/"index.html",written,title="Covers80 歌词差异损失 · 五折试听" if with_edit else "Covers80 五折试听 · 10 对",
         intro="每个外层测试折各抽 2 对。左边是官方原始查询，右边是冻结本折参数后选出的参考区间，两侧各 30 秒。试听判断可导出。",
         storage_key="care-covers80-edit-loss-b40-v1" if with_edit else "care-covers80-nested-fitted-b40-v1")
    with ZipFile(folder.with_suffix(".zip"),"x",compression=ZIP_DEFLATED) as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                archive.write(path,path.relative_to(folder).as_posix())
    print(summary["metrics"])


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("labels",type=Path)
    p.add_argument("folder",type=Path)
    p.add_argument("--workers",type=int,default=5)
    p.add_argument("--max-shift-sec",type=float)
    p.add_argument("--lyric-edit-loss",action="store_true")
    args=p.parse_args()
    if args.max_shift_sec is not None and args.max_shift_sec<0:
        p.error("--max-shift-sec must be nonnegative")
    create(args.labels,args.folder,args.workers,args.max_shift_sec,args.lyric_edit_loss)
