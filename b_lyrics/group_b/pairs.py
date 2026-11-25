"""Fold-specific calibration, candidate lyric scoring and conditional cover typing."""
import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np

from .cache import MANIFESTS, OUTPUT, ROOT, load_stage, read, rows, write
from .lyrics import gate, iou, jaccard, local_alignment, reliable, similarity
from .position_fusion import select_reference_interval

DEFAULTS=dict(min_seconds=3.,min_power=1e-5,min_ratio=.10,lang_conf=.80,lang_margin=.20,
              min_iou=.50,position_margin=.10,original=.65,rewrite=.25)


@lru_cache(maxsize=1)
def detector():
    import fasttext
    return fasttext.load_model(str(ROOT/"weights/fasttext/lid.176.bin"))


class Corpus:
    def __init__(self,config,output=OUTPUT):
        self.cfg={**DEFAULTS,**config}
        self.output=output
        self.queries={r["work_id"]:r for r in rows(MANIFESTS/"covers80_queries.jsonl")}
        self.refs={r["work_id"]:r for r in rows(MANIFESTS/"covers80_references.jsonl")}
        self.cache={}

    def evidence(self,work,side):
        if (work,side) in self.cache:
            return self.cache[work,side]
        row=(self.queries if side=="query" else self.refs)[work]
        record=load_stage(side,"text",row,self.output)
        lines=[l for l in record["lines"] if l.get("vector")]
        decision=None
        top=record["language_top2"]
        if side=="query":
            vocals=load_stage(side,"vocals",row,self.output)
            decision=gate(record["lines"],vocals,self.cfg["min_seconds"],self.cfg["min_power"],self.cfg["min_ratio"])
            ids=set(decision["line_ids"])
            lines=[l for l in lines if l["line_id"] in ids] if decision["enabled"] else []
            text=" ".join(l["clean"] for l in lines)
            top=[(label.removeprefix("__label__"),float(prob)) for prob,label in
                 detector().f.predict(text+"\n",2,0.,"strict")] if text else []
        lang=reliable(top,self.cfg["lang_conf"],self.cfg["lang_margin"])
        self.cache[work,side]=(lines,lang,decision)
        return lines,lang,decision

    def calibrate(self,fit_ids):
        samples={"semantic":[],"same_language":[]}
        for work in fit_ids:
            q,ql,_=self.evidence(work,"query")
            if not q:
                continue
            for other in fit_ids:
                if other==work:
                    continue
                r,rl,_=self.evidence(other,"reference")
                if r:
                    samples["semantic"].append(similarity(q,r,False).ravel())
                    if ql and ql==rl:
                        samples["same_language"].append(similarity(q,r,True).ravel())
        return {k:float(np.quantile(np.concatenate(v),.95)) if v else None for k,v in samples.items()}

    def score_candidate(self,work,other,background,semantic_only=False):
        q,ql,decision=self.evidence(work,"query")
        r,rl,_=self.evidence(other,"reference")
        same=bool(ql and ql==rl and not semantic_only and background["same_language"] is not None)
        b=background["same_language" if same else "semantic"]
        result=dict(work_id=work,reference_work_id=other,gate=decision,query_language=ql,reference_language=rl,
                    score=None,matches=[],branch="same_language" if same else "semantic")
        if not q or not r or b is None:
            result["reason"]=decision["reason"] or ("no_reference_lines" if not r else "missing_calibration")
            return result
        scores=similarity(q,r,same)
        raw,path=local_alignment(scores,b)
        result.update(score=raw,matches=[evidence([q[i]],[r[j]],float(scores[i,j])) for i,j in path])
        return result


def evidence(q,r,score):
    return dict(query_lines=[l["line_id"] for l in q],reference_lines=[l["line_id"] for l in r],
                query_sec=[q[0]["start_sec"],q[-1]["end_sec"]],reference_sec=[r[0]["start_sec"],r[-1]["end_sec"]],
                query_text=" ".join(l["clean"] for l in q),reference_text=" ".join(l["clean"] for l in r),score=score)


def music_pairs(q,r,blocks,region,cfg,mode):
    # Each block contains only contiguous reliable anchors, in seconds on both axes.
    anchors=[]
    for block in blocks:
        x,y=np.array(block,dtype=float).T
        if len(x)<2 or not np.isfinite([x,y]).all() or (np.diff(x)<=0).any() or (np.diff(y)<0).any():
            raise ValueError("Invalid reliable music-path block")
        anchors.append((x,y))
    proposals=[]
    for i in range(len(q)):
        for k in (1,2):
            if i+k>len(q):
                continue
            group=q[i:i+k]
            if k==2 and group[1]["line_id"]!=group[0]["line_id"]+1:
                continue
            a,b=group[0]["start_sec"],group[-1]["end_sec"]
            choices={}
            for x,y in anchors:
                if not x[0]<=a<b<=x[-1]:
                    continue
                projected=np.interp([a,b],x,y)
                for j in range(len(r)):
                    for size in ((1,2) if k==1 else (1,)):
                        group_r=r[j:j+size]
                        if len(group_r)!=size or (size==2 and group_r[1]["line_id"]!=group_r[0]["line_id"]+1):
                            continue
                        interval=(group_r[0]["start_sec"],group_r[-1]["end_sec"])
                        if region and iou(interval,region)==0:
                            continue
                        overlap=iou(projected,interval)
                        if overlap<=0:
                            continue
                        score=overlap if mode=="music" else .5*(overlap+float(similarity(group,group_r,False).mean()))
                        item=evidence(group,group_r,score)
                        item["mapped_iou"]=overlap
                        if (j,size) not in choices or choices[j,size]["score"]<score:
                            choices[j,size]=item
            options=sorted(choices.values(),key=lambda e:-e["score"])
            if options:
                best=options[0]
                distinct=[e for e in options[1:] if not set(e["reference_lines"]) & set(best["reference_lines"])]
                if best["mapped_iou"]>=cfg["min_iou"] and (not distinct or best["mapped_iou"]-max(e["mapped_iou"] for e in distinct)>=cfg["position_margin"]):
                    proposals.append(best)
    selected=[]
    for p in sorted(proposals,key=lambda e:-e["score"]):
        def ordered(other):
            q,r=p["query_lines"],p["reference_lines"]
            u,v=other["query_lines"],other["reference_lines"]
            return (q[-1]<u[0] and r[-1]<v[0]) or (q[0]>u[-1] and r[0]>v[-1])
        if all(ordered(other) for other in selected):
            selected.append(p)
    return sorted(selected,key=lambda p:p["query_lines"])


def predict(corpus,request,background,mode):
    work,other=request["work_id"],request["reference_work_id"]
    result=corpus.score_candidate(work,other,background,semantic_only=(mode=="semantic"))
    result.update(display=[],type_status="pending")
    region=request.get("reference_interval")
    if (mode=="fusion" and request.get("confirmed",False) and region is None
            and request.get("position_candidates")):
        config=request["position_config"]
        fused=select_reference_interval(request["position_candidates"],corpus.refs[other]["duration_sec"],
                                        config["weights"],start_bias_sec=config.get("start_bias_sec",0.),
                                        use_lyrics=result["gate"]["enabled"])
        if fused is not None:
            region=fused["reference_interval"]
            result.update(reference_interval=region,position_fusion=fused)
    if not result["gate"]["enabled"]:
        result["type_status"]="skipped"
        return result
    if not request.get("confirmed",False):
        return result
    q,ql,_=corpus.evidence(work,"query")
    r,rl,_=corpus.evidence(other,"reference")
    can_locate=request.get("context","oracle")=="oracle" or region is not None
    music=music_pairs(q,r,request.get("blocks",[]),region,corpus.cfg,mode) if can_locate and mode!="semantic" else []
    result["display"]=([m for m in result["matches"] if not region or iou(m["reference_sec"],region)>0]
                       if can_locate and mode=="semantic" else music)
    if ql and rl and ql!=rl:
        result.update(type_status="answered",cover_type="different_language")
    elif ql and ql==rl:
        vals=[]
        seen_q,seen_r=set(),set()
        for item in music:
            a,b=item["query_text"],item["reference_text"]
            if a not in seen_q and b not in seen_r:
                vals.append(jaccard(a,b))
                seen_q.add(a)
                seen_r.add(b)
        result["lexical_evidence"]=vals
        result["lexical_median"]=float(np.median(vals)) if vals else None
        result["high_overlap_fraction"]=sum(s>=corpus.cfg["original"] for s in vals)/len(vals) if vals else None
        if len(vals)>=2:
            if min(vals)>=corpus.cfg["original"]:
                result.update(type_status="answered",cover_type="original")
            elif max(vals)<=corpus.cfg["rewrite"] and request.get("transcripts_reliable") is True:
                result.update(type_status="answered",cover_type="rewrite")
    return result


def fold_sets(outer,inner):
    folds=rows(MANIFESTS/"covers80_folds.jsonl")
    fold=next(f for f in folds if f["outer_fold"]==outer)
    if inner is None:
        fit,heldout=fold["development_work_ids"],fold["test_work_ids"]
    else:
        sub=next(s for s in fold["inner"] if s["inner_fold"]==inner)
        fit,heldout=sub["fit_work_ids"],sub["validation_work_ids"]
    if set(fit)&set(heldout) or len(set(fit))!=(64 if inner is None else 48) or len(set(heldout))!=16:
        raise ValueError("Invalid fold partition")
    return fit,heldout


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--outer",required=True,type=int,choices=range(1,6))
    p.add_argument("--inner",type=int,choices=range(1,5))
    p.add_argument("--requests",type=Path,required=True)
    p.add_argument("--config",type=Path)
    p.add_argument("--mode",choices=["semantic","music","fusion"],default="fusion")
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    fit,heldout=fold_sets(args.outer,args.inner)
    config=read(args.config) if args.config else {}
    if "outer_fold" in config and config["outer_fold"]!=args.outer:
        p.error("Gate calibration belongs to a different outer fold")
    corpus=Corpus(config.get("config",config))
    requests=rows(args.requests)
    allowed=set(fit+heldout) if args.inner else set(corpus.refs)
    if any(r["work_id"] not in heldout or r["reference_work_id"] not in allowed for r in requests):
        p.error("Requested pair outside the held-out query set or eligible reference library")
    bg=corpus.calibrate(fit)
    write(args.out,dict(outer_fold=args.outer,inner_fold=args.inner,fit_work_ids=fit,config=corpus.cfg,
                        background=bg,mode=args.mode,predictions=[predict(corpus,r,bg,args.mode) for r in requests]))
