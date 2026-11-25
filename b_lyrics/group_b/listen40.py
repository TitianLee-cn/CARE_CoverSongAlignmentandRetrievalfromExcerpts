"""Find 30-second listening candidates for the official 40 B-side pairs."""
import argparse
import html
import json
from concurrent.futures import ProcessPoolExecutor
from math import gcd
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import fftconvolve, resample_poly, stft

from .cache import MANIFESTS, OUTPUT, ROOT, cache_path, load_stage, read, rows, write
from .lyrics import clean_line, tokens
from .media import decode
from .review_pack import crop

STEP = .1


def chroma(row, query=False):
    wave, rate, _ = decode(ROOT/"datasets"/row["path"])
    if (rate,len(wave),wave.shape[1])!=(row["sample_rate"],row["num_samples"],row["channels"]):
        raise ValueError(f"Audio metadata differs: {row['recording_id']}")
    if query:
        wave=wave[row["start_sample"]:row["end_sample"]]
    sr=12000
    divisor=gcd(sr,rate)
    audio=resample_poly(wave.mean(1),sr//divisor,rate//divisor)
    freq,_,spectrum=stft(audio,fs=sr,nperseg=4096,noverlap=2896)
    harmonic=median_filter(np.abs(spectrum[1:]),size=(1,7))
    spectrum=np.log1p(harmonic/np.maximum(median_filter(harmonic,size=(13,1)),2e-5))
    freq=freq[1:]
    midi=69+12*np.log2(freq/440)
    pitch=np.rint(midi)
    basis=np.eye(12,dtype=np.float32)[pitch.astype(int)%12]
    near=np.cos((midi-pitch)*np.pi)**2
    bands=[]
    for start,end in ((55,180),(180,680),(680,2400)):
        weights=(basis*((start<=freq)&(freq<end))[:,None]*near[:,None]).astype(np.float32)
        band=np.sqrt(np.maximum(gaussian_filter1d(spectrum.T @ weights,1.8,axis=0),0))
        band-=band.mean(1,keepdims=True)
        bands.append(band/np.maximum(np.linalg.norm(band,axis=1,keepdims=True),.03))
    return np.stack(bands,axis=1)/np.sqrt(len(bands))


def audio_candidates(query, reference, duration):
    proposals=[]
    for speed in np.round(np.arange(.55,1.61,.05),2):
        length=round(30*speed/STEP)
        qt=query[np.minimum(np.rint(np.arange(length)/speed).astype(int),len(query)-1)]
        limit=min(len(reference)-length+1,int((duration-30*speed)/STEP)+1)
        if limit<1:
            continue
        for shift in range(12):
            shifted=np.roll(qt,shift,axis=2)
            scores=fftconvolve(reference,shifted[::-1],mode="valid",axes=0).sum(axis=(1,2))[:limit]/length
            for _ in range(3):
                idx=int(scores.argmax())
                if scores[idx]<=-1:
                    break
                proposals.append((float(scores[idx]),idx*STEP,float(speed)))
                scores[max(0,idx-45):idx+46]=-1
    selected=[]
    for _,start,speed in sorted(proposals,reverse=True):
        mid=start+15*speed
        if all(abs(mid-(a+15*s))>2.5 for a,s in selected):
            selected.append((start,speed))
        if len(selected)==12:
            break
    return selected


def lyric_phrases(lines):
    phrases={}
    segment=[]
    last_end=None
    for line in sorted(lines,key=lambda x:x["start_sec"]):
        words=tokens(clean_line(line["text"]))
        duration=line["end_sec"]-line["start_sec"]
        if duration<=0 or not words:
            continue
        if last_end is not None and line["start_sec"]-last_end>3:
            segment=[]
        last_end=line["end_sec"]
        for idx,word in enumerate(words):
            t=line["start_sec"]+(idx+.5)*duration/len(words)
            segment.append((word,t))
            for length in range(3,min(5,len(segment))+1):
                span=segment[-length:]
                phrase=tuple(w for w,_ in span)
                if len(set(phrase))>=2 and 0<=span[-1][1]-span[0][1]<=8:
                    phrases.setdefault(phrase,[]).append((span[0][1]+span[-1][1])/2)
        segment=segment[-4:]
    return phrases


def lyric_anchors(query, ref):
    anchors=[]
    for stage in ("kiki","independent-qwen","independent-whisper"):
        if stage!="kiki" and not (cache_path("query",stage,query).is_file()
                                  and cache_path("reference",stage,ref).is_file()):
            continue
        q=lyric_phrases(load_stage("query",stage,query)["lines"])
        r=lyric_phrases(load_stage("reference",stage,ref)["lines"])
        for phrase, query_times in q.items():
            for t in query_times:
                anchors.extend((t,u,phrase,stage) for u in r.get(phrase,[]) if 0<t<30 and 0<u<ref["duration_sec"])
    return anchors


def lyric_candidates(anchors):
    if not anchors:
        return []
    q,r=np.asarray([(a,b) for a,b,_,_ in anchors]).T
    bucket=(q//3).astype(int)
    proposals=[]
    for speed in np.round(np.arange(.55,1.61,.05),2):
        starts=r-q*speed
        for center_bin in np.unique(np.round(starts*2)):
            close=np.abs(starts-center_bin/2)<=1.6
            distinct=np.unique(bucket[close])
            if len(distinct)<2:
                continue
            center=float(np.median(starts[close]))
            span=float(q[close].max()-q[close].min())
            residual=float(np.median(np.abs(starts[close]-center)))
            proposals.append((len(distinct),span,-residual,-abs(speed-1),center,float(speed)))
    selected=[]
    for _,_,_,_,start,speed in sorted(proposals,reverse=True):
        if any(abs(start+15*speed-a-15*s)<2 and abs(speed-s)<.12 for a,s,_ in selected):
            continue
        selected.append((start,speed,"ordered_lyrics"))
        if len(selected)==24:
            break
    return selected


def lyric_evidence(anchors,start,speed):
    # Keep one representative anchor per three seconds to limit repeated words and overlapping n-grams.
    support={}
    models=set()
    for q,r,phrase,stage in anchors:
        error=abs(r-(start+q*speed))
        if error>1.6:
            continue
        bucket=int(q//3)
        rank=(len(phrase),-error)
        models.add(stage)
        if bucket not in support or rank>support[bucket][0]:
            support[bucket]=(rank,dict(query_sec=q,reference_sec=r,phrase=" ".join(phrase),model=stage))
    ordered=[v for _,v in sorted(support.values(),key=lambda x:x[1]["query_sec"])]
    chains=[]
    for idx,anchor in enumerate(ordered):
        prior=[p for p in chains[:idx] if p[-1]["reference_sec"]<anchor["reference_sec"]]
        chains.append(max(prior,key=len,default=[])+[anchor])
    result=max(chains,key=len,default=[])
    if len({r["phrase"] for r in result})<2:
        return [],[]
    return result,sorted(models)


def rank_windows(query, ref, request, paths):
    q=chroma(query,query=True)
    r=chroma(ref)
    duration=ref["duration_sec"]
    anchors=lyric_anchors(query,ref)
    candidates=[(start,speed,"independent_audio") for start,speed in audio_candidates(q,r,duration)]
    candidates+=lyric_candidates(anchors)
    melody=cache_path("annotation","melody-v2",query)
    if melody.is_file():
        candidates.extend((x["start_sec"],x["tempo_ratio"],"independent_pitch") for x in read(melody))
    for path in paths:
        a,b=path["reference_span_sec"]
        candidates.append((a,(b-a)/30,"A_music"))
    for block in request["blocks"]:
        (a,b),(c,d)=block[0],block[-1]
        if c-a>=15 and d>b:
            speed=(d-b)/(c-a)
            candidates.append((b-a*speed,speed,"A_reliable_block"))
    scores=[]
    seen=set()
    for start,speed,origin in candidates:
        if not .48<=speed<=1.85 or start < -3 or start+30*speed>duration+3:
            continue
        key=(round(start,1),round(speed,2))
        if key in seen:
            continue
        seen.add(key)
        coords=np.rint(start/STEP+np.arange(len(q))*speed).astype(int)
        valid=(0<=coords)&(coords<len(r))
        if valid.sum()<len(q)*.85:
            continue
        qf,rf=q[valid],r[coords[valid]]
        agreement=np.array([np.sum(np.roll(qf,shift,axis=2)*rf,axis=(1,2)) for shift in range(12)])
        means=agreement.mean(axis=1)
        shift=int(np.argmax(means))
        thirds=[float(part.mean()) for part in np.array_split(agreement[shift],3)]
        audio_score=.8*float(means[shift])+.2*min(thirds)
        matches,models=lyric_evidence(anchors,start,speed)
        buckets=len(matches)
        span=matches[-1]["query_sec"]-matches[0]["query_sec"] if buckets>1 else 0.
        lyric_score=min(1.,buckets/4)*min(1.,span/12)
        mid=start+15*speed
        agrees=any(abs(mid-sum(p["reference_span_sec"])/2)<=3 for p in paths)
        crop_start=min(duration-30,max(0.,mid-15))
        scores.append(dict(score=float(audio_score+.32*lyric_score+.035*agrees),
            listening_start_sec=crop_start,alignment_start_sec=start,tempo_ratio=speed,
            acoustic_score=audio_score,acoustic_thirds=thirds,
            lyric_time_buckets=buckets,lyric_span_sec=span,lyric_models=models,
            lyric_anchors=matches,a_music_agrees=agrees,source=origin))
    ranked=[]
    for item in sorted(scores,key=lambda x:-x["score"]):
        if all(abs(item["listening_start_sec"]-r["listening_start_sec"])>=4 for r in ranked):
            ranked.append(item)
        if len(ranked)==3:
            break
    return ranked


def find_one(task):
    query,ref,request,paths=task
    candidates=rank_windows(query,ref,request,paths)
    top=candidates[0]
    margin=top["score"]-candidates[1]["score"] if len(candidates)>1 else 0.
    lyric_supported=top["lyric_time_buckets"]>=3 and top["lyric_span_sec"]>=8
    repeated=len(candidates)>1 and all(c["lyric_time_buckets"]>=3 and c["lyric_span_sec"]>=8
        for c in candidates[:2]) and margin<.075
    support=lyric_supported or (top["a_music_agrees"] and top["acoustic_score"]>=.42)
    strong=support and margin>=.045 and min(top["acoustic_thirds"])>=.15
    return dict(work_id=query["work_id"],title=query["work_name"],official_index=query["official_index"],
        query_recording_id=query["recording_id"],reference_recording_id=ref["recording_id"],
        source_hashes=dict(query=query["sha256"],reference=ref["sha256"]),
        query_source_sec=[query["start_sec"],query["end_sec"]],
        review_status="multi_source_candidate" if strong and not repeated else "review",
        same_lyric_sequence_support=lyric_supported,competing_lyric_location=repeated,
        margin=margin,candidates=candidates)


def create(folder,workers):
    queries={r["work_id"]:r for r in rows(MANIFESTS/"covers80_queries.jsonl") if 41<=r["official_index"]<=80}
    refs={r["work_id"]:r for r in rows(MANIFESTS/"covers80_references.jsonl")}
    pair_dir=OUTPUT/"cover_pair_alignment"
    requests={r["work_id"]:r for r in read(pair_dir/"oracle_requests.json")}
    audit=read(pair_dir/"audit.json")
    paths={r["work_id"]:r["paths"] for r in audit["items"]}
    tasks=[(q,refs[w],requests[w],paths[w]) for w,q in queries.items()]
    if len(tasks)!=40:
        raise ValueError("Expected 40 official B-side pairs")
    folder.mkdir(parents=True,exist_ok=False)
    (folder/"audio").mkdir()
    results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(find_one,tasks):
            work=result["work_id"]
            q,r=queries[work],refs[work]
            start=round(result["candidates"][0]["listening_start_sec"]*r["sample_rate"])
            result["reference_source_sec"]=[start/r["sample_rate"],start/r["sample_rate"]+30]
            result["query_audio"]=f"audio/{work}_input_30s.wav"
            result["reference_audio"]=f"audio/{work}_reference_30s.wav"
            crop(q,q["start_sample"],q["end_sample"],folder/result["query_audio"])
            crop(r,start,start+30*r["sample_rate"],folder/result["reference_audio"])
            results.append(result)
            print(work,result["review_status"],result["reference_source_sec"],flush=True)
    write(folder/"manifest.json",dict(context="Official correct recordings; automatic listening candidates",
          a_alignment_sha256=audit["archive_sha256"],
          method="Whole-track harmonic chroma, ordered lyric anchors from Kiki/Qwen/Whisper, independent pitch, and A paths; 30 s playback centred on selected alignment",
          entries=results))
    with (folder/"dataset_candidates.jsonl").open("w",encoding="utf-8") as stream:
        for x in results:
            work=x["work_id"]
            q,r=queries[work],refs[work]
            start=round(x["reference_source_sec"][0]*r["sample_rate"])
            record=dict(work_id=work,official_index=x["official_index"],query_id=q["query_id"],
                query_recording_id=q["recording_id"],reference_recording_id=r["recording_id"],
                query_path=q["path"],reference_path=r["path"],source_hashes=x["source_hashes"],
                query_samples=[q["start_sample"],q["end_sample"]],query_sample_rate=q["sample_rate"],
                reference_samples=[start,start+30*r["sample_rate"]],reference_sample_rate=r["sample_rate"],
                query_source_sec=x["query_source_sec"],reference_source_sec=x["reference_source_sec"],
                alternatives=[c["listening_start_sec"] for c in x["candidates"][1:]],
                evidence_status=x["review_status"],annotation_status="automatic_candidate",
                human_confirmed_reference_sec=None)
            stream.write(json.dumps(record,ensure_ascii=False)+"\n")
    page(folder/"index.html",results)
    with ZipFile(folder.with_suffix(".zip"),"x",compression=ZIP_DEFLATED) as archive:
        for file in sorted(folder.rglob("*")):
            if file.is_file():
                archive.write(file,file.relative_to(folder).as_posix())


def page(path,results,title="Covers80 后 40 对",intro=None,storage_key="care-covers80-b40-lyrics-v2"):
    cards=[]
    for r in results:
        q0,q1=r["query_source_sec"]
        r0,r1=r["reference_source_sec"]
        status="多来源支持" if r["review_status"]=="multi_source_candidate" else "待复核"
        alternatives="、".join(f"{c['listening_start_sec']:.2f}–{c['listening_start_sec']+30:.2f} 秒"
                               for c in r["candidates"][1:])
        title=html.escape(r["title"].replace("_"," "))
        anchors=r["candidates"][0]["lyric_anchors"]
        notes="；".join(f"{a['query_sec']:.1f} → {a['reference_sec']:.1f} 秒「{html.escape(a['phrase'])}」" for a in anchors)
        ambiguity=" · 备选位置也有同词支持" if r["competing_lyric_location"] else ""
        cards.append(f"""<article data-work="{r['work_id']}">
<h2>{title} <small>第 {r['official_index']} 对 · {status}</small></h2>
<div class="pair"><section><h3>原始查询</h3><p>原录音 {q0:.2f}–{q1:.2f} 秒</p>
<audio controls preload="none" src="{r['query_audio']}"></audio></section>
<section><h3>参考候选</h3><p>原录音 {r0:.2f}–{r1:.2f} 秒</p>
<audio controls preload="none" src="{r['reference_audio']}"></audio></section></div>
<p>其他候选位置 {alternatives or "无"}{ambiguity}</p>
<p>按顺序命中的歌词线索（查询片段内 → 参考原录音） {notes or "暂无"}</p>
<div class="marks"><label>试听结论 <select><option value="">未检查</option><option value="correct">对应</option>
<option value="wrong">不对应</option><option value="uncertain">不确定</option></select></label>
<label>修正起点（秒） <input class="start" type="number" min="0" step=".01"></label>
<label>备注 <input class="note"></label></div></article>""")
    path.write_text("""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Covers80 · 后 40 对试听</title>
<style>
body{font:16px/1.5 system-ui,sans-serif;max-width:1000px;margin:auto;padding:22px;background:#f2f5f7;color:#17232e}
article{background:white;border:1px solid #d5dee5;border-radius:8px;padding:16px;margin:16px 0}
h2,h3{margin:3px 0}small{font-weight:normal;color:#475569}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}
section{background:#f2f5f7;padding:10px}.marks{display:flex;gap:12px;flex-wrap:wrap}audio{width:100%}
select,input,button{font:inherit}input.start{width:7rem}@media(max-width:650px){.pair{display:block}}
</style>"""+f"<h1>{html.escape(title)}</h1><p>{html.escape(intro or '每对提供官方输入与参考录音中的连续 30 秒候选。标为待复核的条目可优先试听。')}</p>"
    +"""<button id="export">导出人工试听记录</button>"""+"".join(cards)+
    "<script>const key="+json.dumps(storage_key)+";</script>"+r"""
<script>
const cards=[...document.querySelectorAll('article')];
let store={}; try { store=JSON.parse(localStorage.getItem(key)||'{}'); } catch {}
function row(c) {
  const t=c.querySelector('.start').valueAsNumber;
  return {work_id:c.dataset.work,verdict:c.querySelector('select').value||null,
    corrected_reference_start_sec:Number.isFinite(t)?t:null,notes:c.querySelector('.note').value};
}
for(const c of cards) {
  const old=store[c.dataset.work];
  if(old){c.querySelector('select').value=old.verdict||'';
    c.querySelector('.start').value=old.corrected_reference_start_sec??'';
    c.querySelector('.note').value=old.notes||'';}
  c.oninput=()=>{store[c.dataset.work]=row(c);try{localStorage.setItem(key,JSON.stringify(store));}catch{}};
}
document.querySelector('#export').onclick=()=>{
  const blob=new Blob([cards.map(c=>JSON.stringify(row(c))).join('\n')+'\n'],{type:'application/jsonl'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='covers80_b40_listening.jsonl';a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href),1000);
};
</script></html>""",encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("folder",type=Path)
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    create(args.folder,args.workers)
