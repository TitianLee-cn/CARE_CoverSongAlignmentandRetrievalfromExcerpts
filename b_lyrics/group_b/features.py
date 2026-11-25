"""Extract local, recording-level predictions. No evaluation labels are loaded."""
import argparse
import hashlib
import re
from functools import lru_cache
from math import gcd
from pathlib import Path

import numpy as np
from .media import decode
import torch

from .cache import HERE, MANIFESTS, NAMES, OUTPUT, ROOT, cache_path, identity, load_stage, read, rows, write
from .lyrics import clean_line, tokens

WEIGHTS = ROOT/"weights"
VERSIONS = {
    "kiki": ["b35a2d893c8c287c6f77da7c2931ac17df0b90df"],
    "asr": ["7278e1e70fe206f11671096ffdd38061171dd6e5","c7cbfc2048c462b0d63a45797104fc9db3ad62b7"],
    "text": ["97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3","lid.176.bin"],
    "vocals": ["955717e8-8726e21a.th"],
}
PARAMS = {"kiki":{"window_sec":60,"overlap_sec":10,"rate":16000,"max_new_tokens":1024},
          "asr":{"pause_sec":0.65,"max_new_tokens":2048},
          "text":{"pool":"last-token","normalize":True,"language_top_k":2},
          "vocals":{"frame_sec":0.02,"shifts":0}}


@lru_cache(maxsize=1)
def models(stage):
    if stage=="kiki":
        from transformers import AutoModelForCausalLM, AutoProcessor
        path=WEIGHTS/"echora-kiki"
        proc=AutoProcessor.from_pretrained(path,trust_remote_code=True,local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(path,trust_remote_code=True,dtype=torch.bfloat16).eval().cuda()
        return proc,model
    if stage=="asr":
        from qwen_asr import Qwen3ASRModel
        return Qwen3ASRModel.from_pretrained(str(WEIGHTS/"qwen3-asr-1.7b"),dtype=torch.bfloat16,device_map="cuda:0",
              forced_aligner=str(WEIGHTS/"qwen3-forcedaligner-0.6b"),
              forced_aligner_kwargs=dict(dtype=torch.bfloat16,device_map="cuda:0"),
              max_inference_batch_size=1,max_new_tokens=PARAMS["asr"]["max_new_tokens"])
    if stage=="vocals":
        from demucs.states import load_model
        path=WEIGHTS/"demucs"/VERSIONS[stage][0]
        with path.open("rb") as f:
            if not hashlib.file_digest(f,"sha256").hexdigest().startswith("8726e21a"):
                raise ValueError("Demucs checkpoint failed checksum")
        return load_model(torch.load(path,map_location="cpu",weights_only=False)).eval().cuda()
    from transformers import AutoModel,AutoTokenizer
    import fasttext
    path=WEIGHTS/"qwen3-embedding-0.6b"
    return (AutoTokenizer.from_pretrained(path,padding_side="left"),AutoModel.from_pretrained(path,dtype=torch.bfloat16).eval().cuda(),
            fasttext.load_model(str(WEIGHTS/"fasttext"/"lid.176.bin")))


def audio(row):
    full, rate, _ = decode(ROOT/"datasets"/row["path"])
    if "sample_rate" in row and (rate!=row["sample_rate"] or len(full)!=row["num_samples"] or
                                 full.shape[1]!=row["channels"]):
        raise ValueError("Decoded audio does not match shared recording metadata")
    start=row.get("start_sample",round(row.get("start_sec",0)*rate))
    end=row.get("end_sample",round(row.get("end_sec",row["duration_sec"])*rate))
    wave=full[start:end]
    if len(wave)!=end-start or not np.isfinite(wave).all():
        raise ValueError("Incomplete or nonfinite input audio")
    return wave,rate


def kiki(wave,sr):
    from scipy.signal import resample_poly
    proc,model=models("kiki")
    mono=resample_poly(wave.mean(1),16000//gcd(sr,16000),sr//gcd(sr,16000))
    duration=len(mono)/16000
    lines,windows=[],[]
    instruction="Transcribe the audio. For each segment, start with the timestamp and speaker ID ([S01], [S02], [S03], ...), then the spoken text, and end with the segment timestamp."
    prompt=proc.apply_chat_template([{"role":"user","content":[{"type":"audio"},{"type":"text","text":instruction}]}],
                                    tokenize=False,add_generation_prompt=True)
    offset=0
    while offset < duration:
        end=min(duration,offset+60)
        chunk=mono[offset*16000:round(end*16000)]
        inputs=proc(text=prompt,audio=chunk,return_tensors="pt").to("cuda")
        inputs["input_features"]=inputs["input_features"].to(torch.bfloat16)
        seq=model.generate(**inputs,do_sample=False,max_new_tokens=1024)
        raw=proc.tokenizer.decode(seq[0,inputs["input_ids"].shape[1]:],skip_special_tokens=True)
        windows.append(dict(start_sec=offset,end_sec=end,raw=raw))
        for a,speaker,text,b in re.findall(r"\[([\d.]+)\]\[(S\d+)\]([^\[]*)\[([\d.]+)\]",raw):
            a,b=float(a),float(b)
            if not 0<=a<b or a>=end-offset or not text.strip():
                continue
            b=min(b,end-offset)
            mid=offset+(a+b)/2
            if (offset and mid<offset+5) or (end<duration and mid>=end-5):
                continue
            lines.append(dict(start_sec=offset+a,end_sec=offset+b,text=text.strip(),speaker=speaker,
                              partial_left=a<.1,partial_right=b>=end-offset-.1))
        if end==duration:
            break
        offset+=50
    return dict(lines=lines,windows=windows)


def asr(wave,sr):
    hyp=models("asr").transcribe(audio=(wave.mean(1),sr),language=None,return_time_stamps=True)[0]
    duration=len(wave)/sr
    units,adjusted=[],[]
    for token in hyp.time_stamps or []:
        start,end=float(token.start_time),float(token.end_time)
        if not np.isfinite([start,end]).all() or end<start or end<0 or start>duration:
            adjusted.append(dict(start_sec=start if np.isfinite(start) else None,
                                 end_sec=end if np.isfinite(end) else None,action="discarded"))
            continue
        if start<0 or end>duration:
            adjusted.append(dict(start_sec=start,end_sec=end,action="clipped_to_recording"))
        units.append(dict(text=token.text,start_sec=max(0.,start),end_sec=min(duration,end)))
    lines=[]
    for unit in units:
        if not lines or unit["start_sec"]-lines[-1]["end_sec"]>=.65:
            lines.append(dict(unit))
        else:
            lines[-1]["text"] += " "+unit["text"]
            lines[-1]["end_sec"]=max(lines[-1]["end_sec"],unit["end_sec"])
    return dict(lines=lines,units=units,text=hyp.text,language=hyp.language,timestamp_adjustments=adjusted)


def vocals(wave,sr):
    from demucs.apply import apply_model
    from demucs.audio import convert_audio
    model=models("vocals")
    mix=convert_audio(torch.from_numpy(wave.T.copy()),sr,model.samplerate,model.audio_channels).cuda()
    ref=mix.mean(0)
    std=ref.std().clamp_min(1e-8)
    stems=apply_model(model,((mix-ref.mean())/std)[None],shifts=0,device="cuda")[0]
    voice=stems[model.sources.index("vocals")]*std+ref.mean()
    hop=round(.02*model.samplerate)
    def power(x):
        return x.unfold(-1,hop,hop).square().mean((0,2))
    energy=power(voice)
    return dict(frame_sec=hop/model.samplerate,vocal_power=energy.tolist(),vocal_to_mix_power=(energy/power(mix).clamp_min(1e-10)).tolist())


def language(text, lid):
    if not text.strip():
        return []
    predictions=lid.f.predict(" ".join(text.split())+"\n",2,0.0,"strict")
    return [(label.removeprefix("__label__"),float(score)) for score,label in predictions]


def text_features(transcription):
    tok,model,lid=models("text")
    lines=[dict(line) for line in transcription["lines"]]
    valid=[line for line in lines if line["clean"]]
    for i in range(0,len(valid),32):
        batch=valid[i:i+32]
        inputs=tok([l["clean"] for l in batch],padding=True,truncation=True,max_length=512,return_tensors="pt").to("cuda")
        vectors=torch.nn.functional.normalize(model(**inputs).last_hidden_state[:,-1].float(),dim=1)
        if not torch.isfinite(vectors).all() or not torch.allclose(vectors.norm(dim=1),torch.ones(len(batch),device="cuda"),atol=1e-5):
            raise ValueError("Invalid embeddings")
        for line,vector in zip(batch,vectors.tolist()):
            line.update(vector=vector,language_top2=language(line["clean"],lid))
    return dict(lines=lines,language_top2=language(" ".join(l["clean"] for l in valid),lid))


def extract(stage,dataset,row,output=OUTPUT):
    target=cache_path(dataset,stage,row,output)
    config=dict(identity=identity(row),stage=stage,model_versions=VERSIONS[stage],params=PARAMS[stage],
                code_hash=hashlib.sha256(Path(__file__).read_bytes()+(HERE/"lyrics.py").read_bytes()+(HERE/"media.py").read_bytes()).hexdigest())
    if stage=="text":
        transcription=load_stage(dataset,"kiki",row,output)
        config["source_hash"]=hashlib.sha256(cache_path(dataset,"kiki",row,output).read_bytes()).hexdigest()
    if target.is_file():
        prior=read(target)
        if all(prior.get(k)==v for k,v in config.items()):
            return target
    if stage=="text":
        result=text_features(transcription)
    else:
        result={"kiki":kiki,"asr":asr,"vocals":vocals}[stage](*audio(row))
    if stage in ("kiki","asr"):
        for i,line in enumerate(sorted(result["lines"],key=lambda l:l["start_sec"]),1):
            clean=clean_line(line["text"])
            line.update(line_id=i,clean=clean,filler_only=bool(tokens(line["text"])) and not clean)
        result["lines"].sort(key=lambda l:l["line_id"])
    write(target,dict(**config,**result))
    return target


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=VERSIONS)
    p.add_argument("dataset",choices=NAMES)
    p.add_argument("--start-index",type=int,default=1)
    p.add_argument("--limit",type=int)
    p.add_argument("--out",type=Path,default=OUTPUT)
    args=p.parse_args()
    records=rows(MANIFESTS/(NAMES[args.dataset]+".jsonl"))
    if not 1<=args.start_index<=len(records) or (args.limit is not None and args.limit<1):
        p.error("Invalid manifest row range")
    batch=records[args.start_index-1:][:args.limit]
    failures=[]
    with torch.inference_mode():
        for row in batch:
            try:
                print(extract(args.stage,args.dataset,row,args.out),flush=True)
            except Exception as exc:
                failures.append(dict(recording_id=row["recording_id"],error=repr(exc)))
                print(f"{row['recording_id']}: {exc!r}",flush=True)
                if len(batch)==1:
                    raise
    write(args.out/"runs"/f"{args.dataset}-{args.stage}-{args.start_index}-{len(batch)}.json",
          dict(requested=len(batch),completed=len(batch)-len(failures),failures=failures))
    raise SystemExit(bool(failures))
