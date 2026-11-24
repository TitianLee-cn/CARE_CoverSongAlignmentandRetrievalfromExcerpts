"""B-group lyric evidence and pair matching. All times are clip-relative seconds."""
import math
import re
import unicodedata

import numpy as np


def tokens(text):
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold(), re.UNICODE)


FILLER = {"oh","ooh","ah","aah","uh","la","na","啦","啊","哦","喔"}


def clean_line(text):
    words = [w for w in tokens(text) if w not in FILLER and not re.fullmatch(r"(?:[啊哦啦喔]+|(?:la){2,}|o+h+|a+h+)", w)]
    return " ".join(words)


def phrase(text):
    return len(tokens(text)) >= 2 or len(re.findall(r"[\u3400-\u9fff]", text)) >= 4


def shingles(text):
    units = list(text.replace(" ", "")) if re.search(r"[\u3400-\u9fff]", text) else tokens(text)
    return set(zip(units,units[1:])) if len(units)>1 else {(word,) for word in units}


def jaccard(a,b):
    left,right=shingles(a),shingles(b)
    return len(left & right)/len(left | right) if left or right else 0.


def iou(a,b):
    overlap=max(0,min(a[1],b[1])-max(a[0],b[0]))
    return overlap/(a[1]-a[0]+b[1]-b[0]-overlap) if overlap else 0.


def gate(lines, vocals, min_seconds=3., min_power=1e-5, min_ratio=.1):
    hop=vocals["frame_sec"]
    active=(np.asarray(vocals["vocal_power"])>=min_power)&(np.asarray(vocals["vocal_to_mix_power"])>=min_ratio)
    covered=[]
    supported=[]
    for line in lines:
        if not line["clean"]:
            continue
        start,end=line["start_sec"],line["end_sec"]
        frames=range(max(0,math.floor(start/hop)), min(len(active),math.ceil(end/hop)))
        parts=[(max(start,k*hop),min(end,(k+1)*hop)) for k in frames if active[k]]
        if parts:
            supported.append(line)
            covered.extend(parts)
    last=0.
    seconds=0.
    for a,b in sorted(covered):
        seconds+=max(0,b-max(a,last))
        last=max(last,b)
    ok=seconds>=min_seconds and any(phrase(l["clean"]) for l in supported)
    reason=None if ok else ("filler_only" if lines and all(l.get("filler_only") for l in lines)
                             else "no_supported_lyrics" if not supported else "too_short_or_fragmentary")
    return dict(enabled=ok,reason=reason,seconds=seconds,line_ids=[l["line_id"] for l in supported])


def reliable(top, confidence=.8, margin=.2):
    if len(top)<2 or top[0][1]<confidence or top[0][1]-top[1][1]<margin:
        return None
    return top[0][0]


def similarity(q,r,lexical):
    if not q or not r:
        return np.empty((len(q),len(r)))
    cos=np.array([l["vector"] for l in q]) @ np.array([l["vector"] for l in r]).T
    scores=(np.clip(cos,-1,1)+1)/2
    if lexical:
        scores=(scores+np.array([[jaccard(a["clean"],b["clean"]) for b in r] for a in q]))/2
    return scores


def local_alignment(scores, background, gap=.15):
    n,m=scores.shape
    dp=np.zeros((n+1,m+1))
    prev=np.zeros_like(dp,dtype=np.int8)
    for i in range(1,n+1):
        for j in range(1,m+1):
            options=(0.,dp[i-1,j-1]+scores[i-1,j-1]-background,dp[i-1,j]-gap,dp[i,j-1]-gap)
            k=int(np.argmax(options))
            dp[i,j],prev[i,j]=options[k],k
    i,j=np.unravel_index(np.argmax(dp),dp.shape)
    raw=float(dp[i,j])/max(n,1)
    path=[]
    while prev[i,j]:
        k=prev[i,j]
        if k==1:
            path.append((int(i-1),int(j-1)))
        i-=int(k in (1,2))
        j-=int(k in (1,3))
    return raw,path[::-1]


def ordered_lines(gold,pred,threshold=.6):
    """Maximise matched-line count then text similarity, without using timestamps."""
    from difflib import SequenceMatcher
    n,m=len(gold),len(pred)
    dp=np.zeros((n+1,m+1))
    prev=np.zeros_like(dp,dtype=np.int8)
    g=[" ".join(tokens(l["text"])) for l in gold]
    p=[" ".join(tokens(l["text"])) for l in pred]
    for i in range(1,n+1):
        for j in range(1,m+1):
            sim=SequenceMatcher(None,g[i-1],p[j-1],autojunk=False).ratio() if g[i-1] and p[j-1] else 0.
            match=dp[i-1,j-1]+min(n,m)+1+sim if sim>=threshold else -1
            choices=(match,dp[i-1,j],dp[i,j-1])
            k=int(np.argmax(choices))
            dp[i,j],prev[i,j]=choices[k],k
    i,j=n,m
    pairs=[]
    while i and j:
        k=prev[i,j]
        if k==0:
            pairs.append((i-1,j-1))
        i-=int(k!=2)
        j-=int(k!=1)
    return pairs[::-1]
