"""Combine audio and lyric evidence for locations in one confirmed recording."""
import math

import numpy as np

WINDOW_SEC = 30.
NEARBY_SEC = 2.
WEIGHT_KEYS = ("lyrics_mix","audio_temperature","agreement_bonus","far_penalty")
FEATURE_KEYS = ("start_sec","audio","a_path","anchors","timed_words","window_words","far_penalty")


def score_positions(candidates, *, lyrics_mix, audio_temperature, agreement_bonus,
                    far_penalty, use_lyrics=True):
    """Return one score per candidate in the supplied order.

    Feature scores are computed before this call. All candidate start times share
    the reference recording's clock. A missing cue is represented with zero.
    """
    parameters = (lyrics_mix,audio_temperature,agreement_bonus,far_penalty)
    if not all(math.isfinite(value) for value in parameters) or not (
            0<=lyrics_mix<=1 and audio_temperature>0 and agreement_bonus>=0 and far_penalty>=0):
        raise ValueError("Invalid position fusion parameters")
    if not candidates:
        return np.empty(0,dtype=float)

    starts,audio,path,anchors,timed,words,penalty = np.asarray(
        [[c[key] for key in FEATURE_KEYS] for c in candidates],dtype=float).T
    if not np.isfinite([starts,audio,path,anchors,timed,words,penalty]).all():
        raise ValueError("Position features must be finite")
    if (starts<0).any() or (penalty>0).any() or not all(
            ((v>=0)&(v<=1)).all() for v in (path,anchors,timed,words)):
        raise ValueError("Position features are outside their supported ranges")

    scores_audio=audio+.04*path
    scores_lyrics=.55*anchors+.30*timed+.15*words if use_lyrics else np.zeros(len(starts))
    nearby=np.abs(starts[:,None]-starts[None,:])<=NEARBY_SEC
    local_audio=np.max(np.where(nearby,scores_audio[None,:],-np.inf),axis=1)
    local_lyrics=np.max(np.where(nearby,scores_lyrics[None,:],-np.inf),axis=1)
    strength=min(1.,float(scores_lyrics.max())/.60)
    lyric=np.exp((local_lyrics-scores_lyrics.max())/.12)
    acoustic=np.exp((local_audio-scores_audio.max())/audio_temperature)
    balance=lyrics_mix*strength
    values=(1-balance)*acoustic+balance*lyric+agreement_bonus*strength*np.minimum(acoustic,lyric)
    values+=.025*((1-balance)*scores_audio+balance*scores_lyrics)+far_penalty*penalty
    return values


def select_reference_interval(candidates, reference_duration_sec, weights, *,
                              start_bias_sec=0., use_lyrics=True):
    """Choose a 30-second proposal after the caller has confirmed the recording."""
    if not math.isfinite(reference_duration_sec) or reference_duration_sec<=0 or not math.isfinite(start_bias_sec):
        raise ValueError("Invalid reference duration or position bias")
    scores=score_positions(candidates,**weights,use_lyrics=use_lyrics)
    if not len(scores):
        return None
    latest=max(0.,reference_duration_sec-WINDOW_SEC)
    if any(candidate["start_sec"]>latest+1e-8 for candidate in candidates):
        raise ValueError("A candidate extends past the reference recording")
    index=int(scores.argmax())
    start=max(0.,min(latest,candidates[index]["start_sec"]+start_bias_sec))
    return dict(reference_interval=[start,min(reference_duration_sec,start+WINDOW_SEC)],
                candidate_index=index,score=float(scores[index]))
