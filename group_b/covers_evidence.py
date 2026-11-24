"""Export fold-specific lyric evidence that can be computed before A's results arrive."""
from collections import Counter

import numpy as np

from .cache import MANIFESTS, OUTPUT, load_stage, rows, write
from .lyrics import gate, similarity
from .pairs import Corpus


def export():
    corpus = Corpus({})
    folds = rows(MANIFESTS/"covers80_folds.jsonl")
    gate_rows = []
    for work, row in corpus.queries.items():
        lines = load_stage("query","kiki",row)["lines"]
        vocals = load_stage("query","vocals",row)
        decisions = {str(seconds):gate(lines,vocals,seconds,corpus.cfg["min_power"],corpus.cfg["min_ratio"])
                     for seconds in (2,3,5)}
        gate_rows.append(dict(work_id=work, query_id=row["query_id"], decisions=decisions))
    gate_counts = {s:dict(Counter("enabled" if row["decisions"][s]["enabled"]
                                  else row["decisions"][s]["reason"] for row in gate_rows))
                   for s in ("2","3","5")}
    write(OUTPUT/"scores"/"covers80-gate-evidence.json",
          dict(config=corpus.cfg, counts=gate_counts, queries=gate_rows))

    # Each different-work matrix is reused across folds. Fold fit membership determines which enter the quantile.
    scores = {}
    for work in corpus.queries:
        q, q_lang, _ = corpus.evidence(work,"query")
        if not q:
            continue
        for other in corpus.refs:
            if work == other:
                continue
            r, r_lang, _ = corpus.evidence(other,"reference")
            if r:
                semantic = similarity(q,r,False).ravel()
                lexical = similarity(q,r,True).ravel() if q_lang and q_lang == r_lang else None
                scores[work,other] = semantic, lexical

    def background(fit):
        allowed = set(fit)
        eligible = [pair for (q,r), pair in scores.items() if q in allowed and r in allowed]
        result = {}
        for i, branch in enumerate(("semantic","same_language")):
            values = [pair[i] for pair in eligible if pair[i] is not None]
            result[branch] = float(np.quantile(np.concatenate(values),.95)) if values else None
        return result

    calibrated, predictions = [], []
    for fold in folds:
        fit = fold["development_work_ids"]
        bg = background(fit)
        calibrated.append(dict(outer_fold=fold["outer_fold"], fit_work_ids=fit, background=bg,
                               inner=[dict(inner_fold=sub["inner_fold"], fit_work_ids=sub["fit_work_ids"],
                                           background=background(sub["fit_work_ids"])) for sub in fold["inner"]]))
        for work in fold["test_work_ids"]:
            predictions.append(dict(outer_fold=fold["outer_fold"],
                                    query_id=corpus.queries[work]["query_id"],
                                    reference_recording_id=corpus.refs[work]["recording_id"],
                                    semantic=corpus.score_candidate(work,work,bg,semantic_only=True),
                                    lyrics=corpus.score_candidate(work,work,bg)))
    write(OUTPUT/"scores"/"covers80-lyric-evidence.json",
          dict(config=corpus.cfg, folds=calibrated, correct_pair_predictions=predictions))
    print(f"Saved {len(predictions)} correct-pair predictions and {len(calibrated)} outer calibrations.")


if __name__ == "__main__":
    export()
