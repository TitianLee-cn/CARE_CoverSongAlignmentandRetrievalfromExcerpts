"""Select a lyric-duration threshold on the four inner validation folds."""
import argparse
from pathlib import Path

from .cache import MANIFESTS, OUTPUT, load_stage, rows, write
from .lyrics import gate
from .pairs import DEFAULTS


def select(outer, labels, output, min_power, min_ratio):
    fold=next(f for f in rows(MANIFESTS/"covers80_folds.jsonl") if f["outer_fold"]==outer)
    records={r["work_id"]:r for r in rows(MANIFESTS/"covers80_queries.jsonl")}
    counts={s:[] for s in (2,3,5)}
    for inner in fold["inner"]:
        held=inner["validation_work_ids"]
        if set(held)&set(fold["test_work_ids"]):
            raise ValueError("Outer test work in validation")
        evidence=[]
        for work in held:
            label=labels.get(work,{}).get("valid_lyrics")
            if type(label) is not bool:
                raise ValueError(f"Missing human gate label: {work}")
            row=records[work]
            evidence.append((label,load_stage("query","kiki",row,output)["lines"],
                              load_stage("query","vocals",row,output)))
        for s in counts:
            opened=closed=pos=0
            for label,lines,vocals in evidence:
                enabled=gate(lines,vocals,s,min_power,min_ratio)["enabled"]
                pos+=label
                opened+=int(enabled and not label)
                closed+=int(not enabled and label)
            counts[s].append(dict(inner=inner["inner_fold"],positive=pos,negative=len(held)-pos,
                                  false_open=opened,false_close=closed,
                                  false_open_rate=opened/(len(held)-pos) if len(held)>pos else None,
                                  false_close_rate=closed/pos if pos else None))
    # Equal error cost, prespecified 3 s wins ties.
    winner=min((3,2,5),key=lambda s:sum(r["false_open"]+r["false_close"] for r in counts[s]))
    return dict(outer_fold=outer,development_work_ids=fold["development_work_ids"],counts=counts,
                config={**DEFAULTS,"min_seconds":winner,"min_power":min_power,"min_ratio":min_ratio})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--outer",type=int,choices=range(1,6),required=True)
    p.add_argument("--labels",type=Path,required=True)
    p.add_argument("--min-power",type=float,default=DEFAULTS["min_power"])
    p.add_argument("--min-ratio",type=float,default=DEFAULTS["min_ratio"])
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    data=rows(args.labels)
    labels={r["work_id"]:r for r in data}
    if len(data)!=len(labels):
        p.error("Duplicate work IDs")
    write(args.out,select(args.outer,labels,OUTPUT,args.min_power,args.min_ratio))
