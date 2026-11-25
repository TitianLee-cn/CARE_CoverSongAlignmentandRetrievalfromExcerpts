"""Evaluate frozen Covers80 predictions against human labels."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from .cache import read, rows, write
from .lyrics import iou

CLASSES=("original","rewrite","different_language")


def ratio(n,d):
    return n/d if d else None


def evaluate(predictions,labels):
    matrix={k:Counter() for k in CLASSES}
    gates=Counter()
    languages=Counter()
    by_class=defaultdict(Counter)
    results=[]
    for work,g in labels.items():
        p=predictions[work]
        hit=p["reference_work_id"]==work
        eligible=g.get("valid_lyrics")
        if type(eligible) is bool:
            gates[("positive" if eligible else "negative")]+=1
            gates["false_open"]+=int(not eligible and p["gate"]["enabled"])
            gates["false_close"]+=int(eligible and not p["gate"]["enabled"])
        actual=g.get("cover_type")
        if actual not in CLASSES:
            continue
        predicted=p.get("cover_type",p["type_status"])
        verdict=predicted if hit else "wrong_reference"
        matrix[actual][verdict]+=1
        c=by_class[actual]
        c["queries"]+=1
        c["answered"]+=int(predicted in CLASSES)
        c["correct"]+=int(hit and predicted==actual)
        if hit and eligible:
            q,r=p.get("query_language"),p.get("reference_language")
            languages["stage1_total"]+=1
            languages["stage1_answered"]+=int(bool(q and r))
            languages["stage1_correct"]+=int(bool(q and r) and ((q!=r)==(actual=="different_language")))
            if actual!="different_language":
                languages["same_total"]+=1
                languages["same_correct"]+=int(predicted==actual)
        # A missing annotation is distinct from an annotated query with no acceptable pair.
        if g.get("pairs") is not None:
            display=p.get("display",[])
            c["display_eligible"]+=1
            c["display_queries"]+=int(bool(display))
            for item in display:
                c["displayed"]+=1
                overlaps=[iou(item["reference_sec"],ref) for pair in g["pairs"]
                          if hit and iou(item["query_sec"],pair["query_sec"])>=.5
                          for ref in pair["acceptable_reference_sec"]]
                best=max(overlaps,default=0.)
                c["iou_sum"]+=best
                c["correct_pairs"]+=int(best>=.5)
        results.append(dict(work_id=work,reference_work_id=p["reference_work_id"],truth=actual,
                            prediction=predicted,top1_correct=hit,joint_correct=hit and predicted==actual))
    fs=[]
    for key in CLASSES:
        c=by_class[key]
        tp=matrix[key][key]
        fp=sum(matrix[j][key] for j in CLASSES if j!=key)
        fn=sum(matrix[key].values())-tp
        fs.append(ratio(2*tp,2*tp+fp+fn))
        by_class[key]=dict(c,recall=ratio(tp,tp+fn),type_coverage=ratio(c["answered"],c["queries"]),
                 answered_accuracy=ratio(c["correct"],c["answered"]),
                 display_coverage=ratio(c["display_queries"],c["display_eligible"]),
                 displayed_accuracy=ratio(c["correct_pairs"],c["displayed"]),
                 displayed_mean_iou=ratio(c["iou_sum"],c["displayed"]))
    return dict(labeled_types=len(results),matrix=matrix,classes=by_class,
                macro_f1=mean(fs) if all(f is not None for f in fs) else None,
                f1_by_class=dict(zip(CLASSES,fs)),gate=dict(gates,false_open_rate=ratio(gates["false_open"],gates["negative"]),
                false_close_rate=ratio(gates["false_close"],gates["positive"])),
                stage1_accuracy=ratio(languages["stage1_correct"],languages["stage1_total"]),
                same_language_stage2_accuracy=ratio(languages["same_correct"],languages["same_total"]),
                stage_counts=languages,joint_accuracy=ratio(sum(r["joint_correct"] for r in results),len(results)),queries=results)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--predictions",type=Path,nargs="+",required=True)
    p.add_argument("--labels",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    pred={}
    for path in args.predictions:
        for row in read(path)["predictions"]:
            if row["work_id"] in pred:
                p.error("Use one selected candidate per held-out query, with no duplicate works")
            pred[row["work_id"]]=row
    raw_labels=rows(args.labels)
    labels={l["work_id"]:l for l in raw_labels}
    if len(labels)!=len(raw_labels) or set(labels)!=set(pred):
        p.error("Prediction and label work IDs must correspond one-to-one")
    write(args.out,evaluate(pred,labels))
