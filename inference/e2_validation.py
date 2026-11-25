"""Paired evaluation of whole-query localization, conditional on a correctly confirmed song."""

from itertools import combinations, product

from align_cover_pairs import span_iou
from common import write_json, write_jsonl
import e2


def label_iou(span, truth):
    return (
        max((span_iou(span, [r["start_sec"], r["end_sec"]]) for r in truth), default=0.0) if span else 0.0
    )


def outcome(options, interval, truth):
    ious = [label_iou(p["interval"], truth) for p in options]
    best = max(ious, default=0.0)
    return dict(
        interval=interval,
        emitted=interval is not None,
        iou=label_iou(interval, truth),
        proposal_hit=best >= 0.5,
        best_iou=best,
        best_good_confidence=max(
            (p["confidence"] for p, i in zip(options, ious) if i >= 0.5), default=None
        ),
    )


def competing(options):
    # Compare primary positions across sources, before confidence filtering.
    first = {}
    for p in options:
        first.setdefault(p["source"], p["interval"])
    return any(span_iou(a, b) < 0.5 for a, b in combinations(first.values(), 2))


def diagnostic(query, old, new, old_span, new_span, truth, lyrics_enabled, rule, *, old_rule=None):
    if [(p["source"], p["interval"]) for p in old] != [(p["source"], p["interval"]) for p in new]:
        raise ValueError(f"Proposal set changed: {query}")
    a, b = outcome(old, old_span, truth), outcome(new, new_span, truth)
    if b["iou"] >= 0.5:
        stage = "hit"
    elif not b["proposal_hit"]:
        stage = "proposal_miss"
    elif b["best_good_confidence"] < rule["threshold"]:
        stage = "confidence_below_threshold"
    else:
        stage = "selection"
    return dict(
        query_id=query,
        lyrics_enabled=lyrics_enabled,
        conflicting=competing(new),
        any_conflict=any(span_iou(p["interval"], q["interval"]) < 0.5 for p, q in combinations(new, 2)),
        old=a,
        new=b,
        new_outcome=stage,
        old_at_new_rule=outcome(old, e2.select(old, **rule), truth),
        new_at_old_rule=outcome(new, e2.select(new, **(old_rule or e2.LEGACY_RULE)), truth),
    )


def summarize(rows):
    report = {}
    for group, members in (
        ("all", rows),
        ("no_lyrics", [r for r in rows if not r["lyrics_enabled"]]),
        ("any_conflict", [r for r in rows if r["any_conflict"]]),
        ("conflicting", [r for r in rows if r["conflicting"]]),
    ):
        group_report = {}
        for version in ("old", "new", "old_at_new_rule", "new_at_old_rule"):
            values = [r[version] for r in members]
            emitted = sum(v["emitted"] for v in values)
            hits = sum(v["iou"] >= 0.5 for v in values)
            group_report[version] = dict(
                denominator=len(values),
                proposal_hits=sum(v["proposal_hit"] for v in values),
                emitted=emitted,
                hits=hits,
                precision_denominator=emitted,
            )
        group_report["new_outcomes"] = {
            stage: sum(r["new_outcome"] == stage for r in members)
            for stage in ("hit", "proposal_miss", "confidence_below_threshold", "selection")
        }
        report[group] = group_report
    return report


def select_rule(eligible, *, options_key="new"):
    if not eligible:
        raise ValueError("No correctly confirmed development queries with interval references")
    sweep = []
    for threshold, margin in product((0.30, 0.40, 0.50, 0.60, 0.70), (0.0, 0.10, 0.15)):
        rule = dict(threshold=threshold, margin=margin)
        spans = [e2.select(r[options_key], **rule) for r in eligible]
        emitted = sum(s is not None for s in spans)
        hits = sum(label_iou(s, r["truth"]) >= 0.5 for s, r in zip(spans, eligible))
        sweep.append(dict(rule=rule, emitted=emitted, hits=hits, f1=2 * hits / (len(eligible) + emitted)))
    best = max(sweep, key=lambda s: (s["f1"], s["hits"], s["rule"]["threshold"], s["rule"]["margin"]))
    return best, sweep


def development(rows, dest, *, tune_old=False):
    eligible = [r for r in rows if r["confirmed"] and r["winner"] == r["work_id"] and r["truth"]]
    best, sweep = select_rule(eligible)
    old_best, old_sweep = (
        select_rule(eligible, options_key="old") if tune_old else ({"rule": e2.LEGACY_RULE}, [])
    )
    rule = best["rule"]
    old_rule = old_best["rule"]
    scored = [
        diagnostic(
            r["query_id"],
            r["old"],
            r["new"],
            e2.select(r["old"], **old_rule),
            e2.select(r["new"], **rule),
            r["truth"],
            r["lyrics_enabled"],
            rule,
            old_rule=old_rule,
        )
        for r in eligible
    ]
    write_jsonl(dest / "development_oof.jsonl", rows)
    write_jsonl(dest / "development_comparison.jsonl", scored)
    write_json(
        dest / "development_metrics.json",
        dict(
            queries=len(rows),
            selection=best,
            sweep=sweep,
            legacy_selection=old_best,
            legacy_sweep=old_sweep,
            groups=summarize(scored),
        ),
    )
    return (rule, old_rule) if tune_old else rule
