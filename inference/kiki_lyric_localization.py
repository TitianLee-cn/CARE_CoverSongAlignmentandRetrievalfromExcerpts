"""Lexical Kiki localization using clip-relative and whole-recording STT lines."""

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import re
import sys
from time import perf_counter_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "b_lyrics"))
from group_b.cache import load_stage
from group_b.lyrics import clean_line, gate, iou, tokens

HAN = re.compile(r"[\u3400-\u9fff]")
UNITS = re.compile(r"[\u3400-\u9fff]|(?:(?![\u3400-\u9fff])[^\W_])+")
QUERY_SECONDS = 30.0


@dataclass(frozen=True)
class Phrase:
    ids: tuple
    span: tuple
    text: str
    units: tuple
    shingles: frozenset

    @property
    def midpoint(self):
        return sum(self.span) / 2

    def long_enough(self, words, han):
        characters = sum(bool(HAN.fullmatch(unit)) for unit in self.units)
        return characters >= han or len(self.units) - characters >= words


@dataclass(frozen=True)
class Match:
    query: Phrase
    reference: Phrase
    points: float
    similarity: float

    def record(self, slope=None, intercept=None):
        return dict(
            query_line_ids=self.query.ids,
            reference_line_ids=self.reference.ids,
            query_sec=self.query.span,
            reference_sec=self.reference.span,
            query_text=self.query.text,
            reference_text=self.reference.text,
            score=self.points,
            jaccard=self.similarity,
            residual_sec=self.residual(slope, intercept) if slope is not None else None,
        )

    def residual(self, slope, intercept):
        return abs(self.reference.midpoint - (slope * self.query.midpoint + intercept))


def read_evidence(corpus, query_work, reference_work):
    """Read raw text rows and apply the existing query vocal gate without embedding filters."""
    query = corpus.queries[query_work]
    reference = corpus.refs[reference_work]
    start = perf_counter_ns()
    try:
        lines = load_stage("query", "text", query, corpus.output)["lines"]
        vocals = load_stage("query", "vocals", query, corpus.output)
    except FileNotFoundError:
        return dict(
            reason="missing_query_lyrics_or_vocals",
            cache_ns=perf_counter_ns() - start,
            gate_ns=0,
            duration=reference["duration_sec"],
        )
    cache_ns = perf_counter_ns() - start
    start = perf_counter_ns()
    decision = gate(
        lines, vocals, corpus.cfg["min_seconds"], corpus.cfg["min_power"], corpus.cfg["min_ratio"]
    )
    gate_ns = perf_counter_ns() - start
    data = dict(gate=decision, gate_ns=gate_ns, duration=reference["duration_sec"])
    if not decision["enabled"]:
        return dict(data, reason="no_effective_lyrics", cache_ns=cache_ns)
    start = perf_counter_ns()
    try:
        reference_lines = load_stage("reference", "text", reference, corpus.output)["lines"]
    except FileNotFoundError:
        return dict(
            data, reason="missing_reference_lyrics", cache_ns=cache_ns + perf_counter_ns() - start
        )
    cache_ns += perf_counter_ns() - start
    supported = set(decision["line_ids"])
    return dict(
        data,
        query_lines=[line for line in lines if line["line_id"] in supported],
        reference_lines=reference_lines,
        cache_ns=cache_ns,
        reason=None,
    )


def phrases(lines, duration):
    clean = []
    for line in lines:
        start, end = line["start_sec"], line["end_sec"]
        if (
            line.get("partial_left")
            or line.get("partial_right")
            or not 0 <= start < end <= duration + 1e-7
        ):
            continue
        normalized = clean_line(line["text"])
        if normalized:
            clean.append((line["line_id"], start, end, line["text"], normalized))
    clean.sort(key=lambda line: line[0])
    output = []
    for i, line in enumerate(clean):
        variants = [(line,)]
        if i + 1 < len(clean):
            successor = clean[i + 1]
            # Original IDs enforce adjacency even after dropping unsupported lines.
            if successor[0] == line[0] + 1 and 0 <= successor[1] - line[2] <= 2:
                variants.append((line, successor))
        for group in variants:
            normalized = " ".join(member[4] for member in group)
            units = tuple(UNITS.findall(normalized) if HAN.search(normalized) else tokens(normalized))
            output.append(
                Phrase(
                    tuple(member[0] for member in group),
                    (line[1], group[-1][2]),
                    " ".join(member[3] for member in group),
                    units,
                    frozenset(zip(units, units[1:])),
                )
            )
    return output


def sentence_matches(query, reference, near_threshold):
    for left in query:
        if not left.long_enough(2, 4):
            continue
        for right in reference:
            if len(left.ids) + len(right.ids) > 3:
                continue
            if left.units == right.units:
                yield Match(left, right, 1.0, 1.0)
            elif near_threshold is not None and left.long_enough(5, 6) and right.long_enough(5, 6):
                similarity = len(left.shingles & right.shingles) / len(left.shingles | right.shingles)
                if similarity >= near_threshold:
                    yield Match(left, right, 0.5, similarity)


def precedes(left, right):
    return (
        left.query.ids[-1] < right.query.ids[0]
        and left.reference.ids[-1] < right.reference.ids[0]
        and left.query.span[1] <= right.query.span[0]
        and left.reference.span[1] <= right.reference.span[0]
    )


def path_quality(path, slope, intercept, first=None):
    if not path:
        return (0, 0, 0, 0, ())
    start = path[0].query.midpoint if first is None else first
    return (
        -sum(m.points for m in path),
        -sum(m.points == 1 for m in path),
        start - path[-1].query.midpoint,
        sum(m.residual(slope, intercept) for m in path),
        tuple((m.query.ids, m.reference.ids) for m in path),
    )


def best_path(matches, seed0, seed1, slope, intercept, tolerance):
    seeds = (seed0, seed1)
    options = [
        m
        for m in matches
        if m.residual(slope, intercept) <= tolerance + 1e-9
        and (
            m in seeds
            or (
                m.query.units not in (seed0.query.units, seed1.query.units)
                and all(precedes(m, s) or precedes(s, m) for s in seeds)
            )
        )
    ]
    anchors = tuple(options.index(seed) for seed in seeds)

    @lru_cache(None)
    def suffix(previous, required, used, first):
        end = anchors[required] + 1 if required < 2 else len(options)
        winner = () if required == 2 else None
        for index in range(previous + 1, end):
            match = options[index]
            if match.query.units in used or (previous >= 0 and not precedes(options[previous], match)):
                continue
            next_required = required + (required < 2 and index == anchors[required])
            rest = suffix(
                index,
                next_required,
                used | {match.query.units},
                match.query.midpoint if first is None else first,
            )
            if rest is not None:
                path = (match, *rest)
                if winner is None or path_quality(path, slope, intercept, first) < path_quality(
                    winner, slope, intercept, first
                ):
                    winner = path
        return winner

    result = suffix(-1, 0, frozenset(), None)
    suffix.cache_clear()
    return result


def fit_mapping(path):
    weight = sum(match.points for match in path)
    qbar = sum(m.points * m.query.midpoint for m in path) / weight
    rbar = sum(m.points * m.reference.midpoint for m in path) / weight
    variance = sum(m.points * (m.query.midpoint - qbar) ** 2 for m in path)
    slope = (
        sum(m.points * (m.query.midpoint - qbar) * (m.reference.midpoint - rbar) for m in path) / variance
    )
    return slope, rbar - slope * qbar


def candidate(path, tolerance, duration):
    slope, intercept = fit_mapping(path)
    trimmed = tuple(m for m in path if m.residual(slope, intercept) <= tolerance + 1e-9)
    if len(trimmed) < 2:
        return None
    if trimmed != path:
        path = trimmed
        slope, intercept = fit_mapping(path)
    quality = path_quality(path, slope, intercept)
    left, right = intercept, QUERY_SECONDS * slope + intercept
    if (
        not 0.67 <= slope <= 1.50
        or any(m.residual(slope, intercept) > tolerance + 1e-9 for m in path)
        or -quality[0] < 1.5
        or -quality[1] < 1
        or -quality[2] < 10
        or path[0].query.span[0] > 10
        or QUERY_SECONDS - path[-1].query.span[1] > 10
        or left < -2
        or right > duration + 2
        or max(0, left) >= min(duration, right)
    ):
        return None
    return quality, dict(
        interval=[max(0, left), min(duration, right)],
        score=-quality[0],
        exact_count=-quality[1],
        query_midpoint_span=-quality[2],
        residual_sum_sec=quality[3],
        slope=slope,
        intercept=intercept,
        matches=[m.record(slope, intercept) for m in path],
    )


def generate(evidence, *, near_threshold, tolerance):
    """Generate distinct candidate positions without reference annotations or E2 scores."""
    if evidence["reason"]:
        return dict(reason=evidence["reason"], candidates=[], sentence_pairs=0)
    query = phrases(evidence["query_lines"], QUERY_SECONDS)
    reference = phrases(evidence["reference_lines"], evidence["duration"])
    if not reference:
        return dict(reason="no_reference_lyrics", candidates=[], sentence_pairs=0)
    matches = sorted(
        sentence_matches(query, reference, near_threshold), key=lambda m: (m.query.ids, m.reference.ids)
    )
    evidence_rows = dict(sentence_pairs=len(matches), matched_lines=[m.record() for m in matches])
    if len({m.query.units for m in matches}) < 2 or not any(m.points == 1 for m in matches):
        return dict(reason="no_two_effective_matches", candidates=[], **evidence_rows)
    candidates = []
    seen_paths = set()
    independent_matches = False
    for left, right in combinations(matches, 2):
        if left.query.units == right.query.units or not precedes(left, right):
            continue
        independent_matches = True
        if right.query.midpoint - left.query.midpoint < 10:
            continue
        slope = (right.reference.midpoint - left.reference.midpoint) / (
            right.query.midpoint - left.query.midpoint
        )
        if not 0.67 <= slope <= 1.50:
            continue
        intercept = left.reference.midpoint - slope * left.query.midpoint
        path = best_path(matches, left, right, slope, intercept, tolerance)
        if path and path not in seen_paths:
            seen_paths.add(path)
            result = candidate(path, tolerance, evidence["duration"])
            if result:
                candidates.append(result)
    distinct = []
    for _, proposal in sorted(candidates, key=lambda item: item[0]):
        if all(iou(proposal["interval"], kept["interval"]) < 0.8 for kept in distinct):
            distinct.append(proposal)
    return dict(
        reason=None if distinct else "time_checks_failed",
        candidates=distinct,
        sentence_pairs=len(matches),
    )


def fuse(generated, original_interval, e2_proposals, e2_threshold, position_margin):
    """Choose a lexical position, then apply the fixed conservative replacement rule."""
    options = generated["candidates"]
    reason = generated["reason"]
    choice = None
    if len(options) == 1:
        choice, reason = options[0], "unique"
    elif options and options[0]["score"] - options[1]["score"] >= position_margin:
        choice, reason = options[0], "text_lead"
    elif options:
        contenders = [p for p in options if options[0]["score"] - p["score"] < position_margin]
        music = [
            p
            for p in e2_proposals
            if p["source"] in {"audio", "harmony"} and p["confidence"] >= e2_threshold
        ]
        supported = [
            p for p in contenders if any(iou(p["interval"], m["interval"]) >= 0.5 for m in music)
        ]
        if len(supported) == 1:
            choice, reason = supported[0], "music_tiebreak"
        else:
            reason = "ambiguous_positions"

    overlap = iou(original_interval, choice["interval"]) if original_interval and choice else None
    final = original_interval
    source = "original_e2" if original_interval else "pending"
    action = "fallback"
    if choice:
        if original_interval is None:
            action = "fill_pending"
        elif overlap >= 0.5:
            action = "adjust_boundary"
        elif choice["exact_count"] >= 2 and reason in {"unique", "text_lead"}:
            action = "replace_conflict"
        else:
            action = "keep_original_conflict"
        if action != "keep_original_conflict":
            final, source = choice["interval"], "kiki"
    return dict(
        selected=choice,
        selection_reason=reason,
        fusion_decision=action,
        final_interval=final,
        interval_source=source,
        original_overlap=overlap,
    )
