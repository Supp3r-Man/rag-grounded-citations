"""Can the retrieved evidence answer the question at all?

The score is the idf-weighted fraction of the question's content words that the
single best retrieved sentence covers. A question about a product's rate limit
is fully covered by the rate-limit sentence; a question about its burst limit
is covered only through the product name, and the unseen word 'burst' carries
the highest idf, so the score is low. The abstention threshold is chosen on dev
questions only (see `calibrate`).
"""
from __future__ import annotations

from typing import Sequence

from .corpus import Passage, Question
from .index import Index
from .text import content_tokens, split_sentences

GRID = [round(0.30 + 0.02 * i, 2) for i in range(36)]  # 0.30 .. 1.00


def weighted_coverage(index: Index, wanted: Sequence[str], have: set[str]) -> float:
    total = sum(index.idf(t) for t in wanted)
    if total == 0:
        return 0.0
    return sum(index.idf(t) for t in wanted if t in have) / total


def rank_sentences(index: Index, question: str, passages: Sequence[Passage]
                   ) -> list[tuple[float, Passage, str]]:
    """Every sentence in the passages, best question coverage first. Ties keep
    retrieval order, so the sort is stable and deterministic."""
    wanted = sorted(set(content_tokens(question, keep_numbers=False)))
    scored = []
    for p in passages:
        for s in split_sentences(p.text):
            have = set(content_tokens(s, keep_numbers=False))
            scored.append((weighted_coverage(index, wanted, have), p, s))
    scored.sort(key=lambda x: -x[0])
    return scored


def best_score(index: Index, question: str, passages: Sequence[Passage]) -> float:
    ranked = rank_sentences(index, question, passages)
    return ranked[0][0] if ranked else 0.0


def _balanced_accuracy(scores_pos: list[float], scores_neg: list[float], thr: float) -> float:
    """Positives are answerable (should score >= thr), negatives should score < thr."""
    tpr = sum(s >= thr for s in scores_pos) / max(1, len(scores_pos))
    tnr = sum(s < thr for s in scores_neg) / max(1, len(scores_neg))
    return (tpr + tnr) / 2


def calibrate(scores_pos: list[float], scores_neg: list[float]) -> float:
    """Threshold on the grid with the best balanced accuracy; ties go to the
    higher threshold. Inputs must come from dev questions only."""
    best, best_thr = -1.0, GRID[0]
    for thr in GRID:
        ba = _balanced_accuracy(scores_pos, scores_neg, thr)
        if ba >= best:
            best, best_thr = ba, thr
    return best_thr


def dev_scores(index: Index, questions: Sequence[Question], retrieve) -> tuple[list[float], list[float]]:
    """(answerable scores, unanswerable scores) using `retrieve(question) -> passages`."""
    pos, neg = [], []
    for q in questions:
        s = best_score(index, q.text, retrieve(q.text))
        (pos if q.answerable else neg).append(s)
    return pos, neg
