"""The grounded-answer pipeline.

    retrieve -> resolve conflicts -> abstain if the evidence cannot answer
             -> generate cited claims -> verify each citation
             -> drop or repair bad claims
             -> abstain if no surviving claim answers the question

Each stage can be switched off, which is how `evaluate.py` measures what every
stage buys.
"""
from __future__ import annotations

from typing import Callable, Sequence

from .answerability import best_score, weighted_coverage
from .corpus import Passage
from .conflict import note_applies, resolve_passages
from .generate import Answer, Claim, Generator
from .index import Index
from .text import content_tokens
from .verify import Verifier


class Pipeline:
    def __init__(self, index: Index, generator: Generator, *,
                 verifier: Verifier | None = None,
                 abstain_threshold: float | None = None,
                 abstain_score: Callable[[str, Sequence[Passage]], float] | None = None,
                 claim_ratio: float = 0.8,
                 resolve: bool = True, k: int = 6):
        self.index = index
        self.generator = generator
        self.verifier = verifier
        self.abstain_threshold = abstain_threshold
        # The evidence gate's score: evidence coverage by default, or any
        # function (question, passages) -> float, e.g. a BM25 top score.
        self.abstain_score = abstain_score or (
            lambda q, ps: best_score(self.index, q, ps))
        # When a verifier is on, the best surviving claim must cover the question
        # at least this fraction as well as the best retrieved sentence does.
        # Otherwise the claim that answered it was rejected and what is left is
        # only context. One fixed constant, not tuned.
        self.claim_ratio = claim_ratio
        self.resolve = resolve
        self.k = k

    def _primary(self, question: str, claims: list[Claim]) -> Claim | None:
        """The claim that best covers the question: the one that answers it."""
        wanted = sorted(set(content_tokens(question, keep_numbers=False)))
        best, best_score = None, -1.0
        for c in claims:
            have = set(content_tokens(c.text, keep_numbers=False))
            s = weighted_coverage(self.index, wanted, have)
            if s > best_score:
                best, best_score = c, s
        return best

    def ask(self, question: str) -> Answer:
        retrieved = self.index.search(question, self.k)
        passages, notes = (resolve_passages(retrieved, self.index)
                           if self.resolve else (retrieved, []))
        score = self.abstain_score(question, passages)
        if self.abstain_threshold is not None and score < self.abstain_threshold:
            return Answer(abstained=True, reason=f"evidence score {score:.2f} below "
                          f"{self.abstain_threshold:.2f}", score=score)
        claims = self.generator.generate(question, passages)
        dropped = []
        if self.verifier is not None:
            kept = []
            for c in claims:
                v = self.verifier.verify(c, retrieved)
                if v.kept:
                    kept.append(Claim(c.text, v.cites))
                else:
                    dropped.append(v)
            claims = kept
        if not claims:
            return Answer(abstained=True, reason="no supported claim", score=score,
                          dropped=dropped)
        primary = self._primary(question, claims)
        if self.verifier is not None:
            have = set(content_tokens(primary.text, keep_numbers=False))
            wanted = sorted(set(content_tokens(question, keep_numbers=False)))
            evidence = best_score(self.index, question, passages)
            if weighted_coverage(self.index, wanted, have) < self.claim_ratio * evidence:
                return Answer(abstained=True, score=score, dropped=dropped,
                              reason="the claim that answered the question was rejected")
        # Only the version notes that explain a claim in this answer.
        note_text = [str(n) for n in notes if any(note_applies(n, c.text) for c in claims)]
        return Answer(claims=claims, primary=primary, notes=note_text,
                      score=score, dropped=dropped)
