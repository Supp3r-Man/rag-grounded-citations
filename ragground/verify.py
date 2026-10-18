"""The citation verifier: does the cited passage actually support the claim?

For each claim and each passage it cites, find the passage sentence that covers
the most of the claim's words, then run independent checks against that sentence:

  unknown_citation   the cited id is not in the corpus
  low_support        too few of the claim's content words appear in the sentence
  number_mismatch    the claim states a number the sentence does not
  unit_mismatch      the claim states a unit (minutes, days...) the sentence does not
  negation_mismatch  the claim and the sentence differ in polarity
  entity_missing     the claim names an entity (a product) the passage never mentions
  superseded         the matched sentence is out of date: a newer version disagrees

A claim is supported if any cited passage passes every check. If none does and
`repair` is on, the verifier looks through the other retrieved passages for one
that passes and re-cites the claim ("repaired"). Otherwise the claim is rejected.

This is lexical, not an entailment model. It cannot see a meaning flip made with
words it has no rule for, and it cannot see a small unsourced addition. The
evaluation in `evaluate.py` measures both.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .conflict import find_superseding
from .corpus import Passage
from .generate import Claim
from .index import Index
from .text import (capitalised_tokens, content_tokens, entities, numbers,
                   polarity, split_sentences, units)


@dataclass(frozen=True)
class Verdict:
    claim: Claim
    status: str                       # supported | repaired | unsupported
    cites: tuple[str, ...]            # citations to use going forward
    reasons: tuple[str, ...] = ()
    coverage: float = 0.0

    @property
    def kept(self) -> bool:
        return self.status != "unsupported"


@dataclass
class Config:
    threshold: float = 0.85
    check_numbers: bool = True
    check_units: bool = True
    check_negation: bool = True
    check_entities: bool = True
    check_superseded: bool = True
    repair: bool = True
    synonyms: bool = True

    def label(self) -> str:
        return ",".join(k for k, v in self.__dict__.items()
                        if isinstance(v, bool) and v) or "lexical"


class Verifier:
    def __init__(self, index: Index, known: frozenset[str], config: Config | None = None):
        self.index = index
        self.known = known
        self.cfg = config or Config()

    def _check(self, claim: Claim, passage: Passage) -> tuple[tuple[str, ...], float]:
        cfg = self.cfg
        wanted = set(content_tokens(claim.text, synonyms=cfg.synonyms))
        best_cov, best_sent = -1.0, ""
        for sent in split_sentences(passage.text):
            have = set(content_tokens(sent, synonyms=cfg.synonyms))
            cov = len(wanted & have) / len(wanted) if wanted else 0.0
            if cov > best_cov:
                best_cov, best_sent = cov, sent
        reasons: list[str] = []
        if best_cov < cfg.threshold:
            reasons.append("low_support")
        if cfg.check_numbers and not numbers(claim.text) <= numbers(best_sent):
            reasons.append("number_mismatch")
        if cfg.check_units and not units(claim.text) <= units(best_sent):
            reasons.append("unit_mismatch")
        if cfg.check_negation and polarity(claim.text) != polarity(best_sent):
            reasons.append("negation_mismatch")
        if cfg.check_entities:
            in_passage = {t.lower() for _, t in capitalised_tokens(passage.text)}
            if not entities(claim.text, self.known) <= in_passage:
                reasons.append("entity_missing")
        if cfg.check_superseded and find_superseding(passage, best_sent, self.index):
            reasons.append("superseded")
        return tuple(reasons), max(best_cov, 0.0)

    def verify(self, claim: Claim, pool: Sequence[Passage] = ()) -> Verdict:
        passing: list[str] = []
        reasons: list[str] = []
        best_cov = 0.0
        for cid in claim.cites:
            passage = self.index.by_id.get(cid)
            if passage is None:
                reasons.append("unknown_citation")
                continue
            r, cov = self._check(claim, passage)
            best_cov = max(best_cov, cov)
            if r:
                reasons.extend(r)
            else:
                passing.append(cid)
        if passing:
            return Verdict(claim, "supported", tuple(passing), (), best_cov)
        if self.cfg.repair:
            for p in pool:
                if p.id in claim.cites:
                    continue
                r, cov = self._check(claim, p)
                if not r:
                    return Verdict(claim, "repaired", (p.id,), tuple(dict.fromkeys(reasons)), cov)
        return Verdict(claim, "unsupported", (), tuple(dict.fromkeys(reasons)), best_cov)
