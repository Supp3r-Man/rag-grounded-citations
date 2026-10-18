"""Conflicting evidence: prefer the newer version of a document, and say so.

A sentence is superseded when the newest version of the same document family
has a counterpart sentence (same words once numbers and negation are removed)
whose numbers or polarity differ. If the newest counterpart agrees with the
sentence, the sentence is current even if a middle version disagreed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

from .corpus import Passage
from .index import Index
from .text import NEGATION, content_tokens, numbers, polarity, split_sentences

JACCARD_MIN = 0.8


@dataclass(frozen=True)
class Note:
    old_passage: str
    old: str
    new_passage: str
    new: str
    old_version: str
    new_version: str
    new_date: str

    def __str__(self) -> str:
        return (f"Superseded: version {self.old_version} said \"{self.old}\"; "
                f"version {self.new_version} ({self.new_date}) says \"{self.new}\" "
                f"[{self.new_passage}]")


def fact_key(sentence: str) -> frozenset[str]:
    """What a sentence is about: its words without numbers and negation cues."""
    return frozenset(t for t in content_tokens(sentence, keep_numbers=False)
                     if t not in NEGATION)


def _jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def note_applies(note: Note, claim_text: str) -> bool:
    """Is this note about the fact the claim states? Used so an answer only
    carries the version notes that explain it, not every superseded sentence
    that happened to be retrieved."""
    return _jaccard(fact_key(note.new), fact_key(claim_text)) >= JACCARD_MIN


_VERSION = re.compile(r"\bversion \d+(?:\.\d+)*", re.IGNORECASE)


def differs(a: str, b: str) -> bool:
    """Do two counterpart sentences disagree? A document's own version number
    (the title line) is not a fact that changed, so it is ignored."""
    a, b = _VERSION.sub("", a), _VERSION.sub("", b)
    return numbers(a) != numbers(b) or polarity(a) != polarity(b)


def find_superseding(passage: Passage, sentence: str,
                     index: Index) -> tuple[Passage, str] | None:
    """The newest counterpart of `sentence` if it disagrees with it, else None."""
    key = fact_key(sentence)
    for newer in index.newer_versions(passage):
        hits = [s2 for s2 in split_sentences(newer.text)
                if _jaccard(key, fact_key(s2)) >= JACCARD_MIN]
        if hits:
            s2 = hits[0]
            return (newer, s2) if differs(sentence, s2) else None
    return None


def resolve_passages(passages: list[Passage],
                     index: Index) -> tuple[list[Passage], list[Note]]:
    """Drop superseded sentences from retrieved passages and pull in the passage
    that supersedes them. Returns trimmed passages and one note per replacement."""
    out: list[Passage] = []
    notes: list[Note] = []
    extra: dict[str, Passage] = {}
    for p in passages:
        keep = []
        for s in split_sentences(p.text):
            sup = find_superseding(p, s, index)
            if sup is None:
                keep.append(s)
                continue
            newer, news = sup
            notes.append(Note(p.id, s, newer.id, news, p.version, newer.version,
                              newer.date))
            extra[newer.id] = newer
        if keep:
            out.append(replace(p, text=" ".join(keep)))
    have = {p.id for p in out}
    for pid, newer in extra.items():
        if pid not in have:
            out.append(newer)
    return out, notes
