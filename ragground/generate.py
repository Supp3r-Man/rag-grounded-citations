"""Answer generation: the claim format, the generator interface, a deterministic
extractive generator, and an adapter for a real model.

A claim is one sentence plus the passage ids it cites. The pipeline only ever
sees `Claim`s, so a real model and the test doubles are interchangeable.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from .answerability import rank_sentences
from .corpus import Passage
from .index import Index


@dataclass(frozen=True)
class Claim:
    text: str
    cites: tuple[str, ...]


@dataclass
class Answer:
    claims: list[Claim] = field(default_factory=list)
    primary: Claim | None = None
    abstained: bool = False
    reason: str = ""
    notes: list[str] = field(default_factory=list)
    score: float = 0.0
    dropped: list = field(default_factory=list)  # rejected Verdicts

    def render(self) -> str:
        if self.abstained:
            return f"I can't answer that from the documents ({self.reason})."
        lines = [f"{c.text} [{', '.join(c.cites)}]" for c in self.claims]
        lines += [f"Note: {n}" for n in self.notes]
        return "\n".join(lines)


class Generator(Protocol):
    def generate(self, question: str, passages: Sequence[Passage]) -> list[Claim]: ...


# Document title lines are headings, not claims an answer should make.
HEADING = re.compile(r"^Product reference for\b")


class ExtractiveGenerator:
    """Copies the best-matching sentence verbatim and cites its passage, then adds
    one context sentence from a different passage. It cannot invent anything, so it
    is the 'clean' generator: whatever goes wrong with it is retrieval, conflicts or
    questions with no answer, not generation faults."""

    def __init__(self, index: Index, n_claims: int = 2, min_score: float = 0.3):
        self.index = index
        self.n_claims = n_claims
        self.min_score = min_score  # a context claim must still be about the question

    def generate(self, question: str, passages: Sequence[Passage]) -> list[Claim]:
        ranked = rank_sentences(self.index, question, passages)
        claims: list[Claim] = []
        used: set[str] = set()
        for score, p, s in ranked:
            if HEADING.match(s) or score < (self.min_score if claims else 0.01) or s in used \
                    or p.id in {c.cites[0] for c in claims}:
                continue
            used.add(s)
            claims.append(Claim(s, (p.id,)))
            if len(claims) == self.n_claims:
                break
        return claims


PROMPT = (
    "Answer the question using only the passages. Reply with JSON: "
    '{{"claims": [{{"text": "<one sentence>", "cites": ["<passage id>"]}}]}}. '
    "Every claim must cite the passage ids that support it. If the passages do "
    'not answer the question reply {{"claims": []}}.\n\nPassages:\n{passages}\n\n'
    "Question: {question}"
)


class ClaudeGenerator:
    """Adapter for a hosted model. `transport` is any function prompt -> text, so
    no SDK or key is needed to import this module. UNTESTED AGAINST A LIVE MODEL:
    only the prompt format and the JSON parsing are exercised, with a fake transport."""

    def __init__(self, transport: Callable[[str], str]):
        self.transport = transport

    def generate(self, question: str, passages: Sequence[Passage]) -> list[Claim]:
        block = "\n".join(f"[{p.id}] {p.text}" for p in passages)
        raw = self.transport(PROMPT.format(passages=block, question=question))
        return parse_claims(raw)


def parse_claims(raw: str) -> list[Claim]:
    """Parse the model's JSON. Malformed output yields no claims, never a crash,
    and the pipeline treats no claims as an abstention."""
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        data = json.loads(raw[start:end])
        out = []
        for item in data["claims"]:
            text = str(item["text"]).strip()
            cites = tuple(str(c) for c in item["cites"])
            if text:
                out.append(Claim(text, cites))
        return out
    except (ValueError, KeyError, TypeError):
        return []
