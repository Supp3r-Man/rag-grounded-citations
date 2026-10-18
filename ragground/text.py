"""Tokenisation and the small linguistic toolkit shared by the retriever,
the generator and the verifier.

Everything here is deliberately plain string handling so each rule can be
read, tested and argued about. There is no model in this file.
"""
from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an the of to for in on by per at with and or be is are was were it its
    this that these those what which who how many much does do did can could will
    would should if when where there any from as up has have had into than then
    so such not no never cannot without neither nor""".split()
)

# Words that flip the meaning of a sentence. Polarity is the parity of these.
NEGATION = frozenset(
    {"not", "no", "never", "cannot", "without", "neither", "nor",
     "unsupported", "disabled"}
)

# Paraphrases the verifier knows about (applied after stemming). Anything a
# generator paraphrases that is NOT in here looks like an unsupported claim.
KNOWN_SYNONYMS = {
    "cap": "limit",
    "call": "request",
    "keep": "retain",
    "retention": "retain",
    "largest": "maximum",
    "reattempt": "retry",
}

UNIT_STEMS = frozenset(
    {"second", "minute", "hour", "day", "kilobyte", "megabyte", "gigabyte"}
)

_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
_WORD = re.compile(r"[a-z_][a-z0-9_\-]*")
_CAPITALISED = re.compile(r"[A-Za-z][A-Za-z0-9_\-]*")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def normalise(text: str) -> str:
    """Lower-case and rewrite a few forms so equal meanings tokenise equally."""
    t = text.lower()
    t = t.replace("cannot", "can not")
    t = re.sub(r"n't\b", " not", t)
    t = re.sub(r"\b(?:time out|times out|timed out|timeouts?)\b", "timeout", t)
    return t


def stem(word: str) -> str:
    """A light suffix stripper. Crude on purpose; consistent on both sides."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def raw_tokens(text: str) -> list[str]:
    return _WORD.findall(normalise(text))


def numbers(text: str) -> frozenset[str]:
    """Numbers in the text as comparable strings ('1,200' -> '1200')."""
    return frozenset(n.replace(",", "") for n in _NUMBER.findall(text))


def content_tokens(
    text: str, *, synonyms: bool = True, keep_numbers: bool = True
) -> list[str]:
    """Stemmed, stopword-free tokens, optionally with numbers kept as tokens."""
    out: list[str] = []
    norm = normalise(text)
    for w in _WORD.findall(norm):
        if w in STOPWORDS:
            continue
        s = stem(w)
        if synonyms:
            s = KNOWN_SYNONYMS.get(s, s)
        out.append(s)
    if keep_numbers:
        out.extend(n.replace(",", "") for n in _NUMBER.findall(norm))
    return out


def units(text: str) -> frozenset[str]:
    return frozenset(t for t in content_tokens(text, keep_numbers=False) if t in UNIT_STEMS)


def polarity(text: str) -> int:
    """0 for an affirmative sentence, 1 for a negated one (parity of cues)."""
    return sum(1 for w in raw_tokens(text) if w in NEGATION) % 2


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    return _SENTENCE_BREAK.split(text) if text else []


def capitalised_tokens(text: str) -> list[tuple[int, str]]:
    """(position, original token) for capitalised tokens. Position 0 is the
    sentence-initial word, whose capital tells us nothing."""
    toks = _CAPITALISED.findall(text)
    return [(i, t) for i, t in enumerate(toks) if t[0].isupper()]


def known_entities(texts: list[str]) -> frozenset[str]:
    """Capitalised tokens that appear mid-sentence anywhere in the corpus."""
    found: set[str] = set()
    for text in texts:
        for sent in split_sentences(text):
            for pos, tok in capitalised_tokens(sent):
                if pos > 0:
                    found.add(tok.lower())
    return frozenset(found)


def entities(text: str, known: frozenset[str]) -> frozenset[str]:
    """Entity-like tokens in a claim: mid-sentence capitals, plus a sentence-initial
    word only if the corpus uses it as a name elsewhere."""
    out: set[str] = set()
    for pos, tok in capitalised_tokens(text):
        if pos > 0 or tok.lower() in known:
            out.add(tok.lower())
    return frozenset(out)
