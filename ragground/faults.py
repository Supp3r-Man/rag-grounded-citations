"""A fault model for generators: the ways a real model's cited answer goes wrong.

`NoisyGenerator` wraps a clean generator and corrupts its claims with seeded,
labelled faults, so the verifier can be scored against known ground truth. This
is a model of failure modes I chose, not measured model behaviour: the fault mix
and rates are knobs (see WEIGHTS), and results depend on them.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Sequence

from .corpus import FEATURES, NUMERIC, PRODUCTS, Passage
from .generate import Claim, Generator
from .text import split_sentences

# Faults whose claim text is still true (only the citation, or the wording, changed).
CONTENT_OK = {"none", "paraphrase_known", "paraphrase_unseen", "reword",
              "wrong_citation", "fabricated_citation"}

WEIGHTS = {
    "none": 0.35,
    "paraphrase_known": 0.06, "paraphrase_unseen": 0.06, "reword": 0.10,
    "wrong_citation": 0.06, "fabricated_citation": 0.04,
    "number_altered": 0.07, "unit_swap": 0.07,
    "negation_dropped": 0.04, "antonym_flip": 0.03,
    "entity_swap": 0.04, "subtle_addition": 0.06,
    "stale_claim": 0.04,
}
EXTRA_CLAIM_RATE = 0.10  # chance of an additional wholly unsourced sentence

PARAPHRASE_KNOWN = [("limit", "cap"), ("requests", "calls"), ("retains", "keeps"),
                    ("maximum", "largest"), ("retries", "reattempts")]
PARAPHRASE_UNSEEN = [("limit", "ceiling"), ("supports", "offers"),
                     ("retains", "stores"), ("accepted", "permitted"),
                     ("failed", "unsuccessful"), ("enforces", "applies")]
UNIT_SWAPS = [("minute", "hour"), ("seconds", "minutes"), ("days", "hours"),
              ("megabytes", "kilobytes")]
ADDITIONS = [" for all regions", " across every workspace",
             " for all customers", " in all environments"]
# Whole-sentence rewordings (faithful). Some stay inside the verifier's synonym
# table and word stems, some do not; that is the point.
REWORD = {
    "rate_limit": "On the {P}, each API key is limited to {v} requests per minute.",
    "timeout": "A request to the {P} times out once {v} seconds have passed.",
    "retention": "Audit logs are kept on the {P} for {v} days.",
    "max_payload": "The {P} accepts payloads of up to {v} megabytes.",
    "retries": "Failed calls are retried by the {P} client up to {v} times.",
    "page_size": "By default, list endpoints on the {P} return {v} items per page.",
}
REWORD_FEATURE = ("{F} is supported on the {P}.", "{F} is not available on the {P}.")


def _reword(text: str) -> str | None:
    for attr, alt in REWORD.items():
        pat = re.escape(NUMERIC[attr][0]).replace(r"\{P\}", "(?P<P>.+)").replace(r"\{v\}", r"(?P<v>\d+)")
        m = re.fullmatch(pat, text)
        if m:
            return alt.format(P=m.group("P"), v=m.group("v"))
    m = re.fullmatch(r"The (?P<P>.+?) (?P<neg>does not support|supports) (?P<F>.+)\.", text)
    if m:
        alt = REWORD_FEATURE[1 if m.group("neg") == "does not support" else 0]
        f = m.group("F")
        return alt.format(F=f[0].upper() + f[1:], P=m.group("P"))
    return None


UNSOURCED = ["This setting is enforced separately in every region.",
             "Enterprise customers can request a higher value.",
             "The value was last reviewed by the platform team.",
             "Changes take effect for new workspaces first."]


@dataclass(frozen=True)
class LabeledClaim:
    claim: Claim
    fault: str
    origin: str | None  # the passage sentence this claim was derived from


def _swap_word(text: str, a: str, b: str) -> str | None:
    new, n = re.subn(rf"\b{re.escape(a)}\b", b, text, count=1)
    return new if n else None


def apply_fault(fault: str, text: str, rng: random.Random) -> str | None:
    """The corrupted text, or None if this fault cannot apply to this sentence."""
    if fault == "none":
        return text
    if fault == "reword":
        return _reword(text)
    if fault.startswith("paraphrase"):
        pool = PARAPHRASE_KNOWN if fault.endswith("known") else PARAPHRASE_UNSEEN
        pairs = pool[:]
        rng.shuffle(pairs)
        for a, b in pairs:
            new = _swap_word(text, a, b)
            if new:
                return new
        return None
    if fault == "number_altered":
        nums = list(re.finditer(r"\d+", text))
        if not nums:
            return None
        m = rng.choice(nums)
        v = int(m.group())
        new = rng.choice([x for x in (v * 2, max(1, v // 2), v + 7) if x != v])
        return text[:m.start()] + str(new) + text[m.end():]
    if fault == "unit_swap":
        for a, b in UNIT_SWAPS:
            new = _swap_word(text, a, b)
            if new:
                return new
        return None
    if fault == "negation_dropped":
        if "does not support" in text:
            return text.replace("does not support", "supports")
        if " supports " in text:
            return text.replace(" supports ", " does not support ")
        return None
    if fault == "antonym_flip":
        if "does not support" in text:
            return text.replace("does not support", "fully handles")
        if " supports " in text:
            return text.replace(" supports ", " lacks support for ")
        return None
    if fault == "entity_swap":
        for p in PRODUCTS:
            if p in text:
                other = rng.choice([q for q in PRODUCTS if q != p])
                return text.replace(p, other)
        return None
    if fault == "subtle_addition":
        return text.rstrip(".") + rng.choice(ADDITIONS) + "."
    raise ValueError(f"unknown fault {fault!r}")


class NoisyGenerator:
    """Wraps a clean generator and corrupts each claim with a seeded fault."""

    def __init__(self, base: Generator, seed: int, weights: dict[str, float] | None = None):
        self.base = base
        self.seed = seed
        self.weights = weights or WEIGHTS

    def _rng(self, question: str, salt: str) -> random.Random:
        h = hashlib.sha256(f"{self.seed}|{question}|{salt}".encode()).hexdigest()
        return random.Random(int(h[:16], 16))

    def generate_labeled(self, question: str, passages: Sequence[Passage],
                         pool: Sequence[Passage] | None = None,
                         stale: Sequence[tuple[str, str]] = ()) -> list[LabeledClaim]:
        """`stale` is (sentence, passage id) pairs for superseded sentences, the
        raw material for the stale_claim fault: a claim that is true of an older
        version and cites it."""
        pool = list(pool if pool is not None else passages)
        clean = self.base.generate(question, passages)
        out: list[LabeledClaim] = []
        for i, c in enumerate(clean):
            rng = self._rng(question, f"claim{i}")
            # Draw a fault; if it cannot apply to this sentence (no number to
            # alter, no unit to swap...) draw again from the rest, so the mix of
            # faults that actually happen follows the weights.
            remaining = dict(self.weights)
            while True:
                fault = rng.choices(list(remaining), list(remaining.values()))[0]
                text, cites = c.text, c.cites
                if fault == "wrong_citation":
                    others = [p for p in pool if c.text not in split_sentences(p.text)]
                    if others:
                        cites = (rng.choice(others).id,)
                        break
                elif fault == "fabricated_citation":
                    cites = (f"{rng.choice(pool).doc_id.split('@')[0]}@9.9#7",)
                    break
                elif fault == "stale_claim":
                    if stale:
                        text, pid = rng.choice(list(stale))
                        cites = (pid,)
                        break
                else:
                    new = apply_fault(fault, text, rng)
                    if new is not None:
                        text = new
                        break
                del remaining[fault]
            out.append(LabeledClaim(Claim(text, cites), fault, c.text))
        if clean and self._rng(question, "extra").random() < EXTRA_CLAIM_RATE:
            rng = self._rng(question, "extra-pick")
            out.append(LabeledClaim(Claim(rng.choice(UNSOURCED), clean[0].cites),
                                    "unsupported_addition", None))
        return out

    def generate(self, question: str, passages: Sequence[Passage]) -> list[Claim]:
        return [lc.claim for lc in self.generate_labeled(question, passages)]
