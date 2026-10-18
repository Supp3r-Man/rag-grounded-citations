"""Measurement: the verifier against labelled faults, abstention, and end to end.

Rules this module follows:
  * Every threshold (verifier support, abstention) is chosen on dev questions
    and dev noise seeds, then scored on held-out products and held-out seeds.
  * Each verifier configuration is calibrated separately, so the lexical-only
    baseline gets its own best threshold and is not a strawman.
  * Confidence intervals resample questions, not claims.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .answerability import GRID, best_score, calibrate, dev_scores
from .conflict import resolve_passages
from .corpus import Corpus, Question, build_corpus, fact_sentence
from .faults import CONTENT_OK, LabeledClaim, NoisyGenerator
from .generate import Claim, ExtractiveGenerator
from .index import Index
from .pipeline import Pipeline
from .stats import ratio_ci
from .text import content_tokens, known_entities, numbers, polarity, split_sentences
from .verify import Config, Verifier

K = 6
DEV_SEEDS = [101, 102, 103, 104, 105]
HELDOUT_SEEDS = [201, 202, 203, 204, 205]
VERIFIER_GRID = [round(0.50 + 0.05 * i, 2) for i in range(11)]


@dataclass
class World:
    corpus: Corpus
    index: Index
    known: frozenset[str]


def build_world(seed: int = 7) -> World:
    corpus = build_corpus(seed)
    return World(corpus, Index(corpus.passages),
                 known_entities([p.text for p in corpus.passages]))


# ---------------------------------------------------------------- ground truth

def gold_sentence(q: Question) -> str:
    """The sentence a correct answer to this question rests on (latest version)."""
    if q.gold.value is not None:
        return fact_sentence(q.attr, q.product, int(q.gold.value))
    return fact_sentence(q.attr, q.product, q.gold.polarity == 0)


def is_correct(text: str, q: Question) -> bool:
    """Does this claim state the current gold fact about the right attribute and
    product? Stale or wrong values, the wrong attribute, the wrong product, and
    the wrong polarity all fail."""
    g = q.gold
    if not set(g.required) <= set(content_tokens(text, keep_numbers=False)):
        return False
    if g.value is not None:
        return g.value in numbers(text)
    return polarity(text) == g.polarity


def mentions_stale(text: str, q: Question) -> bool:
    return q.gold.value is not None and any(s in numbers(text) for s in q.gold.stale) \
        and q.gold.value not in numbers(text)


def truly_supports(world: World, cites, origin: str | None) -> bool:
    if origin is None:
        return False
    for c in cites:
        p = world.index.by_id.get(c)
        if p is not None and origin in split_sentences(p.text):
            return True
    return False


def claim_is_bad(world: World, lc: LabeledClaim, cites) -> bool:
    return lc.fault not in CONTENT_OK or not truly_supports(world, cites, lc.origin)


# ---------------------------------------------------------- labelled claim sets

@dataclass
class Item:
    q: Question
    pool: list
    labeled: list[LabeledClaim]


def collect(world: World, questions, seeds) -> list[Item]:
    base = ExtractiveGenerator(world.index)
    items = []
    for q in questions:
        if not q.answerable:
            continue
        retrieved = world.index.search(q.text, K)
        passages, notes = resolve_passages(retrieved, world.index)
        stale = [(n.old, n.old_passage) for n in notes]
        for s in seeds:
            labeled = NoisyGenerator(base, s).generate_labeled(
                q.text, passages, pool=retrieved, stale=stale)
            items.append(Item(q, retrieved, labeled))
    return items


@dataclass
class VerifierResult:
    name: str
    threshold: float
    n: int = 0
    bad_before: int = 0
    kept: int = 0
    bad_after: int = 0
    faithful: int = 0
    false_reject: int = 0
    repaired: int = 0
    prec_before: float = 0.0
    prec_after: float = 0.0
    supported_before: float = 0.0
    per_fault: dict = field(default_factory=dict)  # fault -> Counter
    groups: dict = field(default_factory=lambda: defaultdict(lambda: [0, 0, 0, 0]))

    def rate(self, num: int, den: int) -> float:
        return num / den if den else 0.0

    @property
    def unsupported_before(self) -> float:
        return self.rate(self.bad_before, self.n)

    @property
    def unsupported_after(self) -> float:
        return self.rate(self.bad_after, self.kept)

    @property
    def catch_rate(self) -> float:
        return self.rate(self.bad_before - self.bad_after, self.bad_before)

    @property
    def false_reject_rate(self) -> float:
        return self.rate(self.false_reject, self.faithful)

    def after_ci(self):
        gs = [(g[1], g[0]) for g in self.groups.values()]
        return ratio_ci(gs)


def score_verifier(world: World, cfg: Config, items: list[Item], name: str = "") -> VerifierResult:
    ver = Verifier(world.index, world.known, cfg)
    r = VerifierResult(name or cfg.label(), cfg.threshold)
    pairs_before = pairs_before_ok = pairs_after = pairs_after_ok = 0
    sup_before = 0
    for it in items:
        g = r.groups[it.q.id]
        for lc in it.labeled:
            v = ver.verify(lc.claim, it.pool)
            bad = claim_is_bad(world, lc, lc.claim.cites)
            r.n += 1
            g[0] += 1
            pc = r.per_fault.setdefault(lc.fault, Counter())
            pc["n"] += 1
            for c in lc.claim.cites:
                pairs_before += 1
                pairs_before_ok += truly_supports(world, (c,), lc.origin)
            sup_before += truly_supports(world, lc.claim.cites, lc.origin)
            if bad:
                r.bad_before += 1
            else:
                r.faithful += 1
            if v.kept:
                r.kept += 1
                g[2] += 1
                r.repaired += v.status == "repaired"
                still_bad = claim_is_bad(world, lc, v.cites)
                for c in v.cites:
                    pairs_after += 1
                    pairs_after_ok += truly_supports(world, (c,), lc.origin)
                if still_bad:
                    r.bad_after += 1
                    g[1] += 1
                    pc["slipped"] += 1
                if bad and not still_bad:
                    pc["repaired"] += 1
                if not bad:
                    pc["kept_ok"] += 1
            else:
                if bad:
                    pc["dropped"] += 1
                else:
                    r.false_reject += 1
                    pc["false_reject"] += 1
    r.prec_before = pairs_before_ok / pairs_before if pairs_before else 0.0
    r.prec_after = pairs_after_ok / pairs_after if pairs_after else 0.0
    r.supported_before = sup_before / r.n if r.n else 0.0
    return r


def calibrate_verifier(world: World, base: Config, dev_items: list[Item]) -> Config:
    """Support threshold with the best balanced accuracy on dev claims: the share
    of bad claims neutralised plus the share of faithful claims kept, halved."""
    best, best_cfg = -1.0, None
    for thr in VERIFIER_GRID:
        cfg = Config(**{**base.__dict__, "threshold": thr})
        r = score_verifier(world, cfg, dev_items)
        neutralised = r.bad_before - r.bad_after
        ba = (r.rate(neutralised, r.bad_before) + r.rate(r.faithful - r.false_reject, r.faithful)) / 2
        if ba >= best:
            best, best_cfg = ba, cfg
    return best_cfg


CONFIGS = {
    "citation id exists only": Config(threshold=0.0, check_numbers=False, check_units=False,
                                      check_negation=False, check_entities=False,
                                      check_superseded=False, repair=False),
    "lexical overlap only": Config(check_numbers=False, check_units=False, check_negation=False,
                                   check_entities=False, check_superseded=False, repair=False),
    "+ numbers and units": Config(check_negation=False, check_entities=False,
                                  check_superseded=False, repair=False),
    "+ negation": Config(check_entities=False, check_superseded=False, repair=False),
    "+ entities": Config(check_superseded=False, repair=False),
    "+ superseded (full)": Config(repair=False),
    "full + citation repair": Config(),
}


def verifier_table(world: World, dev_items, held_items) -> list[VerifierResult]:
    out = []
    for name, base in CONFIGS.items():
        cfg = base if name == "citation id exists only" else calibrate_verifier(world, base, dev_items)
        out.append(score_verifier(world, cfg, held_items, name))
    return out


# ------------------------------------------------------------------ abstention

def abstain_scores(world: World, questions):
    def retrieve(text):
        return resolve_passages(world.index.search(text, K), world.index)[0]
    return dev_scores(world.index, questions, retrieve)


def bm25_scores(world: World, questions):
    pos, neg = [], []
    for q in questions:
        top = world.index.scored(q.text, 1)
        s = top[0][0] if top else 0.0
        (pos if q.answerable else neg).append(s)
    return pos, neg


def bm25_threshold(pos, neg) -> float:
    cands = sorted(set(pos + neg))
    best, best_t = -1.0, 0.0
    for t in cands:
        tpr = sum(s >= t for s in pos) / len(pos)
        tnr = sum(s < t for s in neg) / len(neg)
        if (tpr + tnr) / 2 >= best:
            best, best_t = (tpr + tnr) / 2, t
    return best_t


@dataclass
class AbstainResult:
    name: str
    threshold: float | None
    abstained_unanswerable: int
    abstained_answerable: int
    n_unanswerable: int
    n_answerable: int

    @property
    def precision(self) -> float:
        d = self.abstained_unanswerable + self.abstained_answerable
        return self.abstained_unanswerable / d if d else 0.0

    @property
    def recall(self) -> float:
        return self.abstained_unanswerable / self.n_unanswerable if self.n_unanswerable else 0.0

    @property
    def answerable_lost(self) -> float:
        return self.abstained_answerable / self.n_answerable if self.n_answerable else 0.0


def _dev_ba(pos, neg, thr) -> float:
    return (sum(s >= thr for s in pos) / len(pos) + sum(s < thr for s in neg) / len(neg)) / 2


@dataclass
class Gate:
    """The evidence gate chosen on dev: which score, which threshold."""
    name: str
    threshold: float
    score_fn: "callable"
    cov_threshold: float   # coverage threshold, also used for the post-verifier claim check
    dev_ba: dict


def select_gate(world: World) -> Gate:
    """Calibrate both candidate scores on dev and keep the one with the better
    dev balanced accuracy. Held-out questions play no part."""
    dev = world.corpus.dev
    cpos, cneg = abstain_scores(world, dev)
    bpos, bneg = bm25_scores(world, dev)
    cov_thr, bm_thr = calibrate(cpos, cneg), bm25_threshold(bpos, bneg)
    dev_ba = {"evidence coverage": _dev_ba(cpos, cneg, cov_thr),
              "top BM25 score": _dev_ba(bpos, bneg, bm_thr)}

    def bm25_fn(q, passages):
        top = world.index.scored(q, 1)
        return top[0][0] if top else 0.0

    def cov_fn(q, passages):
        return best_score(world.index, q, passages)

    if dev_ba["top BM25 score"] > dev_ba["evidence coverage"]:
        return Gate("top BM25 score", bm_thr, bm25_fn, cov_thr, dev_ba)
    return Gate("evidence coverage", cov_thr, cov_fn, cov_thr, dev_ba)


def abstention_table(world: World) -> tuple[list[AbstainResult], Gate]:
    held = world.corpus.heldout
    gate = select_gate(world)
    bm_thr = bm25_threshold(*bm25_scores(world, world.corpus.dev))
    cov_pos, cov_neg = abstain_scores(world, held)
    bm_pos, bm_neg = bm25_scores(world, held)
    rows = [
        AbstainResult("never abstain", None, 0, 0, len(cov_neg), len(cov_pos)),
        AbstainResult("top BM25 score threshold", bm_thr,
                      sum(s < bm_thr for s in bm_neg), sum(s < bm_thr for s in bm_pos),
                      len(bm_neg), len(bm_pos)),
        AbstainResult("evidence coverage threshold", gate.cov_threshold,
                      sum(s < gate.cov_threshold for s in cov_neg),
                      sum(s < gate.cov_threshold for s in cov_pos),
                      len(cov_neg), len(cov_pos)),
    ]
    return rows, gate


# ----------------------------------------------------------------- end to end

class Recorder(NoisyGenerator):
    """NoisyGenerator that remembers each claim's label so final claims can be scored."""

    def __init__(self, base, seed):
        super().__init__(base, seed)
        self.labels: dict[str, LabeledClaim] = {}

    def generate(self, question, passages):
        lcs = self.generate_labeled(question, passages)
        self.labels = {lc.claim.text: lc for lc in lcs}
        return [lc.claim for lc in lcs]


REWORDINGS = ("reword", "paraphrase_known", "paraphrase_unseen")


def judged_text(text: str, rec) -> str:
    """A faithful rewording changes the wording, not the fact, and my string check
    for the right attribute cannot read every rewording. So score the sentence it
    was reworded from. Faults that change the fact are scored as they are."""
    lc = rec.labels.get(text) if rec is not None else None
    return lc.origin if lc is not None and lc.fault in REWORDINGS and lc.origin else text


@dataclass
class E2E:
    name: str
    per_q: dict = field(default_factory=lambda: defaultdict(Counter))

    def ratio(self, num_key: str, den_keys, kinds=None):
        gs = []
        for qid, c in self.per_q.items():
            if kinds is not None and c["_kind"] not in kinds:
                continue
            gs.append((c[num_key], sum(c[k] for k in den_keys)))
        return ratio_ci(gs)


def run_e2e(world: World, name: str, make: "callable", questions, seeds=(None,)) -> E2E:
    """`make(seed)` returns (pipeline, recorder_or_None). Outcomes are tallied per
    question and averaged over seeds by summing (ratios are ratios of sums)."""
    res = E2E(name)
    for q in questions:
        c = res.per_q[q.id]
        c["_kind"] = q.kind
        for s in seeds:
            pipe, rec = make(s)
            a = pipe.ask(q.text)
            if q.answerable:
                c["ans_n"] += 1
                if a.abstained:
                    c["ans_abstain"] += 1
                else:
                    text = judged_text(a.primary.text, rec) if a.primary else ""
                    ok = a.primary is not None and is_correct(text, q)
                    c["ans_correct" if ok else "ans_wrong"] += 1
                    if q.kind == "conflict" and a.primary is not None \
                            and mentions_stale(text, q):
                        c["stale"] += 1
                if q.kind == "conflict":
                    c["conf_n"] += 1
                    if not a.abstained and a.primary is not None \
                            and is_correct(judged_text(a.primary.text, rec), q):
                        c["conf_correct"] += 1
            else:
                c["un_n"] += 1
                c["un_abstain" if a.abstained else "un_answered"] += 1
            if not a.abstained:
                for cl in a.claims:
                    c["claims"] += 1
                    if rec is not None and cl.text in rec.labels:
                        lc = rec.labels[cl.text]
                        c["bad_claims"] += claim_is_bad(world, lc, cl.cites)
    return res


def e2e_table(world: World) -> dict[str, list[E2E]]:
    held = world.corpus.heldout
    gate = select_gate(world)
    dev_items = collect(world, world.corpus.dev, DEV_SEEDS)
    base_cfg = calibrate_verifier(world, Config(), dev_items)
    no_repair = Config(**{**base_cfg.__dict__, "repair": False})
    ext = ExtractiveGenerator(world.index)

    def clean(resolve, gated, verify):
        def make(_):
            v = Verifier(world.index, world.known, base_cfg) if verify else None
            return Pipeline(world.index, ext, verifier=v,
                            abstain_threshold=gate.threshold if gated else None,
                            abstain_score=gate.score_fn,
                            resolve=resolve, k=K), None
        return make

    def noisy(cfg):
        def make(seed):
            rec = Recorder(ext, seed)
            v = Verifier(world.index, world.known, cfg) if cfg else None
            return Pipeline(world.index, rec, verifier=v, abstain_threshold=gate.threshold,
                            abstain_score=gate.score_fn,
                            k=K), rec
        return make

    clean_rows = [
        run_e2e(world, "naive RAG (no resolve, no abstain, no verify)", clean(False, False, False), held),
        run_e2e(world, "+ conflict resolution", clean(True, False, False), held),
        run_e2e(world, "+ abstention", clean(True, True, False), held),
        run_e2e(world, "+ citation verifier", clean(True, True, True), held),
    ]
    noisy_rows = [
        run_e2e(world, "noisy generator, no verifier", noisy(None), held, HELDOUT_SEEDS),
        run_e2e(world, "noisy + verifier (drop only)", noisy(no_repair), held, HELDOUT_SEEDS),
        run_e2e(world, "noisy + verifier (drop + repair)", noisy(base_cfg), held, HELDOUT_SEEDS),
    ]
    return {"clean": clean_rows, "noisy": noisy_rows}


def retrieval_recall(world: World, questions) -> tuple[float, float]:
    """Share of answerable questions whose gold sentence is among the top-K
    passages, before and after conflict resolution."""
    raw = res = n = 0
    for q in questions:
        if not q.answerable:
            continue
        n += 1
        g = gold_sentence(q)
        retrieved = world.index.search(q.text, K)
        raw += any(g in split_sentences(p.text) for p in retrieved)
        passages, _ = resolve_passages(retrieved, world.index)
        res += any(g in split_sentences(p.text) for p in passages)
    return raw / n, res / n


def slipped_examples(world: World, cfg: Config, items: list[Item], limit: int = 6):
    """Bad claims the verifier kept: (fault, claim text, original sentence)."""
    ver = Verifier(world.index, world.known, cfg)
    out, seen = [], set()
    for it in items:
        for lc in it.labeled:
            v = ver.verify(lc.claim, it.pool)
            if v.kept and claim_is_bad(world, lc, v.cites) and lc.claim.text not in seen:
                seen.add(lc.claim.text)
                out.append((lc.fault, lc.claim.text, lc.origin))
    return out[:limit]


def rejected_faithful(world: World, cfg: Config, items: list[Item], limit: int = 6):
    """Faithful claims the verifier rejected: (fault, claim text, reasons)."""
    ver = Verifier(world.index, world.known, cfg)
    out, seen = [], set()
    for it in items:
        for lc in it.labeled:
            if claim_is_bad(world, lc, lc.claim.cites):
                continue
            v = ver.verify(lc.claim, it.pool)
            if not v.kept and lc.claim.text not in seen:
                seen.add(lc.claim.text)
                out.append((lc.fault, lc.claim.text, v.reasons))
    return out[:limit]
