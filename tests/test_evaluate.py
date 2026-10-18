import ragground.evaluate as ev
from ragground.faults import LabeledClaim
from ragground.generate import Claim
from ragground.stats import ratio_ci, wilson
from ragground.verify import Config

from conftest import NIMBUS_RATE

V2 = "nimbus-reference@2.0#0"


def test_wilson_interval_brackets_the_estimate_and_handles_edges():
    lo, hi = wilson(7, 10)
    assert lo < 0.7 < hi
    assert wilson(0, 0) == (0.0, 1.0)
    assert wilson(10, 10)[1] == 1.0


def test_ratio_ci_resamples_groups_and_is_deterministic():
    groups = [(1, 2), (0, 2), (2, 2), (1, 2)]
    est, lo, hi = ratio_ci(groups)
    assert est == 0.5 and lo <= est <= hi and (est, lo, hi) == ratio_ci(groups)
    assert ratio_ci([]) == (0.0, 0.0, 0.0)


def pick(world, kind="conflict"):
    return next(q for q in world.corpus.heldout if q.kind == kind and q.gold.value)


def test_is_correct_rejects_stale_wrong_product_and_wrong_polarity(world):
    q = pick(world)
    good = ev.gold_sentence(q)
    assert ev.is_correct(good, q)
    stale = good.replace(q.gold.value, q.gold.stale[0])
    assert not ev.is_correct(stale, q) and ev.mentions_stale(stale, q)
    from ragground.corpus import PRODUCTS
    other = next(p for p in PRODUCTS if p != q.product)
    assert not ev.is_correct(good.replace(q.product, other), q)
    bq = next(q for q in world.corpus.heldout if q.answerable and q.gold.polarity is not None)
    yes = ev.gold_sentence(bq)
    flipped = yes.replace("does not support", "supports") if "does not" in yes else \
        yes.replace("supports", "does not support")
    assert ev.is_correct(yes, bq) and not ev.is_correct(flipped, bq)


def test_claim_is_bad_for_wrong_citation_but_not_for_a_faithful_paraphrase(world):
    cirrus = next(p.id for p in world.corpus.passages if p.id.startswith("cirrus"))
    sent = world.index.by_id[cirrus].text.split(". ")[0] + "."
    # sentence taken from its own passage -> a citation to another passage is wrong
    other = next(p.id for p in world.corpus.passages if p.id != cirrus and sent not in p.text)
    ok = LabeledClaim(Claim(sent, (cirrus,)), "none", sent)
    wrong = LabeledClaim(Claim(sent, (other,)), "wrong_citation", sent)
    para = LabeledClaim(Claim(sent, (cirrus,)), "paraphrase_unseen", sent)
    assert not ev.claim_is_bad(world, ok, ok.claim.cites)
    assert ev.claim_is_bad(world, wrong, wrong.claim.cites)
    assert not ev.claim_is_bad(world, para, para.claim.cites)
    assert ev.claim_is_bad(world, LabeledClaim(Claim(sent, (cirrus,)), "number_altered", sent), (cirrus,))


def test_score_verifier_counts_on_a_handmade_set(world):
    q = pick(world)
    gold = ev.gold_sentence(q)
    pid = next(p.id for p in world.corpus.passages if gold in p.text and p.version == "2.0")
    altered = gold.replace(q.gold.value, "99999")
    items = [ev.Item(q, list(world.index.passages), [
        LabeledClaim(Claim(gold, (pid,)), "none", gold),
        LabeledClaim(Claim(altered, (pid,)), "number_altered", gold),
        LabeledClaim(Claim(gold, ("nope@9.9#7",)), "fabricated_citation", gold),
    ])]
    r = ev.score_verifier(world, Config(threshold=0.8), items)
    assert (r.n, r.bad_before, r.faithful) == (3, 2, 1)
    assert r.kept == 2 and r.bad_after == 0 and r.repaired == 1 and r.false_reject == 0
    assert r.per_fault["number_altered"]["dropped"] == 1
    assert r.per_fault["fabricated_citation"]["repaired"] == 1
    r2 = ev.score_verifier(world, Config(threshold=0.8, repair=False), items)
    assert r2.kept == 1 and r2.per_fault["fabricated_citation"]["dropped"] == 1


def test_citation_id_only_baseline_keeps_altered_claims(world):
    q = pick(world)
    gold = ev.gold_sentence(q)
    pid = next(p.id for p in world.corpus.passages if gold in p.text and p.version == "2.0")
    items = [ev.Item(q, [], [LabeledClaim(Claim(gold.replace(q.gold.value, "99999"), (pid,)),
                                          "number_altered", gold)])]
    r = ev.score_verifier(world, ev.CONFIGS["citation id exists only"], items)
    assert r.kept == 1 and r.bad_after == 1


def test_collect_is_deterministic_and_skips_unanswerable_questions(world):
    qs = world.corpus.heldout[:20]
    a, b = ev.collect(world, qs, [201]), ev.collect(world, qs, [201])
    assert [[lc.claim for lc in it.labeled] for it in a] == [[lc.claim for lc in it.labeled] for it in b]
    assert all(it.q.answerable for it in a)


def test_calibration_never_sees_held_out_questions(world, monkeypatch):
    seen = []
    real_a, real_b = ev.abstain_scores, ev.bm25_scores
    monkeypatch.setattr(ev, "abstain_scores", lambda w, qs: (seen.extend(qs), real_a(w, qs))[1])
    monkeypatch.setattr(ev, "bm25_scores", lambda w, qs: (seen.extend(qs), real_b(w, qs))[1])
    gate = ev.select_gate(world)
    assert seen and all(q.split == "dev" for q in seen)
    assert gate.name in ("top BM25 score", "evidence coverage")


def test_bm25_threshold_separates_hand_made_groups():
    t = ev.bm25_threshold([12.0, 15.0, 11.0], [3.0, 5.0, 8.0])
    assert 8.0 < t <= 11.0


def test_retrieval_recall_is_high_on_this_corpus(world):
    raw, resolved = ev.retrieval_recall(world, world.corpus.heldout)
    assert raw >= 0.95 and resolved >= raw


def test_verifier_row_for_dropped_claims_reports_the_fault_in_slipped_examples(world):
    items = ev.collect(world, world.corpus.heldout[:40], [201])
    ex = ev.slipped_examples(world, Config(threshold=0.8), items, 50)
    # only the faults a lexical verifier is structurally weak on may slip through
    assert {f for f, _, _ in ex} <= {"subtle_addition", "antonym_flip"}
    cfg = ev.CONFIGS["citation id exists only"]
    assert len(ev.slipped_examples(world, cfg, items, 500)) > len(ex)


def test_a_faithful_reword_is_judged_by_the_fact_it_came_from(world):
    q = pick(world)
    gold = ev.gold_sentence(q)
    reword = LabeledClaim(Claim("It is " + q.gold.value + " here.", (V2,)), "reword", gold)
    rec = type("R", (), {"labels": {reword.claim.text: reword}})()
    assert ev.judged_text(reword.claim.text, rec) == gold and ev.is_correct(gold, q)
    altered = LabeledClaim(Claim(gold.replace(q.gold.value, "99999"), (V2,)), "number_altered", gold)
    rec2 = type("R", (), {"labels": {altered.claim.text: altered}})()
    assert ev.judged_text(altered.claim.text, rec2) == altered.claim.text
    assert ev.judged_text("anything", None) == "anything"
