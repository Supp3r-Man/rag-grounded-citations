import json

from ragground.answerability import (GRID, best_score, calibrate, rank_sentences,
                                     weighted_coverage)
from ragground.conflict import resolve_passages
from ragground.evaluate import is_correct, mentions_stale
from ragground.generate import ClaudeGenerator, Claim, ExtractiveGenerator
from ragground.pipeline import Pipeline
from ragground.verify import Config, Verifier

from conftest import NIMBUS_RATE, P


def test_answerable_questions_score_higher_than_unanswerable_ones_on_average(world):
    def mean_score(qs):
        s = [best_score(world.index, q.text, resolve_passages(world.index.search(q.text), world.index)[0])
             for q in qs]
        return sum(s) / len(s)
    dev = world.corpus.dev
    assert mean_score([q for q in dev if q.answerable]) > mean_score([q for q in dev if not q.answerable]) + 0.3


def test_off_topic_question_scores_zero(world):
    q = next(q for q in world.corpus.dev if q.kind == "offtopic")
    assert best_score(world.index, q.text, world.index.search(q.text)) < 0.2


def test_calibrate_picks_a_threshold_between_the_two_groups_and_ties_go_high():
    pos, neg = [0.9, 1.0, 0.95], [0.2, 0.3, 0.4]
    thr = calibrate(pos, neg)
    assert 0.4 < thr <= 0.9 and thr in GRID
    assert thr == max(t for t in GRID if 0.4 < t <= 0.9)  # tie goes to the higher threshold


def test_rank_sentences_ties_keep_retrieval_order(mini):
    index, _ = mini
    a, b = index.by_id["nimbus-reference@2.0#0"], index.by_id["nimbus-reference@1.0#0"]
    ranked = rank_sentences(index, "mutual TLS", [a, b])
    tls = [p.id for s, p, _ in ranked if s == ranked[0][0]]
    assert tls == [a.id, b.id]


def test_weighted_coverage_is_zero_when_nothing_is_wanted(mini):
    index, _ = mini
    assert weighted_coverage(index, [], {"x"}) == 0.0


def make(world, gen=None, **kw):
    return Pipeline(world.index, gen or ExtractiveGenerator(world.index), **kw)


def question(world, kind, split="heldout"):
    return [q for q in world.corpus.questions if q.kind == kind and q.split == split]


def test_conflict_question_gets_the_newest_value_and_a_note(world):
    pipe = make(world)
    hits = 0
    for q in question(world, "conflict"):
        a = pipe.ask(q.text)
        assert not a.abstained and is_correct(a.primary.text, q)
        hits += bool(a.notes)
    assert hits > 0  # at least some needed a version note


def test_without_conflict_resolution_some_answers_are_stale(world):
    pipe = make(world, resolve=False)
    stale = sum(mentions_stale(pipe.ask(q.text).primary.text, q)
                for q in world.corpus.questions if q.kind == "conflict" and q.gold.value)
    assert stale > 0


def test_off_topic_question_abstains_when_gated(world):
    pipe = make(world, abstain_threshold=0.5)
    a = pipe.ask("What is the capital of France?")
    assert a.abstained and "evidence score" in a.reason
    assert "can't answer" in a.render()


def test_no_gate_answers_a_no_answer_question_with_the_nearest_sentence(world):
    q = "What is the burst limit of the Nimbus API?"
    a = make(world).ask(q)
    assert not a.abstained and "burst" not in a.primary.text  # confident, and not an answer


def test_a_question_with_no_lexical_overlap_retrieves_nothing_and_abstains(world):
    a = make(world).ask("What is the capital of France?")
    assert a.abstained and a.reason == "no supported claim"


class _Fixed:
    def __init__(self, *claims):
        self.claims = list(claims)

    def generate(self, question, passages):
        return list(self.claims)


def asked(world, claims, **kw):
    q = next(q for q in world.corpus.heldout if q.kind == "answerable" and q.gold.value)
    from ragground.evaluate import gold_sentence
    gold = gold_sentence(q)
    pid = next(p.id for p in world.corpus.passages if gold in p.text and p.version == "2.0")
    pipe = Pipeline(world.index, _Fixed(*[c(gold, pid) for c in claims]), **kw)
    return q, gold, pid, pipe.ask(q.text)


def test_verifier_drops_a_claim_with_an_altered_number_and_abstains(world):
    cfg = Config(threshold=0.8)
    ver = Verifier(world.index, world.known, cfg)
    alter = lambda gold, pid: Claim(gold.replace(world_digit(gold), "9999"), (pid,))
    q, gold, pid, a = asked(world, [alter], verifier=ver)
    assert a.abstained and a.dropped and "number_mismatch" in a.dropped[0].reasons


def world_digit(text):
    import re
    return re.search(r"\d+", text).group()


def test_a_wrong_citation_is_repaired_in_the_final_answer(world):
    ver = Verifier(world.index, world.known, Config(threshold=0.8))
    other = next(p.id for p in world.corpus.passages if p.id.startswith("quartz") or p.id.startswith("sable"))
    q, gold, pid, a = asked(world, [lambda g, p: Claim(g, (other,))], verifier=ver)
    assert not a.abstained and a.claims[0].cites[0] != other


def test_a_correct_cited_claim_passes_through_unchanged(world):
    ver = Verifier(world.index, world.known, Config(threshold=0.8))
    q, gold, pid, a = asked(world, [lambda g, p: Claim(g, (p,))], verifier=ver)
    assert not a.abstained and a.primary.text == gold and not a.dropped


def test_no_claims_from_the_generator_is_an_abstention(world):
    a = make(world, _Fixed()).ask("What is the rate limit of the Nimbus API?")
    assert a.abstained and a.reason == "no supported claim"


def test_answer_renders_citations_and_notes(world):
    q = next(q for q in question(world, "conflict") if make(world).ask(q.text).notes)
    text = make(world).ask(q.text).render()
    assert "[" in text and "Note: Superseded" in text


def test_pipeline_works_with_a_model_adapter_via_a_fake_transport(world):
    q = next(q for q in world.corpus.heldout if q.kind == "answerable" and q.gold.value)
    from ragground.evaluate import gold_sentence
    gold = gold_sentence(q)
    pid = next(p.id for p in world.corpus.passages if gold in p.text and p.version == "2.0")
    gen = ClaudeGenerator(lambda prompt: json.dumps({"claims": [{"text": gold, "cites": [pid]}]}))
    ver = Verifier(world.index, world.known, Config(threshold=0.8))
    a = Pipeline(world.index, gen, verifier=ver).ask(q.text)
    assert not a.abstained and a.claims[0].cites == (pid,)


def test_verifier_abstains_when_only_a_context_claim_survives(world):
    """The claim that answers the question is gone; a faithful but irrelevant
    claim is left. Without the verifier stage that leftover is returned as the
    answer; with it the pipeline abstains."""
    from ragground.evaluate import gold_sentence
    from ragground.text import split_sentences
    q = next(q for q in world.corpus.heldout if q.kind == "answerable" and q.gold.value)
    gold = gold_sentence(q)
    other = next((s, p) for p in world.corpus.passages if q.product in p.text
                 and "reference" in p.doc_id and p.version == "2.0"
                 for s in split_sentences(p.text)
                 if s != gold and s.startswith("The ") and " version " not in s
                 and q.gold.value not in s)
    claim = Claim(other[0], (other[1].id,))
    plain = Pipeline(world.index, _Fixed(claim)).ask(q.text)
    assert not plain.abstained and plain.primary.text == other[0]
    ver = Verifier(world.index, world.known, Config(threshold=0.8))
    gated = Pipeline(world.index, _Fixed(claim), verifier=ver).ask(q.text)
    assert gated.abstained and "rejected" in gated.reason


def test_notes_only_explain_claims_that_are_in_the_answer(world):
    pipe = make(world)
    seen = 0
    for q in world.corpus.questions:
        if q.kind != "conflict":
            continue
        a = pipe.ask(q.text)
        for n in a.notes:
            seen += 1
            assert "version 1.0.\"" not in n and "Product reference" not in n
            assert any(n.split('says "')[1].split('"')[0] == c.text for c in a.claims)
    assert seen > 0
