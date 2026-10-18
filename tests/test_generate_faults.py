import json
import random

import pytest

from ragground.faults import CONTENT_OK, WEIGHTS, NoisyGenerator, apply_fault
from ragground.generate import ClaudeGenerator, Claim, ExtractiveGenerator, parse_claims
from ragground.conflict import resolve_passages

from conftest import NIMBUS_RATE, NIMBUS_TLS, P

TLS_NO = "The Nimbus API does not support mutual TLS."


def test_extractive_puts_the_answer_sentence_first_and_cites_its_passage(mini):
    index, _ = mini
    passages = [index.by_id["nimbus-reference@2.0#0"], index.by_id["nimbus-reference@2.0#1"]]
    claims = ExtractiveGenerator(index).generate("What is the rate limit of the Nimbus API?", passages)
    assert claims[0].text == NIMBUS_RATE
    assert claims[0].cites == ("nimbus-reference@2.0#0",)


def test_extractive_second_claim_comes_from_a_different_passage(mini):
    index, _ = mini
    passages = [index.by_id["nimbus-reference@2.0#0"], index.by_id["nimbus-reference@2.0#1"]]
    claims = ExtractiveGenerator(index).generate("What is the rate limit of the Nimbus API?", passages)
    assert len(claims) == 2 and claims[0].cites != claims[1].cites


def test_extractive_never_invents_text(world):
    ext = ExtractiveGenerator(world.index)
    for q in world.corpus.heldout[:30]:
        passages, _ = resolve_passages(world.index.search(q.text), world.index)
        for c in ext.generate(q.text, passages):
            assert any(c.text in p.text for p in passages if p.id == c.cites[0])


@pytest.mark.parametrize("fault,text,check", [
    ("number_altered", NIMBUS_RATE, lambda new: "600" not in new and "requests per minute" in new),
    ("unit_swap", NIMBUS_RATE, lambda new: "per hour" in new),
    ("negation_dropped", NIMBUS_TLS, lambda new: "does not support" in new),
    ("negation_dropped", TLS_NO, lambda new: "does not" not in new and "supports" in new),
    ("antonym_flip", NIMBUS_TLS, lambda new: "lacks support for" in new),
    ("antonym_flip", TLS_NO, lambda new: "fully handles" in new),
    ("entity_swap", NIMBUS_RATE, lambda new: "Nimbus API" not in new and "rate limit" in new),
    ("subtle_addition", NIMBUS_RATE, lambda new: new.startswith(NIMBUS_RATE[:-1]) and len(new) > len(NIMBUS_RATE)),
    ("reword", NIMBUS_RATE, lambda new: new == "On the Nimbus API, each API key is limited to 600 requests per minute."),
    ("reword", TLS_NO, lambda new: new == "Mutual TLS is not available on the Nimbus API."),
    ("paraphrase_known", NIMBUS_RATE, lambda new: "cap" in new or "calls" in new),
    ("paraphrase_unseen", NIMBUS_RATE, lambda new: "ceiling" in new or "applies" in new),
])
def test_each_fault_does_what_it_says(fault, text, check):
    new = apply_fault(fault, text, random.Random(0))
    assert new is not None and new != text and check(new)


@pytest.mark.parametrize("fault,text", [
    ("number_altered", NIMBUS_TLS),
    ("unit_swap", NIMBUS_TLS),
    ("negation_dropped", NIMBUS_RATE),
    ("entity_swap", "Nothing to swap here."),
    ("reword", "An unrecognised sentence."),
])
def test_inapplicable_fault_returns_none(fault, text):
    assert apply_fault(fault, text, random.Random(0)) is None


def test_unknown_fault_is_an_error():
    with pytest.raises(ValueError):
        apply_fault("gremlins", NIMBUS_RATE, random.Random(0))


class _Fixed:
    def generate(self, question, passages):
        return [Claim(NIMBUS_RATE, ("nimbus-reference@2.0#0",))]


def test_noisy_generator_is_deterministic_per_seed_and_varies_across_seeds(mini):
    index, _ = mini
    pool = list(index.passages)
    run = lambda seed: [lc.fault for q in range(60) for lc in
                        NoisyGenerator(_Fixed(), seed).generate_labeled(f"q{q}", pool)]
    assert run(1) == run(1)
    assert run(1) != run(2)


def test_fault_mix_follows_the_weights(mini):
    index, _ = mini
    pool = list(index.passages)
    labels = [lc.fault for q in range(1500) for lc in
              NoisyGenerator(_Fixed(), 5).generate_labeled(f"question {q}", pool)
              if lc.fault != "unsupported_addition"]
    none_share = labels.count("none") / len(labels)
    assert abs(none_share - WEIGHTS["none"]) < 0.06
    assert {"wrong_citation", "fabricated_citation", "number_altered"} <= set(labels)


def test_content_ok_faults_keep_the_text_true():
    assert CONTENT_OK == {"none", "paraphrase_known", "paraphrase_unseen", "reword",
                          "wrong_citation", "fabricated_citation"}


def test_stale_claim_uses_the_supplied_stale_sentences(mini):
    index, _ = mini
    pool = list(index.passages)
    stale = [("The Nimbus API enforces a rate limit of 300 requests per minute per API key.",
              "nimbus-reference@1.0#0")]
    seen = [lc for q in range(400) for lc in NoisyGenerator(_Fixed(), 9).generate_labeled(
        f"q{q}", pool, stale=stale) if lc.fault == "stale_claim"]
    assert seen and all(lc.claim.cites == ("nimbus-reference@1.0#0",) and "300" in lc.claim.text
                        for lc in seen)


def test_wrong_citation_never_cites_a_passage_that_holds_the_sentence(mini):
    index, _ = mini
    pool = list(index.passages)
    for q in range(300):
        for lc in NoisyGenerator(_Fixed(), 3).generate_labeled(f"q{q}", pool):
            if lc.fault == "wrong_citation":
                assert NIMBUS_RATE not in index.by_id[lc.claim.cites[0]].text


def test_claude_adapter_round_trips_json_through_a_fake_transport(mini):
    index, _ = mini
    seen = {}

    def transport(prompt):
        seen["prompt"] = prompt
        return "Sure!\n" + json.dumps({"claims": [{"text": NIMBUS_RATE,
                                                   "cites": ["nimbus-reference@2.0#0"]}]})

    claims = ClaudeGenerator(transport).generate("rate limit?", [index.by_id["nimbus-reference@2.0#0"]])
    assert claims == [Claim(NIMBUS_RATE, ("nimbus-reference@2.0#0",))]
    assert "[nimbus-reference@2.0#0]" in seen["prompt"] and "rate limit?" in seen["prompt"]


@pytest.mark.parametrize("raw", ["", "no json here", '{"claims": "nope"}', '{"claims": [{"x": 1}]}',
                                 '{"claims": [{"text": "  ", "cites": []}]}'])
def test_malformed_model_output_gives_no_claims(raw):
    assert parse_claims(raw) == []


def test_extractive_never_emits_a_document_title_as_a_claim(world):
    ext = ExtractiveGenerator(world.index)
    for q in world.corpus.questions[:60]:
        for c in ext.generate(q.text, world.index.search(q.text)):
            assert not c.text.startswith("Product reference for")
