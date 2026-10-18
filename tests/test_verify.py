import pytest

from ragground.generate import Claim
from ragground.verify import Config, Verifier

from conftest import NIMBUS_RATE, NIMBUS_TLS

V2 = "nimbus-reference@2.0#0"
V1 = "nimbus-reference@1.0#0"
CIRRUS = "cirrus-reference@2.0#0"


def verifier(mini, **kw):
    index, known = mini
    return Verifier(index, known, Config(**{"threshold": 0.8, **kw})), list(index.passages)


def test_verbatim_claim_is_supported(mini):
    v, pool = verifier(mini)
    r = v.verify(Claim(NIMBUS_RATE, (V2,)), pool)
    assert r.status == "supported" and r.cites == (V2,) and r.coverage == 1.0


def test_unknown_citation_is_rejected_without_repair(mini):
    v, pool = verifier(mini, repair=False)
    r = v.verify(Claim(NIMBUS_RATE, ("nimbus-reference@9.9#7",)), pool)
    assert r.status == "unsupported" and "unknown_citation" in r.reasons


def test_fabricated_citation_is_repaired_to_the_passage_that_holds_the_claim(mini):
    v, pool = verifier(mini)
    r = v.verify(Claim(NIMBUS_RATE, ("nimbus-reference@9.9#7",)), pool)
    assert r.status == "repaired" and r.cites == (V2,)


def test_wrong_citation_is_repaired_or_dropped_depending_on_config(mini):
    claim = Claim(NIMBUS_RATE, (CIRRUS,))
    v, pool = verifier(mini)
    assert v.verify(claim, pool).status == "repaired"
    v, pool = verifier(mini, repair=False)
    assert v.verify(claim, pool).status == "unsupported"


def test_altered_number_is_caught_by_the_number_check(mini):
    v, pool = verifier(mini, repair=False)
    r = v.verify(Claim(NIMBUS_RATE.replace("600", "900"), (V2,)), pool)
    assert r.status == "unsupported" and "number_mismatch" in r.reasons


def test_altered_number_slips_past_lexical_overlap_alone(mini):
    v, pool = verifier(mini, repair=False, check_numbers=False, check_units=False,
                       check_negation=False, check_entities=False, check_superseded=False)
    assert v.verify(Claim(NIMBUS_RATE.replace("600", "900"), (V2,)), pool).kept


def test_swapped_unit_is_caught(mini):
    v, pool = verifier(mini, repair=False)
    r = v.verify(Claim(NIMBUS_RATE.replace("minute", "hour"), (V2,)), pool)
    assert "unit_mismatch" in r.reasons and not r.kept


def test_dropped_negation_is_caught(mini):
    v, pool = verifier(mini, repair=False)
    r = v.verify(Claim("The Nimbus API does not support mutual TLS.", (V2,)), pool)
    assert "negation_mismatch" in r.reasons and not r.kept


def test_swapped_entity_is_caught(mini):
    v, pool = verifier(mini, repair=False, threshold=0.5)
    r = v.verify(Claim("The Cirrus Queue supports mutual TLS.", (V2,)), pool)
    assert "entity_missing" in r.reasons and not r.kept


def test_claim_true_only_of_an_older_version_is_rejected_as_superseded(mini):
    old = Claim("The Nimbus API enforces a rate limit of 300 requests per minute per API key.", (V1,))
    v, pool = verifier(mini, repair=False)
    r = v.verify(old, pool)
    assert "superseded" in r.reasons and not r.kept
    v, pool = verifier(mini, repair=False, check_superseded=False)
    assert v.verify(old, pool).kept


def test_known_paraphrase_is_accepted(mini):
    v, pool = verifier(mini, threshold=0.95)
    r = v.verify(Claim("The Nimbus API enforces a rate cap of 600 requests per minute per API key.", (V2,)), pool)
    assert r.kept


# --- documented limits of a lexical verifier. These tests pin behaviour that is
# --- WRONG on purpose, so nobody mistakes the verifier for an entailment model.

def test_limit_meaning_flip_with_no_cue_word_is_accepted(mini):
    v, pool = verifier(mini)
    assert v.verify(Claim("The Nimbus API lacks support for mutual TLS.", (V2,)), pool).kept


def test_limit_small_unsourced_addition_is_accepted(mini):
    v, pool = verifier(mini)
    claim = Claim(NIMBUS_RATE.rstrip(".") + " for all regions.", (V2,))
    assert v.verify(claim, pool).kept


def test_limit_heavy_unseen_reword_of_a_true_claim_is_rejected(mini):
    v, pool = verifier(mini, threshold=0.85, repair=False)
    r = v.verify(Claim("A request to the Nimbus API times out once 60 seconds have passed.",
                       ("nimbus-reference@2.0#1",)), pool)
    assert r.status == "unsupported" and r.reasons == ("low_support",)


def test_raising_the_threshold_never_accepts_more(mini):
    claims = [Claim(NIMBUS_RATE.rstrip(".") + " for all regions.", (V2,)),
              Claim(NIMBUS_RATE, (V2,)), Claim("The Nimbus API lacks support for mutual TLS.", (V2,))]
    kept = []
    for thr in (0.5, 0.7, 0.8, 0.9, 1.0):
        v, pool = verifier(mini, threshold=thr, repair=False)
        kept.append(sum(v.verify(c, pool).kept for c in claims))
    assert kept == sorted(kept, reverse=True)


def test_a_repaired_citation_always_passes_every_check(mini):
    v, pool = verifier(mini)
    r = v.verify(Claim(NIMBUS_TLS, (CIRRUS,)), pool)
    assert r.status == "repaired"
    again = v.verify(Claim(NIMBUS_TLS, r.cites), pool)
    assert again.status == "supported"


def test_coverage_exactly_at_the_threshold_is_accepted(mini):
    """4 of the claim's 5 content words are in the sentence: coverage 0.8."""
    claim = Claim("The Nimbus API supports mutual banana.", (V2,))
    v, pool = verifier(mini, threshold=0.8, repair=False)
    r = v.verify(claim, pool)
    assert r.coverage == 0.8 and r.kept
    v, pool = verifier(mini, threshold=0.85, repair=False)
    assert not v.verify(claim, pool).kept
