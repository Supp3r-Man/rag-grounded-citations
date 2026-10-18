from ragground.conflict import Note, fact_key, find_superseding, resolve_passages
from ragground.index import Index

from conftest import NIMBUS_RATE, NIMBUS_TLS, P


def test_search_finds_gold_passage_for_most_answerable_questions(world):
    from ragground.evaluate import gold_sentence
    from ragground.text import split_sentences
    qs = [q for q in world.corpus.questions if q.answerable]
    hit = sum(any(gold_sentence(q) in split_sentences(p.text)
                  for p in world.index.search(q.text, 6)) for q in qs)
    assert hit / len(qs) >= 0.95


def test_search_is_deterministic_including_ties(world):
    q = "What is the rate limit of the Nimbus API?"
    assert [p.id for p in world.index.search(q)] == [p.id for p in world.index.search(q)]


def test_newer_versions_are_newest_first_and_exclude_self(mini):
    index, _ = mini
    v1 = index.by_id["nimbus-reference@1.0#0"]
    assert [p.version for p in index.newer_versions(v1)] == ["2.0", "2.0"]
    v2 = index.by_id["nimbus-reference@2.0#0"]
    assert index.newer_versions(v2) == []


def test_unseen_words_have_the_highest_idf(mini):
    index, _ = mini
    assert index.idf("burst") > index.idf("limit") > 0


def test_changed_number_is_superseded(mini):
    index, _ = mini
    old = index.by_id["nimbus-reference@1.0#0"]
    sup = find_superseding(old, "The Nimbus API enforces a rate limit of 300 requests per minute per API key.", index)
    assert sup is not None and sup[0].version == "2.0" and "600" in sup[1]


def test_unchanged_sentence_is_not_superseded(mini):
    index, _ = mini
    old = index.by_id["nimbus-reference@1.0#0"]
    assert find_superseding(old, NIMBUS_TLS, index) is None


def test_boolean_flip_is_a_conflict():
    passages = [
        P("x-reference@1.0#0", "The Nimbus API does not support mutual TLS.", "1.0"),
        P("x-reference@2.0#0", "The Nimbus API supports mutual TLS.", "2.0"),
    ]
    index = Index(passages)
    sup = find_superseding(passages[0], passages[0].text, index)
    assert sup is not None and sup[1] == "The Nimbus API supports mutual TLS."


def test_value_reverted_in_newest_version_is_current():
    """600 -> 900 -> 600: version 1 agrees with the newest, version 2 does not."""
    mk = lambda v, n: P(f"x-reference@{v}.0#0", NIMBUS_RATE.replace("600", str(n)), f"{v}.0")
    passages = [mk(1, 600), mk(2, 900), mk(3, 600)]
    index = Index(passages)
    assert find_superseding(passages[0], passages[0].text, index) is None
    assert find_superseding(passages[1], passages[1].text, index) is not None


def test_other_products_are_not_counterparts():
    a = fact_key("The Nimbus API enforces a rate limit of 600 requests per minute per API key.")
    b = fact_key("The Cirrus Queue enforces a rate limit of 600 requests per minute per API key.")
    assert len(a & b) / len(a | b) < 0.8


def test_resolve_trims_old_sentence_and_pulls_in_the_newer_passage(mini):
    index, _ = mini
    old = index.by_id["nimbus-reference@1.0#0"]
    out, notes = resolve_passages([old], index)
    texts = " ".join(p.text for p in out)
    assert "300 requests" not in texts and "600 requests" in texts
    assert any(p.id == "nimbus-reference@2.0#0" for p in out)
    assert len(notes) == 1 and isinstance(notes[0], Note)
    assert "version 1.0" in str(notes[0]) and "version 2.0" in str(notes[0])


def test_resolve_leaves_current_passages_alone(mini):
    index, _ = mini
    cur = index.by_id["nimbus-reference@2.0#0"]
    out, notes = resolve_passages([cur], index)
    assert out == [cur] and notes == []


def test_a_documents_own_version_number_is_not_a_changed_fact():
    passages = [
        P("x-reference@1.0#0", "Product reference for the Nimbus API, version 1.0.", "1.0"),
        P("x-reference@2.0#0", "Product reference for the Nimbus API, version 2.0.", "2.0"),
    ]
    index = Index(passages)
    assert find_superseding(passages[0], passages[0].text, index) is None
    assert resolve_passages([passages[0]], index)[1] == []
