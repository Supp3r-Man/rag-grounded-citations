import re

from ragground.corpus import FEATURES, NUMERIC, build_corpus, fact_sentence
from ragground.evaluate import gold_sentence
from ragground.index import version_key
from ragground.text import content_tokens, split_sentences


def test_same_seed_same_corpus():
    a, b = build_corpus(3), build_corpus(3)
    assert [p.text for p in a.passages] == [p.text for p in b.passages]
    assert [q.text for q in a.questions] == [q.text for q in b.questions]
    assert [p.text for p in a.passages] != [p.text for p in build_corpus(4).passages]


def test_splits_are_disjoint_by_product(world):
    dev = {q.product for q in world.corpus.dev if q.product}
    held = {q.product for q in world.corpus.heldout if q.product}
    assert dev and held and not dev & held


def test_every_gold_sentence_is_in_the_latest_version(world):
    for q in world.corpus.questions:
        if not q.answerable:
            continue
        g = gold_sentence(q)
        homes = [p for p in world.corpus.passages if g in split_sentences(p.text)]
        assert homes, q.text
        fam = homes[0].family
        newest = max((p for p in world.corpus.passages if p.family == fam),
                     key=lambda p: version_key(p.version))
        assert any(p.version == newest.version for p in homes), q.text


def test_conflict_questions_have_a_stale_value_in_an_older_version(world):
    numeric = [q for q in world.corpus.questions if q.kind == "conflict" and q.gold.value]
    assert len(numeric) > 10
    for q in numeric:
        assert q.gold.stale and q.gold.value not in q.gold.stale
        old_sentences = {fact_sentence(q.attr, q.product, int(v)) for v in q.gold.stale}
        older = [p for p in world.corpus.passages
                 if p.version != "3.0" and q.product in p.text and "reference" in p.doc_id]
        assert any(s in split_sentences(p.text) for s in old_sentences for p in older), q.text


def test_unanswerable_facts_are_not_documented_for_that_product(world):
    for q in world.corpus.questions:
        if q.kind != "unanswerable" or q.attr is None:
            continue
        own = NUMERIC[q.attr][3] if q.attr in NUMERIC else FEATURES[q.attr][1]
        for p in world.corpus.passages:
            if q.product in p.text and "reference" in p.doc_id:
                assert not set(own) <= set(content_tokens(p.text, keep_numbers=False)), (q.text, p.text)


def test_offtopic_questions_name_no_product(world):
    off = [q for q in world.corpus.questions if q.kind == "offtopic"]
    assert len(off) == 8 and all(q.product is None for q in off)


def test_no_payment_or_identity_style_data(world):
    blob = " ".join(p.text for p in world.corpus.passages) + " " + \
           " ".join(q.text for q in world.corpus.questions)
    assert not re.search(r"\d{9,}", blob)             # no long digit runs
    assert not re.search(r"\d{3}-\d{2}-\d{4}", blob)  # no SSN-shaped strings
    assert not re.search(r"@\w+\.\w+", blob)           # no email addresses


def test_passage_ids_are_unique(world):
    ids = [p.id for p in world.corpus.passages]
    assert len(ids) == len(set(ids))
