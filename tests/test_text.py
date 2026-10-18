from ragground.text import (content_tokens, entities, known_entities, normalise,
                            numbers, polarity, split_sentences, stem, units)


def test_timeout_forms_tokenise_alike():
    a = content_tokens("Requests to the X time out after 30 seconds.")
    b = content_tokens("What is the request timeout of the X?")
    assert "timeout" in a and "timeout" in b
    assert normalise("It timed out") == "it timeout"


def test_stem_handles_plural_ed_and_ies():
    assert stem("requests") == "request"
    assert stem("accepted") == "accept"
    assert stem("retries") == "retry"
    assert stem("access") == "access"  # double s is not a plural


def test_numbers_strip_thousands_separators():
    assert numbers("limit of 1,200 requests, retry 3 times") == frozenset({"1200", "3"})


def test_polarity_is_parity_of_negation_cues():
    assert polarity("The X supports Y.") == 0
    assert polarity("The X does not support Y.") == 1
    assert polarity("The X doesn't support Y.") == 1
    assert polarity("It is not unsupported.") == 0  # two cues cancel


def test_units_ignore_non_units():
    assert units("600 requests per minute") == frozenset({"minute"})
    assert units("retains logs for 30 days") == frozenset({"day"})


def test_split_sentences_keeps_decimal_version_numbers():
    text = "Product reference for the X, version 2.0. The X supports Y."
    assert split_sentences(text) == ["Product reference for the X, version 2.0.",
                                     "The X supports Y."]


def test_entities_skip_sentence_initial_unless_known():
    known = known_entities(["The Nimbus API supports it.", "Requests are fine."])
    assert "nimbus" in known and "requests" not in known
    assert entities("Requests to the Nimbus API fail.", known) == frozenset({"nimbus", "api"})
    assert "nimbus" in entities("Nimbus API fails.", known)


def test_known_synonyms_collapse():
    assert content_tokens("rate cap") == content_tokens("rate limit")
    assert content_tokens("keeps logs") == content_tokens("retains logs")


def test_content_tokens_drop_function_words_and_negation():
    assert content_tokens("The X does not support Y", keep_numbers=False) == ["x", "support", "y"]
