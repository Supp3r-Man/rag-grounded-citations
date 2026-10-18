import pytest

from ragground.corpus import Passage
from ragground.evaluate import build_world
from ragground.index import Index
from ragground.text import known_entities


@pytest.fixture(scope="session")
def world():
    return build_world(7)


def P(pid, text, version="2.0", date="2026-01-10", family=None):
    doc = pid.split("#")[0]
    return Passage(pid, doc, family or doc.split("@")[0], version, date, text)


NIMBUS_RATE = "The Nimbus API enforces a rate limit of 600 requests per minute per API key."
NIMBUS_TLS = "The Nimbus API supports mutual TLS."


@pytest.fixture()
def mini():
    """A hand-built corpus small enough to reason about line by line."""
    passages = [
        P("nimbus-reference@2.0#0", f"{NIMBUS_RATE} {NIMBUS_TLS}"),
        P("nimbus-reference@2.0#1", "Requests to the Nimbus API time out after 60 seconds."),
        P("nimbus-reference@1.0#0",
          "The Nimbus API enforces a rate limit of 300 requests per minute per API key. "
          "The Nimbus API supports mutual TLS.", version="1.0", date="2025-03-10"),
        P("cirrus-reference@2.0#0",
          "The Cirrus Queue enforces a rate limit of 600 requests per minute per API key."),
    ]
    index = Index(passages)
    return index, known_entities([p.text for p in passages])
