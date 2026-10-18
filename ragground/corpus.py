"""A synthetic, seeded corpus of fictional software documentation.

Sixteen made-up products each have a versioned reference doc (1.0, 2.0 and
sometimes 3.0). A fact can change between versions, which is what creates the
conflicting-evidence questions. Questions come in four kinds:

  answerable    one current value, no older version disagrees
  conflict      older versions state a different value; the newest is correct
  unanswerable  the product exists but this fact is not documented for it
  offtopic      nothing in the corpus is relevant

Nothing here is real: no personal data, no payment data, no health data.
Products are split into dev and held-out halves by product, so no held-out
fact is ever seen while tuning.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

PRODUCTS = [
    "Nimbus API", "Cirrus Queue", "Stratus Cache", "Aurora Gateway",
    "Zephyr Scheduler", "Quartz Auth", "Ember Search", "Tidal Streams",
    "Orbit Webhooks", "Helix Vault", "Lumen Metrics", "Pixel Render",
    "Atlas Router", "Cobalt Deploy", "Marlin Mailer", "Sable Storage",
]

# attr id -> (sentence template, [question templates], numeric choices or None,
#             tokens that must appear in an answer about this attribute)
NUMERIC = {
    "rate_limit": (
        "The {P} enforces a rate limit of {v} requests per minute per API key.",
        ["What is the rate limit of the {P}?",
         "How many requests per minute is each API key limited to on the {P}?"],
        [100, 250, 300, 500, 600, 1000, 1200, 2000], ("rate",)),
    "timeout": (
        "Requests to the {P} time out after {v} seconds.",
        ["After how many seconds do requests to the {P} time out?",
         "What is the request timeout of the {P}?"],
        [10, 15, 30, 45, 60, 90], ("timeout",)),
    "retention": (
        "The {P} retains audit logs for {v} days.",
        ["How many days does the {P} retain audit logs?",
         "What is the log retention period of the {P}?"],
        [7, 14, 30, 90, 180, 365], ("log",)),
    "max_payload": (
        "The maximum payload size accepted by the {P} is {v} megabytes.",
        ["What is the maximum payload size the {P} accepts?",
         "What is the largest payload, in megabytes, that the {P} will accept?"],
        [1, 2, 5, 10, 25, 50], ("payload",)),
    "retries": (
        "The {P} client retries failed calls up to {v} times.",
        ["How many times does the {P} client retry failed calls?",
         "What is the retry count for failed calls in the {P} client?"],
        [2, 3, 4, 5, 6, 8], ("retry",)),
    "page_size": (
        "List endpoints on the {P} return {v} items per page by default.",
        ["How many items per page do list endpoints on the {P} return by default?",
         "What is the default page size of list endpoints on the {P}?"],
        [10, 20, 25, 50, 100, 200], ("page",)),
}

FEATURES = {
    "mtls": ("mutual TLS", ("tls",)),
    "ip_allowlist": ("IP allowlists", ("allowlist",)),
    "batch": ("batch requests", ("batch",)),
    "dry_run": ("dry-run mode", ("dry-run",)),
    "sso": ("single sign-on", ("sign-on",)),
}
FEATURE_QUESTIONS = [
    "Does the {P} support {feat}?",
    "Is {feat} supported by the {P}?",
]

ALL_ATTRS = list(NUMERIC) + list(FEATURES)

# Facts that are never documented for any product: pure "no answer" questions
# about a product that does exist.
NEVER_DOCUMENTED = [
    "What is the burst limit of the {P}?",
    "What uptime percentage does the {P} guarantee?",
    "How much does the {P} cost per million requests?",
    "In which regions is the {P} hosted?",
    "What programming language is the {P} written in?",
]
OFF_TOPIC = [
    "Who is the chief executive of the company?",
    "What is the capital of France?",
    "How do I reset the password on my laptop?",
    "What is the weather in Paris today?",
    "Which football team won the last world cup?",
    "How do I bake sourdough bread?",
    "What is the tallest mountain on Earth?",
    "How many moons does Jupiter have?",
]

FILLER = [
    "Contact your workspace administrator to change account-level settings.",
    "These values apply to every production workspace.",
    "Sandbox workspaces may be configured differently.",
    "See the changelog for the history of these settings.",
]

DATES = ["2025-03-10", "2025-08-15", "2026-01-20"]


@dataclass(frozen=True)
class Passage:
    id: str
    doc_id: str
    family: str
    version: str
    date: str
    text: str


@dataclass(frozen=True)
class Gold:
    value: str | None          # numeric answer as a string, or None for yes/no
    polarity: int | None       # 0 = supported, 1 = not supported, None for numeric
    stale: tuple[str, ...]     # older values that are now wrong
    required: tuple[str, ...]  # canonical tokens an answer about this fact contains


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    kind: str                  # answerable | conflict | unanswerable | offtopic
    split: str                 # dev | heldout
    product: str | None
    attr: str | None
    gold: Gold | None

    @property
    def answerable(self) -> bool:
        return self.kind in ("answerable", "conflict")


@dataclass
class Corpus:
    seed: int
    passages: list[Passage]
    questions: list[Question]
    products: dict[str, str] = field(default_factory=dict)  # product -> split

    def split(self, name: str) -> list[Question]:
        return [q for q in self.questions if q.split == name]

    @property
    def dev(self) -> list[Question]:
        return self.split("dev")

    @property
    def heldout(self) -> list[Question]:
        return self.split("heldout")


def slug(product: str) -> str:
    return product.lower().replace(" ", "-")


def fact_sentence(attr: str, product: str, value) -> str:
    if attr in NUMERIC:
        return NUMERIC[attr][0].format(P=product, v=value)
    feat = FEATURES[attr][0]
    if value:
        return f"The {product} supports {feat}."
    return f"The {product} does not support {feat}."


def question_text(attr: str, product: str, rng: random.Random) -> str:
    if attr in NUMERIC:
        return rng.choice(NUMERIC[attr][1]).format(P=product)
    return rng.choice(FEATURE_QUESTIONS).format(P=product, feat=FEATURES[attr][0])


def _required(attr: str, product: str) -> tuple[str, ...]:
    from .text import content_tokens

    own = NUMERIC[attr][3] if attr in NUMERIC else FEATURES[attr][1]
    return tuple(own) + tuple(content_tokens(product, keep_numbers=False))


def build_corpus(seed: int = 7) -> Corpus:
    rng = random.Random(seed)
    order = PRODUCTS[:]
    rng.shuffle(order)
    half = len(order) // 2
    split_of = {p: ("dev" if i < half else "heldout") for i, p in enumerate(order)}

    passages: list[Passage] = []
    questions: list[Question] = []
    qn = 0

    def new_q(text, kind, p, attr, gold):
        nonlocal qn
        qn += 1
        questions.append(Question(f"q{qn:03d}", text, kind, split_of[p] if p else None,
                                  p, attr, gold))

    for product in PRODUCTS:
        s = slug(product)
        documented = rng.sample(ALL_ATTRS, 8)
        undocumented = [a for a in ALL_ATTRS if a not in documented]
        n_versions = 3 if rng.random() < 0.4 else 2
        history: dict[str, list] = {}
        for a in documented:
            first = rng.choice(NUMERIC[a][2]) if a in NUMERIC else rng.random() < 0.5
            vals = [first]
            for _ in range(n_versions - 1):
                cur = vals[-1]
                if rng.random() < 0.25:
                    if a in NUMERIC:
                        cur = rng.choice([c for c in NUMERIC[a][2] if c != cur])
                    else:
                        cur = not cur
                vals.append(cur)
            history[a] = vals

        chunks = [documented[i:i + 2] for i in range(0, len(documented), 2)]
        fillers = [rng.choice(FILLER) for _ in chunks]
        jitter = rng.randint(0, 9)
        for vi in range(n_versions):
            ver = f"{vi + 1}.0"
            date = DATES[vi][:-2] + f"{10 + jitter:02d}"
            doc_id = f"{s}-reference@{ver}"
            for ci, chunk in enumerate(chunks):
                sents = []
                if ci == 0:
                    sents.append(f"Product reference for the {product}, version {ver}.")
                sents += [fact_sentence(a, product, history[a][vi]) for a in chunk]
                sents.append(fillers[ci])
                passages.append(Passage(f"{doc_id}#{ci}", doc_id, f"{s}-reference",
                                        ver, date, " ".join(sents)))
        passages.append(Passage(
            f"{s}-quickstart@1.0#0", f"{s}-quickstart@1.0", f"{s}-quickstart", "1.0",
            DATES[0],
            f"Getting started with the {product}. Create a workspace, generate an "
            f"API key and send your first request. The {product} dashboard shows "
            f"usage once traffic arrives."))

        for a in documented:
            latest = history[a][-1]
            stale = []
            for v in history[a][:-1]:
                if a in NUMERIC and v != latest and str(v) not in stale:
                    stale.append(str(v))
            if a in NUMERIC:
                gold = Gold(str(latest), None, tuple(stale), _required(a, product))
                conflict = bool(stale)
            else:
                gold = Gold(None, 0 if latest else 1, (), _required(a, product))
                conflict = any(v != latest for v in history[a][:-1])
            new_q(question_text(a, product, rng),
                  "conflict" if conflict else "answerable", product, a, gold)
        for a in rng.sample(undocumented, 2):
            new_q(question_text(a, product, rng), "unanswerable", product, a, None)
        new_q(rng.choice(NEVER_DOCUMENTED).format(P=product), "unanswerable",
              product, None, None)

    for i, text in enumerate(OFF_TOPIC):
        q = Question(f"q{qn + 1 + i:03d}", text, "offtopic",
                     "dev" if i % 2 == 0 else "heldout", None, None, None)
        questions.append(q)

    return Corpus(seed, passages, questions, split_of)
