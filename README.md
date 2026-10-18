# rag-grounded-citations

![ci](https://github.com/Supp3r-Man/rag-grounded-citations/actions/workflows/ci.yml/badge.svg)

The generation side of RAG: answer a question with **sentence-level citations**, then **check every citation** against the passage it points to, **repair or drop** the ones that fail, **prefer the newer document version** when sources conflict, and **abstain** when the documents do not answer the question.

> **What is verified, and what is not.** 109 tests pass locally on Python 3.13 and in GitHub Actions on Linux with Python 3.11 and 3.13 (standard library only). I also broke the code on purpose in 26 places (each verifier check, the conflict rules, calibration, the correctness check...) and the tests failed every time; two mutants first survived and I added tests for them. **The corpus is synthetic** (16 invented products, templated sentences), and **the "model" is not a model**: a deterministic extractive generator, wrapped in a fault injector whose fault mix and rates I chose. Every number below says how well the pipeline handles *those* faults. It does not say how a real LLM behaves. The Claude adapter (`ClaudeGenerator`) is **untested against a live model**: only its prompt and JSON parsing are exercised, with a fake transport. The verifier is lexical, not an entailment model; the sections below show where it fails.

## Pipeline

```
question -> BM25 top-6 passages
         -> conflict resolution   drop sentences a newer version of the same doc contradicts,
                                  pull the newer passage in, keep a note saying so
         -> evidence gate         abstain if the best evidence score is below a dev-calibrated threshold
         -> generator             cited claims (one sentence + passage ids)
         -> citation verifier     per claim: supported / repaired (re-cited) / unsupported
         -> answer check          abstain if the claim that answered the question was rejected
```

Every stage can be switched off, which is how the tables below measure what each one buys.

### The verifier (`ragground/verify.py`)

For each claim and each passage it cites, find the passage sentence covering most of the claim's content words, then run independent checks against that sentence:

| Check | Catches |
|---|---|
| `unknown_citation` | a cited id that is not in the corpus |
| `low_support` | too few of the claim's words appear in the sentence (threshold chosen on dev) |
| `number_mismatch` | the claim states a number the sentence does not |
| `unit_mismatch` | minutes vs hours, days vs hours, megabytes vs kilobytes |
| `negation_mismatch` | the claim and the sentence differ in polarity |
| `entity_missing` | the claim names a product the passage never mentions |
| `superseded` | the sentence is out of date: a newer version of the doc disagrees |

A claim is **supported** if any cited passage passes every check. If none does, the verifier searches the other retrieved passages for one that does and **repairs** the citation; otherwise the claim is **rejected**.

### Conflicting sources (`ragground/conflict.py`)

Versions of a document family (`nimbus-api-reference@1.0`, `@2.0`, `@3.0`) are linked by metadata, not by retrieval. A sentence is superseded when the *newest* version's counterpart (same words once numbers and negation are removed, Jaccard >= 0.8) has different numbers or polarity. The newest version decides: a value that went 600 -> 900 -> 600 is current in v1 and stale in v2 (tested). The answer carries a note for each superseded fact it states, for example (real output): `Note: Superseded: version 1.0 said "The Nimbus API supports mutual TLS."; version 2.0 (2025-08-11) says "The Nimbus API does not support mutual TLS." [nimbus-api-reference@2.0#0]`. Two bugs I found by reading that output: a document's own title line ("... version 1.0.") was being flagged as a changed fact, and the answer listed notes for facts nobody asked about; both are fixed and tested.

## Results (`python -m ragground demo`, full output in `results/demo_output.txt`)

Corpus: 168 passages, 16 invented products split **by product** into 8 dev and 8 held-out. Held-out questions: 43 answerable, 21 with conflicting versions, 24 about a documented product but an undocumented fact, 4 off-topic. Retrieval is easy here (the gold sentence is in the top 6 for 100% of questions), so these numbers isolate generation, not retrieval.

Every threshold is chosen on dev products and dev noise seeds, then scored on held-out products and held-out seeds. Confidence intervals resample **questions**, not claims.

### 1. The verifier against injected faults (663 held-out claims, 45% of them faulty)

Each configuration is calibrated separately on dev, so the lexical baseline gets its own best threshold.

| Configuration | Threshold | Bad claims kept (of claims kept) | Bad claims caught | Faithful claims rejected | Citation precision |
|---|---|---|---|---|---|
| nothing (45.1% of claims are bad) | | 45.1% | | | 80.4% |
| citation id exists only | n/a | 40.4% [37-44] | 10.4% | 0.0% | 80.4 -> 84.3% |
| lexical overlap only | 0.80 | 16.0% [13-19] | 64.5% | 2.5% | 80.4 -> 93.9% |
| + numbers and units | 0.80 | 10.3% [8-13] | 77.3% | 2.5% | 80.4 -> 93.9% |
| + negation | 0.80 | 7.4% [5-10] | 83.6% | 2.5% | 80.4 -> 94.1% |
| + entities | 0.75 | 8.9% [7-11] | 80.3% | 0.0% | 80.4 -> 94.3% |
| + superseded (full) | 0.75 | **3.9% [2-6]** | 91.3% | 0.0% | 80.4 -> 100.0% |
| full + citation repair | 0.75 | 3.9% [2-6] | 91.3% | 0.0% | 80.4 -> 100.0% |

Read it honestly:
- The `+ entities` row is *worse* than `+ negation` (8.9% vs 7.4%), and the entity check by itself catches nothing the word-overlap threshold did not already catch. The row got worse because dev recalibration moved the threshold from 0.80 to 0.75. Threshold choice moves these numbers by several points: while building this repo, two small fixes to the corpus handling moved the "lexical overlap only" row by about 7 points (to a 0.90 threshold with 17.7% faithful claims rejected) before I fixed the cause. Do not read the rows as a precise ladder.
- The biggest single step is `superseded` (8.9% -> 3.9%), but the stale claims were injected by me (`stale_claim`), so it shows the check works, not how often a real model cites a stale version.
- Repair does not change how many bad claims survive; it changes how many *good* claims are kept (it turns 78 wrong or fabricated citations into correct ones instead of dropping the claim).

Per fault type, full verifier (n = held-out claims):

| Fault | n | Outcome |
|---|---|---|
| none (verbatim) | 255 | 255 kept |
| reword / paraphrase (known and unseen words) | 109 | 109 kept, 0 rejected |
| wrong citation / fabricated citation | 78 | 78 repaired |
| number altered, unit swapped, negation dropped, entity swapped, stale claim, unsourced sentence added | 155 | 155 dropped |
| **subtle addition** ("... per API key for all regions") | 54 | 32 dropped, **22 slipped through (41%)** |
| **antonym flip** ("supports" -> "lacks support for") | 12 | 8 dropped, **4 slipped through (33%)** |

### 2. Abstention: the boring baseline won

| Gate (threshold chosen on dev) | Threshold | Precision | Recall | Answerable questions lost |
|---|---|---|---|---|
| never abstain | | | 0% | 0/64 |
| top BM25 score | 10.43 | 92.9% | 92.9% | 2/64 |
| my evidence-coverage score | 0.66 | 78.1% | 89.3% | 7/64 |

I built an idf-weighted question-coverage score expecting it to beat a raw BM25 score threshold. It did not, on dev (0.933 vs 0.974 balanced accuracy) or on held-out, so the pipeline uses the BM25 gate. Two caveats: raw BM25 is not length-normalised, so its threshold is tied to this corpus and these question lengths; and the gate is fooled when a question reuses another product's words: "How many items per page do list endpoints on the Nimbus API return by default?" is answered with the *Atlas Router* sentence (score 11.52 > 10.43). That is a confident answer about the wrong product. A check that the answer names the product the question names would fix it; I did not add one.

### 3. End to end (held-out products; 95% CI resamples questions)

Clean generator (extractive, cannot invent text): what conflicts and no-answer questions cost.

| Pipeline | Correct | Wrong | Abstained | Stale on conflict questions | No-answer questions answered |
|---|---|---|---|---|---|
| naive RAG | 81.2% [71-90] | 18.8% [10-29] | 0% | 38.1% [19-57] | 85.7% [71-97] |
| + conflict resolution | 100% | 0% | 0% | 0% | 85.7% |
| + abstention gate | 96.9% [92-100] | 0% | 3.1% | 0% | 7.1% [0-17] |
| + citation verifier | 96.9% | 0% | 3.1% | 0% | 7.1% |

Noisy generator (the fault injector, 5 held-out seeds): what the verifier buys.

| Pipeline | Correct | Wrong | Abstained | Unsupported claims in the answer |
|---|---|---|---|---|
| noisy, no verifier | 82.5% [77-87] | 14.4% [11-18] | 3.1% | 41.2% [38-45] |
| + verifier, drop only | 60.3% [54-66] | 0.0% | 39.7% [34-46] | 6.4% [4-9] |
| + verifier, drop + repair | 75.6% [69-81] | 0.0% | 24.4% [19-31] | 6.1% [4-9] |

The trade is visible: the wrong-answer rate goes from 14.4% to 0.0%, and 24.4% of answerable questions become abstentions instead of answers (down from 39.7% without repair). Two honest footnotes: (a) the 6.1% unsupported claims left in answers are, by the table above, subtle additions and antonym flips (the only faults that slip through); a subtle addition on the answer claim keeps the right value, so the correctness check still counts that answer as correct; (b) the naive-RAG and conflict numbers are easy by construction: templated sentences, retrieval recall 100%, and a superseding sentence that is always findable through version metadata. Real documentation does not line up that neatly.

## What does not work, and where the evidence is thin

- **The fault model is mine.** Real model errors are not independent draws from a menu. The verifier threshold was fit on this fault distribution (different products, questions and seeds, same distribution).
- **Faithful claims are verbatim or lightly reworded.** A real model paraphrases much harder, so the false-reject rate (0.0-2.5% here) is a lower bound. A heavy reword is rejected: the test `test_limit_heavy_unseen_reword_of_a_true_claim_is_rejected` pins it.
- **A lexical verifier cannot see these.** Subtle unsourced additions and meaning flips with no cue word get through (41% and 33% above). Tests `test_limit_*` pin that wrong behaviour so nobody mistakes this for entailment. An NLI model or an LLM judge is the fix; I did not build or measure one.
- **Small samples.** 64 answerable and 28 no-answer held-out questions. The five noise seeds are correlated (same questions), so they are not five times the data; the intervals resample questions to reflect that.
- **Ground truth comes from construction**, not human labels: I know which claims are bad because I made them bad.
- **No live model, no live retrieval at scale**, BM25 over 168 passages.

## Run it

```
python -m ragground demo      # every number above (about 90 s)
python -m ragground ask "What is the rate limit of the Nimbus API?"
python -m ragground faults    # one example of each injected fault
python -m ragground corpus
python -m pytest              # 109 tests, ~2 s
```

No third-party packages, no API key, no network. To try a real model, give `ClaudeGenerator` any `prompt -> text` function and pass it to `Pipeline` (untested live).

## Layout

| File | Purpose |
|---|---|
| `ragground/text.py` | tokens, stems, numbers, units, negation polarity, entities |
| `ragground/corpus.py` | seeded synthetic corpus, versions, dev / held-out questions |
| `ragground/index.py` | BM25 plus a version-family lookup |
| `ragground/conflict.py` | supersession and conflict resolution with notes |
| `ragground/generate.py` | claim format, extractive generator, `ClaudeGenerator` adapter |
| `ragground/faults.py` | the fault injector and its labels |
| `ragground/verify.py` | the citation verifier |
| `ragground/answerability.py` | evidence score and dev calibration |
| `ragground/pipeline.py` | the pipeline |
| `ragground/evaluate.py` | held-out measurement, ablations, bootstrap intervals |
