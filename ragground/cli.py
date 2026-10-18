"""Command line: `demo` (every number in the README), `ask`, `faults`, `corpus`."""
from __future__ import annotations

import argparse
import random
from collections import Counter

from . import evaluate as ev
from .conflict import resolve_passages
from .faults import WEIGHTS, apply_fault
from .generate import ExtractiveGenerator
from .pipeline import Pipeline
from .verify import Config, Verifier


def pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def ci(t) -> str:
    return f"{100 * t[0]:5.1f}% [{100 * t[1]:.0f}-{100 * t[2]:.0f}]"


def cmd_corpus(world: ev.World) -> None:
    c = world.corpus
    print(f"{len(c.passages)} passages, {len({p.family for p in c.passages})} document families, "
          f"{len(c.products)} fictional products (seed {c.seed})")
    for split in ("dev", "heldout"):
        k = Counter(q.kind for q in c.split(split))
        print(f"  {split:8s} " + ", ".join(f"{n} {kind}" for kind, n in sorted(k.items())))
    raw, res = ev.retrieval_recall(world, c.heldout)
    print(f"  held-out retrieval: gold sentence in top-{ev.K} for {pct(raw)} of answerable "
          f"questions ({pct(res)} after conflict resolution)")


def cmd_ask(world: ev.World, question: str) -> None:
    gate = ev.select_gate(world)
    ext = ExtractiveGenerator(world.index)
    cfg = ev.calibrate_verifier(world, Config(), ev.collect(world, world.corpus.dev, ev.DEV_SEEDS))
    pipe = Pipeline(world.index, ext, verifier=Verifier(world.index, world.known, cfg),
                    abstain_threshold=gate.threshold, abstain_score=gate.score_fn)
    print(pipe.ask(question).render())


def cmd_faults() -> None:
    rng = random.Random(0)
    sample = "The Nimbus API enforces a rate limit of 600 requests per minute per API key."
    flag = "The Nimbus API does not support mutual TLS."
    print(f"original: {sample}")
    for f in WEIGHTS:
        if f in ("wrong_citation", "fabricated_citation", "none"):
            continue
        text = flag if f in ("negation_dropped", "antonym_flip") else sample
        print(f"  {f:20s} {apply_fault(f, text, rng)}")


def cmd_demo() -> None:
    world = ev.build_world()
    print("== corpus ==")
    cmd_corpus(world)

    dev = ev.collect(world, world.corpus.dev, ev.DEV_SEEDS)
    held = ev.collect(world, world.corpus.heldout, ev.HELDOUT_SEEDS)
    mix = Counter(lc.fault for it in held for lc in it.labeled)
    n = sum(mix.values())
    print(f"\n== labelled claims (held-out products, seeds {ev.HELDOUT_SEEDS}): {n} claims ==")
    print("  " + ", ".join(f"{k} {v}" for k, v in mix.most_common()))

    print("\n== verifier configurations (each calibrated on dev claims only) ==")
    print(f"{'configuration':26s} {'thr':>4s} {'unsupported before':>19s} {'after (kept)':>20s} "
          f"{'bad caught':>11s} {'faithful rejected':>18s} {'cite precision':>15s}")
    rows = ev.verifier_table(world, dev, held)
    for r in rows:
        print(f"{r.name:26s} {r.threshold:4.2f} {pct(r.unsupported_before):>19s} "
              f"{ci(r.after_ci()):>20s} {pct(r.catch_rate):>11s} "
              f"{pct(r.false_reject_rate):>18s} "
              f"{100 * r.prec_before:5.1f}->{100 * r.prec_after:5.1f}%")
    full = rows[-1]

    print("\n== per fault type, full verifier with citation repair ==")
    print(f"{'fault':22s} {'n':>4s}  outcome")
    for f, c in sorted(full.per_fault.items(), key=lambda kv: -kv[1]["n"]):
        parts = [f"{k} {v}" for k, v in c.items() if k != "n"]
        print(f"{f:22s} {c['n']:4d}  " + ", ".join(parts))

    print("\n== bad claims the full verifier kept ==")
    for fault, text, origin in ev.slipped_examples(world, full_cfg(rows), held, 6):
        print(f"  [{fault}] {text}\n      from: {origin}")
    print("\n== faithful claims the full verifier rejected ==")
    for fault, text, reasons in ev.rejected_faithful(world, full_cfg(rows), held, 4):
        print(f"  [{fault}] {text}  ({', '.join(reasons)})")

    print("\n== abstention (threshold chosen on dev, scored on held-out) ==")
    arows, gate = ev.abstention_table(world)
    print(f"  dev balanced accuracy: " + ", ".join(f"{k} {v:.3f}" for k, v in gate.dev_ba.items())
          + f"  -> pipeline gate: {gate.name} (threshold {gate.threshold:.2f})")
    print(f"{'gate':30s} {'threshold':>9s} {'precision':>10s} {'recall':>7s} {'answerable lost':>16s}")
    for a in arows:
        thr = "-" if a.threshold is None else f"{a.threshold:.2f}"
        print(f"{a.name:30s} {thr:>9s} {pct(a.precision):>10s} {pct(a.recall):>7s} "
              f"{pct(a.answerable_lost):>9s} ({a.abstained_answerable}/{a.n_answerable})")

    print("\n== end to end on held-out products (95% CI resamples questions) ==")
    tables = ev.e2e_table(world)
    ans = ("ans_correct", "ans_wrong", "ans_abstain")
    print(f"{'pipeline':50s} {'correct':>16s} {'wrong':>16s} {'abstained':>16s} "
          f"{'stale (conflict)':>16s} {'no-answer qs answered':>22s} {'unsupported claims':>19s}")
    for group in ("clean", "noisy"):
        print(f"-- {group} generator" + (" (5 held-out noise seeds)" if group == "noisy" else ""))
        for e in tables[group]:
            print(f"{e.name:50s} {ci(e.ratio('ans_correct', ans)):>16s} "
                  f"{ci(e.ratio('ans_wrong', ans)):>16s} {ci(e.ratio('ans_abstain', ans)):>16s} "
                  f"{ci(e.ratio('stale', ('conf_n',), ('conflict',))):>16s} "
                  f"{ci(e.ratio('un_answered', ('un_n',))):>22s} "
                  f"{(ci(e.ratio('bad_claims', ('claims',))) if group == 'noisy' else '-'):>19s}")


def full_cfg(rows) -> Config:
    r = rows[-1]
    return Config(threshold=r.threshold)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ragground")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="print every number in the README")
    sub.add_parser("corpus", help="corpus and split summary")
    sub.add_parser("faults", help="one example of each injected fault")
    a = sub.add_parser("ask", help="answer one question with citations")
    a.add_argument("question")
    args = ap.parse_args(argv)
    if args.cmd == "demo":
        cmd_demo()
    elif args.cmd == "corpus":
        cmd_corpus(ev.build_world())
    elif args.cmd == "faults":
        cmd_faults()
    else:
        cmd_ask(ev.build_world(), args.question)
    return 0
