"""BM25 retrieval over passages, plus a metadata lookup for newer versions of a
document family. The version lookup does not depend on what retrieval returned,
which is what lets conflict resolution find a superseding value even when the
newer passage did not make the top-k."""
from __future__ import annotations

import hashlib
import math
from collections import Counter

from .corpus import Passage
from .text import content_tokens


def version_key(version: str) -> tuple[int, ...]:
    return tuple(int(x) for x in version.split("."))


class Index:
    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75):
        self.passages = list(passages)
        self.by_id = {p.id: p for p in self.passages}
        self.k1, self.b = k1, b
        self._tokens = [content_tokens(p.text) for p in self.passages]
        self._tf = [Counter(t) for t in self._tokens]
        self._len = [len(t) for t in self._tokens]
        self._avg = sum(self._len) / max(1, len(self._len))
        df: Counter = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        self._df = df
        self.n = len(self.passages)
        self._families: dict[str, list[Passage]] = {}
        for p in self.passages:
            self._families.setdefault(p.family, []).append(p)

    def idf(self, token: str) -> float:
        """Out-of-vocabulary tokens get the highest idf: a word the corpus has
        never seen is the strongest evidence a question is about something else."""
        df = self._df.get(token, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scored(self, query: str, k: int = 6) -> list[tuple[float, Passage]]:
        q = content_tokens(query, keep_numbers=False)
        scores = []
        for i, p in enumerate(self.passages):
            s = 0.0
            for t in set(q):
                f = self._tf[i].get(t, 0)
                if not f:
                    continue
                norm = f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * self._len[i] / self._avg))
                s += self.idf(t) * norm
            if s > 0:
                # Equal scores (the same passage in two versions) are ordered by a
                # hash of id+query: arbitrary, but neither oldest- nor newest-first.
                tie = hashlib.sha1((p.id + "|" + query).encode()).hexdigest()
                scores.append((s, tie, p))
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [(s, p) for s, _, p in scores[:k]]

    def search(self, query: str, k: int = 6) -> list[Passage]:
        return [p for _, p in self.scored(query, k)]

    def newer_versions(self, passage: Passage) -> list[Passage]:
        """Passages in the same family with a higher version, newest first."""
        mine = version_key(passage.version)
        newer = [p for p in self._families.get(passage.family, [])
                 if version_key(p.version) > mine]
        return sorted(newer, key=lambda p: version_key(p.version), reverse=True)
