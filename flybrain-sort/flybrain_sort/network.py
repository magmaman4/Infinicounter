"""Batcher's odd-even merge sorting network.

A sorting network is a fixed wiring of compare-exchange gates; which gates fire
never depends on the data, only their outcomes do. That makes it a natural fit
for a neural comparator: the wiring is fixed up front and every data-dependent
decision is handed to the fly brain.
"""

from __future__ import annotations


def batcher_pairs(n: int) -> list[tuple[int, int]]:
    """Compare-exchange gates (i, j), i < j, that sort ``n`` wires ascending.

    Built for the next power of two; gates touching the padding wires are
    dropped, which is valid because padding acts as +infinity and never moves.
    """
    size = 1
    while size < n:
        size *= 2
    pairs = []
    p = 1
    while p < size:
        k = p
        while k >= 1:
            for j in range(k % p, size - k, 2 * k):
                for i in range(min(k, size - j - k)):
                    a, b = i + j, i + j + k
                    if a // (2 * p) == b // (2 * p) and b < n:
                        pairs.append((a, b))
            k //= 2
        p *= 2
    return pairs


def layers(pairs, width: int | None = None) -> list[list[tuple[int, int]]]:
    """Group gates into rounds that touch disjoint wires, in order.

    Each round holds at most ``width`` gates (the number of comparators that
    can run side by side in one brain simulation).
    """
    rounds: list[list[tuple[int, int]]] = []
    busy: list[set[int]] = []
    for a, b in pairs:
        # earliest round after the last one that used either wire
        start = 0
        for r in range(len(rounds) - 1, -1, -1):
            if a in busy[r] or b in busy[r]:
                start = r + 1
                break
        r = start
        while r < len(rounds) and width is not None and len(rounds[r]) >= width:
            r += 1
        if r == len(rounds):
            rounds.append([])
            busy.append(set())
        rounds[r].append((a, b))
        busy[r].update((a, b))
    return rounds
