import itertools
import random

import pytest

from flybrain_sort.network import batcher_pairs, layers


def run(pairs, xs):
    xs = list(xs)
    for i, j in pairs:
        if xs[i] > xs[j]:
            xs[i], xs[j] = xs[j], xs[i]
    return xs


@pytest.mark.parametrize("n", range(1, 13))
def test_batcher_sorts_every_binary_input(n):
    # 0-1 principle: a network that sorts all 0/1 inputs sorts everything
    pairs = batcher_pairs(n)
    assert all(0 <= i < j < n for i, j in pairs)
    for bits in itertools.product((0, 1), repeat=n):
        assert run(pairs, bits) == sorted(bits)


@pytest.mark.parametrize("width", [1, 2, 3, None])
def test_layers_preserve_order_and_are_disjoint(width):
    pairs = batcher_pairs(16)
    rounds = layers(pairs, width)
    assert sum(len(r) for r in rounds) == len(pairs)
    for r in rounds:
        wires = [w for g in r for w in g]
        assert len(wires) == len(set(wires))
        if width:
            assert len(r) <= width
    flat = [g for r in rounds for g in r]
    rng = random.Random(0)
    for _ in range(200):
        xs = [rng.random() for _ in range(16)]
        assert run(flat, xs) == sorted(xs)


def test_depth_for_power_of_two():
    assert len(layers(batcher_pairs(8))) == 6
    assert len(batcher_pairs(8)) == 19
