"""Measure how well the fly brain compares and sorts.

    python benchmarks/benchmark.py [--trials 10] [--n 8]
"""

import argparse
import time

import numpy as np

from flybrain_sort import FlyBrainSorter


def comparator_resolution(sorter, gaps, trials, center=150.0):
    """Fraction of single comparisons answered correctly vs. rate gap (Hz)."""
    c = sorter.comparators[0]
    rng = np.random.default_rng(1)
    rows = []
    for gap in gaps:
        right = 0
        for t in range(trials):
            lo, hi = center - gap / 2, center + gap / 2
            big_left = bool(rng.integers(2))
            drive = c.drive(hi, lo) if big_left else c.drive(lo, hi)
            s = sorter.brain.simulate(drive, sorter.duration, record=[c.left_readout, c.right_readout], seed=100 + t)
            nl, nr = s.count(c.left_readout), s.count(c.right_readout)
            right += (nl > nr) == big_left and nl != nr
        rows.append((gap, right / trials))
    return rows


def sort_accuracy(sorter, n, trials):
    rng = np.random.default_rng(2)
    exact, taus = 0, []
    for _ in range(trials):
        x = rng.choice(1000, n, replace=False).tolist()
        r = sorter.sort(x)
        exact += r.values == sorted(x)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        disc = sum(r.values[i] > r.values[j] for i, j in pairs)
        taus.append(1 - 2 * disc / len(pairs))
    return exact / trials, float(np.mean(taus))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    sorter = FlyBrainSorter()
    c = sorter.comparators[0]
    print(f"comparator: {c.sensory_type} -> {c.readout_type}, {sorter.duration:g} ms per decision")
    print("\n| rate gap (Hz) | value gap (% of range) | correct |\n|---|---|---|")
    span = sorter.rate_range[1] - sorter.rate_range[0]
    for gap, acc in comparator_resolution(sorter, [2, 5, 10, 20, 40], args.trials):
        print(f"| {gap:g} | {100 * gap / span:.1f}% | {100 * acc:.0f}% |")
    exact, tau = sort_accuracy(sorter, args.n, args.trials)
    print(f"\nsorting {args.trials} random lists of {args.n} distinct integers in [0, 1000):")
    print(f"  perfectly sorted: {100 * exact:.0f}%   mean Kendall tau: {tau:.3f}")
    print(f"  wall clock: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
