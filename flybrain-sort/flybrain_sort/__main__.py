"""Command line: ``python -m flybrain_sort 42 7 19 3``."""

from __future__ import annotations

import argparse
import time

from .sorter import FlyBrainSorter


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="flybrain_sort",
        description="Sort numbers using a whole-brain simulation of the FlyWire fruit fly connectome.",
    )
    ap.add_argument("values", nargs="+", type=float, help="numbers to sort")
    ap.add_argument("--duration", type=float, default=200.0, help="brain time per round, ms (default 200)")
    ap.add_argument("--rates", type=float, nargs=2, default=(60.0, 240.0), metavar=("LO", "HI"),
                    help="sensory firing-rate range values are mapped onto, Hz (default 60 240)")
    ap.add_argument("--mode", choices=["regular", "poisson"], default="regular",
                    help="sensory spike trains: periodic with random phase, or Poisson")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-q", "--quiet", action="store_true", help="print only the sorted values")
    args = ap.parse_args(argv)

    values = [int(v) if v.is_integer() else v for v in args.values]
    t0 = time.time()
    sorter = FlyBrainSorter(rate_range=tuple(args.rates), duration=args.duration,
                            seed=args.seed, mode=args.mode, verbose=not args.quiet)
    if not args.quiet:
        names = ", ".join(f"{c.sensory_type} -> {c.readout_type}" for c in sorter.comparators)
        print(f"connectome: {sorter.brain.n:,} neurons, {sorter.brain.weights.nnz:,} connections")
        print(f"comparators: {names}")
    result = sorter.sort(values)
    if not args.quiet:
        for d in result.decisions:
            i, j = d.wires
            print(f"  round {d.round + 1:>2}  wires {i}<->{j}  readout spikes {d.spikes[0]:>3} vs {d.spikes[1]:>3}"
                  f"  {'swap' if d.swapped else 'keep'}")
        print(f"{result.simulations} whole-brain simulations, {result.brain_time_ms / 1000:g} s of fly time, "
              f"{time.time() - t0:.1f} s wall clock")
        print(f"monotonic: {result.correct}")
    print(" ".join(str(v) for v in result.values))


if __name__ == "__main__":
    main()
