"""Sort numbers with the simulated fly brain.

Pipeline for ``FlyBrainSorter().sort(values)``:

1. **Transduction.** Each value is mapped linearly onto a sensory firing rate
   (``rate_range``, Hz). This is the only place the numbers themselves are
   touched; from here on they exist only as stimulus rates.
2. **Wiring.** Batcher's odd-even merge network fixes which pairs of items get
   compared, round by round (see :mod:`flybrain_sort.network`).
3. **Deciding.** For every round, one whole-brain simulation is run in which
   each compare gate gets its own mirror-pair comparator: the left sensory
   population is driven at item A's rate and the right at item B's (sides
   picked at random per gate). After ``duration`` ms of brain time the item
   whose readout neuron fired more spikes is taken to be the larger one; if the
   counts tie, the earlier first spike wins; if that ties too, the items stay
   put.
4. **Routing.** Items are swapped exactly when the brain said so.

The fly brain is therefore the only thing that ever decides which of two items
is larger. The sort can be wrong when two values are closer than the
comparator's resolution, just like a real animal's discrimination threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import comparators as cmp
from . import data
from .brain import Connectome
from .network import batcher_pairs, layers


@dataclass
class Decision:
    round: int
    wires: tuple  # (i, j) positions being compared, i < j
    comparator: str
    spikes: tuple  # readout spike counts for wire i and wire j
    swapped: bool


@dataclass
class SortResult:
    values: list  # the input, sorted ascending by the fly brain
    order: list  # values[k] == input[order[k]]
    decisions: list = field(default_factory=list)
    simulations: int = 0
    brain_time_ms: float = 0.0

    @property
    def correct(self) -> bool:
        return all(a <= b for a, b in zip(self.values, self.values[1:]))


class FlyBrainSorter:
    def __init__(
        self,
        brain: Connectome | None = None,
        comparators=None,
        rate_range=(60.0, 240.0),
        duration=200.0,
        max_comparators=None,
        seed=0,
        mode="regular",
        cache: Path | None = None,
        verbose=False,
    ):
        self.brain = brain if brain is not None else Connectome.load()
        self.rate_range = rate_range
        self.duration = duration
        self.seed = seed
        self.mode = mode
        self.verbose = verbose
        if comparators is None:
            comparators = self._comparators(cache)
        if not comparators:
            raise RuntimeError("no usable comparators found in this connectome")
        self.comparators = list(comparators)[:max_comparators]

    def _comparators(self, cache):
        cache = cache or data.data_dir() / "comparators.json"
        if cache.exists():
            return cmp.load(cache)
        if self.verbose:
            print("screening mirror-pair comparators in the whole brain ...")
        found = cmp.screen(
            self.brain,
            cmp.discover(self.brain),
            rates=np.linspace(*self.rate_range, 4),
            duration=self.duration,
            verbose=self.verbose,
        )
        if cache.parent.exists():
            cmp.save(found, cache)
        return found

    def rates(self, values) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        lo, hi = self.rate_range
        span = x.max() - x.min()
        if span == 0:
            return np.full(len(x), (lo + hi) / 2)
        return lo + (x - x.min()) / span * (hi - lo)

    def sort(self, values) -> SortResult:
        values = list(values)
        n = len(values)
        result = SortResult(values=values[:], order=list(range(n)))
        if n < 2:
            return result
        rate = self.rates(values)
        rng = np.random.default_rng(self.seed)
        wires = list(range(n))  # wires[k] = index of the input item on wire k
        for r, gates in enumerate(layers(batcher_pairs(n), width=len(self.comparators))):
            drive, plan = {}, []
            for (i, j), c in zip(gates, self.comparators):
                flip = bool(rng.integers(2))  # which hemisphere gets wire i
                a, b = rate[wires[i]], rate[wires[j]]
                drive.update(c.drive(b, a) if flip else c.drive(a, b))
                plan.append((i, j, c, flip))
            record = [x for _, _, c, _ in plan for x in (c.left_readout, c.right_readout)]
            spikes = self.brain.simulate(
                drive, self.duration, record=record, seed=self.seed + 7919 * r, mode=self.mode
            )
            result.simulations += 1
            result.brain_time_ms += self.duration
            for i, j, c, flip in plan:
                ro_i, ro_j = (c.right_readout, c.left_readout) if flip else (c.left_readout, c.right_readout)
                ni, nj = spikes.count(ro_i), spikes.count(ro_j)
                if ni != nj:
                    swap = ni > nj
                else:
                    swap = spikes.first(ro_i) < spikes.first(ro_j)
                if swap:
                    wires[i], wires[j] = wires[j], wires[i]
                result.decisions.append(Decision(r, (i, j), c.sensory_type, (ni, nj), swap))
        result.order = wires
        result.values = [values[k] for k in wires]
        return result


def fly_sort(values, **kwargs) -> list:
    """Convenience wrapper: sort ``values`` ascending with the fly brain."""
    return FlyBrainSorter(**kwargs).sort(values).values
