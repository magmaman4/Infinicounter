# flybrain-sort

A sorting algorithm in which **every comparison is decided by a simulation of
the complete adult fruit fly brain**: all 138,639 neurons and ~15 million
synaptic connections of the [FlyWire](https://flywire.ai) connectome.

```
$ python -m flybrain_sort 42 7 19 3 88 61 25 50
connectome: 138,639 neurons, 15,091,983 connections
comparators: BM_InOm -> AN_GNG_89
  round  1  wires 0<->1  readout spikes  48 vs  13  swap
  round  2  wires 2<->3  readout spikes  30 vs   9  swap
  round  3  wires 4<->5  readout spikes  64 vs  55  swap
  ...
  round 18  wires 3<->4  readout spikes  47 vs  36  swap
  round 19  wires 5<->6  readout spikes  50 vs  56  keep
19 whole-brain simulations, 3.8 s of fly time, 36.1 s wall clock
monotonic: True
3 7 19 25 42 50 61 88
```

## How it works

Python never compares two of your numbers. It only

1. **transduces** each number into a sensory firing rate (a linear map onto
   60-240 Hz, like turning stimulus intensity into spikes),
2. **routes** items through a fixed [Batcher odd-even merge sorting
   network](https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort)
   (the gates are fixed in advance and never look at the data), and
3. **listens** to two neurons and swaps a pair of items when the brain says so.

Every data-dependent decision comes from the simulated brain.

### The brain's comparator

Many fly sensory neurons come as mirror pairs: a left population wired to a
neuron in the left hemisphere and a right population wired to its twin on the
right. `flybrain_sort.comparators.discover` scans the FlyWire annotations for
every such lateralized sensory cell type and picks the readouts from the
wiring alone. `screen` then simulates each candidate in the full brain and
keeps only the ones that respond monotonically and symmetrically.

Out of 14 candidates found in FlyWire v783, one passes:

| | |
|---|---|
| sensory neurons | `BM_InOm`: mechanosensory neurons of the interommatidial bristles on the eye, 555 left / 558 right |
| readout | `AN_GNG_89`: one ascending neuron per hemisphere in the gnathal ganglion |
| wiring | ~1,786 synapses per side, **zero** crossing to the other side |

The other 13 fail because the real connectome is not perfectly mirror
symmetric. For example, `LB3d` taste neurons drive their left readout about
three times harder than the right one. A biased comparator would sort wrongly,
so it is rejected rather than corrected for.

To compare items *a* and *b*, the left bristle neurons fire at *a*'s rate and
the right ones at *b*'s (which side gets which is picked at random per
comparison), the whole brain runs for 200 ms, and the item whose
`AN_GNG_89` fired more spikes is the larger. Ties go to the earlier first
spike. When several comparators pass screening, the independent comparisons of
a network round share one simulation, one comparator each. With the single
comparator FlyWire offers, each simulation decides one comparison. Sorting *n* items takes O(*n* log² *n*) simulations (19 for 8 items).

### The brain model

`flybrain_sort.brain` is a numpy re-implementation of the leaky
integrate-and-fire model of [Shiu et al. 2024, *Nature*](https://doi.org/10.1038/s41586-024-07763-9),
which reproduces a range of known sensorimotor behaviors from the
connectome alone. It uses their published parameters (resting potential
-52 mV, threshold -45 mV, membrane time constant 20 ms, synaptic time constant
5 ms, delay 1.8 ms, 0.275 mV per synapse) and every synapse's predicted
neurotransmitter sign. Sub-threshold dynamics are integrated exactly on a
0.1 ms grid. The stimulus spreads to about 113,000 neurons within three
synapses, so the whole brain is simulated at every step. That takes about
1.5-2 s of wall clock per 200 ms of fly time on a 4-core CPU.

## Accuracy

Like any animal, the comparator has a discrimination threshold: values that
map to nearly the same firing rate can come out in the wrong order. From
`python benchmarks/benchmark.py --trials 10` on FlyWire v783:

Single comparisons (10 trials each, around 150 Hz):

| rate gap (Hz) | value gap (% of range) | correct |
|---|---|---|
| 2 | 1.1% | 60% |
| 5 | 2.8% | 90% |
| 10 | 5.6% | 90% |
| 20 | 11.1% | 100% |
| 40 | 22.2% | 100% |

Full sorts of 10 random lists of 8 distinct integers in [0, 1000): **8 of 10
perfectly sorted**, mean Kendall tau 0.971. The errors were adjacent items
whose values were close together. A longer `--duration` gives the readout more
spikes to count and sharpens discrimination.

## Usage

```bash
pip install -e .            # numpy, scipy, pandas, pyarrow
python -m flybrain_sort 5 3 9 1 7
```

```python
from flybrain_sort import FlyBrainSorter

sorter = FlyBrainSorter()            # loads the connectome, finds comparators
result = sorter.sort([42, 7, 19, 3])
result.values      # [3, 7, 19, 42]
result.decisions   # every comparison with the readout spike counts
```

On first use about 135 MB of connectome data is downloaded into `./data`
(set `FLYBRAIN_DATA` to put it somewhere else). Comparator screening runs
once and is cached in `data/comparators.json`.

Options: `--duration` (ms of brain time per round; longer means more
evidence and finer discrimination), `--rates LO HI`, `--mode poisson`
(Poisson instead of regular sensory spike trains), `--seed`.

## Tests

```bash
pytest
```

The tests use a small synthetic connectome, so they run in about a second
without downloading anything. They check the sorting network
exhaustively with the 0-1 principle and run the full discover, screen and sort
pipeline on a toy brain.

## Data and credits

* Connectome: FlyWire Consortium, Dorkenwald et al. 2024, *Nature* 634, 124-138.
* Cell types: Schlegel et al. 2024, *Nature* 634, 139-152
  ([flywire_annotations](https://github.com/flyconnectome/flywire_annotations)).
* Model and connectivity tables: Shiu et al. 2024, *Nature* 634, 210-219
  ([Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)).

The data is downloaded from these sources and is not redistributed here.
Please cite them if you use this project.
