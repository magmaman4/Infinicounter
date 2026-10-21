"""Fast tests on a small synthetic brain with the same structure as FlyWire."""

import numpy as np
import pandas as pd
import scipy.sparse as sp

from flybrain_sort import FlyBrainSorter
from flybrain_sort import comparators as cmp
from flybrain_sort.brain import Connectome


def toy_brain(n_sensory=40, n_other=200, seed=0):
    """Mirror-symmetric sensory -> readout pathways inside random background wiring."""
    rng = np.random.default_rng(seed)
    left = np.arange(n_sensory)
    right = left + n_sensory
    rl, rr = 2 * n_sensory, 2 * n_sensory + 1
    n = 2 * n_sensory + 2 + n_other
    rows, cols, vals = [], [], []
    for group, readout in ((left, rl), (right, rr)):
        rows += group.tolist()
        cols += [readout] * len(group)
        vals += [8.0] * len(group)
    # sparse background: readouts drive the rest, plus weak random recurrence
    bg = np.arange(2 * n_sensory + 2, n)
    for r in (rl, rr):
        tgt = rng.choice(bg, 20, replace=False)
        rows += [r] * 20
        cols += tgt.tolist()
        vals += rng.choice([-2.0, 2.0], 20).tolist()
    m = 3 * n_other
    rows += rng.choice(bg, m).tolist()
    cols += rng.choice(bg, m).tolist()
    vals += rng.choice([-1.0, 1.0], m).tolist()
    w = sp.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(n, n))
    neurons = pd.DataFrame({
        "root_id": np.arange(n),
        "super_class": ["sensory"] * (2 * n_sensory) + ["central"] * (n - 2 * n_sensory),
        "cell_class": None,
        "cell_type": ["toy_SN"] * (2 * n_sensory) + ["toy_RO", "toy_RO"] + [None] * n_other,
        "side": ["left"] * n_sensory + ["right"] * n_sensory + ["left", "right"] + ["center"] * n_other,
        "pos_x": rng.uniform(0, 1e5, n),
        "pos_y": rng.uniform(0, 5e4, n),
        "pos_z": rng.uniform(0, 5e3, n),
    })
    return Connectome(w, neurons)


def test_simulation_rate_increases_readout_spikes():
    brain = toy_brain()
    (c,) = cmp.discover(brain)
    lo = brain.simulate(c.drive(40, 40), 100, record=[c.left_readout])
    hi = brain.simulate(c.drive(200, 200), 100, record=[c.left_readout])
    assert 0 < lo.count(c.left_readout) < hi.count(c.left_readout)


def test_discover_and_screen_find_the_mirror_pair():
    brain = toy_brain()
    found = cmp.discover(brain)
    assert [(c.sensory_type, c.readout_type) for c in found] == [("toy_SN", "toy_RO")]
    kept = cmp.screen(brain, found, duration=100)
    assert len(kept) == 1 and kept[0].gain > 0


def test_sorter_sorts_with_the_toy_brain(tmp_path):
    brain = toy_brain()
    sorter = FlyBrainSorter(brain, cache=tmp_path / "c.json", duration=100)
    values = [5, 1, 9, 3, 7, 2, 8]
    result = sorter.sort(values)
    assert result.values == sorted(values)
    assert sorted(result.order) == list(range(len(values)))
    assert [values[k] for k in result.order] == result.values
    assert result.simulations == len({d.round for d in result.decisions})


def test_trivial_inputs():
    sorter = FlyBrainSorter(toy_brain(), comparators=cmp.discover(toy_brain()), duration=50)
    assert sorter.sort([]).values == []
    assert sorter.sort([4]).values == [4]
    assert sorter.sort([2, 2, 2]).values == [2, 2, 2]
