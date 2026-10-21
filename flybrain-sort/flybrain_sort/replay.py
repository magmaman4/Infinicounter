"""Record a fly-brain sort as a replay for the viewer (``flybrain_sort.gui``).

A replay is plain JSON: the frontal-view positions of a sample of brain
neurons, and for every brain simulation the comparison it decided plus which
neurons spiked in each time bin.
"""

from __future__ import annotations

import numpy as np

from .sorter import FlyBrainSorter

FRAME_MS = 4.0  # spike raster bin
MAX_ACTIVE = 9000  # most active neurons drawn with their spikes
BACKGROUND = 16000  # silent neurons drawn as the brain's outline


def _project(neurons, idx):
    """Frontal view: x = left-right, y = dorsal-ventral (FlyWire voxels)."""
    x = neurons.pos_x.to_numpy()[idx] * 4
    y = neurons.pos_y.to_numpy()[idx] * 4
    return x, y


def record(sorter: FlyBrainSorter, values, seed=0) -> dict:
    brain = sorter.brain
    rounds = []

    def observer(r, decisions, spikes, rates):
        rounds.append((decisions, spikes))

    result = sorter.sort(values, observer=observer)

    nrn = brain.neurons
    placed = np.flatnonzero(nrn.pos_x.notna().to_numpy() & nrn.pos_y.notna().to_numpy())
    has_pos = np.zeros(brain.n, bool)
    has_pos[placed] = True

    # neurons to animate: the most active ones, plus every comparator neuron
    counts = np.zeros(brain.n, np.int64)
    for _, spikes in rounds:
        np.add.at(counts, spikes.neurons, 1)
    key = set()
    for c in sorter.comparators:
        key.update(c.left_inputs + c.right_inputs + [c.left_readout, c.right_readout])
    active = np.flatnonzero((counts > 0) & has_pos)
    active = active[np.argsort(-counts[active])][:MAX_ACTIVE]
    shown = np.unique(np.concatenate([active, np.fromiter(key, np.int64)]))
    shown = shown[has_pos[shown]]

    rng = np.random.default_rng(seed)
    quiet = np.setdiff1d(placed, shown)
    quiet = rng.choice(quiet, min(BACKGROUND, len(quiet)), replace=False)

    x, y = _project(nrn, np.concatenate([shown, quiet]))
    x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
    scale = 1000 / max(x1 - x0, y1 - y0)

    def q(ix):
        px, py = _project(nrn, ix)
        return np.stack([(px - x0) * scale, (py - y0) * scale], 1).round().astype(int).ravel().tolist()

    local = {int(g): k for k, g in enumerate(shown)}
    role = np.zeros(len(shown), np.int8)  # 0 brain, 1 left sensory, 2 right sensory, 3 left readout, 4 right readout
    for c in sorter.comparators:
        for ids, code in ((c.left_inputs, 1), (c.right_inputs, 2), ([c.left_readout], 3), ([c.right_readout], 4)):
            for g in ids:
                if g in local:
                    role[local[g]] = code

    sensory = np.fromiter((g for c in sorter.comparators for g in c.left_inputs + c.right_inputs), np.int64)
    n_frames = int(np.ceil(sorter.duration / FRAME_MS))
    out_rounds = []
    for decisions, spikes in rounds:
        # driven sensory neurons fire at their stimulus rates; the viewer redraws them from those
        keep = np.isin(spikes.neurons, shown) & ~np.isin(spikes.neurons, sensory)
        ids = np.searchsorted(shown, spikes.neurons[keep])
        frame = np.minimum((spikes.times[keep] / FRAME_MS).astype(int), n_frames - 1)
        frames = [[] for _ in range(n_frames)]
        for f, i in sorted(set(zip(frame.tolist(), ids.tolist()))):
            frames[f].append(i)
        comps = []
        for d, c in zip(decisions, sorter.comparators):
            left_ro, right_ro = c.left_readout, c.right_readout
            comps.append(dict(
                wires=list(d.wires),
                items=list(d.items),
                rates=[round(float(r), 1) for r in d.rates],
                flipped=d.flipped,
                spikes=list(d.spikes),
                swapped=d.swapped,
                readout_times={
                    "left": spikes.times[spikes.neurons == left_ro].round(1).tolist(),
                    "right": spikes.times[spikes.neurons == right_ro].round(1).tolist(),
                },
            ))
        out_rounds.append(dict(comparisons=comps, total_spikes=spikes.total_spikes, frames=frames))

    c = sorter.comparators[0]
    return dict(
        values=list(values),
        sorted=result.values,
        order=result.order,
        correct=result.correct,
        duration_ms=sorter.duration,
        frame_ms=FRAME_MS,
        rate_range=list(sorter.rate_range),
        brain=dict(neurons=brain.n, connections=int(brain.weights.nnz)),
        comparator=dict(sensory=c.sensory_type, readout=c.readout_type,
                        sensory_per_side=[len(c.left_inputs), len(c.right_inputs)], synapses=c.synapses),
        shown=q(shown),
        roles=role.tolist(),
        background=q(quiet),
        rounds=out_rounds,
    )
