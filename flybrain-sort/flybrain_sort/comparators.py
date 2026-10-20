"""Find the fly brain's own left-vs-right comparators.

Many sensory neuron types exist as a mirror pair: a left population wired to a
left downstream neuron and a right population wired to that neuron's
right-hemisphere twin, with (almost) no crossing between the two. Driving the
left population at a rate encoding value ``a`` and the right one at a rate
encoding ``b`` turns that pair into a comparator: whichever hemisphere's
readout neuron fires more spikes saw the larger value.

Nothing is hand-picked. :func:`discover` scans every lateralized sensory cell
type in the annotations and picks its readouts from the wiring alone;
:func:`screen` then simulates each candidate in the whole brain and keeps the
ones that behave as fair comparators (monotonic, left/right symmetric). Real
connectomes are not perfectly mirror-symmetric, so most candidates fail.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .brain import Connectome


@dataclass
class Comparator:
    sensory_type: str  # FlyWire cell type of the driven sensory neurons
    readout_type: str  # FlyWire cell type of the readout neurons
    left_inputs: list  # neuron indices
    right_inputs: list
    left_readout: int
    right_readout: int
    synapses: int = 0  # sensory -> readout synapses (mean of both sides)
    asymmetry: float = 0.0  # screening result, spikes
    gain: float = 0.0  # screening result, spikes per Hz

    def drive(self, rate_left: float, rate_right: float) -> dict:
        d = dict.fromkeys(self.left_inputs, rate_left)
        d.update(dict.fromkeys(self.right_inputs, rate_right))
        return d


def discover(brain: Connectome, min_neurons=5, min_synapses=100, max_crossing=0.05):
    """Candidate mirror-pair comparators, strongest first."""
    nrn = brain.neurons
    sensory = nrn[(nrn.super_class == "sensory") & nrn.side.isin(["left", "right"])]
    sensory = sensory[sensory.cell_type.notna()]
    found, used = [], set()
    for ctype, group in sensory.groupby("cell_type"):
        left = group.index[group.side == "left"].to_numpy()
        right = group.index[group.side == "right"].to_numpy()
        if min(len(left), len(right)) < min_neurons:
            continue
        from_l = np.asarray(brain.weights[left].sum(0)).ravel()
        from_r = np.asarray(brain.weights[right].sum(0)).ravel()
        rl = int(np.argmax(from_l - from_r))
        rr = int(np.argmax(from_r - from_l))
        if rl == rr or rl in used or rr in used:
            continue
        rtype = nrn.cell_type.iloc[rl]
        if not isinstance(rtype, str) or rtype != nrn.cell_type.iloc[rr]:
            continue
        own = min(from_l[rl], from_r[rr])
        cross = max(from_r[rl], from_l[rr])
        if own < min_synapses or cross > max_crossing * own:
            continue
        used.update((rl, rr))
        found.append(
            Comparator(
                ctype,
                rtype,
                left.tolist(),
                right.tolist(),
                rl,
                rr,
                synapses=int(round((from_l[rl] + from_r[rr]) / 2)),
            )
        )
    return sorted(found, key=lambda c: -c.synapses)


def screen(
    brain: Connectome,
    candidates,
    rates=(60.0, 120.0, 180.0, 240.0),
    duration=200.0,
    max_asymmetry=0.1,
    min_gain=0.05,
    seed=0,
    verbose=False,
):
    """Keep candidates that respond monotonically and symmetrically.

    All candidates are driven together, both sides at the same rate, at each
    test rate. A candidate passes if each readout's spike count rises with the
    rate (``gain`` spikes/Hz) and the left/right count difference stays below
    ``max_asymmetry`` of the response range.
    """
    record = [c.left_readout for c in candidates] + [c.right_readout for c in candidates]
    curves = np.zeros((len(candidates), len(rates), 2))
    for j, r in enumerate(rates):
        drive = {}
        for c in candidates:
            drive.update(c.drive(r, r))
        spikes = brain.simulate(drive, duration, record=record, seed=seed + j)
        for i, c in enumerate(candidates):
            curves[i, j] = spikes.count(c.left_readout), spikes.count(c.right_readout)
    kept = []
    for c, curve in zip(candidates, curves):
        span = curve[-1].min() - curve[0].max()
        c.gain = float(span / (rates[-1] - rates[0]))
        c.asymmetry = float(np.abs(curve[:, 0] - curve[:, 1]).max())
        monotonic = (np.diff(curve, axis=0) > 0).all()
        ok = monotonic and c.gain >= min_gain and c.asymmetry <= max_asymmetry * max(span, 1)
        if verbose:
            shown = " ".join(f"{int(l)}/{int(r)}" for l, r in curve)
            print(f"  {'PASS' if ok else 'fail'} {c.sensory_type:>16} -> {c.readout_type:<12} {shown}")
        if ok:
            kept.append(c)
    return sorted(kept, key=lambda c: c.asymmetry / max(c.gain, 1e-9))


def save(comparators, path: Path):
    path.write_text(json.dumps([asdict(c) for c in comparators], indent=1))


def load(path: Path):
    return [Comparator(**d) for d in json.loads(path.read_text())]
