"""Whole-brain leaky integrate-and-fire model of the adult fruit fly.

Every one of the 138,639 proofread FlyWire neurons is simulated, connected by
the ~15 million connectome edges with their measured synapse counts and
predicted neurotransmitter sign (excitatory +, inhibitory -). Neuron and
synapse parameters are the ones published by Shiu et al. (2024), whose model
reproduces many known sensorimotor pathways; this is a dependency-light numpy
re-implementation of that model (their reference version uses Brian 2).

    dv/dt = (v_0 - v + g) / t_mbr
    dg/dt = -g / tau
    presynaptic spike  ->  g_post += w_syn * (signed synapse count), after t_dly
    v > v_th           ->  spike; v = v_rst; g = 0; refractory for t_rfc

The linear sub-threshold dynamics are integrated exactly on a 0.1 ms grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import data

# Shiu et al. 2024 (units: mV, ms)
PARAMS = dict(
    v_0=-52.0,  # resting potential
    v_rst=-52.0,  # reset potential
    v_th=-45.0,  # spike threshold
    t_mbr=20.0,  # membrane time constant
    tau=5.0,  # synaptic time constant
    t_rfc=2.2,  # refractory period
    t_dly=1.8,  # synaptic delay
    w_syn=0.275,  # mV per synapse
    dt=0.1,  # integration step
)


@dataclass
class Connectome:
    """Signed synapse-count matrix plus per-neuron annotations.

    ``weights[i, j]`` is the signed number of synapses from neuron ``i`` onto
    neuron ``j`` (CSR, rows = presynaptic). ``neurons`` is indexed by the same
    integer neuron index and carries ``root_id`` and FlyWire annotations
    (``cell_type``, ``side``, ``super_class``, ...).
    """

    weights: sp.csr_matrix
    neurons: pd.DataFrame

    @property
    def n(self) -> int:
        return self.weights.shape[0]

    @classmethod
    def load(cls, directory=None) -> "Connectome":
        """Load FlyWire v783, downloading it (~135 MB) on first use."""
        paths = data.fetch_all(directory)
        comp = pd.read_csv(paths["Completeness_783.csv"], index_col=0)
        con = pd.read_parquet(
            paths["Connectivity_783.parquet"],
            columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
        )
        n = len(comp)
        w = sp.csr_matrix(
            (
                con["Excitatory x Connectivity"].to_numpy(np.float32),
                (con["Presynaptic_Index"].to_numpy(), con["Postsynaptic_Index"].to_numpy()),
            ),
            shape=(n, n),
        )
        ann = pd.read_csv(
            paths["Supplemental_file1_neuron_annotations.tsv"],
            sep="\t",
            usecols=["root_id", "super_class", "cell_class", "cell_type", "side", "pos_x", "pos_y", "pos_z"],
            dtype=str,
        )
        ann["root_id"] = ann["root_id"].astype(np.int64)
        for axis in ("pos_x", "pos_y", "pos_z"):  # FlyWire voxels (4 x 4 x 40 nm)
            ann[axis] = pd.to_numeric(ann[axis], errors="coerce")
        neurons = pd.DataFrame({"root_id": comp.index.to_numpy(np.int64)})
        neurons = neurons.merge(ann.drop_duplicates("root_id"), on="root_id", how="left")
        return cls(w, neurons)

    def where(self, **match) -> np.ndarray:
        """Indices of neurons whose annotation columns equal the given values."""
        mask = np.ones(self.n, bool)
        for col, val in match.items():
            mask &= (self.neurons[col] == val).to_numpy()
        return np.flatnonzero(mask)

    def simulate(self, drive, duration=100.0, record=(), seed=0, mode="regular", params=None):
        """Run the whole brain for ``duration`` ms.

        ``drive`` maps neuron index -> firing rate (Hz) imposed on that neuron,
        the way an optogenetic or sensory stimulus would. ``mode`` is
        ``"regular"`` (periodic trains with random phase) or ``"poisson"``.
        Spikes of the neurons in ``record`` are returned.
        """
        return simulate(self.weights, drive, duration, record, seed, mode, params)


@dataclass
class SpikeRecord:
    """Spikes from the recorded neurons: parallel arrays of times and indices."""

    times: np.ndarray
    neurons: np.ndarray
    total_spikes: int = 0
    counts: dict = field(default_factory=dict)

    def count(self, i: int) -> int:
        return self.counts.get(int(i), 0)

    def first(self, i: int) -> float:
        hit = self.times[self.neurons == i]
        return float(hit[0]) if len(hit) else np.inf


def simulate(weights, drive, duration=100.0, record=(), seed=0, mode="regular", params=None):
    p = {**PARAMS, **(params or {})}
    if mode not in ("regular", "poisson"):
        raise ValueError(f"unknown drive mode {mode!r}")
    rng = np.random.default_rng(seed)
    n = weights.shape[0]
    dt = p["dt"]
    steps = int(round(duration / dt))
    delay = max(1, int(round(p["t_dly"] / dt)))
    refractory_steps = int(round(p["t_rfc"] / dt))
    indptr, indices = weights.indptr, weights.indices
    w = (weights.data * p["w_syn"]).astype(np.float32)

    # exact propagator for the linear (v, g) system over one step:
    #   v' - v_0 = (v - v_0) * em + g * c,   g' = g * eg
    f32 = np.float32
    em, eg = f32(np.exp(-dt / p["t_mbr"])), f32(np.exp(-dt / p["tau"]))
    c = f32(p["tau"] / (p["tau"] - p["t_mbr"]) * (np.exp(-dt / p["tau"]) - np.exp(-dt / p["t_mbr"])))
    v0, v_th, v_rst = f32(p["v_0"]), f32(p["v_th"]), f32(p["v_rst"])

    u = np.zeros(n, f32)  # v - v_0
    g = np.zeros(n, f32)
    tmp = np.empty(n, f32)
    ref_until = np.zeros(n, np.int32)  # first step a neuron integrates again
    active = np.empty(n, bool)
    pending = np.zeros((delay, n), f32)  # ring buffer of delayed synaptic input

    src = np.fromiter(drive.keys(), np.int64, len(drive))
    rate = np.fromiter(drive.values(), np.float64, len(drive))
    src, rate = src[rate > 0], rate[rate > 0]
    period = 1000.0 / rate
    next_spike = rng.random(len(src)) * period
    p_spike = rate * dt / 1000.0

    rec_mask = np.zeros(n, bool)
    rec_mask[np.asarray(list(record), dtype=np.int64)] = True
    rec_t, rec_i, total = [], [], 0
    th = v_th - v0

    for step in range(steps):
        t = step * dt
        slot = step % delay
        g += pending[slot]
        pending[slot].fill(0)

        np.less_equal(ref_until, step, out=active)
        np.multiply(g, c, out=tmp)
        tmp += u * em
        np.copyto(u, tmp, where=active)
        np.multiply(g, eg, out=tmp)
        np.copyto(g, tmp, where=active)

        fired = np.flatnonzero(u > th)
        if mode == "regular":
            due = next_spike <= t
            forced = src[due]
            next_spike[due] += period[due]
        else:
            forced = src[rng.random(len(src)) < p_spike]
        if len(forced):
            fired = np.union1d(fired, forced)
        if not len(fired):
            continue

        u[fired] = v_rst - v0
        g[fired] = 0.0
        ref_until[fired] = step + 1 + refractory_steps
        ref_until[forced] = 0  # driven neurons follow their stimulus
        total += len(fired)

        lo, hi = indptr[fired], indptr[fired + 1]
        if (hi - lo).any():
            seg = np.concatenate([np.arange(s, e) for s, e in zip(lo, hi)])
            np.add.at(pending[slot], indices[seg], w[seg])

        hit = fired[rec_mask[fired]]
        if len(hit):
            rec_t.append(np.full(len(hit), t))
            rec_i.append(hit)

    times = np.concatenate(rec_t) if rec_t else np.zeros(0)
    neurons = np.concatenate(rec_i) if rec_i else np.zeros(0, np.int64)
    ids, cnt = np.unique(neurons, return_counts=True)
    return SpikeRecord(times, neurons, total, dict(zip(ids.tolist(), cnt.tolist())))
