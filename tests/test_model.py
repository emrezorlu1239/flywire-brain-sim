"""
Fast sanity checks on tiny hand-made networks (no download needed).

    pytest -q
"""

import numpy as np
import torch

from flybrain.data import Connectome
from flybrain.model import BrainSimulator, LIFParams, Stimulus


def tiny(pre, post, w, n):
    """Build a Connectome from an edge list."""
    pre, post, w = map(np.asarray, (pre, post, w))
    order = np.argsort(pre, kind="stable")
    pre, post, w = pre[order], post[order], w[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, pre + 1, 1)
    sign = np.zeros(n, dtype=np.int8)
    sign[pre] = np.sign(w)
    return Connectome(np.arange(n), torch.from_numpy(np.cumsum(indptr)),
                      torch.from_numpy(post.astype(np.int64)),
                      torch.from_numpy(w.astype(np.float32)), sign)


def test_silent_without_input():
    c = tiny([0], [1], [100], 2)
    res = BrainSimulator(c, LIFParams(n_trials=2, duration_ms=200)).run([])
    assert len(res.neurons) == 0


def test_stimulated_neuron_follows_poisson_rate():
    c = tiny([0], [1], [0], 2)
    res = BrainSimulator(c, LIFParams(n_trials=8, duration_ms=2000), seed=0).run([Stimulus("s", [0], 100)])
    assert 90 < res.rates()[0] < 110


def test_strong_synapse_drives_partner_after_delay():
    # 200 synapses x 0.275 mV = 55 mV of synaptic drive g. The input goes into g, and v
    # then charges up with tau_m = 20 ms, so the partner fires a few ms after the delay.
    c = tiny([0], [1], [200], 2)
    p = LIFParams(n_trials=1, duration_ms=300)
    res = BrainSimulator(c, p, seed=0).run([Stimulus("s", [0], 50)])
    t0 = res.times_ms[res.neurons == 0]
    t1 = res.times_ms[res.neurons == 1]
    assert len(t1) > 0
    lag = t1[0] - t0[0]
    assert p.delay_ms < lag < p.delay_ms + 6.0


def test_inhibition_suppresses():
    # 0 excites 2, 1 inhibits 2
    c = tiny([0, 1], [2, 2], [60, -200], 3)
    p = LIFParams(n_trials=4, duration_ms=1000)
    exc_only = BrainSimulator(c, p, seed=0).run([Stimulus("e", [0], 150)]).rates()[2]
    both = BrainSimulator(c, p, seed=0).run([Stimulus("e", [0], 150), Stimulus("i", [1], 150)]).rates()[2]
    assert exc_only > 10 and both < exc_only / 3


def test_silencing_removes_downstream_activity():
    c = tiny([0], [1], [200], 2)
    p = LIFParams(n_trials=2, duration_ms=500)
    res = BrainSimulator(c, p, seed=0).run([Stimulus("s", [0], 100)], silence=[0])
    assert res.rates()[1] == 0


def test_refractory_inputs_are_ignored():
    # Neuron 1 gets a huge kick from 0 and a second one 1 ms later (inside its 2.2 ms
    # refractory window). It must spike only once per pair.
    c = tiny([0, 2], [1, 1], [200, 200], 3)
    sim = BrainSimulator(c, LIFParams(n_trials=1, duration_ms=50), seed=0)
    res = sim.run([Stimulus("a", [0], 20), Stimulus("b", [2], 20)])
    t1 = np.sort(res.times_ms[res.neurons == 1])
    assert np.all(np.diff(t1) >= 2.2 - 1e-6)
